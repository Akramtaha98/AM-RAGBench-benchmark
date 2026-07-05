"""
End-to-end demo run: retrieve -> generate -> score -> summarize.

Runs on toy_dataset.jsonl with BM25Retriever + MockGenerator by default, so it
executes anywhere with just `pip install rank_bm25 pandas`. This proves the
pipeline wiring is correct. IT DOES NOT PRODUCE REAL RESULTS — swap in
DenseRetriever and ClaudeGenerator/HFGenerator (see retrieval.py, generation.py)
once you have the real corpus and API access, then re-run.

Usage:
    python run_pipeline.py --retriever bm25 --generator mock
    python run_pipeline.py --retriever dense --generator claude
"""

import argparse
import json
import sys
from collections import defaultdict

from schema import load_jsonl
from retrieval import build_retriever
from generation import build_generator
from faithfulness import score_batch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="demo/toy_dataset.jsonl")
    ap.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    ap.add_argument("--generator", choices=["mock", "claude", "hf", "ollama"], default="mock")
    ap.add_argument("--hf-model", default=None, help="model name if --generator hf")
    ap.add_argument("--ollama-model", default="qwen3.5:2b",
                     help="model name if --generator ollama (free/local, no API key)")
    ap.add_argument("--judge", choices=["heuristic", "llm"], default="heuristic",
                     help="heuristic (default) is a crude rule-based label, fine for "
                          "smoke-testing the harness only. Use --judge llm for anything "
                          "that goes in the paper.")
    ap.add_argument("--judge-model", default="qwen3.5:4b",
                     help="model name if --judge llm (free/local via Ollama). 4b "
                          "recommended over 2b for this role since judgment quality "
                          "matters here, unlike question drafting where every record "
                          "gets human review anyway.")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--out", default="demo/results.jsonl")
    args = ap.parse_args()

    records = load_jsonl(args.data)
    print(f"Loaded {len(records)} records from {args.data}")

    # Index each language's passages separately (retrieval should not cross languages
    # for this benchmark -- the point is testing within-language RAG faithfulness).
    by_lang_passages = defaultdict(dict)
    for r in records:
        by_lang_passages[r.language][r.gold_passage_id] = r.gold_passage_text

    retrievers = {}
    for lang, passage_map in by_lang_passages.items():
        ids = list(passage_map.keys())
        texts = list(passage_map.values())
        retrievers[lang] = build_retriever(args.retriever, ids, texts)
    print(f"Built {args.retriever} retriever for languages: {list(retrievers.keys())}")

    gen_kwargs = {}
    if args.generator == "hf":
        if not args.hf_model:
            sys.exit("--hf-model is required when --generator hf")
        gen_kwargs["model_name"] = args.hf_model
    if args.generator == "ollama":
        gen_kwargs["model_name"] = args.ollama_model
    generator = build_generator(args.generator, **gen_kwargs)
    print(f"Built generator: {args.generator}")

    rows_for_scoring = []
    retrieval_hits = 0
    for r in records:
        retrieved = retrievers[r.language].retrieve(r.question, k=args.k)
        retrieved_ids = [p.passage_id for p in retrieved]
        hit = r.gold_passage_id in retrieved_ids
        retrieval_hits += int(hit)

        # Oracle-retrieval condition: force the gold passage regardless of retriever output.
        # This is what separates "retrieval failure" from "generation failure" (RQ3).
        top_passage = retrieved[0].text if retrieved else r.gold_passage_text

        if args.generator == "mock":
            gen = generator.generate(r.question, top_passage, gold_answer=r.gold_answer)
        else:
            gen = generator.generate(r.question, top_passage)

        rows_for_scoring.append({
            "id": r.id,
            "language": r.language,
            "model_name": gen.model_name,
            "question": r.question,
            "generated_answer": gen.text,
            "passage": top_passage,
            "gold_answer": r.gold_answer,
            "retrieval_hit": hit,
        })

    judge = None
    if args.judge == "llm":
        from faithfulness import OllamaJudge
        judge = OllamaJudge(model_name=args.judge_model)
        print(f"Using LLM judge: ollama/{args.judge_model} "
              f"(this is the real faithfulness scorer -- {len(rows_for_scoring)} judge "
              f"calls will be made, one per record)")
    else:
        print("Using heuristic judge -- fine for a smoke test, NOT for paper numbers "
              "(re-run with --judge llm for a real faithfulness score)")

    faithfulness_results = score_batch(rows_for_scoring, judge=judge)

    with open(args.out, "w", encoding="utf-8") as f:
        for row, fres in zip(rows_for_scoring, faithfulness_results):
            row["faithfulness_label"] = fres.label
            row["lexical_overlap"] = fres.lexical_overlap
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(rows_for_scoring)} scored rows to {args.out}")
    print(f"Retrieval recall@{args.k}: {retrieval_hits}/{len(records)} "
          f"({100 * retrieval_hits / len(records):.1f}%)")

    label_counts = defaultdict(int)
    for fres in faithfulness_results:
        label_counts[fres.label] += 1
    standard_labels = ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "ABSTAINED"]
    heading = ("Faithfulness label distribution (DEMO DATA -- not real results):"
               if args.judge == "heuristic" else
               "Faithfulness label distribution (LLM judge -- still spot-check a human "
               "agreement sample before reporting as final):")
    print(f"\n{heading}")
    for label in standard_labels:
        print(f"  {label:20s} {label_counts.get(label, 0)}")
    # Any label outside the standard five (e.g. JUDGE_PARSE_FAILED) is shown
    # separately rather than silently folded into a real category -- if this is
    # non-zero, check the printed raw-output snippets above to see why the judge
    # model's response didn't parse.
    extras = {k: v for k, v in label_counts.items() if k not in standard_labels}
    if extras:
        print("  --- non-standard labels (investigate before trusting the numbers above) ---")
        for label, count in extras.items():
            print(f"  {label:20s} {count}")


if __name__ == "__main__":
    main()
