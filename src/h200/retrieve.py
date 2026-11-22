"""Retrieval for the H200 experiments. Writes one JSONL row per benchmark record with the top-k
retrieved passages, so generation (vllm_generate.py) is decoupled from retrieval.

Two index modes, per language and domain:
  gold  : the benchmark's own unique gold passages (as in the paper's original experiments)
  full  : gold passages PLUS a distractor pool (the full Quran: every verse; Wikipedia: extra paragraphs)

Retrievers: bm25 | any sentence-transformers model (e.g. paraphrase-multilingual-mpnet-base-v2,
intfloat/multilingual-e5-large, BAAI/bge-m3). Optional cross-encoder rerank of the top-N.

Example (full Quran + Wikipedia pool, E5-large, rerank):
  python3 src/h200/retrieve.py --data data/verified/quran_verified.jsonl --domain quran \
     --pool-file quran_passages.jsonl --retriever intfloat/multilingual-e5-large \
     --rerank BAAI/bge-reranker-v2-m3 --out results/h200/retr/quran_full_e5large_rerank.jsonl
"""
import argparse, json, os, re, sys
from collections import defaultdict
import numpy as np

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def tok(t): return re.findall(r"\w+", t.lower())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--domain", choices=["quran", "wiki"], required=True)
    ap.add_argument("--pool-file", nargs="*", default=[],
                    help="distractor passages (jsonl with 'language' and 'text'); omit for gold-only index")
    ap.add_argument("--retriever", default="bm25")
    ap.add_argument("--rerank", default=None)
    ap.add_argument("--rerank-n", type=int, default=20)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = load(a.data)
    pool = defaultdict(dict)  # lang -> {text: text}
    for r in recs: pool[r["language"]][r["gold_passage_text"].strip()] = r["gold_passage_text"]
    gold_texts = {l: set(v) for l, v in pool.items()}
    for pf in a.pool_file:
        for p in load(pf):
            t = p["text"].strip()
            if t and p["language"] in pool: pool[p["language"]].setdefault(t, t)
    sizes = {l: len(v) for l, v in pool.items()}
    print("index sizes:", sizes, "(gold:", {l: len(v) for l, v in gold_texts.items()}, ")")
    ce = None
    if a.rerank:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(a.rerank, max_length=512, device="cuda")
    out = []
    for lang in pool:
        texts = list(pool[lang]); qs = [r for r in recs if r["language"] == lang]
        questions = [r["question"] for r in qs]
        if a.retriever == "bm25":
            from rank_bm25 import BM25Okapi
            bm = BM25Okapi([tok(t) for t in texts])
            top = [np.argsort(-bm.get_scores(tok(q)))[:max(a.rerank_n, a.k)] for q in questions]
        else:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(a.retriever, device="cuda"); m.half()
            e5 = "e5" in a.retriever.lower()
            P = m.encode([("passage: " if e5 else "") + t for t in texts], batch_size=a.batch,
                         normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=True)
            Q = m.encode([("query: " if e5 else "") + q for q in questions], batch_size=a.batch,
                         normalize_embeddings=True, convert_to_tensor=True)
            import torch
            sc = Q @ P.T
            top = torch.topk(sc, min(max(a.rerank_n, a.k), len(texts)), dim=1).indices.cpu().numpy()
        for r, q, idx in zip(qs, questions, top):
            cand = [texts[i] for i in idx]
            if ce is not None:
                s = ce.predict([(q, c) for c in cand], batch_size=64)
                cand = [c for _, c in sorted(zip(-np.array(s), cand), key=lambda x: x[0])]
            cand = cand[:a.k]
            gold = r["gold_passage_text"].strip()
            out.append({"id": r["id"], "language": lang, "question": q, "gold_answer": r["gold_answer"],
                        "gold_passage": r["gold_passage_text"], "retrieved": cand,
                        "top1_hit": bool(cand) and cand[0].strip() == gold,
                        "retrieval_hit": any(c.strip() == gold for c in cand),
                        "index_size": sizes[lang], "retriever": a.retriever + ("+" + a.rerank if a.rerank else "")})
    order = {r["id"]: i for i, r in enumerate(recs)}
    out.sort(key=lambda x: order[x["id"]])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(out)
    print(f"recall@1 {sum(r['top1_hit'] for r in out)/n:.3f}  recall@{a.k} {sum(r['retrieval_hit'] for r in out)/n:.3f}  -> {a.out}")

if __name__ == "__main__":
    main()
