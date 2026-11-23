"""Run every generation job for ONE model inside a single vLLM process (model is loaded once).
Same prompt / sampling / output format as vllm_generate.py. Finished outputs are skipped, so re-running resumes.

  python3 src/h200/gen_batch.py --model Qwen/Qwen2.5-7B-Instruct --util 0.28
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generation import RAG_PROMPT_TEMPLATE

R = "results/h200"
GOLD = "paraphrase-multilingual-mpnet-base-v2"


def done(p):
    return os.path.exists(p) and os.path.getsize(p) > 0


def jobs(model):
    n, out = os.path.basename(model), []
    for dom in ("quran", "wiki"):
        g = f"{R}/retr/{dom}_gold_{GOLD}.jsonl"
        f = f"{R}/retr/{dom}_full_multilingual-e5-large.jsonl"
        out += [(f"{R}/gen/{dom}_{n}_goldmpnet.jsonl", g, False, 0.0, 1),
                (f"{R}/gen/{dom}_{n}_fulle5.jsonl", f, False, 0.0, 1),
                (f"{R}/gen/{dom}_{n}_oracle.jsonl", g, True, 0.0, 1)]
        if n == "Qwen2.5-7B-Instruct":  # sampled repeats for the variation estimate
            out += [(f"{R}/gen/{dom}_{n}_goldmpnet_t07_s{s}.jsonl", g, False, 0.7, s) for s in (1, 2, 3)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--util", type=float, default=0.90, help="fraction of GPU memory this process may use")
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=400)
    a = ap.parse_args()
    todo = [j for j in jobs(a.model) if not done(j[0])]
    if not todo:
        print("nothing to do for", a.model)
        return
    for _, retr, *_ in todo:  # retrieval files are produced earlier in the pipeline; wait if still missing
        while not done(retr):
            print("waiting for", retr, flush=True)
            time.sleep(20)
    from vllm import LLM, SamplingParams
    def load(mml):
        return LLM(a.model, dtype="bfloat16", max_model_len=mml, trust_remote_code=True,
                   gpu_memory_utilization=a.util, seed=1)
    try:
        llm = load(a.max_model_len)
    except ValueError as e:  # e.g. models whose native context is shorter than 4096
        print("retrying with max_model_len=2048:", str(e)[:200], flush=True)
        llm = load(2048)
    for out, retr, oracle, temp, seed in todo:
        rows = [json.loads(l) for l in open(retr, encoding="utf-8")]
        msgs, passages = [], []
        for r in rows:
            p = r["gold_passage"] if oracle else (r["retrieved"][0] if r["retrieved"] else r["gold_passage"])
            passages.append(p)
            msgs.append([{"role": "user", "content": RAG_PROMPT_TEMPLATE.format(passage=p, question=r["question"])}])
        sp = SamplingParams(temperature=temp, top_p=1.0, max_tokens=a.max_tokens, seed=seed)
        try:
            outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
        except TypeError:
            outs = llm.chat(msgs, sp)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        tmp = out + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r, p, o in zip(rows, passages, outs):
                f.write(json.dumps({"id": r["id"], "language": r["language"], "model_name": a.model,
                    "question": r["question"], "generated_answer": o.outputs[0].text.strip(), "passage": p,
                    "gold_answer": r["gold_answer"], "retrieval_hit": r["retrieval_hit"], "top1_hit": r["top1_hit"],
                    "oracle": oracle, "temperature": temp, "seed": seed,
                    "index_size": r.get("index_size"), "retriever": r.get("retriever")}, ensure_ascii=False) + "\n")
        os.replace(tmp, out)  # atomic: a half-written file is never mistaken for a finished one
        print("wrote", len(rows), "->", out, flush=True)


if __name__ == "__main__":
    main()
