"""Generate answers with a vLLM model. Input: a retrieval file from retrieve.py (or --oracle to use the
gold passage). Same prompt as the paper (src/generation.py RAG_PROMPT_TEMPLATE). Output rows match the
v2 result format so src/v2_analysis.py-style analysis works unchanged (faithfulness_label is added by
vllm_judge.py).

  python3 src/h200/vllm_generate.py --retrieval results/h200/retr/quran_gold_mpnet.jsonl \
      --model Qwen/Qwen2.5-7B-Instruct --temperature 0 --seed 1 --out results/h200/gen/quran_qwen7b_mpnet.jsonl
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from generation import RAG_PROMPT_TEMPLATE

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--quant", default=None, help="e.g. fp8 for 70B+ models")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.retrieval, encoding="utf-8")]
    from vllm import LLM, SamplingParams
    llm = LLM(a.model, dtype="bfloat16", quantization=a.quant, tensor_parallel_size=a.tp,
              max_model_len=a.max_model_len, seed=a.seed, trust_remote_code=True, gpu_memory_utilization=0.90)
    sp = SamplingParams(temperature=a.temperature, top_p=a.top_p, max_tokens=a.max_tokens, seed=a.seed)
    msgs, passages = [], []
    for r in rows:
        p = r["gold_passage"] if a.oracle else (r["retrieved"][0] if r["retrieved"] else r["gold_passage"])
        passages.append(p)
        msgs.append([{"role": "user", "content": RAG_PROMPT_TEMPLATE.format(passage=p, question=r["question"])}])
    try:
        outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(msgs, sp)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, p, o in zip(rows, passages, outs):
            f.write(json.dumps({"id": r["id"], "language": r["language"], "model_name": a.model,
                "question": r["question"], "generated_answer": o.outputs[0].text.strip(), "passage": p,
                "gold_answer": r["gold_answer"], "retrieval_hit": r["retrieval_hit"], "top1_hit": r["top1_hit"],
                "oracle": a.oracle, "temperature": a.temperature, "seed": a.seed,
                "index_size": r.get("index_size"), "retriever": r.get("retriever")}, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "->", a.out)

if __name__ == "__main__":
    main()
