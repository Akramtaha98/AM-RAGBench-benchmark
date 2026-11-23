"""Judge ALL generation files in one vLLM process (judge model is loaded once).
Same prompts, constrained labels and output files as vllm_judge.py:
  <name>_faith.jsonl  (faithfulness) -> <name>_3way.jsonl (three-way) -> <name>_full.jsonl (correctness)
Finished outputs are skipped, so re-running resumes.

  python3 src/h200/judge_batch.py --judge Qwen/Qwen2.5-72B-Instruct --quant fp8
"""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vllm_judge import LABELS, FIELD, CORRECT_PROMPT, THREE_WAY, LLM_JUDGE_PROMPT_TEMPLATE


def done(p):
    return os.path.exists(p) and os.path.getsize(p) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True)
    ap.add_argument("--quant", default=None)
    ap.add_argument("--util", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--glob", default="results/h200/gen/*.jsonl")
    ap.add_argument("--outdir", default="results/h200/judged")
    a = ap.parse_args()
    files = sorted(glob.glob(a.glob))
    print(len(files), "generation files to judge", flush=True)
    from vllm import LLM, SamplingParams
    def make_sp(mode):
        try:
            from vllm.sampling_params import GuidedDecodingParams
            return SamplingParams(temperature=0.0, max_tokens=12, guided_decoding=GuidedDecodingParams(choice=LABELS[mode]))
        except Exception:  # newer vLLM API
            from vllm.sampling_params import StructuredOutputsParams
            return SamplingParams(temperature=0.0, max_tokens=12, structured_outputs=StructuredOutputsParams(choice=LABELS[mode]))
    sps = {m: make_sp(m) for m in LABELS}
    llm = LLM(a.judge, dtype="bfloat16", quantization=a.quant, tensor_parallel_size=1,
              max_model_len=a.max_model_len, trust_remote_code=True, gpu_memory_utilization=a.util)
    os.makedirs(a.outdir, exist_ok=True)
    for n, F in enumerate(files, 1):
        B = os.path.basename(F)[:-len(".jsonl")]
        chain = [("faithfulness", F, f"{a.outdir}/{B}_faith.jsonl"),
                 ("three-way", f"{a.outdir}/{B}_faith.jsonl", f"{a.outdir}/{B}_3way.jsonl"),
                 ("correctness", f"{a.outdir}/{B}_3way.jsonl", f"{a.outdir}/{B}_full.jsonl")]
        for mode, src, dst in chain:
            if done(dst):
                continue
            rows = [json.loads(l) for l in open(src, encoding="utf-8")]
            prompts = []
            for r in rows:
                if mode == "faithfulness":
                    t = LLM_JUDGE_PROMPT_TEMPLATE.format(passage=r["passage"], question=r["question"], answer=r["generated_answer"])
                elif mode == "three-way":
                    t = THREE_WAY.format(passage=r["passage"], question=r["question"], answer=r["generated_answer"])
                else:
                    t = CORRECT_PROMPT.format(question=r["question"], gold=r["gold_answer"], answer=r["generated_answer"])
                prompts.append([{"role": "user", "content": t}])
            try:
                outs = llm.chat(prompts, sps[mode], chat_template_kwargs={"enable_thinking": False})
            except TypeError:
                outs = llm.chat(prompts, sps[mode])
            tmp = dst + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                for r, o in zip(rows, outs):
                    r[FIELD[mode]] = o.outputs[0].text.strip().upper()
                    r["judge_model"] = a.judge
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            os.replace(tmp, dst)
            print(f"[{n}/{len(files)}] {mode}: wrote {len(rows)} -> {dst}", flush=True)
    print("judging complete", flush=True)


if __name__ == "__main__":
    main()
