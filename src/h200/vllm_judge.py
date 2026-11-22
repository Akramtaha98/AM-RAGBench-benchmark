"""LLM-judge faithfulness labelling (and optional answer-correctness labelling) with vLLM.

Uses constrained decoding (guided choice), so the output is always one of the labels and there are no
parse failures. Faithfulness prompt = the paper's LLM_JUDGE_PROMPT_TEMPLATE, unchanged.
Modes:
  --mode faithfulness : labels SUPPORTED / PARTIALLY_SUPPORTED / UNSUPPORTED / CONTRADICTED / ABSTAINED
  --mode correctness  : labels CORRECT / INCORRECT / ABSTAINED against the gold answer
  --mode three-way    : SUPPORTED / NOT_SUPPORTED / ABSTAINED (collapsed scheme, review item B7)

  python3 src/h200/vllm_judge.py --in results/h200/gen/x.jsonl --judge Qwen/Qwen2.5-72B-Instruct --quant fp8 \
      --mode faithfulness --out results/h200/judged/x_judge72b.jsonl
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from faithfulness import LLM_JUDGE_PROMPT_TEMPLATE

CORRECT_PROMPT = """You are grading an answer against a reference answer.

Question: {question}
Reference (gold) answer: {gold}
Model's answer: {answer}

The question may be in Arabic or Malay. Decide whether the model's answer conveys the same answer as the
reference (a paraphrase, translation, or extra harmless detail is fine). Choose exactly one of:
CORRECT, INCORRECT, ABSTAINED
(ABSTAINED = the model said it does not know.) Respond with only the single label word."""

THREE_WAY = LLM_JUDGE_PROMPT_TEMPLATE.split("Classify")[0] + """Classify the model's answer into exactly one of:
SUPPORTED, NOT_SUPPORTED, ABSTAINED

- SUPPORTED: the answer's claims are all stated in the passage (extra wording that adds no new fact is fine).
- NOT_SUPPORTED: the answer contains any claim that is absent from, or contradicts, the passage.
- ABSTAINED: the model declined to answer / said it doesn't know.

Respond with only the single label word, nothing else."""

LABELS = {"faithfulness": ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "ABSTAINED"],
          "correctness": ["CORRECT", "INCORRECT", "ABSTAINED"],
          "three-way": ["SUPPORTED", "NOT_SUPPORTED", "ABSTAINED"]}
FIELD = {"faithfulness": "faithfulness_label", "correctness": "correctness_label", "three-way": "threeway_label"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--mode", choices=list(LABELS), default="faithfulness")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    from vllm import LLM, SamplingParams
    try:
        from vllm.sampling_params import GuidedDecodingParams
        sp = SamplingParams(temperature=0.0, max_tokens=12, guided_decoding=GuidedDecodingParams(choice=LABELS[a.mode]))
    except Exception:   # newer vLLM API
        from vllm.sampling_params import StructuredOutputsParams
        sp = SamplingParams(temperature=0.0, max_tokens=12, structured_outputs=StructuredOutputsParams(choice=LABELS[a.mode]))
    llm = LLM(a.judge, dtype="bfloat16", quantization=a.quant, tensor_parallel_size=a.tp,
              max_model_len=a.max_model_len, trust_remote_code=True, gpu_memory_utilization=0.90)
    prompts = []
    for r in rows:
        if a.mode == "faithfulness":
            t = LLM_JUDGE_PROMPT_TEMPLATE.format(passage=r["passage"], question=r["question"], answer=r["generated_answer"])
        elif a.mode == "three-way":
            t = THREE_WAY.format(passage=r["passage"], question=r["question"], answer=r["generated_answer"])
        else:
            t = CORRECT_PROMPT.format(question=r["question"], gold=r["gold_answer"], answer=r["generated_answer"])
        prompts.append([{"role": "user", "content": t}])
    try:
        outs = llm.chat(prompts, sp, chat_template_kwargs={"enable_thinking": False})
    except TypeError:
        outs = llm.chat(prompts, sp)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r, o in zip(rows, outs):
            r[FIELD[a.mode]] = o.outputs[0].text.strip().upper()
            r["judge_model"] = a.judge
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", len(rows), "->", a.out)

if __name__ == "__main__":
    main()
