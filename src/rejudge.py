"""Re-judge EXISTING generations with a different judge model (same answers, new labels).
Usage: python3 src/rejudge.py results/quran_results.jsonl --judge-model llama3.1:8b --out results/quran_rejudged_llama.jsonl
Prints raw agreement and Cohen's kappa between the original and the new judge."""
import argparse, json, sys
from collections import Counter
sys.path.insert(0, "src")
from faithfulness import OllamaJudge

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("--judge-model", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
rows = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
j = OllamaJudge(model_name=a.judge_model)
for i, r in enumerate(rows, 1):
    r["judge2_label"] = j.label(r["question"], r["passage"], r["generated_answer"])
    r["judge2_model"] = a.judge_model
    if i % 50 == 0: print(i, "/", len(rows), flush=True)
with open(a.out, "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
pairs = [(r["faithfulness_label"], r["judge2_label"]) for r in rows]
n = len(pairs); agree = sum(x == y for x, y in pairs) / n
c1 = Counter(x for x, _ in pairs); c2 = Counter(y for _, y in pairs)
pe = sum(c1[k] * c2[k] for k in c1) / n**2
print(f"n={n} agreement={agree:.3f} kappa={(agree - pe) / (1 - pe):.3f}")
