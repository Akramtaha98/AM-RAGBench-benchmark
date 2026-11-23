"""Agreement between the original annotator (pass 1) and the second annotator (pass 2), plus judge comparison on the full-corpus rows.
Inputs (results/human_eval/): overlap_pass2.xlsx (annotator 2), annotation_{ar,ms}.xlsx (pass 1, calibration overlap rows),
overlap_fullcorpus_pass1.xlsx (pass 1, full-corpus rows), overlap_pass2_key_private.csv.
Output: pass2_agreement.json and a printed summary. Reports, per source and language: n, raw agreement, Cohen's kappa, Gwet AC1,
for 5-way faithfulness, 3-way faithfulness and correctness, with bootstrap CIs; and for the full-corpus rows the agreement of the 72B judge
with each human pass. Usage: python3 src/human_eval/pass2_agreement.py [--dir results/human_eval]"""
import csv, json, os, sys
from collections import defaultdict
import numpy as np
from openpyxl import load_workbook
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
D = sys.argv[sys.argv.index("--dir")+1] if "--dir" in sys.argv else os.path.join(ROOT, "results", "human_eval")
rng = np.random.default_rng(3)
F5 = ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "ABSTAINED"]
t3 = lambda l: l if l in ("SUPPORTED", "ABSTAINED") else "NOT_SUPPORTED"
def sheet(p, fcol, ccol):
    ws = load_workbook(p, data_only=True)["ANNOTATE"]; out = {}
    for r in ws.iter_rows(min_row=2, values_only=True):
        if r[0]: out[r[0]] = ((r[fcol] or "").strip().upper() or None, (r[ccol] or "").strip().upper() or None)
    return out
def stats(a, b):
    n = len(a); po = np.mean([x == y for x, y in zip(a, b)]); labs = sorted(set(a) | set(b)); K = len(labs)
    pa = {l: np.mean([x == l for x in a]) for l in labs}; pb = {l: np.mean([y == l for y in b]) for l in labs}
    pe = sum(pa[l]*pb[l] for l in labs); kap = (po-pe)/(1-pe) if pe < 1 else float("nan")
    pi = {l: (pa[l]+pb[l])/2 for l in labs}; pe1 = sum(p*(1-p) for p in pi.values())/(K-1) if K > 1 else 1; ac1 = (po-pe1)/(1-pe1) if pe1 < 1 else float("nan")
    return float(po), float(kap), float(ac1)
def ci(a, b, i, B=2000):
    v = []
    for _ in range(B):
        ix = rng.integers(0, len(a), len(a)); v.append(stats([a[j] for j in ix], [b[j] for j in ix])[i])
    return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
key = {r["item"]: r for r in csv.DictReader(open(os.path.join(D, "overlap_pass2_key_private.csv"), encoding="utf-8"))}
p2 = sheet(os.path.join(D, "overlap_pass2.xlsx"), 5, 6)
p1 = {}
for lang in ("ar", "ms"):
    s = sheet(os.path.join(D, f"annotation_{lang}.xlsx"), 4, 5)
    for k, v in key.items():
        if v["source"] == "calibration_overlap" and v["language"] == lang: p1[k] = s.get(v["orig_item"], (None, None))
fc = os.path.join(D, "overlap_fullcorpus_pass1.xlsx")
if os.path.exists(fc):
    for k, v in sheet(fc, 5, 6).items(): p1[k] = v
res = {}
for src in ("calibration_overlap", "fullcorpus_miss"):
    for lang in ("ar", "ms", "all"):
        ids = [k for k, v in key.items() if v["source"] == src and (lang == "all" or v["language"] == lang) and p1.get(k, (None,))[0] and p2.get(k, (None,))[0] and p1[k][1] and p2[k][1]]
        if len(ids) < 5: continue
        d = {"n": len(ids)}
        for nm, f in (("faith5", lambda t: t[0]), ("faith3", lambda t: t3(t[0])), ("correct", lambda t: t[1])):
            a = [f(p1[k]) for k in ids]; b = [f(p2[k]) for k in ids]; po, ka, ac = stats(a, b)
            d[nm] = {"agreement": po, "kappa": ka, "AC1": ac, "AC1_ci": ci(a, b, 2)}
        if src == "fullcorpus_miss":
            for who, pp in (("pass1", p1), ("pass2", p2)):
                d[f"judge_vs_{who}_faith5"] = float(np.mean([pp[k][0] == key[k]["judge72_faith"] for k in ids]))
        res[f"{src}|{lang}"] = d
json.dump(res, open(os.path.join(D, "pass2_agreement.json"), "w"), indent=1)
for k, v in res.items(): print(k, "n=%d" % v["n"], {m: (round(v[m]["agreement"], 3), round(v[m]["kappa"], 2), round(v[m]["AC1"], 2)) for m in ("faith5", "faith3", "correct")})
print("saved", os.path.join(D, "pass2_agreement.json"), "(rows with disagreement should then be adjudicated by a third person)")
