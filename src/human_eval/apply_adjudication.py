"""Apply the filled disagreements_44.xlsx. Never overwrites original files.
Writes data/verified/adjudicated/<file>.jsonl (KEEP unchanged, EDIT replaced, REMOVE dropped, only for rows with IN RELEASE=yes),
results/human_eval/adjudication_summary.json and data/verified/adjudicated/CHANGELOG.csv.
Usage: python3 src/human_eval/apply_adjudication.py [path/to/disagreements_44.xlsx] [--out dir]"""
import csv, json, os, sys
from collections import Counter
from openpyxl import load_workbook
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
src = args[0] if args else os.path.join(ROOT, "results", "human_eval", "disagreements_44.xlsx")
OUTD = sys.argv[sys.argv.index("--out")+1] if "--out" in sys.argv else os.path.join(ROOT, "data", "verified", "adjudicated")
ws = load_workbook(src, data_only=True)["ADJUDICATE"]
H = [c.value for c in ws[1]]; rows = [dict(zip(H, r)) for r in ws.iter_rows(min_row=2, values_only=True) if r[0]]
bad = [r["ID"] for r in rows if r["FINAL ADJUDICATION"] not in ("KEEP_AS_IS", "EDIT", "REMOVE")]
assert not bad, f"rows without a valid FINAL ADJUDICATION: {bad}"
for r in rows:
    if r["FINAL ADJUDICATION"] == "EDIT": assert r["CORRECTED QUESTION (if EDIT)"] or r["CORRECTED GOLD ANSWER (if EDIT)"], f"EDIT without correction: {r['ID']}"
dec = {r["ID"]: r for r in rows if r["IN RELEASE"] == "yes"}
os.makedirs(OUTD, exist_ok=True); log = []; counts = Counter()
split = os.path.join(ROOT, "data", "verified", "licence_split")
for fn in sorted(os.listdir(split)):
    out = []
    for l in open(os.path.join(split, fn), encoding="utf-8"):
        rec = json.loads(l); d = dec.get(rec["id"])
        if not d: out.append(rec); continue
        a = d["FINAL ADJUDICATION"]; counts[a] += 1
        if a == "REMOVE": log.append([rec["id"], "REMOVE", rec["question"], "", rec["gold_answer"], ""]); continue
        if a == "EDIT":
            q = d["CORRECTED QUESTION (if EDIT)"] or rec["question"]; g = d["CORRECTED GOLD ANSWER (if EDIT)"] or rec["gold_answer"]
            log.append([rec["id"], "EDIT", rec["question"], q, rec["gold_answer"], g]); rec["question"], rec["gold_answer"] = q, g; rec["verification_pass"] = "adjudicated"
        out.append(rec)
    with open(os.path.join(OUTD, fn), "w", encoding="utf-8") as f:
        for rec in out: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
with open(os.path.join(OUTD, "CHANGELOG.csv"), "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f); w.writerow(["id", "action", "old_question", "new_question", "old_gold_answer", "new_gold_answer"]); w.writerows(log)
side = Counter()
for r in rows:
    p = "approve" if r["PRIMARY DECISION"] in ("approve", "edit") else "reject"; s = {"valid": "approve", "edit": "edit", "invalid": "reject"}[r["SECOND ANNOTATOR DECISION"]]
    f_ = {"KEEP_AS_IS": "approve", "EDIT": "edit", "REMOVE": "reject"}[r["FINAL ADJUDICATION"]]
    side["agrees_with_primary"] += (f_ == p); side["agrees_with_second"] += (f_ == s)
summ = {"n_rows": len(rows), "released_rows": len(dec), "final_decision_counts_released": dict(counts), "all_rows_final": dict(Counter(r["FINAL ADJUDICATION"] for r in rows)),
        "final_agrees_with_primary": side["agrees_with_primary"], "final_agrees_with_second": side["agrees_with_second"]}
json.dump(summ, open(os.path.join(ROOT, "results", "human_eval", "adjudication_summary.json"), "w"), indent=1); print(json.dumps(summ, indent=1))
print("released files rewritten in", OUTD, "- re-run src/h200_analysis.py with these ids excluded, or use the sensitivity script, before updating the paper")
