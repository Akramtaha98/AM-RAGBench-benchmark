"""Build blind annotation sheets to calibrate the 72B LLM judge against human labels.

Population: the baseline condition (gold-only MPNet top-1, greedy decoding) of the five generators
(results/h200/judged/{domain}_{model}_goldmpnet_full.jsonl), 5 x 1,140 = 5,700 generations.
Sampling: stratified by domain x language x judge label (the 72B judge's 5-way label).
  per domain-language cell (4 cells): SUPPORTED 30, PARTIALLY_SUPPORTED 20, UNSUPPORTED 15, ABSTAINED 15,
  CONTRADICTED = all available (the judge almost never uses it). Total about 360 items.
Annotators are BLIND to the judge's labels and to the generator. Stratum sizes are stored in the key file so
that rates can be re-weighted to the full population (calibrate_judge.py).
Overlap: --overlap N items per language are flagged for a second annotator (inter-annotator agreement).

Outputs (results/human_eval/):
  annotation_ar.xlsx, annotation_ms.xlsx : what annotators see + dropdown label columns
  key_private.csv                        : item_id -> judge labels, generator, hit, stratum sizes (DO NOT SHARE)
Usage: python3 src/human_eval/make_sheets.py [--per-cell 30 20 15 15] [--overlap 60] [--seed 7]
"""
import argparse, csv, glob, json, os, random
from collections import defaultdict
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct", "Llama-3.1-8B-Instruct", "gemma-2-9b-it", "Sailor2-8B-Chat"]
FAITH = ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "ABSTAINED"]
GUIDE = """ANNOTATION GUIDELINES (read fully before starting)

Task. For each row you see a QUESTION, the PASSAGE the model was given, and the MODEL ANSWER. You give two labels.
Work blind: you do not see any automatic label, and the model's name is hidden. Do not look anything up.

LABEL 1: faithfulness (use only the PASSAGE; ignore whether the answer is true in the real world; ignore the hidden GOLD column until Label 1 is done)
  SUPPORTED            Every claim in the answer is stated in (or directly follows from) the passage. Rewording is fine.
  PARTIALLY_SUPPORTED  The core claim is in the passage, but the answer also adds some detail that is not in the passage.
  UNSUPPORTED          The answer's main claim is not in the passage (invented, from outside knowledge, or off-topic).
  CONTRADICTED         The answer states something that the passage directly contradicts.
  ABSTAINED            The model declines to answer or says it does not know (in any language), with no substantive claim.
Rules. (1) If the answer both abstains and then guesses, label the guess (SUPPORTED/PARTIALLY/UNSUPPORTED). (2) If the answer is
correct in the real world but not in the passage, it is NOT SUPPORTED. (3) Quran: judge against the given verse only, not tafsir
or other verses. (4) Language of the answer does not matter. (5) When torn between two labels, pick the stricter one
(order from loose to strict: SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED) and write why in NOTES.

LABEL 2: correctness (expand the hidden GOLD ANSWER column, columns grouped to the right of the answer)
  CORRECT     The answer conveys the same answer as the gold answer (paraphrase, translation, harmless extra detail are fine).
  INCORRECT   The answer differs from, or conflicts with, the gold answer.
  ABSTAINED   The model declined to answer.
Rule. If the gold answer is itself wrong or the question cannot be answered from the passage, still compare with the gold answer and
write 'BAD_ITEM' in NOTES.

Practicalities. Label in the dropdown cells (green). Do not edit other columns. Rows are in random order. Take breaks: 20 to 25 items
per 30 minutes is typical. Do the sheet in at most 3 sittings and do NOT revisit finished rows. 'OVERLAP=yes' rows are also labelled by
another annotator, so treat them like all others. Record time spent in the TIME sheet.
"""

def load(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cell", nargs=4, type=int, default=[30, 20, 15, 15], help="SUPPORTED PARTIALLY UNSUPPORTED ABSTAINED")
    ap.add_argument("--overlap", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    pool = defaultdict(list)           # (domain, lang, label) -> list of rows
    N = defaultdict(int)               # (domain, lang, label) -> population size
    for dom in ("quran", "wiki"):
        for m in MODELS:
            for r in load(os.path.join(ROOT, "results", "h200", "judged", f"{dom}_{m}_goldmpnet_full.jsonl")):
                r["_dom"], r["_model"] = dom, m
                pool[(dom, r["language"], r["faithfulness_label"])].append(r); N[(dom, r["language"], r["faithfulness_label"])] += 1
    quota = dict(zip(["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "ABSTAINED"], a.per_cell))
    chosen = []
    for dom in ("quran", "wiki"):
        for lang in ("ar", "ms"):
            for lab in FAITH:
                rows = pool[(dom, lang, lab)]; k = quota.get(lab, len(rows))
                for r in rng.sample(rows, min(k, len(rows))): chosen.append(r)
    out = os.path.join(ROOT, "results", "human_eval"); os.makedirs(out, exist_ok=True)
    key = []
    by_lang = defaultdict(list)
    for r in chosen: by_lang[r["language"]].append(r)
    for lang, rows in by_lang.items():
        rng.shuffle(rows)
        ov = set(id(x) for x in rng.sample(rows, min(a.overlap, len(rows))))
        wb = Workbook(); ws = wb.active; ws.title = "ANNOTATE"
        hdr = ["ITEM", "QUESTION", "PASSAGE (the only evidence)", "MODEL ANSWER", "LABEL 1: faithfulness", "LABEL 2: correctness", "NOTES", "OVERLAP", "GOLD ANSWER (expand after Label 1)"]
        ws.append(hdr)
        for c in ws[1]: c.font = Font(bold=True); c.alignment = Alignment(wrap_text=True, vertical="top")
        for i, r in enumerate(rows, 1):
            iid = f"{lang.upper()}-{i:03d}"
            ws.append([iid, r["question"], r["passage"], r["generated_answer"], None, None, None, "yes" if id(r) in ov else "no", r["gold_answer"]])
            key.append({"item_id": iid, "language": lang, "domain": r["_dom"], "generator": r["_model"], "id": r["id"],
                        "top1_hit": r["top1_hit"], "judge72_faith": r["faithfulness_label"], "judge72_threeway": r["threeway_label"],
                        "judge72_correct": r["correctness_label"], "overlap": "yes" if id(r) in ov else "no",
                        "stratum_N": N[(r["_dom"], r["language"], r["faithfulness_label"])],
                        "stratum_n": sum(1 for x in rows if x["_dom"] == r["_dom"] and x["faithfulness_label"] == r["faithfulness_label"])})
        n = len(rows)
        dv1 = DataValidation(type="list", formula1='"' + ",".join(FAITH) + '"', allow_blank=True); dv2 = DataValidation(type="list", formula1='"CORRECT,INCORRECT,ABSTAINED"', allow_blank=True)
        ws.add_data_validation(dv1); ws.add_data_validation(dv2); dv1.add(f"E2:E{n+1}"); dv2.add(f"F2:F{n+1}")
        green = PatternFill("solid", fgColor="E2F0D9")
        for row in ws.iter_rows(min_row=2, max_row=n+1):
            for c in row: c.alignment = Alignment(wrap_text=True, vertical="top")
            for c in (row[4], row[5], row[6]): c.fill = green
        for col, w in zip("ABCDEFGHI", (9, 40, 60, 50, 22, 18, 25, 9, 30)): ws.column_dimensions[col].width = w
        ws.column_dimensions["I"].hidden = True; ws.freeze_panes = "B2"
        if lang == "ar":
            for row in ws.iter_rows(min_row=2, max_row=n+1, min_col=2, max_col=4):
                for c in row: c.alignment = Alignment(wrap_text=True, vertical="top", horizontal="right", readingOrder=2)
        g = wb.create_sheet("GUIDELINES"); [g.append([line]) for line in GUIDE.split("\n")]; g.column_dimensions["A"].width = 150
        t = wb.create_sheet("TIME"); t.append(["sitting", "start", "end", "items done"])
        wb.save(os.path.join(out, f"annotation_{lang}.xlsx")); print(lang, n, "items; overlap", len(ov))
    with open(os.path.join(out, "key_private.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    print("wrote", len(key), "key rows; stratum sizes (population N):")
    for k, v in sorted(N.items()): print("  ", k, v)

if __name__ == "__main__":
    main()
