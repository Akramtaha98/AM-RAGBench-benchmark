"""Build the ready-to-fill sheets for the open review items. Output in results/human_eval/:
  disagreements_44.xlsx            adjudication of the 44 unresolved second-annotator disagreements
  quran_specialist_review.xlsx     tafsir/specialist verification of the 369 released Quran records
  miss_audit.xlsx                  audit of full-corpus retrieval misses for alternative supporting passages
  overlap_pass2.xlsx               blind second pass (calibration overlap rows + full-corpus misses), for annotator 2
  overlap_fullcorpus_pass1.xlsx    the full-corpus rows only, for the original annotator (so both annotators label them)
  overlap_pass2_key_private.csv    item -> source, judge labels (DO NOT SHARE with annotators)
Usage: python3 src/human_eval/make_review_sheets.py [--seed 13]"""
import argparse, csv, json, os, random, re
from collections import defaultdict
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
OUT = os.path.join(ROOT, "results", "human_eval")
MODELS = ["Qwen2.5-7B-Instruct", "Qwen2.5-32B-Instruct", "Llama-3.1-8B-Instruct", "gemma-2-9b-it", "Sailor2-8B-Chat"]
FAITH = ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "ABSTAINED"]
GREEN = PatternFill("solid", fgColor="E2F0D9")
HDR = Font(bold=True)
def L(p): return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
def wrap(ws, n, rtl_cols=(), widths=None, hidden=()):
    for c in ws[1]: c.font = HDR; c.alignment = Alignment(wrap_text=True, vertical="top")
    for row in ws.iter_rows(min_row=2, max_row=n+1):
        for c in row: c.alignment = Alignment(wrap_text=True, vertical="top")
    for i, w in enumerate(widths or [], 1): ws.column_dimensions[chr(64+i)].width = w
    for h in hidden: ws.column_dimensions[h].hidden = True
    ws.freeze_panes = "B2"
def dropdown(ws, col, n, options):
    dv = DataValidation(type="list", formula1='"' + ",".join(options) + '"', allow_blank=True); ws.add_data_validation(dv); dv.add(f"{col}2:{col}{n+1}")
def fill(ws, cols, n):
    for row in ws.iter_rows(min_row=2, max_row=n+1):
        for c in row:
            if c.column_letter in cols: c.fill = GREEN
def text_sheet(wb, title, text):
    g = wb.create_sheet(title); [g.append([x]) for x in text.split("\n")]; g.column_dimensions["A"].width = 150

# ---------------------------------------------------------------- 1. disagreements
def disagreements():
    d = list(csv.DictReader(open(os.path.join(ROOT, "review", "reannotation", "disagreements_for_adjudication.csv"), encoding="utf-8-sig")))
    rel = {r["id"]: r["in_released_benchmark"] for r in csv.DictReader(open(os.path.join(ROOT, "data", "verified", "unadjudicated_disagreements.csv"), encoding="utf-8"))}
    flags = {r["id"] for r in csv.DictReader(open(os.path.join(ROOT, "Computation work", "answerability_flags.csv"), encoding="utf-8-sig"))}
    wb = Workbook(); ws = wb.active; ws.title = "ADJUDICATE"
    ws.append(["ID", "LANGUAGE", "DOMAIN", "QUESTION", "PASSAGE", "GOLD ANSWER", "PRIMARY DECISION", "SECOND ANNOTATOR DECISION", "SECOND ANNOTATOR NOTES",
               "IN RELEASE", "ALSO IN ANSWERABILITY FLAGS", "FINAL ADJUDICATION", "CORRECTED QUESTION (if EDIT)", "CORRECTED GOLD ANSWER (if EDIT)", "ADJUDICATOR NOTES", "ADJUDICATOR"])
    for r in d:
        ws.append([r["id"], r["language"], r["domain"], r["question"], r["passage"], r["gold_answer"], r["original"], r["second"], r["second_notes"],
                   rel.get(r["id"], ""), "yes" if r["id"] in flags else "no", None, None, None, None, None])
    n = len(d); wrap(ws, n, widths=[22, 9, 8, 45, 60, 30, 12, 14, 35, 9, 12, 20, 40, 30, 35, 12]); fill(ws, set("LMNOP"), n)
    dropdown(ws, "L", n, ["KEEP_AS_IS", "EDIT", "REMOVE"])
    text_sheet(wb, "GUIDELINES", """ADJUDICATION GUIDELINES
Decide for each record whether the released text (question + gold answer) is correct as it stands.
  KEEP_AS_IS  The question is clear, the gold answer is correct, and both are supported by the passage.
  EDIT        The record is salvageable: write the corrected question and/or gold answer in the green columns.
  REMOVE      The question cannot be answered from the passage, the gold answer is wrong, or the item is ambiguous beyond repair.
Rules. (1) Judge against the PASSAGE only. (2) Decide independently of which annotator you personally trust; read both decisions and notes, then the passage.
(3) For Quran records, if the answer needs tafsir or another verse, choose REMOVE (or EDIT to a question the verse answers). (4) 'IN RELEASE = no' rows
were already rejected in primary review: record the decision anyway, no change to the release follows.
After filling: python3 src/human_eval/apply_adjudication.py  (writes adjudicated release files and a changelog; no original file is overwritten).""")
    wb.save(os.path.join(OUT, "disagreements_44.xlsx")); print("disagreements_44.xlsx", n)

# ---------------------------------------------------------------- 2. quran specialist
def quran_specialist():
    flags = {r["id"] for r in csv.DictReader(open(os.path.join(ROOT, "Computation work", "answerability_flags.csv"), encoding="utf-8-sig"))}
    dis = {r["id"] for r in csv.DictReader(open(os.path.join(ROOT, "data", "verified", "unadjudicated_disagreements.csv"), encoding="utf-8"))}
    rows = []
    for f in ("quran_ar_CC-BY-4.0.jsonl", "quran_ms_CC-BY-NC-4.0.jsonl"):
        rows += L(os.path.join(ROOT, "data", "verified", "licence_split", f))
    def prio(r): return "HIGH" if (r["id"] in flags or r["id"] in dis) else "normal"
    rows.sort(key=lambda r: (prio(r) != "HIGH", r["language"], r["id"]))
    wb = Workbook(); ws = wb.active; ws.title = "REVIEW"
    ws.append(["ID", "LANGUAGE", "SURAH:AYAH", "PRIORITY", "QUESTION", "PASSAGE (as released)", "GOLD ANSWER", "SPECIALIST VERDICT", "CORRECTED QUESTION / ANSWER (if needed)", "NOTES (tafsir source if used)"])
    for r in rows:
        m = re.match(r"quran-(\d+:\d+(?:-\d+)?)", r["id"]); ws.append([r["id"], r["language"], m.group(1) if m else "", prio(r), r["question"], r["gold_passage_text"], r["gold_answer"], None, None, None])
    n = len(rows); wrap(ws, n, widths=[22, 9, 11, 9, 45, 60, 30, 28, 45, 40]); fill(ws, set("HIJ"), n)
    dropdown(ws, "H", n, ["VALID", "VALID_WITH_EDIT", "INVALID", "TAFSIR_DEPENDENT", "NOT_ANSWERABLE_FROM_PASSAGE"])
    text_sheet(wb, "GUIDELINES", """SPECIALIST (TAFSIR / QURANIC SCIENCES) REVIEW
Verify each question-answer pair against the verse text shown. You are checking the benchmark record, not giving religious guidance.
  VALID                        Question clear, gold answer correct, answerable from this verse text alone.
  VALID_WITH_EDIT              Correct in substance but needs a wording fix: write it in the green column.
  INVALID                      Gold answer wrong, or the question misreads the verse.
  TAFSIR_DEPENDENT             Correct only with exegesis, context, or another verse (not answerable from the verse text alone).
  NOT_ANSWERABLE_FROM_PASSAGE  The verse text does not contain the answer (for example the subject is named in the surrounding verses).
PRIORITY = HIGH rows are the answerability-flagged records and the records with unresolved second-annotator disagreement: review these first (a few hours).
The Malay rows show the Basmeih translation: judge them against the translation as given; note any translation issue in NOTES.
Fill the REVIEWER sheet (name, qualification, date, time spent) so the paper can describe the review accurately.""")
    r = wb.create_sheet("REVIEWER"); [r.append(x) for x in (["Name", None], ["Qualification / institution", None], ["Date(s)", None], ["Hours spent", None], ["Compensated (yes/no)", None], ["Rows reviewed", None]) ]
    r.column_dimensions["A"].width = 30; r.column_dimensions["B"].width = 50
    wb.save(os.path.join(OUT, "quran_specialist_review.xlsx")); print("quran_specialist_review.xlsx", n, "rows;", sum(prio(x) == "HIGH" for x in rows), "high priority")

# ---------------------------------------------------------------- full-corpus miss pool
def fullcorpus_misses():
    pool = defaultdict(list)
    for dom in ("quran", "wiki"):
        for m in MODELS:
            for r in L(os.path.join(ROOT, "results", "h200", "judged", f"{dom}_{m}_fulle5_full.jsonl")):
                if not r["top1_hit"]: r["_dom"], r["_model"] = dom, m; pool[(dom, r["language"])].append(r)
    return pool
def gold_passages():
    g = {}
    for f in os.listdir(os.path.join(ROOT, "data", "verified", "licence_split")):
        for r in L(os.path.join(ROOT, "data", "verified", "licence_split", f)): g[r["id"]] = r["gold_passage_text"]
    return g

def miss_audit(rng, pool, gold):
    wb = Workbook(); ws = wb.active; ws.title = "AUDIT"
    ws.append(["ITEM", "DOMAIN", "LANGUAGE", "QUESTION", "GOLD ANSWER", "GOLD PASSAGE", "RETRIEVED TOP-1 PASSAGE (full-corpus E5)", "ALTERNATIVE SUPPORT", "NEAR-DUPLICATE OF GOLD PASSAGE", "NOTES"])
    key = []; i = 0; per = 40
    for (dom, lang), rows in sorted(pool.items()):
        seen = {}; 
        for r in rows: seen.setdefault(r["id"], r)       # one record per question (retrieval is generator independent)
        uniq = list(seen.values()); cor = [r for r in uniq if r["correctness_label"] == "CORRECT"]; oth = [r for r in uniq if r["correctness_label"] != "CORRECT"]
        pick = rng.sample(cor, min(per//2, len(cor))); pick += rng.sample(oth, min(per-len(pick), len(oth)))
        rng.shuffle(pick)
        for r in pick:
            i += 1; iid = f"MA-{i:03d}"
            ws.append([iid, dom, lang, r["question"], r["gold_answer"], gold.get(r["id"], ""), r["passage"], None, None, None])
            key.append({"item": iid, "id": r["id"], "generator_shown_in_key_only": r["_model"], "judge72_correct": r["correctness_label"], "judge72_faith": r["faithfulness_label"]})
    n = i; wrap(ws, n, widths=[8, 8, 9, 45, 30, 55, 55, 22, 18, 35]); fill(ws, set("HIJ"), n)
    dropdown(ws, "H", n, ["FULL", "PARTIAL", "NONE"]); dropdown(ws, "I", n, ["yes", "no"])
    text_sheet(wb, "GUIDELINES", """RETRIEVAL-MISS AUDIT
The retriever returned a passage that is NOT the designated gold passage. Decide whether it nevertheless supports the gold answer.
  FULL     The retrieved passage by itself contains or directly implies the gold answer (a legitimate alternative).
  PARTIAL  It contains part of the answer or a related claim, not enough alone.
  NONE     It does not support the gold answer.
NEAR-DUPLICATE = the retrieved passage repeats the gold wording (repeated Quranic formulae, duplicated Wikipedia sentences).
Judge from the passages only; do not look things up. Half of the rows are cases where an automatic judge scored a generated answer CORRECT, half are not; this is not shown.
Result use: the share of FULL (and FULL+PARTIAL) gives the fraction of 'misses' that are not real failures; report it with a Wilson interval.""")
    wb.save(os.path.join(OUT, "miss_audit.xlsx"))
    with open(os.path.join(OUT, "miss_audit_key_private.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(key[0])); w.writeheader(); w.writerows(key)
    print("miss_audit.xlsx", n)

# ---------------------------------------------------------------- overlap pass 2
def annotate_sheet(wb, title, rows, first=False):
    ws = wb.active if first else wb.create_sheet(title); ws.title = title
    ws.append(["ITEM", "LANGUAGE", "QUESTION", "PASSAGE (the only evidence)", "MODEL ANSWER", "LABEL 1: faithfulness", "LABEL 2: correctness", "NOTES", "GOLD ANSWER (expand after Label 1)"])
    for r in rows: ws.append(r)
    n = len(rows); wrap(ws, n, widths=[12, 9, 40, 60, 50, 22, 18, 25, 30], hidden=["I"]); fill(ws, set("FGH"), n)
    dropdown(ws, "F", n, FAITH); dropdown(ws, "G", n, ["CORRECT", "INCORRECT", "ABSTAINED"])
    for row in ws.iter_rows(min_row=2, max_row=n+1):
        if row[1].value == "ar":
            for c in row[2:5]: c.alignment = Alignment(wrap_text=True, vertical="top", horizontal="right", readingOrder=2)
    return ws
def guide_text():
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "make_sheets.py"), encoding="utf-8").read()
    m = re.search(r'GUIDE = """(.*?)"""', src, re.S); return m.group(1)
def overlap_pass2(rng, pool):
    key = list(csv.DictReader(open(os.path.join(OUT, "key_private.csv"), encoding="utf-8")))
    A = {}
    for lang in ("ar", "ms"):
        ws = load_workbook(os.path.join(OUT, "blank_templates", f"annotation_{lang}.xlsx"))["ANNOTATE"]
        for r in ws.iter_rows(min_row=2, values_only=True): A[r[0]] = r
    rows2, rows1, keyout = [], [], []; i = 0
    for k in key:
        if k["overlap"] != "yes": continue
        r = A[k["item_id"]]; i += 1; iid = f"P2-{k['language'].upper()}-{i:03d}"
        rows2.append([iid, k["language"], r[1], r[2], r[3], None, None, None, r[8]]); keyout.append({"item": iid, "source": "calibration_overlap", "orig_item": k["item_id"], "language": k["language"], "judge72_faith": k["judge72_faith"], "judge72_correct": k["judge72_correct"], "generator": k["generator"], "id": k["id"]})
    quota = {"SUPPORTED": 8, "PARTIALLY_SUPPORTED": 4, "UNSUPPORTED": 4, "ABSTAINED": 4}
    for (dom, lang), rows in sorted(pool.items()):
        by = defaultdict(list)
        for r in rows: by[r["faithfulness_label"]].append(r)
        pick = []
        for lab, q in quota.items(): pick += rng.sample(by[lab], min(q, len(by[lab])))
        for r in pick:
            i += 1; iid = f"P2-{lang.upper()}-{i:03d}"
            row = [iid, lang, r["question"], r["passage"], r["generated_answer"], None, None, None, r["gold_answer"]]
            rows2.append(row); rows1.append(row)
            keyout.append({"item": iid, "source": "fullcorpus_miss", "orig_item": "", "language": lang, "judge72_faith": r["faithfulness_label"], "judge72_correct": r["correctness_label"], "generator": r["_model"], "id": r["id"]})
    rng.shuffle(rows2); rng.shuffle(rows1)
    for fn, rows, who in (("overlap_pass2.xlsx", rows2, "annotator 2: label ONLY the rows in your language (filter the LANGUAGE column)"),
                          ("overlap_fullcorpus_pass1.xlsx", rows1, "the original annotator: label ONLY the rows in your language")):
        wb = Workbook(); annotate_sheet(wb, "ANNOTATE", rows, first=True); text_sheet(wb, "GUIDELINES", guide_text() + f"\n\nThis file is for {who}.\nDo not discuss items with the other annotator before both files are finished.")
        t = wb.create_sheet("TIME"); t.append(["sitting", "start", "end", "items done"]); wb.save(os.path.join(OUT, fn))
    with open(os.path.join(OUT, "overlap_pass2_key_private.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(keyout[0])); w.writeheader(); w.writerows(keyout)
    print("overlap_pass2.xlsx", len(rows2), "rows (", sum(1 for x in keyout if x["source"] == "calibration_overlap"), "calibration overlap +", len(rows1), "full-corpus misses )")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=13); a = ap.parse_args(); rng = random.Random(a.seed)
    os.makedirs(OUT, exist_ok=True)
    pool = fullcorpus_misses(); gold = gold_passages()
    disagreements(); quran_specialist(); miss_audit(rng, pool, gold); overlap_pass2(rng, pool)
