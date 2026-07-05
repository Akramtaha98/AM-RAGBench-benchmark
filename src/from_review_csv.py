"""
Step 6, part 2: merge a completed review spreadsheet (from to_review_csv.py)
back into a benchmark-ready jsonl file, validated against schema.py.

Row handling:
  decision == "approve" -> kept as-is, verification_pass set to --verification-pass
  decision == "edit"    -> question/gold_answer replaced with corrected_question/
                           corrected_answer (falls back to the original if a
                           correction field was left blank), verification_pass
                           set to --verification-pass
  decision == "reject"  -> dropped entirely, NOT written to output
  blank / anything else -> dropped, treated as "not reviewed yet" (not silently
                           included -- re-run once review is actually complete)

Usage:
    python from_review_csv.py review/quran_review.csv --out verified/quran_verified.jsonl \
        --verification-pass primary --annotator-id akram
    python from_review_csv.py review/ms_wiki_review.csv --out verified/ms_wiki_verified.jsonl \
        --verification-pass primary --annotator-id malay_annotator_1
"""

import argparse
import csv
import json
import sys

from schema import BenchmarkRecord


VALID_DECISIONS = {"approve", "edit", "reject"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("review_csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verification-pass", default="primary",
                     choices=["primary", "secondary", "mt_plus_spotcheck"],
                     help="what tier of verification this reviewer represents -- see "
                          "dataset_construction_plan.md Step 4")
    ap.add_argument("--annotator-id", default=None,
                     help="pseudonymous id of the human reviewer, e.g. 'malay_annotator_1' "
                          "-- overwrites the draft's annotator_id to record who actually "
                          "verified it, not who drafted it")
    args = ap.parse_args()

    n_total = 0
    n_approved = 0
    n_edited = 0
    n_rejected = 0
    n_unreviewed = 0
    n_schema_problems = 0
    out_records = []

    with open(args.review_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_total += 1
            decision = (row.get("decision") or "").strip().lower()

            if decision not in VALID_DECISIONS:
                n_unreviewed += 1
                continue
            if decision == "reject":
                n_rejected += 1
                continue

            question = row["question"]
            answer = row["gold_answer"]
            if decision == "edit":
                n_edited += 1
                if (row.get("corrected_question") or "").strip():
                    question = row["corrected_question"].strip()
                if (row.get("corrected_answer") or "").strip():
                    answer = row["corrected_answer"].strip()
            else:
                n_approved += 1

            record = BenchmarkRecord(
                id=row["id"],
                language=row["language"],
                domain=row["domain"],
                question=question,
                gold_passage_id=row.get("gold_passage_id", row["id"]),
                gold_passage_text=row["gold_passage_text"],
                gold_answer=answer,
                source_citation=row.get("source_citation", ""),
                annotator_id=args.annotator_id or row.get("draft_model", "unknown"),
                verification_pass=args.verification_pass,
                notes=(row.get("annotator_notes") or None),
            )
            problems = [p for p in record.validate() if not p.startswith("WARNING")]
            if problems:
                n_schema_problems += 1
                print(f"[{record.id}] SCHEMA PROBLEM, skipped: {problems}")
                continue

            out_records.append(record)

    with open(args.out, "w", encoding="utf-8") as out_f:
        for r in out_records:
            out_f.write(json.dumps(
                {k: v for k, v in r.__dict__.items() if v is not None},
                ensure_ascii=False) + "\n")

    print(f"\n{n_total} rows read from {args.review_csv}:")
    print(f"  {n_approved} approved, {n_edited} edited, {n_rejected} rejected, "
          f"{n_unreviewed} not yet reviewed (skipped)")
    if n_schema_problems:
        print(f"  {n_schema_problems} had schema problems after review and were skipped -- "
              f"see messages above")
    print(f"\nWrote {len(out_records)} benchmark-ready records to {args.out} "
          f"(verification_pass='{args.verification_pass}')")
    if n_unreviewed:
        print(f"\n{n_unreviewed} rows were skipped because 'decision' was blank or invalid -- "
              f"these are NOT in the output yet. Finish reviewing them and re-run this script.")


if __name__ == "__main__":
    main()
