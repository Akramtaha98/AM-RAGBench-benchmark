"""
Step 6, part 1: convert a drafts/*.jsonl file into a spreadsheet an annotator
can actually work in (Excel, Google Sheets, Numbers) -- no coding required.

Flagged records (from lint_drafts.py) are sorted to the top, so your
annotator reviews the highest-risk records first.

The annotator fills in ONE column: `decision`, using exactly one of:
  approve  -- question/answer are correct as-is
  edit     -- question/answer need a fix; put the fix in `corrected_question`
              and/or `corrected_answer` (leave either blank to keep the
              original for that field)
  reject   -- this record should not be in the benchmark at all

Leave `decision` blank to mean "not reviewed yet" -- blank rows are dropped
when you run from_review_csv.py, so nothing half-reviewed silently sneaks in.

Usage:
    python to_review_csv.py drafts/quran_questions_flagged.jsonl --out review/quran_review.csv
    (if you didn't run lint_drafts.py first, this also works directly on a
    *_draft.jsonl file -- it just won't have flags to sort by)
"""

import argparse
import csv
import json


COLUMNS = [
    "id", "language", "domain", "question", "gold_answer", "gold_passage_text",
    "source_citation", "draft_model", "lint_flags",
    "decision", "corrected_question", "corrected_answer", "annotator_notes",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drafts_file")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = []
    with open(args.drafts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # Flagged records first (most likely to need attention), unflagged after.
    records.sort(key=lambda r: 0 if r.get("lint_flags") else 1)

    # utf-8-sig (BOM) so Excel opens Arabic/Malay text correctly instead of mojibake.
    with open(args.out, "w", encoding="utf-8-sig", newline="") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            row = dict(r)
            flags = row.get("lint_flags", [])
            row["lint_flags"] = "; ".join(flags) if flags else ""
            row.setdefault("draft_model", "")
            row["decision"] = ""
            row["corrected_question"] = ""
            row["corrected_answer"] = ""
            row["annotator_notes"] = ""
            writer.writerow(row)

    n_flagged = sum(1 for r in records if r.get("lint_flags"))
    print(f"Wrote {len(records)} records to {args.out} ({n_flagged} flagged, sorted to top).")
    print("Open this in Excel/Numbers/Google Sheets. Have the annotator fill in the "
          "'decision' column (approve / edit / reject) for every row, using "
          "'corrected_question' / 'corrected_answer' only for edits.")
    print("When review is done, run from_review_csv.py to merge decisions back into the "
          "benchmark format.")


if __name__ == "__main__":
    main()
