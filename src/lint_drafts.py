"""
Lint pass over LLM-drafted question files (Step 5 output) BEFORE they go to
human verification (Step 6). This does NOT auto-correct anything -- it only
flags suspicious records so your annotator can triage them first, rather
than reading every record with equal attention.

Flags applied (each is a hint, not a rejection -- a human still decides):
  - REPEATED_CHAR: the same character appears 3+ times in a row anywhere in
    the question, answer, or passage (e.g. "للله" -- 3 lams in a row -- which
    is not a valid Arabic sequence; this is what caught the duplicate-lam
    typo from an earlier batch).
  - ANSWER_NOT_IN_PASSAGE: the answer text (or a close variant) doesn't
    appear in the gold passage at all -- possible hallucinated answer.
  - VERY_SHORT_QUESTION / VERY_SHORT_ANSWER: fewer than 3 characters,
    usually a sign of a malformed generation.

Usage:
    python lint_drafts.py drafts/quran_questions_draft.jsonl
    python lint_drafts.py drafts/quran_questions_draft.jsonl --out drafts/quran_questions_flagged.jsonl
"""

import argparse
import json
import re
import sys

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # no-op fallback if tqdm isn't installed
        return iterable


REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")


def lint_record(record: dict) -> list[str]:
    flags = []

    for field in ("question", "gold_answer", "gold_passage_text"):
        text = record.get(field, "")
        if REPEATED_CHAR_RE.search(text):
            flags.append(f"REPEATED_CHAR in {field}")

    question = record.get("question", "")
    answer = record.get("gold_answer", "")
    passage = record.get("gold_passage_text", "")

    if len(question.strip()) < 3:
        flags.append("VERY_SHORT_QUESTION")
    if len(answer.strip()) < 2:
        flags.append("VERY_SHORT_ANSWER")

    # Loose substring check, case-insensitive. Many correct answers are
    # paraphrased rather than verbatim, so this is a hint to check, not proof
    # of an error -- don't auto-reject on this alone.
    if answer.strip() and answer.strip().lower() not in passage.lower():
        flags.append("ANSWER_NOT_VERBATIM_IN_PASSAGE (may be a fine paraphrase -- check by hand)")

    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("drafts_file")
    ap.add_argument("--out", default=None,
                     help="if given, writes a copy of the input with a 'lint_flags' field added "
                          "to every record (empty list if clean)")
    args = ap.parse_args()

    total = 0
    flagged = 0
    flag_counts: dict[str, int] = {}
    out_records = []

    with open(args.drafts_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in tqdm(lines, desc="linting", unit="record"):
        record = json.loads(line)
        total += 1
        flags = lint_record(record)
        if flags:
            flagged += 1
            print(f"[{record.get('id', '?')}] {', '.join(flags)}")
            for flag in flags:
                key = flag.split(" in ")[0].split(" (")[0]
                flag_counts[key] = flag_counts.get(key, 0) + 1
        record["lint_flags"] = flags
        out_records.append(record)

    print(f"\n{flagged}/{total} records flagged for a closer look "
          f"({total - flagged} passed with no flags).")
    if flag_counts:
        print("Flag breakdown:")
        for key, count in sorted(flag_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {key}: {count}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as out_f:
            for record in out_records:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"\nWrote annotated copy (with 'lint_flags' field) to {args.out}")
        print("Have your annotator review flagged records first -- it's the fastest way to "
              "catch the highest-impact errors early.")


if __name__ == "__main__":
    main()
