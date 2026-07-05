"""
Step 4a: extract Quran passages from Tanzil text files into paired
Arabic/Malay passage records (verse-level -- already naturally segmented,
no chunking needed).

Tanzil's "Text (with aya numbers)" download format is pipe-delimited, one
line per verse:

    sura|aya|verse text

Get the files from:
  - Arabic:  https://tanzil.net/download/  -> choose a text type (e.g.
             "Simple Clean") + "Text (with aya numbers)"
  - Malay:   https://tanzil.net/trans/  -> Basmeih translation + same format

If your downloaded file is NOT in that pipe-delimited format (e.g. you picked
"Text (without aya numbers)", which is just one verse per line with no sura/aya
prefix), this script will now tell you so explicitly and show you the first
few raw lines, instead of silently writing an empty output file.

Usage:
    python extract_quran.py --ar quran-arabic.txt --ms quran-malay-basmeih.txt \
        --out quran_passages.jsonl
"""

import argparse
import json


def load_tanzil_file(path: str) -> dict:
    """
    Returns {(sura, aya): text}.
    Prints a diagnostic and raises SystemExit with a clear message if the
    file doesn't look like the expected sura|aya|text pipe format, instead
    of silently returning an empty dict.
    """
    verses = {}
    total_lines = 0
    non_comment_lines = 0
    sample_lines = []

    with open(path, "r", encoding="utf-8-sig") as f:  # utf-8-sig strips a BOM if present
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            non_comment_lines += 1
            if len(sample_lines) < 5:
                sample_lines.append(line)

            parts = line.split("|")
            if len(parts) != 3:
                continue
            sura, aya, text = parts
            try:
                verses[(int(sura), int(aya))] = text.strip()
            except ValueError:
                continue

    if not verses:
        print(f"\n--- COULD NOT PARSE {path} AS sura|aya|text ---")
        print(f"Total lines in file: {total_lines}, non-comment lines: {non_comment_lines}")
        print("First few non-comment lines found (so you can see the actual format):")
        for s in sample_lines:
            print(f"  {s!r}")
        print(
            "\nThis usually means the download was 'Text (without aya numbers)' "
            "(one verse per line, no 'sura|aya|' prefix) rather than 'Text (with aya "
            "numbers)'. Go back to the Tanzil download page and re-download using the "
            "'with aya numbers' option, which produces the sura|aya|text format this "
            "script expects. If the lines above already look like sura|aya|text and this "
            "is still failing, the file may use a different delimiter -- paste the sample "
            "lines above back for a fix.\n"
        )
    return verses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ar", required=True, help="Tanzil Arabic text file")
    ap.add_argument("--ms", required=True, help="Tanzil Malay (Basmeih) text file")
    ap.add_argument("--out", default="quran_passages.jsonl")
    args = ap.parse_args()

    ar_verses = load_tanzil_file(args.ar)
    ms_verses = load_tanzil_file(args.ms)

    if not ar_verses or not ms_verses:
        raise SystemExit(
            "Stopping: one or both files failed to parse (see diagnostics above). "
            "Fix the input file(s) and re-run -- no output was written."
        )

    ar_keys = set(ar_verses.keys())
    ms_keys = set(ms_verses.keys())
    common = ar_keys & ms_keys
    missing_in_ms = ar_keys - ms_keys
    missing_in_ar = ms_keys - ar_keys

    print(f"Arabic verses: {len(ar_verses)}")
    print(f"Malay verses: {len(ms_verses)}")
    print(f"Aligned pairs: {len(common)}")
    if missing_in_ms:
        print(f"WARNING: {len(missing_in_ms)} verses in Arabic file missing from Malay file "
              f"(check both files are complete downloads of the same numbering scheme)")
    if missing_in_ar:
        print(f"WARNING: {len(missing_in_ar)} verses in Malay file missing from Arabic file")

    with open(args.out, "w", encoding="utf-8") as f:
        for (sura, aya) in sorted(common):
            for lang, text in (("ar", ar_verses[(sura, aya)]), ("ms", ms_verses[(sura, aya)])):
                record = {
                    "passage_id": f"quran-{sura}:{aya}-{lang}",
                    "language": lang,
                    "domain": "quran",
                    "sura": sura,
                    "aya": aya,
                    "text": text,
                    "source_citation": (
                        f"Quran {sura}:{aya} (Arabic, Tanzil.net Uthmani/Simple text, CC BY 3.0)"
                        if lang == "ar" else
                        f"Quran {sura}:{aya} (Malay, Abdullah Muhammad Basmeih translation via "
                        f"Tanzil.net, CC BY 3.0)"
                    ),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(common) * 2} passage records ({len(common)} verses x 2 languages) "
          f"to {args.out}")
    print("Next: feed quran_passages.jsonl into question generation (Step 5).")


if __name__ == "__main__":
    main()
