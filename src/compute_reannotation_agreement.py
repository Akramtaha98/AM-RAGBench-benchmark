"""
Computes inter-annotator agreement between the original single-annotator
decision (approve/edit/reject, recorded during initial benchmark
construction) and a new, independent annotator's blind re-verification
decision (valid/edit/invalid) on the same stratified sample of records.

This is the real second/third-annotator validation the paper's Limitations
section flags as missing -- run this once you have a completed CSV back
from each new annotator (exported from arabic_reannotation_review.html /
malay_reannotation_review.html, in review/reannotation/).

Usage:
    python compute_reannotation_agreement.py arabic_reannotation_v1_completed.csv ../review/reannotation/arabic_original_decisions.json
    python compute_reannotation_agreement.py malay_reannotation_v1_completed.csv ../review/reannotation/malay_original_decisions.json
"""

import sys
import csv
import json
from collections import Counter, defaultdict

# valid/edit/invalid (new annotator's 3-way schema) <-> approve/edit/reject
# (original annotator's 3-way schema during initial construction) -- same
# underlying meaning, different button labels since the new annotator is
# reviewing already-released records rather than raw drafts.
LABEL_MAP = {"valid": "approve", "edit": "edit", "invalid": "reject"}


def cohens_kappa(pairs):
    """
    pairs: list of (rater_a_label, rater_b_label) tuples, already on a
    shared label space. Standard two-rater Cohen's kappa, per Artstein &
    Poesio (2008): kappa = (Po - Pe) / (1 - Pe).
    """
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    labels = sorted(set(a for a, b in pairs) | set(b for a, b in pairs))
    a_counts = Counter(a for a, b in pairs)
    b_counts = Counter(b for a, b in pairs)
    pe = sum((a_counts[l] / n) * (b_counts[l] / n) for l in labels)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: python compute_reannotation_agreement.py <completed_csv> <original_decisions.json>")

    completed_path, original_path = sys.argv[1], sys.argv[2]

    with open(completed_path, encoding="utf-8-sig") as f:
        completed = list(csv.DictReader(f))
    with open(original_path, encoding="utf-8") as f:
        original = json.load(f)

    pairs = []
    by_domain = defaultdict(list)
    missing = 0
    for row in completed:
        rid = row["id"]
        new_raw = row.get("new_annotator_decision", "").strip()
        if not new_raw:
            missing += 1
            continue
        if rid not in original:
            print(f"  WARNING: {rid} not found in original decisions file, skipping")
            continue
        new_label = LABEL_MAP.get(new_raw, new_raw)
        orig_label = original[rid]["decision"]
        pairs.append((orig_label, new_label))
        by_domain[original[rid]["domain"]].append((orig_label, new_label))

    n_total = len(completed)
    print(f"Total sampled records: {n_total}")
    print(f"Missing/undecided (excluded from agreement): {missing}")
    print(f"Scored pairs: {len(pairs)}\n")

    if not pairs:
        print("No decisions to score yet.")
        return

    agree = sum(1 for a, b in pairs if a == b)
    print(f"Overall raw agreement: {agree}/{len(pairs)} ({100*agree/len(pairs):.1f}%)")
    kappa = cohens_kappa(pairs)
    print(f"Cohen's kappa (Artstein & Poesio 2008 conventions): {kappa:.3f}\n")

    print("Confusion matrix (rows=original decision, cols=new annotator decision):")
    labels = ["approve", "edit", "reject"]
    header = "            " + "  ".join(f"{l:>8s}" for l in labels)
    print(header)
    for row_label in labels:
        counts = Counter(b for a, b in pairs if a == row_label)
        line = f"{row_label:>10s}  " + "  ".join(f"{counts.get(l,0):8d}" for l in labels)
        print(line)

    print("\nBy domain:")
    for domain, dpairs in by_domain.items():
        dagree = sum(1 for a, b in dpairs if a == b)
        dkappa = cohens_kappa(dpairs)
        print(f"  {domain:8s} n={len(dpairs):3d}  agreement={100*dagree/len(dpairs):.1f}%  kappa={dkappa:.3f}")

    print("\nDisagreements (original -> new), for manual adjudication:")
    for row, (a, b) in zip((r for r in completed if r.get("new_annotator_decision","").strip()), pairs):
        if a != b:
            print(f"  {row['id']}: {a} -> {b}"
                  + (f"  [note: {row['new_annotator_notes']}]" if row.get("new_annotator_notes") else ""))


if __name__ == "__main__":
    main()
