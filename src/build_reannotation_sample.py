import csv, random, json
from collections import defaultdict

random.seed(42)
TARGET_TOTAL = 150

def load(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

quran = load("/sessions/charming-sharp-babbage/mnt/uploads/quran_review_completed.csv")
wiki = load("/sessions/charming-sharp-babbage/mnt/uploads/42ba2aaf-f65d-4ec8-aa72-3a0b53614e37-1783180912566_wiki_review_completed.csv")
for r in quran: r["domain"] = "quran"
for r in wiki: r["domain"] = "wiki"
all_rows = quran + wiki

def build_for_language(lang):
    rows = [r for r in all_rows if r["language"] == lang]
    edit_reject = [r for r in rows if r["decision"] in ("edit", "reject")]
    approve = [r for r in rows if r["decision"] == "approve"]

    n_fill = max(TARGET_TOTAL - len(edit_reject), 0)
    n_per_domain = n_fill // 2

    sampled_approve = []
    for domain in ["quran", "wiki"]:
        dom_approve = [r for r in approve if r["domain"] == domain]
        # median split by passage length within this domain's approve pool
        lengths = sorted(len(r["gold_passage_text"]) for r in dom_approve)
        median_len = lengths[len(lengths)//2] if lengths else 0
        short = [r for r in dom_approve if len(r["gold_passage_text"]) <= median_len]
        long_ = [r for r in dom_approve if len(r["gold_passage_text"]) > median_len]
        random.shuffle(short); random.shuffle(long_)
        half = n_per_domain // 2
        sampled_approve.extend(short[:half])
        sampled_approve.extend(long_[:(n_per_domain - half)])

    sample = edit_reject + sampled_approve
    random.shuffle(sample)
    return sample, {r["id"]: r for r in rows}  # full answer key for this language

for lang, label in [("ar", "arabic"), ("ms", "malay")]:
    sample, answer_key = build_for_language(lang)
    print(f"{label}: n={len(sample)}  "
          f"edit/reject={sum(1 for r in sample if r['decision'] in ('edit','reject'))}  "
          f"quran={sum(1 for r in sample if r['domain']=='quran')}  "
          f"wiki={sum(1 for r in sample if r['domain']=='wiki')}")

    # blind CSV for the new annotator (no decision/corrected fields exposed)
    blind_path = f"/tmp/reannotation/{label}_reannotation_blind.csv"
    with open(blind_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id","domain","language","question","gold_answer",
                                          "gold_passage_text","source_citation",
                                          "new_annotator_decision","new_corrected_question",
                                          "new_corrected_answer","new_annotator_notes"])
        w.writeheader()
        for r in sample:
            # BUGFIX (found during first reannotation round): for records whose
            # original decision was 'edit', the released/final text lives in
            # corrected_question/corrected_answer, not question/gold_answer
            # (those hold the pre-correction draft). Show the annotator what is
            # actually published in the benchmark, falling back to the draft
            # field only if no correction was recorded for that specific field.
            final_question = (r.get("corrected_question", "") or "").strip() or r["question"]
            final_answer = (r.get("corrected_answer", "") or "").strip() or r["gold_answer"]
            w.writerow({
                "id": r["id"], "domain": r["domain"], "language": r["language"],
                "question": final_question, "gold_answer": final_answer,
                "gold_passage_text": r["gold_passage_text"],
                "source_citation": r["source_citation"],
                "new_annotator_decision": "", "new_corrected_question": "",
                "new_corrected_answer": "", "new_annotator_notes": "",
            })

    # answer key (original decision), kept separate, NOT given to annotator
    key_path = f"/tmp/reannotation/{label}_original_decisions.json"
    with open(key_path, "w", encoding="utf-8") as f:
        json.dump({rid: {"decision": r["decision"], "domain": r["domain"]}
                   for rid, r in answer_key.items() if rid in {s["id"] for s in sample}},
                  f, ensure_ascii=False, indent=2)

    print(f"  wrote {blind_path}")
    print(f"  wrote {key_path}")
