"""
Benchmark record schema for the Arabic-Malay RAG faithfulness benchmark.

Every question in the real dataset should validate against this schema before
it is admitted into the benchmark. Use `validate_record()` as a gate at the
end of the annotation pipeline (see dataset_construction_plan.md, Step 5).
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class BenchmarkRecord:
    id: str                       # stable unique id, e.g. "ar-wiki-000123"
    language: str                 # "ar" | "ms" | "en"
    domain: str                   # "quran" | "wikipedia_general"
    question: str
    gold_passage_id: str          # points into the source corpus, not inlined here
    gold_passage_text: str        # snapshot of the passage text at annotation time
    gold_answer: str
    source_citation: str          # e.g. "Quran 2:255 (Malay: Basmeih translation via Tanzil.net)"
                                   # or "ar.wikipedia.org/?curid=... (article id)"
    annotator_id: str             # pseudonymous id, not a real name; "claude_draft" for
                                   # not-yet-verified LLM-drafted records (see generate_questions.py)
    verification_pass: str        # "draft_unverified" | "primary" | "secondary" | "mt_plus_spotcheck"
                                   # "draft_unverified": LLM-drafted, not yet reviewed by a human --
                                   #   NOT part of the benchmark yet, must be promoted or rejected
                                   #   in Step 6 (human verification).
                                   # "primary"/"secondary"/"mt_plus_spotcheck": see
                                   # dataset_construction_plan.md Step 4 re: the
                                   # single-language-annotator mitigation
    difficulty: Optional[str] = None   # "easy" | "medium" | "hard" (optional, post-hoc)
    notes: Optional[str] = None
    draft_model: Optional[str] = None  # exact model that drafted this record, e.g.
                                        # "qwen3.5:4b" or "claude-sonnet-4-5" -- set by
                                        # generate_questions.py, useful if you mix models
                                        # across runs and want to audit/report it later

    def validate(self) -> list[str]:
        """Return a list of validation problems; empty list = valid."""
        problems = []
        if self.language not in {"ar", "ms", "en"}:
            problems.append(f"unexpected language code: {self.language}")
        if self.verification_pass not in {"draft_unverified", "primary", "secondary",
                                           "mt_plus_spotcheck"}:
            problems.append(f"unexpected verification_pass value: {self.verification_pass}")
        if not self.question.strip():
            problems.append("empty question")
        if not self.gold_answer.strip():
            problems.append("empty gold_answer")
        if not self.gold_passage_text.strip():
            problems.append("empty gold_passage_text")
        if self.gold_answer.lower() not in self.gold_passage_text.lower() \
                and self.verification_pass == "primary":
            # Soft check only — many correct answers are paraphrased, not substrings.
            # Flag for manual review rather than hard-fail.
            problems.append("WARNING: gold_answer not found verbatim in gold_passage_text "
                             "(may be fine if answer is paraphrased — flag for spot-check)")
        return problems

    def is_benchmark_ready(self) -> bool:
        """True only once a human has verified this record (i.e. not a raw LLM draft)."""
        return self.verification_pass != "draft_unverified" and not self.validate()


def load_jsonl(path: str) -> list[BenchmarkRecord]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(BenchmarkRecord(**obj))
    return records


def save_jsonl(records: list[BenchmarkRecord], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # quick self-test
    r = BenchmarkRecord(
        id="demo-000",
        language="en",
        domain="wikipedia_general",
        question="What is the capital of Malaysia?",
        gold_passage_id="en-wiki-malaysia-0001",
        gold_passage_text="Kuala Lumpur is the capital and largest city of Malaysia.",
        gold_answer="Kuala Lumpur",
        source_citation="en.wikipedia.org/wiki/Malaysia (placeholder)",
        annotator_id="demo_annotator_1",
        verification_pass="primary",
    )
    print(r.validate() or "OK: no problems found")
    print("is_benchmark_ready:", r.is_benchmark_ready())
