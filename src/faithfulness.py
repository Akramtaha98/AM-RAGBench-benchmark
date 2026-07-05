"""
Faithfulness / hallucination scoring.

Two scorers, meant to be used together (report both, don't collapse to one number):

  - lexical_overlap_score : cheap, fast, language-agnostic-ish sanity check.
                            Good for catching obvious total misses, not nuanced enough
                            to publish as the sole metric.
  - llm_judge_prompt       : the prompt template for an LLM-judge faithfulness classifier
                             (FActScore/RAGTruth-style categorical label). Wire this to
                             ClaudeGenerator or another strong model as the judge. Every
                             LLM-judge label should still be spot-checked by a human
                             annotator on a random sample (report the agreement rate) —
                             do not present LLM-judge labels as ground truth on their own.

Label schema (adapted from FActScore / RAGTruth):
  SUPPORTED           - claim is directly backed by the gold/retrieved passage
  PARTIALLY_SUPPORTED - claim is broadly consistent but adds unsupported detail
  UNSUPPORTED         - claim is not present in the passage (hallucination)
  CONTRADICTED        - claim directly contradicts the passage
  ABSTAINED           - model said it doesn't know / declined to answer
"""

from __future__ import annotations
from dataclasses import dataclass
import re


LABELS = ["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED", "ABSTAINED"]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def lexical_overlap_score(generated_answer: str, passage: str) -> float:
    """Fraction of generated-answer tokens that appear in the passage. Crude but free."""
    gen_tokens = set(re.findall(r"\w+", _normalize(generated_answer)))
    passage_tokens = set(re.findall(r"\w+", _normalize(passage)))
    if not gen_tokens:
        return 0.0
    return len(gen_tokens & passage_tokens) / len(gen_tokens)


def heuristic_label(generated_answer: str, passage: str, gold_answer: str) -> str:
    """
    Cheap rule-based label, used only for the runnable demo in this repo.
    Replace with the LLM-judge (see llm_judge_prompt below) + human spot-check
    for anything that goes in the paper.
    """
    norm_gen = _normalize(generated_answer)
    if norm_gen in {"i don't know.", "i don't know", "لا أعرف", "saya tidak tahu"}:
        return "ABSTAINED"
    if _normalize(gold_answer) in norm_gen:
        return "SUPPORTED"
    overlap = lexical_overlap_score(generated_answer, passage)
    if overlap < 0.2:
        return "UNSUPPORTED"
    return "PARTIALLY_SUPPORTED"


LLM_JUDGE_PROMPT_TEMPLATE = """You are auditing a retrieval-augmented answer for faithfulness.

Passage (the ONLY source of truth the model was given):
{passage}

Question: {question}

Model's answer: {answer}

Classify the model's answer into exactly one of:
SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED, ABSTAINED

Definitions:
- SUPPORTED: every claim in the answer is directly stated in the passage.
- PARTIALLY_SUPPORTED: the core claim is in the passage, but the answer adds
  extra detail not present in the passage.
- UNSUPPORTED: the answer's claim is not present anywhere in the passage.
- CONTRADICTED: the answer directly contradicts something stated in the passage.
- ABSTAINED: the model declined to answer / said it doesn't know.

Respond with only the single label word, nothing else."""


@dataclass
class FaithfulnessResult:
    record_id: str
    language: str
    model_name: str
    label: str
    lexical_overlap: float


# Sentinel used when the judge model's output couldn't be parsed into one of the
# five real labels, even after a retry. Deliberately NOT one of the five labels,
# so a parsing failure never gets silently miscounted as a real classification --
# it shows up in the label distribution print as its own line, same lesson as the
# generate_questions.py num_predict/thinking-mode bug (silent truncation looked
# like "everything failed" instead of being visible as a distinct, diagnosable issue).
JUDGE_PARSE_FAILED = "JUDGE_PARSE_FAILED"

_VALID_LABEL_RE = re.compile(
    r"\b(SUPPORTED|PARTIALLY_SUPPORTED|UNSUPPORTED|CONTRADICTED|ABSTAINED)\b"
)


class OllamaJudge:
    """
    Real LLM-judge faithfulness classifier, using Ollama (free/local, no API
    key) -- same setup as the rest of this project. Wraps LLM_JUDGE_PROMPT_TEMPLATE
    and asks the model to output exactly one of the five labels.

    Recommended model: qwen3.5:4b rather than 2b for this specific role. The
    judge's classification IS the thing being reported in the paper, so it's
    worth the slower speed for better judgment quality here, even though 2b was
    fine for drafting benchmark questions (where every record gets human review
    anyway). Still: every LLM-judge run should have a human-checked agreement
    sample reported in the paper -- these labels are not ground truth on their
    own, see dataset_construction_plan.md.

    Same thinking-mode handling as OllamaBackend (generate_questions.py) and
    OllamaGenerator (generation.py): Qwen3/Qwen3.5 default to a <think>...</think>
    reasoning trace that can eat the token budget before reaching the actual
    label, so think=False is passed (with a fallback for older ollama-python
    versions) and any leaked <think> block is stripped before label extraction.
    """

    def __init__(self, model_name: str = "qwen3.5:4b"):
        import ollama
        self._ollama = ollama
        self.model_name = model_name
        self._supports_think_kwarg = True

    def _call(self, prompt: str) -> str:
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 200},
        )
        if self._supports_think_kwarg:
            kwargs["think"] = False
        try:
            resp = self._ollama.chat(**kwargs)
        except TypeError:
            self._supports_think_kwarg = False
            kwargs.pop("think", None)
            resp = self._ollama.chat(**kwargs)
        text = resp["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    def label(self, question: str, passage: str, answer: str) -> str:
        prompt = LLM_JUDGE_PROMPT_TEMPLATE.format(
            passage=passage, question=question, answer=answer
        )
        last_raw = None
        for attempt in range(2):
            call_prompt = prompt if attempt == 0 else (
                prompt + "\n\nReminder: respond with ONLY the single label word, "
                         "nothing else."
            )
            try:
                raw = self._call(call_prompt)
            except Exception as e:  # noqa: BLE001 -- fall through to parse-failure path
                last_raw = f"<backend error: {e}>"
                continue
            last_raw = raw
            match = _VALID_LABEL_RE.search(raw.upper())
            if match:
                return match.group(1)
        print(f"  [judge] could not parse a label after retry -- raw output: "
              f"{(last_raw or '')[:200]!r}")
        return JUDGE_PARSE_FAILED


def score_batch(rows: list[dict], judge=None) -> list[FaithfulnessResult]:
    """
    rows: list of dicts with keys id, language, model_name, generated_answer,
          passage, gold_answer, question.

    judge:
      None (default)   -> the cheap heuristic labeler. Fine for smoke-testing
                          the harness end-to-end; NOT credible enough to report
                          as the paper's faithfulness number (see heuristic_label
                          docstring -- it can't detect CONTRADICTED at all, and
                          conflates "reasonable paraphrase" with "wrong answer
                          that shares a few words with whatever got retrieved").
      OllamaJudge(...)  -> the real LLM-judge. Use this for any number that
                          actually goes in the paper.
    """
    results = []
    iterator = rows
    try:
        from tqdm import tqdm
        iterator = tqdm(rows, desc="faithfulness scoring", unit="record")
    except ImportError:
        pass

    for row in iterator:
        overlap = lexical_overlap_score(row["generated_answer"], row["passage"])
        if judge is not None:
            label = judge.label(
                row.get("question", ""), row["passage"], row["generated_answer"]
            )
        else:
            label = heuristic_label(row["generated_answer"], row["passage"], row["gold_answer"])
        results.append(FaithfulnessResult(
            record_id=row["id"],
            language=row["language"],
            model_name=row["model_name"],
            label=label,
            lexical_overlap=overlap,
        ))
    return results
