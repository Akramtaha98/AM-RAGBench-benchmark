"""
Step 5: generate candidate questions from passages.

Two backends:
  --backend ollama (default) : fully local, free, no API key, no per-call cost.
                                Good for testing the pipeline and for generating
                                the bulk of draft questions at zero cost.
  --backend claude           : Anthropic API, costs money per call, but generally
                                higher quality. Use for a final quality-check pass
                                on a sample, not for the whole corpus.

Per the dataset construction plan, questions must be drafted independently
PER LANGUAGE from the passage in that language -- never generate an English
question and translate it, since that introduces translation artifacts that
would contaminate faithfulness scoring later. This script enforces that by
prompting the model to write the question in the same language as the
passage it's given, one passage at a time.

v2 prompt change: an earlier version of this prompt let small/local models
default to "how many words/sentences" counting questions on short passages
(e.g. single Quran verses), which produced both bad question types AND
outright wrong counts (hallucinated word/sentence counts). The prompt below
explicitly forbids counting/meta questions and tells the model to return
FEWER than --questions-per-passage (even zero) rather than invent one on a
passage that's too short to support real comprehension questions. If you're
generating from very short passages (single Quran verses especially),
expect a meaningful fraction of passages to legitimately yield 0 questions
-- that's the model doing the right thing, not a bug. If you consistently
get 0s across most short verses, consider grouping a few consecutive verses
into one passage instead of one verse per passage (see extract_quran.py).

Output records are DRAFTS, not verified benchmark entries. Every record gets
`verification_pass: "draft_unverified"` (see schema.py) and MUST go through
Step 6 (human verification) before it counts as part of the benchmark. This
script does not replace annotation, it only produces candidates for
annotators to review, edit, or reject -- that's true regardless of which
backend drafted them.

--- Ollama setup (free, local, no API key) ---

1. Install Ollama: https://ollama.com/download
2. Pull a multilingual model. Recommended for this project:
     ollama pull qwen3.5:4b
   Qwen3.5 officially lists both Arabic and Malay among its supported
   languages (confirmed on its language list, not just inferred from broad
   pretraining), which makes it a better fit here than most alternatives.
   Use `qwen3.5:4b` on a laptop without a dedicated GPU; `qwen3.5:9b` if you
   have 16GB+ RAM and want higher quality at the cost of speed. If 4b is too
   slow, `qwen3.5:2b` trades some quality for noticeably faster generation --
   reasonable for the draft stage since every record still gets human review.
   Every draft still needs human verification in Step 6 regardless of model
   choice -- this just affects how much cleanup the drafts need.
3. Install the Python client: pip install ollama
4. Make sure the Ollama app/service is running (it usually auto-starts after
   install), then just run this script -- no API key needed, nothing to pay.

--- Claude setup (costs money, higher quality) ---

    pip install anthropic
    export ANTHROPIC_API_KEY=...

Usage (start small either way):
    python generate_questions.py --passages quran_passages.jsonl \
        --out quran_questions_draft.jsonl --limit 20 --questions-per-passage 2
    python generate_questions.py --backend claude --passages quran_passages.jsonl \
        --out quran_questions_draft_claude.jsonl --limit 20 --questions-per-passage 2
"""

import argparse
import json
import re
import sys
import time

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):  # no-op fallback if tqdm isn't installed
        return iterable


PROMPT_TEMPLATE = """You will be given a single passage of text. Write UP TO {n} distinct \
question-answer pairs that test understanding of the MEANING and CONTENT of this passage.

Rules:
- Write the question(s) and answer(s) in the SAME language as the passage below. \
Do not translate or switch languages.
- Every answer must be directly supported by the passage -- do not use outside knowledge.
- Ask about WHO, WHAT, WHERE, or WHY based on the content: entities, actions, attributes, \
relationships, or events described in the passage.
- NEVER ask about word count, letter count, sentence count, or "what is the first/last \
word/sentence" -- these are not comprehension questions and you are unreliable at counting.
- If the passage is too short or repetitive to support {n} distinct, non-trivial \
comprehension questions, return FEWER questions (even an empty list) rather than inventing \
a counting question or a near-duplicate of another question. Quality and correctness matter \
far more than hitting exactly {n}.

Passage (language: {language}):
{passage_text}

Respond with ONLY a JSON array, no other text, in this exact form (can be empty: []):
[{{"question": "...", "answer": "..."}}, ...]"""


THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_block(raw_text: str) -> str:
    """
    Qwen3/Qwen3.5 models default to "thinking mode": they emit a <think>...</think>
    reasoning trace before the real answer. Ollama's `think=False` chat option is
    supposed to suppress this, but it's not fully reliable for every Qwen3-family
    model/version (see ollama/ollama-python#529, ollama/ollama#10492), so this is
    a defensive second layer: if a <think> block is present anyway, cut it out
    before we go looking for the JSON array.
    """
    return THINK_BLOCK_RE.sub("", raw_text)


def extract_json_array(raw_text: str):
    """
    Models are asked for JSON-only output, but sometimes wrap it in markdown
    code fences or add a stray sentence. Pull out the first top-level [...]
    block and parse that, rather than assuming raw_text is clean JSON.
    """
    raw_text = strip_think_block(raw_text)
    match = re.search(r"\[.*\]", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def load_passages(path: str, limit: int):
    passages = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            passages.append(json.loads(line))
            if len(passages) >= limit:
                break
    return passages


class ClaudeBackend:
    def __init__(self, model: str):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if hasattr(b, "text"))


class OllamaBackend:
    def __init__(self, model: str):
        import ollama
        self._ollama = ollama
        self.model = model
        # Qwen3/Qwen3.5 models default to "thinking mode" in Ollama: before the
        # real answer, they emit a <think>...</think> reasoning trace. On short
        # Quran verses that trace is short enough to fit inside the num_predict
        # budget, leaving room for the actual JSON afterward. On longer
        # Wikipedia passages the trace can be much longer, and if num_predict
        # is too small the response gets cut off INSIDE the <think> block --
        # so every single call fails to parse, which is exactly the 100%
        # failure pattern seen on ar_wiki/ms_wiki generation runs. think=False
        # asks Ollama to suppress the reasoning trace entirely; this is not
        # 100% reliable on every Qwen3-family build (see ollama/ollama-python
        # issue #529), so we try it but fall back gracefully if this
        # ollama-python version doesn't support the kwarg yet.
        self._supports_think_kwarg = True

    def complete(self, prompt: str) -> str:
        kwargs = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            # Cap output length. Raised from 400 -> 900: 400 was tuned against
            # short Quran verses and silently truncated every response on
            # longer Wikipedia passages (see note above), which is why that
            # run produced 0/200 parsed records for both languages.
            options={"num_predict": 900},
        )
        if self._supports_think_kwarg:
            kwargs["think"] = False
        try:
            resp = self._ollama.chat(**kwargs)
        except TypeError:
            # Installed ollama-python version predates the `think` kwarg.
            self._supports_think_kwarg = False
            kwargs.pop("think", None)
            resp = self._ollama.chat(**kwargs)
        return resp["message"]["content"]


def build_backend(name: str, model: str):
    if name == "claude":
        try:
            return ClaudeBackend(model)
        except ImportError:
            sys.exit("Missing dependency. Run: pip install anthropic")
    if name == "ollama":
        try:
            return OllamaBackend(model)
        except ImportError:
            sys.exit("Missing dependency. Run: pip install ollama\n"
                      "Also make sure the Ollama app is installed and running: "
                      "https://ollama.com/download, then `ollama pull qwen3.5:4b`")
    raise ValueError(f"unknown backend: {name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", required=True, help="a *_passages.jsonl file from Step 4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=20,
                     help="how many passages to process (start small)")
    ap.add_argument("--questions-per-passage", type=int, default=2,
                     help="a MAXIMUM, not a target -- the model is told it's fine to "
                          "return fewer (even 0) for short/thin passages")
    ap.add_argument("--backend", choices=["ollama", "claude"], default="ollama",
                     help="ollama = free/local (default), claude = paid API, higher quality")
    ap.add_argument("--model", default=None,
                     help="defaults to qwen3.5:4b for ollama, claude-sonnet-4-5 for claude")
    ap.add_argument("--sleep", type=float, default=0.2,
                     help="seconds to sleep between calls (higher for claude to respect "
                          "rate limits; ollama running locally doesn't need this but a small "
                          "pause is harmless)")
    args = ap.parse_args()

    model = args.model or ("qwen3.5:4b" if args.backend == "ollama" else "claude-sonnet-4-5")
    backend = build_backend(args.backend, model)
    print(f"Using backend={args.backend}, model={model}")
    if args.backend == "ollama":
        print("Free/local run -- no API cost. First call may be slow while Ollama loads "
              "the model into memory; subsequent calls are faster.")

    passages = load_passages(args.passages, args.limit)
    print(f"Loaded {len(passages)} passages from {args.passages} (--limit {args.limit})")

    n_ok = 0
    n_failed = 0
    n_empty = 0
    n_questions_written = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, passage in enumerate(tqdm(passages, desc="generating", unit="passage")):
            language_name = {"ar": "Arabic", "ms": "Malay", "en": "English"}.get(
                passage["language"], passage["language"]
            )
            prompt = PROMPT_TEMPLATE.format(
                n=args.questions_per_passage,
                language=language_name,
                passage_text=passage["text"],
            )
            qa_pairs = None
            last_error = None
            last_raw_text = None
            # Retry once on parse failure -- small local models sometimes add stray
            # prose around the JSON on the first try; a second attempt often succeeds.
            for attempt in range(2):
                try:
                    call_prompt = prompt if attempt == 0 else (
                        prompt + "\n\nReminder: respond with ONLY the JSON array, "
                                 "no explanation, no markdown code fence."
                    )
                    raw_text = backend.complete(call_prompt)
                    last_raw_text = raw_text
                    qa_pairs = extract_json_array(raw_text)
                    if qa_pairs is not None:
                        break
                except Exception as e:  # noqa: BLE001 -- log and retry/continue, don't kill the run
                    last_error = e
                    qa_pairs = None

            if qa_pairs is None:
                if last_error is not None:
                    print(f"[{i+1}/{len(passages)}] backend error on "
                          f"{passage['passage_id']} (after retry): {last_error}")
                else:
                    print(f"[{i+1}/{len(passages)}] Could not parse JSON from model output for "
                          f"{passage['passage_id']} (after retry), skipping.")
                    if last_raw_text:
                        snippet = last_raw_text[:300].replace("\n", " ")
                        print(f"    raw output snippet: {snippet!r}")
                        if "<think>" in last_raw_text and "</think>" not in last_raw_text:
                            print("    (looks like the response was cut off mid-<think> block -- "
                                  "the model ran out of num_predict tokens before finishing its "
                                  "reasoning trace and never reached the actual answer)")
                n_failed += 1
                time.sleep(args.sleep)
                continue

            if qa_pairs is None:
                print(f"[{i+1}/{len(passages)}] Could not parse JSON from model output for "
                      f"{passage['passage_id']}, skipping.")
                n_failed += 1
                time.sleep(args.sleep)
                continue

            if len(qa_pairs) == 0:
                n_empty += 1
            else:
                n_ok += 1

            for j, qa in enumerate(qa_pairs):
                if "question" not in qa or "answer" not in qa:
                    continue
                record = {
                    "id": f"{passage['passage_id']}-q{j}",
                    "language": passage["language"],
                    "domain": passage["domain"],
                    "question": qa["question"],
                    "gold_passage_id": passage["passage_id"],
                    "gold_passage_text": passage["text"],
                    "gold_answer": qa["answer"],
                    "source_citation": passage.get("source_citation", ""),
                    "annotator_id": f"{args.backend}_draft",
                    "draft_model": model,  # exact model that drafted this record, e.g.
                                           # "qwen3.5:4b" or "qwen3.5:2b" -- lets you audit
                                           # or report which model produced which records
                                           # if you mix models across runs for speed
                    "verification_pass": "draft_unverified",
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_questions_written += 1

            time.sleep(args.sleep)

    print(f"\nDone. {n_ok} passages produced questions, {n_empty} legitimately empty "
          f"(too short/thin for a real question), {n_failed} failed/skipped.")
    print(f"{n_questions_written} total question records written to {args.out}")
    if n_empty > len(passages) * 0.4:
        print(f"\nNOTE: {n_empty}/{len(passages)} passages produced zero questions. If this "
              f"passage set is single Quran verses, consider grouping a few consecutive "
              f"verses into one passage (extract_quran.py) rather than one verse each -- "
              f"very short passages often don't have enough content for a real question.")
    print("IMPORTANT: every record has verification_pass='draft_unverified'. These are NOT "
          "part of the benchmark yet -- they must go through Step 6 human verification (native "
          "speakers review question clarity, answer correctness, and gold-passage sufficiency) "
          "before use, regardless of which backend drafted them.")


if __name__ == "__main__":
    main()
