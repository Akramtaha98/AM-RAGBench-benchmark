"""
Generators for the RAG faithfulness benchmark.

  - ClaudeGenerator : real Anthropic API call. Needs ANTHROPIC_API_KEY in env.
  - HFGenerator     : real local/HF-hosted open-source model call
                      (e.g. Jais for Arabic, Sailor/SEA-LION for Malay, or a
                      general multilingual model like Qwen2.5 or Llama-3.1).
  - MockGenerator   : deterministic canned outputs, used ONLY to exercise the
                      pipeline end-to-end without API keys or GPU. Never use
                      MockGenerator output as if it were a real experimental
                      result — it exists to prove the harness works.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence


RAG_PROMPT_TEMPLATE = """Answer the question using ONLY the information in the passage below.
If the passage does not contain the answer, say "I don't know" rather than guessing.

Passage:
{passage}

Question: {question}

Answer:"""


@dataclass
class Generation:
    text: str
    model_name: str


class ClaudeGenerator:
    def __init__(self, model: str = "claude-sonnet-4-5"):
        import anthropic
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
        self.model = model

    def generate(self, question: str, passage: str) -> Generation:
        prompt = RAG_PROMPT_TEMPLATE.format(passage=passage, question=question)
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in resp.content if hasattr(block, "text"))
        return Generation(text=text.strip(), model_name=self.model)


class HFGenerator:
    """
    Local/HF-hosted open-source model generator. Point `model_name` at:
      - "inceptionai/jais-13b-chat"                 (Arabic-centric)
      - "sail/Sailor-7B-Chat" or an aisingapore SEA-LION checkpoint (Malay/SEA)
      - "Qwen/Qwen2.5-7B-Instruct" or "meta-llama/Llama-3.1-8B-Instruct" (general)
    Requires a GPU for reasonable speed at 7B+ scale.
    """

    def __init__(self, model_name: str):
        from transformers import pipeline
        self.model_name = model_name
        self.pipe = pipeline("text-generation", model=model_name, device_map="auto")

    def generate(self, question: str, passage: str) -> Generation:
        prompt = RAG_PROMPT_TEMPLATE.format(passage=passage, question=question)
        out = self.pipe(prompt, max_new_tokens=256, do_sample=False)
        text = out[0]["generated_text"][len(prompt):].strip()
        return Generation(text=text, model_name=self.model_name)


class OllamaGenerator:
    """
    Free/local generator via Ollama -- consistent with the rest of this project's
    zero-cost setup (same backend used in generate_questions.py). Recommended
    default for the real evaluation run since it needs no API key and no GPU.

    Same thinking-mode caveat as generate_questions.py's OllamaBackend: Qwen3/
    Qwen3.5 models emit a <think>...</think> reasoning trace by default, which
    can eat the token budget on longer passages if not suppressed. This class
    passes think=False (with a graceful fallback for older ollama-python
    versions) and strips any <think> block that leaks through anyway.
    """

    def __init__(self, model_name: str = "qwen3.5:2b"):
        import ollama
        self._ollama = ollama
        self.model_name = model_name
        self._supports_think_kwarg = True

    def generate(self, question: str, passage: str) -> Generation:
        import re
        prompt = RAG_PROMPT_TEMPLATE.format(passage=passage, question=question)
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 400},
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
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        return Generation(text=text, model_name=self.model_name)


class MockGenerator:
    """
    Deterministic stand-in so the pipeline can run without API keys/GPUs.
    Behavior, by design, so the demo output is legible:
      - ~70% of the time: returns the gold answer verbatim (simulates a faithful hit)
      - ~15% of the time: returns a plausible-looking but unsupported claim (simulates
        a hallucination) by returning a fixed distractor string
      - ~15% of the time: returns "I don't know" (simulates an abstention)
    This is NOT a language model and produces NO real experimental signal.
    """

    def __init__(self, seed: int = 0):
        import random
        self._rng = random.Random(seed)
        self.model_name = "mock-generator (NOT a real model, demo only)"

    def generate(self, question: str, passage: str, gold_answer: str = "") -> Generation:
        r = self._rng.random()
        if r < 0.70 and gold_answer:
            text = gold_answer
        elif r < 0.85:
            text = "The answer is 1962."  # fixed distractor, deliberately often wrong
        else:
            text = "I don't know."
        return Generation(text=text, model_name=self.model_name)


def build_generator(kind: str, **kwargs):
    if kind == "claude":
        return ClaudeGenerator(**kwargs)
    if kind == "hf":
        return HFGenerator(**kwargs)
    if kind == "ollama":
        return OllamaGenerator(**kwargs)
    if kind == "mock":
        return MockGenerator(**kwargs)
    raise ValueError(f"unknown generator kind: {kind}")
