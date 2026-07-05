"""
Retrieval stack for the Arabic-Malay RAG benchmark.

Two retrievers:
  - BM25Retriever   : sparse lexical baseline, rank_bm25, no downloads, runs anywhere.
  - DenseRetriever   : sentence-transformers embeddings + FAISS, needs
                       `pip install sentence-transformers faiss-cpu` and a model
                       download on first run (~470MB for a multilingual MiniLM model).
                       This is the one to use for the real experiments.

Both expose the same interface: .index(passages) then .retrieve(query, k).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import re


def _tokenize(text: str) -> list[str]:
    # Simple whitespace + punctuation split. Good enough for BM25 on Arabic/Malay/English;
    # swap in a language-specific tokenizer (e.g. CAMeL Tools for Arabic) for production use.
    return re.findall(r"\w+", text.lower())


@dataclass
class RetrievedPassage:
    passage_id: str
    text: str
    score: float


class BM25Retriever:
    def __init__(self, passage_ids: Sequence[str], passages: Sequence[str]):
        from rank_bm25 import BM25Okapi

        self.passage_ids = list(passage_ids)
        self.passages = list(passages)
        tokenized = [_tokenize(p) for p in self.passages]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedPassage]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [
            RetrievedPassage(self.passage_ids[i], self.passages[i], float(scores[i]))
            for i in ranked
        ]


class DenseRetriever:
    """
    Sentence-transformers + FAISS dense retriever. This is the recommended
    retriever for the real evaluation once you have the full corpus, since it
    handles paraphrase and cross-lingual-ish matches BM25 misses.

    Recommended model for Arabic + Malay + English:
        'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    or, for stronger multilingual retrieval quality:
        'BAAI/bge-m3'
    """

    def __init__(self, passage_ids: Sequence[str], passages: Sequence[str],
                 model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import faiss

        self.passage_ids = list(passage_ids)
        self.passages = list(passages)
        self.model = SentenceTransformer(model_name)

        embeddings = self.model.encode(self.passages, normalize_embeddings=True,
                                        show_progress_bar=False)
        embeddings = np.asarray(embeddings, dtype="float32")
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedPassage]:
        import numpy as np

        q_emb = self.model.encode([query], normalize_embeddings=True,
                                   show_progress_bar=False)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, idx = self.index.search(q_emb, k)
        return [
            RetrievedPassage(self.passage_ids[i], self.passages[i], float(scores[0][j]))
            for j, i in enumerate(idx[0])
        ]


def build_retriever(kind: str, passage_ids: Sequence[str], passages: Sequence[str]):
    if kind == "bm25":
        return BM25Retriever(passage_ids, passages)
    if kind == "dense":
        return DenseRetriever(passage_ids, passages)
    raise ValueError(f"unknown retriever kind: {kind}")
