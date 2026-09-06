"""BM25 keyword retrieval.

Pure Python, no models, no network. This is the arm that runs and produces
committed results in any environment, which is why the chunking experiment
is a complete result on its own rather than something waiting on hardware.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from ..ingest.models import Chunk
from .base import Scored

TOKEN = re.compile(r"[a-z0-9]+")

# Deliberately minimal. An aggressive stoplist would strip "not", which in a
# standards document flips the meaning of the clause being retrieved.
STOPWORDS = frozenset(
    ["a", "an", "the", "of", "to", "in", "for", "on", "at", "by", "is", "are", "was", "were", "be", "been", "and", "or", "as", "it", "its", "this", "that", "these", "those", "with", "from", "into"]
)


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


@dataclass
class BM25Retriever:
    k1: float = 1.5
    b: float = 0.75
    name: str = "bm25"
    _chunks: list[Chunk] = field(default_factory=list, repr=False)
    _bm25: BM25Okapi | None = field(default=None, repr=False)

    def index(self, chunks: Sequence[Chunk]) -> None:
        self._chunks = list(chunks)
        corpus = [tokenize(c.text) for c in self._chunks]
        # rank_bm25 divides by average document length; an empty corpus would
        # raise deep inside the library rather than here.
        if not corpus:
            raise ValueError("cannot index an empty corpus")
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b)

    def search(self, query: str, top_k: int) -> list[Scored]:
        if self._bm25 is None:
            raise RuntimeError("index() must be called before search()")
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            (Scored(c, float(s)) for c, s in zip(self._chunks, scores, strict=True)),
            key=lambda s: (-s.score, s.chunk.ordinal),
        )
        return ranked[:top_k]
