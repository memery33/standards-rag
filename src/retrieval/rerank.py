"""Optional reranking stage.

Two rerankers, both config-selectable and both off by default.

LexicalOverlapReranker runs anywhere and is honest about what it is: a
feature-based reorderer, not a neural cross-encoder. It is reported under
its own name so a run using it is never read as a cross-encoder result.

CrossEncoderReranker is the real thing and is implemented but unrun here --
it needs a model backend this environment cannot reach.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from .base import Scored, normalise
from .lexical import tokenize

SENTENCE = re.compile(r"(?<=[.!?])\s+")


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: Sequence[Scored], top_k: int) -> list[Scored]: ...


@dataclass
class NoOpReranker:
    name: str = "none"

    def rerank(self, query: str, candidates: Sequence[Scored], top_k: int) -> list[Scored]:
        return list(candidates)[:top_k]


@dataclass
class LexicalOverlapReranker:
    """Reorder by best-matching sentence rather than whole-chunk score.

    A long chunk can outrank a short one on aggregate term frequency while
    burying the actual answer. Scoring the single best sentence rewards a
    chunk that states the answer plainly, which is the property that matters
    for citation.

    Explicitly NOT a cross-encoder. It carries no learned relevance model.
    """

    weight: float = 0.5  # blend against the first-stage score
    name: str = "lexical_overlap"

    def _best_sentence_score(self, query_terms: set[str], text: str) -> float:
        if not query_terms:
            return 0.0
        best = 0.0
        for sentence in SENTENCE.split(text):
            terms = set(tokenize(sentence))
            if not terms:
                continue
            overlap = len(query_terms & terms)
            if not overlap:
                continue
            # Coverage of the query, damped by sentence length so a long
            # sentence cannot win on breadth alone.
            best = max(best, overlap / len(query_terms) * (overlap / (overlap + len(terms - query_terms)) ** 0.5))
        return best

    def rerank(self, query: str, candidates: Sequence[Scored], top_k: int) -> list[Scored]:
        if not candidates:
            return []
        query_terms = set(tokenize(query))
        first_stage = {s.chunk.chunk_id: s.score for s in normalise(candidates)}
        rescored = [
            Scored(
                s.chunk,
                self.weight * self._best_sentence_score(query_terms, s.chunk.text)
                + (1 - self.weight) * first_stage[s.chunk.chunk_id],
            )
            for s in candidates
        ]
        return sorted(rescored, key=lambda s: (-s.score, s.chunk.ordinal))[:top_k]


@dataclass
class CrossEncoderReranker:
    """A real cross-encoder, scoring (query, chunk) jointly.

    NOT EXERCISED IN THIS REPO'S COMMITTED RUNS. Requires a model backend
    that was unreachable from the environment that produced them.
    """

    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    name: str = "cross_encoder"

    def rerank(self, query: str, candidates: Sequence[Scored], top_k: int) -> list[Scored]:
        from sentence_transformers import CrossEncoder

        encoder = CrossEncoder(self.model)
        scores = encoder.predict([(query, s.chunk.text) for s in candidates])
        rescored = [Scored(s.chunk, float(score)) for s, score in zip(candidates, scores, strict=True)]
        return sorted(rescored, key=lambda s: (-s.score, s.chunk.ordinal))[:top_k]
