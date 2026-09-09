"""Retriever interface and shared types."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..ingest.models import Chunk


@dataclass(frozen=True)
class Scored:
    chunk: Chunk
    score: float


class Retriever(Protocol):
    name: str

    def index(self, chunks: Sequence[Chunk]) -> None: ...

    def search(self, query: str, top_k: int) -> list[Scored]: ...


def normalise(scored: Sequence[Scored]) -> list[Scored]:
    """Min-max normalise scores to [0, 1].

    Needed before fusing lexical and dense scores, which live on
    incomparable scales. A degenerate spread collapses to zero rather than
    dividing by it -- that case means the retriever ranked nothing, and
    inventing a spread would manufacture confidence the run does not have.
    """
    if not scored:
        return []
    lo = min(s.score for s in scored)
    hi = max(s.score for s in scored)
    if hi - lo < 1e-12:
        return [Scored(s.chunk, 0.0) for s in scored]
    return [Scored(s.chunk, (s.score - lo) / (hi - lo)) for s in scored]
