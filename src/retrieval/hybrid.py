"""Hybrid retrieval: lexical and dense fused.

Two fusion modes, both config-selectable. Weighted score fusion needs the
component scores normalised first since BM25 and cosine live on different
scales; reciprocal rank fusion ignores scores entirely and is robust to that
mismatch, which is usually the safer default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..ingest.models import Chunk
from .base import Retriever, Scored, normalise


@dataclass
class HybridRetriever:
    lexical: Retriever
    dense: Retriever
    alpha: float = 0.5  # weight on lexical, score fusion only
    fusion: str = "rrf"  # "rrf" | "weighted"
    rrf_k: int = 60
    candidate_multiplier: int = 4
    name: str = "hybrid"

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.lexical.index(chunks)
        self.dense.index(chunks)
        self._size = len(chunks)

    def search(self, query: str, top_k: int) -> list[Scored]:
        # Fuse over a wider candidate pool than requested: a chunk ranked
        # just outside top_k by both retrievers can still win after fusion.
        pool = min(max(top_k * self.candidate_multiplier, top_k), self._size)
        lex = self.lexical.search(query, pool)
        den = self.dense.search(query, pool)

        by_id: dict[str, Chunk] = {}
        fused: dict[str, float] = {}

        if self.fusion == "rrf":
            for results in (lex, den):
                for rank, scored in enumerate(results, start=1):
                    by_id[scored.chunk.chunk_id] = scored.chunk
                    fused[scored.chunk.chunk_id] = fused.get(scored.chunk.chunk_id, 0.0) + 1.0 / (
                        self.rrf_k + rank
                    )
        elif self.fusion == "weighted":
            for results, weight in ((normalise(lex), self.alpha), (normalise(den), 1 - self.alpha)):
                for scored in results:
                    by_id[scored.chunk.chunk_id] = scored.chunk
                    fused[scored.chunk.chunk_id] = (
                        fused.get(scored.chunk.chunk_id, 0.0) + weight * scored.score
                    )
        else:
            raise ValueError(f"unknown fusion mode {self.fusion!r}")

        ranked = sorted(
            (Scored(by_id[cid], score) for cid, score in fused.items()),
            key=lambda s: (-s.score, s.chunk.ordinal),
        )
        return ranked[:top_k]
