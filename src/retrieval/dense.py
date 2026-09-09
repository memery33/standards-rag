"""Dense vector retrieval over an embedding backend.

Implemented but unrun in the sandbox: the only backend wired up is Ollama on
localhost, which is not reachable here. Runs produced against it land in the
same results/ format, so they stay comparable with the BM25 runs.

Switching embedding model is a config change: {"embedder": {"name":
"ollama", "params": {"model": "nomic-embed-text"}}}.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from ..ingest.models import Chunk
from .base import Scored


class Embedder(Protocol):
    name: str
    model: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


@dataclass
class OllamaEmbedder:
    """Embeddings from a local Ollama server.

    NOT EXERCISED IN THIS REPO'S COMMITTED RUNS. No Ollama instance was
    reachable from the environment that produced them, so this path is
    unverified against a live server and is labelled as such wherever its
    results would appear.
    """

    model: str = "nomic-embed-text"
    host: str = "http://localhost:11434"
    timeout: float = 120.0
    name: str = "ollama"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        import httpx

        vectors: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for text in texts:
                response = client.post(
                    f"{self.host}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                vectors.append(response.json()["embedding"])
        return np.asarray(vectors, dtype=np.float32)


@dataclass
class HashingEmbedder:
    """A deterministic offline embedder, for testing the dense plumbing only.

    Character n-gram hashing into a fixed-width vector. It is not a semantic
    model and must never be presented as an embedding-model result -- it
    exists so the dense and hybrid code paths can be exercised and tested
    without a server, not to stand in for one.
    """

    dim: int = 256
    ngram: int = 4
    model: str = "hashing-ngram"
    name: str = "hashing"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            lowered = f" {text.lower()} "
            for i in range(max(len(lowered) - self.ngram + 1, 0)):
                gram = lowered[i : i + self.ngram]
                out[row, hash(gram) % self.dim] += 1.0
        return out


def cosine(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(vector)
    denom[denom == 0] = 1e-12
    return (matrix @ vector) / denom


@dataclass
class DenseRetriever:
    embedder: Embedder
    name: str = "dense"
    _chunks: list[Chunk] = field(default_factory=list, repr=False)
    _matrix: np.ndarray | None = field(default=None, repr=False)

    def index(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            raise ValueError("cannot index an empty corpus")
        self._chunks = list(chunks)
        self._matrix = self.embedder.embed([c.text for c in self._chunks])

    def search(self, query: str, top_k: int) -> list[Scored]:
        if self._matrix is None:
            raise RuntimeError("index() must be called before search()")
        scores = cosine(self._matrix, self.embedder.embed([query])[0])
        ranked = sorted(
            (Scored(c, float(s)) for c, s in zip(self._chunks, scores, strict=True)),
            key=lambda s: (-s.score, s.chunk.ordinal),
        )
        return ranked[:top_k]
