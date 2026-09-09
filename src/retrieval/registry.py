"""Config-driven construction of the retrieval stack.

The whole point: an experiment names a component in config.json and nothing
in the calling code changes. Backends unreachable from a given machine fail
here, at construction, with a message naming the fix -- not deep inside a
run that has already written half a results directory.
"""

from __future__ import annotations

from ..config import Component
from .base import Retriever
from .dense import DenseRetriever, Embedder, HashingEmbedder, OllamaEmbedder
from .hybrid import HybridRetriever
from .lexical import BM25Retriever
from .rerank import (
    CrossEncoderReranker,
    LexicalOverlapReranker,
    NoOpReranker,
    Reranker,
)

EMBEDDERS = {"ollama": OllamaEmbedder, "hashing": HashingEmbedder}
RERANKERS = {
    "none": NoOpReranker,
    "lexical_overlap": LexicalOverlapReranker,
    "cross_encoder": CrossEncoderReranker,
}


def build_embedder(spec: Component) -> Embedder:
    if spec.name in {"none", ""}:
        raise ValueError(
            "this retriever needs an embedder; set config.embedder, e.g. "
            '{"name": "ollama", "params": {"model": "nomic-embed-text"}}'
        )
    if spec.name not in EMBEDDERS:
        raise KeyError(f"unknown embedder {spec.name!r}; have {sorted(EMBEDDERS)}")
    return EMBEDDERS[spec.name](**spec.params)


def build_reranker(spec: Component) -> Reranker:
    if spec.name not in RERANKERS:
        raise KeyError(f"unknown reranker {spec.name!r}; have {sorted(RERANKERS)}")
    return RERANKERS[spec.name](**spec.params)


def build_retriever(spec: Component, embedder_spec: Component) -> Retriever:
    if spec.name == "bm25":
        return BM25Retriever(**spec.params)
    if spec.name == "dense":
        return DenseRetriever(embedder=build_embedder(embedder_spec), **spec.params)
    if spec.name == "hybrid":
        params = dict(spec.params)
        return HybridRetriever(
            lexical=BM25Retriever(),
            dense=DenseRetriever(embedder=build_embedder(embedder_spec)),
            **params,
        )
    raise KeyError(f"unknown retriever {spec.name!r}; have ['bm25', 'dense', 'hybrid']")
