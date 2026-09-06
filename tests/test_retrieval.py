"""Retrieval and generation tests.

Focused on the behaviours the eval depends on: config-driven construction,
rank ordering, fusion, and the refusal decision.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Component, Config
from src.generation.answer import (
    ExtractiveGenerator,
    build_generator,
    build_idf,
    parse_citations,
    support_score,
)
from src.ingest.chunkers import SectionAwareChunker
from src.ingest.parser import parse_document
from src.retrieval.base import Scored, normalise
from src.retrieval.dense import DenseRetriever, HashingEmbedder
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.lexical import BM25Retriever, tokenize
from src.retrieval.registry import build_reranker, build_retriever
from src.retrieval.rerank import LexicalOverlapReranker, NoOpReranker

CORPUS = "data/raw/itcp-fundamentals.md"


@pytest.fixture(scope="module")
def chunks():
    doc = parse_document(Path(CORPUS).read_text(), "itcp")
    return SectionAwareChunker().chunk(doc)


@pytest.fixture(scope="module")
def bm25(chunks):
    r = BM25Retriever()
    r.index(chunks)
    return r


# --- tokenisation ----------------------------------------------------------


def test_negation_survives_tokenisation():
    """'not' must not be stoplisted: it inverts a requirement."""
    assert "not" in tokenize("Workers must not enter the swing radius")


def test_stopwords_are_removed():
    assert "the" not in tokenize("the spotter")
    assert "spotter" in tokenize("the spotter")


# --- BM25 ------------------------------------------------------------------


def test_bm25_ranks_the_right_section_first(bm25):
    top = bm25.search("What training and conduct is required of spotters?", 1)[0]
    assert top.chunk.primary[-1].startswith("Personnel responsibility matrix")


def test_bm25_respects_top_k(bm25):
    assert len(bm25.search("spotter", 3)) == 3


def test_bm25_ranking_is_monotonic(bm25):
    scores = [s.score for s in bm25.search("blind area equipment operator", 5)]
    assert scores == sorted(scores, reverse=True)


def test_bm25_rejects_use_before_index():
    with pytest.raises(RuntimeError):
        BM25Retriever().search("x", 1)


def test_bm25_rejects_empty_corpus():
    with pytest.raises(ValueError):
        BM25Retriever().index([])


# --- dense and hybrid (offline embedder; plumbing only) --------------------


def test_dense_retriever_runs_with_a_deterministic_embedder(chunks):
    r = DenseRetriever(embedder=HashingEmbedder())
    r.index(chunks)
    results = r.search("spotter training", 3)
    assert len(results) == 3
    assert [s.score for s in results] == sorted([s.score for s in results], reverse=True)


def test_hybrid_rrf_returns_chunks_from_both_arms(chunks):
    h = HybridRetriever(lexical=BM25Retriever(), dense=DenseRetriever(embedder=HashingEmbedder()))
    h.index(chunks)
    assert len(h.search("worker visibility apparel", 5)) == 5


def test_hybrid_rejects_an_unknown_fusion_mode(chunks):
    h = HybridRetriever(
        lexical=BM25Retriever(), dense=DenseRetriever(embedder=HashingEmbedder()), fusion="nope"
    )
    h.index(chunks)
    with pytest.raises(ValueError):
        h.search("x", 3)


def test_normalise_handles_a_degenerate_spread(chunks):
    flat = [Scored(chunks[0], 1.0), Scored(chunks[1], 1.0)]
    assert [s.score for s in normalise(flat)] == [0.0, 0.0]
    assert normalise([]) == []


# --- reranking -------------------------------------------------------------


def test_noop_reranker_preserves_order(bm25):
    candidates = bm25.search("spotter", 5)
    assert [s.chunk.chunk_id for s in NoOpReranker().rerank("spotter", candidates, 5)] == [
        s.chunk.chunk_id for s in candidates
    ]


def test_lexical_reranker_keeps_the_requested_number(bm25):
    candidates = bm25.search("night milling spotter", 8)
    assert len(LexicalOverlapReranker().rerank("night milling spotter", candidates, 3)) == 3


def test_lexical_reranker_handles_no_candidates():
    assert LexicalOverlapReranker().rerank("q", [], 3) == []


# --- config-driven construction -------------------------------------------


def test_registry_builds_from_config():
    assert build_retriever(Component("bm25"), Component("none")).name == "bm25"
    assert build_reranker(Component("lexical_overlap")).name == "lexical_overlap"
    assert build_generator(Component("extractive")).name == "extractive"


def test_dense_without_an_embedder_fails_with_an_actionable_message():
    with pytest.raises(ValueError, match="embedder"):
        build_retriever(Component("dense"), Component("none"))


def test_unknown_components_are_rejected():
    for build, spec in [
        (lambda: build_retriever(Component("nope"), Component("none")), None),
        (lambda: build_reranker(Component("nope")), None),
        (lambda: build_generator(Component("nope")), None),
    ]:
        with pytest.raises(KeyError):
            build()


def test_config_round_trips_and_rejects_unknown_keys():
    cfg = Config.from_dict({"run_name": "x", "retriever": {"name": "bm25"}, "ks": [1, 3]})
    assert cfg.ks == (1, 3)
    assert Config.from_dict(cfg.to_dict()).fingerprint() == cfg.fingerprint()
    with pytest.raises(KeyError):
        Config.from_dict({"nonsense": 1})


def test_fingerprint_changes_when_a_varied_component_changes():
    a = Config()
    b = Config()
    b.retriever = Component("hybrid")
    assert a.fingerprint() != b.fingerprint()


# --- generation and refusal ------------------------------------------------


def test_idf_weighting_penalises_out_of_corpus_terms(chunks):
    """The reason plain term coverage was not enough.

    An out-of-corpus question shares common words with the corpus, so
    unweighted coverage rates it as answerable. IDF weighting must score it
    below an in-corpus question about the same subject.
    """
    idf = build_idf(chunks)
    r = BM25Retriever()
    r.index(chunks)
    oov = support_score(
        "What does OSHA 29 CFR 1926 require for internal traffic control plans?",
        r.search("OSHA 1926 internal traffic control", 5),
        idf,
        len(chunks),
    )
    in_corpus = support_score(
        "What training and conduct is required of spotters?",
        r.search("spotter training conduct", 5),
        idf,
        len(chunks),
    )
    assert oov < in_corpus


def test_generator_refuses_below_threshold_and_cites_nothing(chunks, bm25):
    g = ExtractiveGenerator()
    g.fit(chunks)
    answer = g.generate("anything at all", bm25.search("anything at all", 5), threshold=1.01)
    assert answer.refused is True
    assert answer.citations == (), "a refusal must not carry a citation"


def test_generator_cites_the_section_it_answered_from(chunks, bm25):
    g = ExtractiveGenerator()
    g.fit(chunks)
    q = "What training and conduct is required of spotters?"
    answer = g.generate(q, bm25.search(q, 5), threshold=0.0)
    assert not answer.refused
    assert answer.citations
    assert any("Personnel responsibility matrix" in p[-1] for p in answer.citations)


def test_generator_keeps_table_rows_whole(chunks, bm25):
    """A row split on its internal periods strands a duty from its role."""
    g = ExtractiveGenerator()
    g.fit(chunks)
    q = "What training and conduct is required of spotters?"
    answer = g.generate(q, bm25.search(q, 5), threshold=0.0)
    if "| Spotters |" in answer.text:
        line = next(ln for ln in answer.text.splitlines() if "| Spotters |" in ln)
        assert line.rstrip().endswith("|"), f"table row was truncated: {line!r}"


def test_empty_context_refuses(chunks):
    g = ExtractiveGenerator()
    g.fit(chunks)
    assert g.generate("anything", [], threshold=0.5).refused is True


def test_parse_citations_reads_a_sources_line():
    text = "Some answer.\n\nSources: ITCP Fundamentals > Daily use; ITCP Fundamentals > Human factors"
    assert parse_citations(text) == (
        ("ITCP Fundamentals", "Daily use"),
        ("ITCP Fundamentals", "Human factors"),
    )
    assert parse_citations("No sources line here.") == ()
