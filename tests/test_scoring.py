"""Scoring tests.

The eval harness is the deliverable, so its arithmetic is the thing that most
needs to be trustworthy. These tests pin the strict/loose distinction, MRR
rank semantics, and the random baseline that keeps a big number on a small
corpus from passing for a good one.
"""

from __future__ import annotations

import pytest

from src.eval.scoring import (
    aggregate,
    covered_paths,
    random_loose_recall,
    random_strict_recall,
    recall_at_k,
    reciprocal_rank,
    score_question,
)
from src.ingest.models import Chunk

A = ("Doc", "Alpha")
B = ("Doc", "Beta")
C = ("Doc", "Gamma")


def make_chunk(ordinal: int, covers: tuple, n_tokens: int = 10) -> Chunk:
    return Chunk(
        chunk_id=f"doc/test/{ordinal:04d}",
        doc_id="doc",
        strategy="test",
        ordinal=ordinal,
        text="x",
        body="x",
        covers=covers,
        primary=covers[0],
        char_start=ordinal,
        char_end=ordinal + 1,
        n_tokens=n_tokens,
    )


# --- strict vs loose -------------------------------------------------------


def test_strict_requires_every_gold_section_loose_requires_one():
    retrieved = [make_chunk(0, (A,)), make_chunk(1, (C,))]
    strict, loose = recall_at_k(retrieved, {A, B}, k=2)
    assert loose is True
    assert strict is False, "one of two gold sections must not count as strict"


def test_strict_and_loose_agree_on_a_single_section_question():
    retrieved = [make_chunk(0, (A,))]
    assert recall_at_k(retrieved, {A}, k=1) == (True, True)
    assert recall_at_k([make_chunk(0, (B,))], {A}, k=1) == (False, False)


def test_strict_satisfied_when_gold_sections_span_several_chunks():
    retrieved = [make_chunk(0, (A,)), make_chunk(1, (B,))]
    assert recall_at_k(retrieved, {A, B}, k=2) == (True, True)
    assert recall_at_k(retrieved, {A, B}, k=1) == (False, True)


def test_k_truncates_the_ranking():
    retrieved = [make_chunk(0, (C,)), make_chunk(1, (A,))]
    assert recall_at_k(retrieved, {A}, k=1) == (False, False)
    assert recall_at_k(retrieved, {A}, k=2) == (True, True)


def test_a_multi_section_chunk_can_satisfy_strict_alone():
    """A window straddling both gold sections is credited for both.

    This is the loose-recall inflation the harness reports paths_per_chunk to
    expose -- correct behaviour here, but never reported without its cost.
    """
    assert recall_at_k([make_chunk(0, (A, B))], {A, B}, k=1) == (True, True)


def test_empty_gold_is_rejected_rather_than_scored():
    with pytest.raises(ValueError):
        recall_at_k([make_chunk(0, (A,))], set(), k=1)
    with pytest.raises(ValueError):
        reciprocal_rank([make_chunk(0, (A,))], set())


# --- MRR -------------------------------------------------------------------


def test_reciprocal_rank_uses_the_first_gold_section():
    retrieved = [make_chunk(0, (C,)), make_chunk(1, (B,)), make_chunk(2, (A,))]
    assert reciprocal_rank(retrieved, {A, B}) == pytest.approx(1 / 2)


def test_reciprocal_rank_is_one_at_rank_one_and_zero_on_a_miss():
    assert reciprocal_rank([make_chunk(0, (A,))], {A}) == 1.0
    assert reciprocal_rank([make_chunk(0, (C,))], {A}) == 0.0


def test_reciprocal_rank_respects_k():
    retrieved = [make_chunk(0, (C,)), make_chunk(1, (A,))]
    assert reciprocal_rank(retrieved, {A}, k=1) == 0.0
    assert reciprocal_rank(retrieved, {A}, k=2) == pytest.approx(0.5)


# --- random baseline -------------------------------------------------------


def test_random_loose_recall_matches_the_closed_form():
    # 1 hit in 10 chunks, k=1 -> 0.1
    assert random_loose_recall(10, 1, 1) == pytest.approx(0.1)
    # 1 hit in 10, k=5 -> 0.5
    assert random_loose_recall(10, 1, 5) == pytest.approx(0.5)


def test_random_loose_recall_is_near_certain_on_a_tiny_corpus():
    """The number that motivates growing the corpus.

    With 15 chunks and 2 hits, drawing 10 at random finds one ~95% of the
    time. A system reporting recall_loose@10 = 0.9 on such a corpus is
    performing worse than chance.
    """
    assert random_loose_recall(15, 2, 10) > 0.9


def test_random_loose_recall_saturates_and_floors():
    assert random_loose_recall(10, 10, 1) == pytest.approx(1.0)
    assert random_loose_recall(10, 0, 5) == 0.0
    assert random_loose_recall(0, 0, 5) == 0.0
    assert random_loose_recall(5, 1, 99) == pytest.approx(1.0), "k clamps to N"


def test_random_strict_recall_is_deterministic_and_bounded():
    corpus = [make_chunk(i, (A,) if i == 0 else (B,) if i == 1 else (C,)) for i in range(10)]
    first = random_strict_recall(corpus, {A, B}, k=5, trials=500, seed=7)
    second = random_strict_recall(corpus, {A, B}, k=5, trials=500, seed=7)
    assert first == second, "fixed seed must make the baseline diffable"
    assert 0.0 <= first <= 1.0
    assert first < random_strict_recall(corpus, {A, B}, k=9, trials=500, seed=7)


def test_random_strict_is_never_above_random_loose():
    corpus = [make_chunk(i, (A,) if i == 0 else (B,) if i == 1 else (C,)) for i in range(12)]
    strict = random_strict_recall(corpus, {A, B}, k=4, trials=2000, seed=3)
    loose = random_loose_recall(12, 2, 4)
    assert strict <= loose + 1e-9


# --- per-question and aggregate -------------------------------------------


def test_score_question_records_rank_context_and_span():
    retrieved = [make_chunk(0, (C,), 12), make_chunk(1, (A, B), 8)]
    score = score_question("q1", retrieved, {A}, ks=(1, 2))
    assert score.first_hit_rank == 2
    assert score.strict[1] is False and score.strict[2] is True
    assert score.rr[2] == pytest.approx(0.5)
    assert score.context_tokens[2] == 20
    assert score.paths_per_chunk == pytest.approx(1.5)


def test_score_question_json_is_serialisable_and_stable():
    score = score_question("q1", [make_chunk(0, (A,))], {A}, ks=(1,))
    payload = score.to_json()
    assert payload["gold"] == ["Doc > Alpha"]
    assert payload["recall_strict"] == {"1": True}
    assert payload["retrieved_ids"] == ["doc/test/0000"]


def test_aggregate_flags_a_score_that_does_not_beat_random():
    """A perfect-looking score on a corpus small enough to exhaust."""
    corpus = [make_chunk(i, (A,) if i < 2 else (C,)) for i in range(3)]
    scores = [score_question("q1", corpus[:3], {A}, ks=(3,))]
    out = aggregate(scores, corpus, [{A}], ks=(3,))
    assert out["recall_loose@3"] == 1.0
    assert out["random_recall_loose@3"] == pytest.approx(1.0)
    assert out["beats_random_loose@3"] is False, (
        "retrieving the whole corpus must not read as a win"
    )


def test_aggregate_reports_a_genuine_win_as_a_win():
    corpus = [make_chunk(i, (A,) if i == 0 else (C,)) for i in range(50)]
    scores = [score_question("q1", [corpus[0]] + corpus[1:], {A}, ks=(1,))]
    out = aggregate(scores, corpus, [{A}], ks=(1,))
    assert out["recall_loose@1"] == 1.0
    assert out["random_recall_loose@1"] == pytest.approx(0.02)
    assert out["beats_random_loose@1"] is True


def test_aggregate_on_no_questions_is_empty_not_a_crash():
    assert aggregate([], [], [], ks=(1,)) == {"n_questions": 0}


def test_covered_paths_unions_across_chunks():
    assert covered_paths([make_chunk(0, (A, B)), make_chunk(1, (C,))]) == {A, B, C}
