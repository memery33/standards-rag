"""Golden-set validation tests.

The golden set is the measuring instrument. If it drifts from the corpus --
a renamed heading, an edited sentence -- every number the harness produces
becomes confidently wrong, silently. These tests fail loudly instead.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.eval.golden import (
    ANSWER_TYPES,
    REFUSAL_CATEGORIES,
    load_questions,
    load_refusals,
    validate,
)
from src.ingest.parser import parse_document

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "raw" / "itcp-fundamentals.md"
QUESTIONS = ROOT / "data" / "golden" / "questions.jsonl"
REFUSALS = ROOT / "data" / "golden" / "refusals.jsonl"


@pytest.fixture(scope="module")
def doc():
    return parse_document(CORPUS.read_text(), "itcp-fundamentals")


@pytest.fixture(scope="module")
def questions():
    return load_questions(QUESTIONS)


@pytest.fixture(scope="module")
def refusals():
    return load_refusals(REFUSALS)


def test_golden_set_matches_the_corpus(questions, refusals, doc):
    """Every heading path exists and every answer span is verbatim."""
    validate(questions, refusals, doc).raise_if_invalid()


def test_ids_are_unique_across_both_files(questions, refusals):
    ids = [q.id for q in questions] + [r.id for r in refusals]
    duplicates = [i for i, n in Counter(ids).items() if n > 1]
    assert not duplicates, f"reused ids: {duplicates}"


def test_answer_types_and_refusal_categories_are_known(questions, refusals):
    assert {q.answer_type for q in questions} <= ANSWER_TYPES
    assert {r.refusal_category for r in refusals} <= REFUSAL_CATEGORIES


def test_all_four_refusal_categories_are_represented(refusals):
    assert {r.refusal_category for r in refusals} == REFUSAL_CATEGORIES


def test_set_is_weighted_toward_failure_modes_not_lookup(questions):
    """The composition claim in the README, enforced.

    A golden set that drifts back toward easy lookups would quietly inflate
    every future run, so the weighting is a test rather than a note.
    """
    lookups = sum(1 for q in questions if q.answer_type == "lookup")
    assert lookups / len(questions) < 0.30, "set has drifted toward lookup"
    hard = sum(1 for q in questions if q.difficulty == "hard")
    assert hard >= len(questions) * 0.40


def test_multi_section_questions_are_well_represented(questions):
    """Strict vs loose recall only means anything if multi-gold exists."""
    multi = [q for q in questions if len(q.source_ids) > 1]
    assert len(multi) >= 8, "too few multi-section questions to exercise strict recall"


def test_cross_reference_is_filterable_on_its_own(questions):
    tagged = [q for q in questions if "cross_reference" in q.tests]
    assert len(tagged) >= 4
    assert any("cross_reference" in q.tests and "synthesis" not in q.tests for q in tagged), (
        "cross_reference must be separable from synthesis, not a subset of it"
    )


def test_gold_targets_are_not_mostly_duplicates(questions):
    distinct = {tuple(sorted(q.source_ids)) for q in questions}
    assert len(distinct) >= len(questions) * 0.70, (
        "too many questions share a retrieval target; the set is rephrasing itself"
    )


def test_refusal_distractors_give_the_retriever_something_to_fabricate_from(refusals):
    """A refusal question with no plausible context proves nothing."""
    with_distractors = [r for r in refusals if r.distractor_paths]
    assert len(with_distractors) == len(refusals)


def test_product_commentary_section_is_never_gold(questions):
    """The trailing 'What this means for Spotter' section is not standards text.

    It is in-corpus and lexically prominent -- 'Spotter' appears 20 times, and
    the 8 in this section are the single largest cluster -- so it is kept as a
    distractor. Citing it as authority for an ITCP requirement is a failure.
    """
    forbidden = ("ITCP Fundamentals", "What this means for Spotter")
    offenders = [q.id for q in questions if forbidden in q.source_ids]
    assert not offenders, f"{offenders} cite product commentary as standards"
