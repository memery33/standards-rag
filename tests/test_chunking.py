"""Chunking tests.

These target the properties the eval depends on: tables survive intact,
heading paths are recoverable from any chunk, and both strategies report
coverage against the same ground truth.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest

from src.ingest.chunkers import (
    FixedWindowChunker,
    SectionAwareChunker,
    build_chunker,
    word_tokens,
)
from src.ingest.parser import parse_document

FIXTURE = Path(__file__).parent / "fixtures" / "mini_itcp.md"


@pytest.fixture
def doc():
    return parse_document(FIXTURE.read_text(), "mini")


# --- parsing ---------------------------------------------------------------


def test_heading_paths_are_nested(doc):
    paths = {n.heading_path for n in doc.nodes()}
    assert ("ITCP Fundamentals", "Operation templates", "Asphalt milling") in paths
    assert ("ITCP Fundamentals", "Worker visibility") in paths


def test_table_is_bound_to_its_owning_section(doc):
    by_path = doc.by_path()
    visibility = by_path[("ITCP Fundamentals", "Worker visibility")]
    assert len(visibility.tables) == 1
    assert visibility.tables[0].n_rows == 2
    assert by_path[("ITCP Fundamentals", "Operation templates")].tables == []


def test_modality_distinguishes_shall_should_must_and_negation(doc):
    by_path = doc.by_path()
    visibility = by_path[("ITCP Fundamentals", "Worker visibility")]
    forms = {m.normative for m in visibility.modality_spans}
    assert "should" in forms
    assert "must not" in forms, "negation must not collapse into bare 'must'"

    milling = by_path[("ITCP Fundamentals", "Operation templates", "Asphalt milling")]
    assert {m.verb for m in milling.modality_spans} == {"shall"}


def test_steps_capture_cross_references(doc):
    steps = doc.by_path()[("ITCP Fundamentals", "The eight-step process")].steps
    assert [s.index for s in steps] == [1, 2, 3]
    assert steps[1].step_refs == (1,)
    assert steps[2].step_refs == (2,)
    assert steps[0].step_refs == (), "step 1 refers to nothing"


# --- section-aware chunking ------------------------------------------------


def test_section_aware_never_splits_a_table(doc):
    chunks = SectionAwareChunker(max_tokens=25, min_tokens=5).chunk(doc)
    table_chunks = [c for c in chunks if "| Class |" in c.body]
    assert len(table_chunks) == 1, "the table appeared in more than one chunk"
    body = table_chunks[0].body
    for row in ("| 2 | Daylight", "| 3 | Night"):
        assert row in body, f"table row {row!r} was lost or split off"


def test_oversize_atomic_block_is_kept_whole_and_flagged(doc):
    chunks = SectionAwareChunker(max_tokens=8, min_tokens=1).chunk(doc)
    table_chunks = [c for c in chunks if "| Class |" in c.body]
    assert len(table_chunks) == 1
    assert table_chunks[0].oversize is True
    assert table_chunks[0].n_tokens > 8


def test_breadcrumb_gives_an_orphaned_clause_its_subject(doc):
    chunks = SectionAwareChunker(max_tokens=40, min_tokens=5).chunk(doc)
    milling = [c for c in chunks if "shall use a dedicated spotter" in c.body]
    assert milling, "expected the milling requirement to be chunked"
    assert milling[0].text.startswith(
        "ITCP Fundamentals > Operation templates > Asphalt milling"
    )
    assert "Asphalt milling" not in milling[0].body.split("\n")[0]


def test_breadcrumb_can_be_disabled_without_changing_ground_truth(doc):
    with_bc = SectionAwareChunker(max_tokens=40, min_tokens=5)
    without = SectionAwareChunker(max_tokens=40, min_tokens=5, prepend_breadcrumb=False)
    a, b = with_bc.chunk(doc), without.chunk(doc)
    assert [c.covers for c in a] == [c.covers for c in b]
    assert [c.body for c in a] == [c.body for c in b]


def test_undersized_sections_merge_with_siblings(doc):
    chunks = SectionAwareChunker(max_tokens=400, min_tokens=200).chunk(doc)
    assert any(len(c.covers) > 1 for c in chunks), "nothing merged"


def test_oversized_section_splits_at_sentence_boundaries(doc):
    chunks = SectionAwareChunker(max_tokens=20, min_tokens=1).chunk(doc)
    milling = [
        c
        for c in chunks
        if c.primary == ("ITCP Fundamentals", "Operation templates", "Asphalt milling")
    ]
    assert len(milling) >= 1
    for chunk in milling:
        assert not chunk.body.startswith(","), "split mid-clause"


# --- fixed window ----------------------------------------------------------


def test_fixed_window_overlaps_and_covers_the_document(doc):
    chunks = FixedWindowChunker(size=30, overlap=10).chunk(doc)
    assert len(chunks) > 1
    for earlier, later in pairwise(chunks):
        assert later.char_start < earlier.char_end, "windows do not overlap"


def test_fixed_window_rejects_overlap_at_or_above_size(doc):
    with pytest.raises(ValueError):
        FixedWindowChunker(size=20, overlap=20).chunk(doc)


def test_fixed_window_straddles_sections_and_says_so(doc):
    chunks = FixedWindowChunker(size=40, overlap=10).chunk(doc)
    assert any(len(c.covers) > 1 for c in chunks), (
        "a naive window over this document must cross section boundaries; "
        "if it does not, coverage is being under-reported"
    )


# --- properties both strategies must satisfy -------------------------------


@pytest.mark.parametrize(
    "chunker",
    [
        FixedWindowChunker(size=40, overlap=10),
        SectionAwareChunker(max_tokens=60, min_tokens=15),
    ],
    ids=["fixed_window", "section_aware"],
)
def test_every_chunk_traces_back_to_a_real_heading_path(doc, chunker):
    known = {n.heading_path for n in doc.nodes()}
    for chunk in chunker.chunk(doc):
        assert chunk.covers, f"{chunk.chunk_id} covers nothing"
        assert set(chunk.covers) <= known
        assert chunk.primary in chunk.covers
        assert chunk.chunk_id.startswith(f"mini/{chunker.name}/")


@pytest.mark.parametrize(
    "chunker",
    [
        FixedWindowChunker(size=40, overlap=10),
        SectionAwareChunker(max_tokens=60, min_tokens=15),
    ],
    ids=["fixed_window", "section_aware"],
)
def test_char_spans_are_traceable_to_the_source(doc, chunker):
    for chunk in chunker.chunk(doc):
        assert 0 <= chunk.char_start < chunk.char_end <= len(doc.source)


def test_both_strategies_reach_every_section(doc):
    """Ground truth is strategy-independent, so neither may lose a section."""
    known = {n.heading_path for n in doc.nodes() if n.text.strip()}
    for chunker in (
        FixedWindowChunker(size=30, overlap=10),
        SectionAwareChunker(max_tokens=60, min_tokens=10),
    ):
        reached = {p for c in chunker.chunk(doc) for p in c.covers}
        missing = known - reached
        assert not missing, f"{chunker.name} never surfaces {missing}"


def test_build_chunker_is_config_driven():
    chunker = build_chunker("section_aware", max_tokens=100, min_tokens=20)
    assert isinstance(chunker, SectionAwareChunker)
    assert chunker.max_tokens == 100
    with pytest.raises(KeyError):
        build_chunker("no_such_chunker")


def test_word_tokens_is_whitespace_delimited():
    assert word_tokens("a b  c\nd") == 4
    assert word_tokens("") == 0
