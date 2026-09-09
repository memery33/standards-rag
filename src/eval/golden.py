"""Loading and validating the golden set.

A golden set that quietly drifts from the corpus produces confident, wrong
numbers, so every record is checked against the parsed document before any
run uses it: heading paths must exist, and answer spans must appear verbatim
in the source. `make eval` refuses to run on an invalid set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..ingest.models import Document, HeadingPath, parse_path

ANSWER_TYPES = {"lookup", "synthesis", "comparison", "conditional"}
REFUSAL_CATEGORIES = {
    "adjacent_domain",
    "false_premise",
    "over_specific_numeric",
    "out_of_jurisdiction",
}


@dataclass(frozen=True)
class GoldenQuestion:
    id: str
    question: str
    answer_spans: tuple[str, ...]
    source_ids: tuple[HeadingPath, ...]
    answer_type: str
    tests: tuple[str, ...]
    difficulty: str
    notes: str

    @property
    def gold(self) -> set[HeadingPath]:
        return set(self.source_ids)


@dataclass(frozen=True)
class RefusalQuestion:
    id: str
    question: str
    refusal_category: str
    notes: str
    distractor_paths: tuple[HeadingPath, ...] = ()


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            joined = "\n  - ".join(self.errors)
            raise ValueError(f"golden set is not valid:\n  - {joined}")


def load_questions(path: str | Path) -> list[GoldenQuestion]:
    out: list[GoldenQuestion] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(
            GoldenQuestion(
                id=row["id"],
                question=row["question"],
                answer_spans=tuple(row["answer_spans"]),
                source_ids=tuple(parse_path(s) for s in row["source_ids"]),
                answer_type=row["answer_type"],
                tests=tuple(row.get("tests", ())),
                difficulty=row["difficulty"],
                notes=row.get("notes", ""),
            )
        )
    return out


def load_refusals(path: str | Path) -> list[RefusalQuestion]:
    out: list[RefusalQuestion] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        out.append(
            RefusalQuestion(
                id=row["id"],
                question=row["question"],
                refusal_category=row["refusal_category"],
                notes=row.get("notes", ""),
                distractor_paths=tuple(
                    parse_path(s) for s in row.get("distractor_paths", ())
                ),
            )
        )
    return out


def validate(
    questions: list[GoldenQuestion],
    refusals: list[RefusalQuestion],
    doc: Document,
) -> ValidationReport:
    """Check the golden set against the corpus it claims to describe."""
    report = ValidationReport()
    known = {node.heading_path for node in doc.nodes()}
    source = doc.source

    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            report.errors.append(f"{q.id}: duplicate id")
        seen.add(q.id)
        if q.answer_type not in ANSWER_TYPES:
            report.errors.append(f"{q.id}: unknown answer_type {q.answer_type!r}")
        if not q.source_ids:
            report.errors.append(f"{q.id}: no source_ids")
        for path in q.source_ids:
            if path not in known:
                report.errors.append(f"{q.id}: heading path not in corpus: {path}")
        if not q.answer_spans:
            report.errors.append(f"{q.id}: no answer_spans")
        for span in q.answer_spans:
            if span not in source:
                report.errors.append(f"{q.id}: answer span not verbatim in corpus: {span[:60]!r}")

    for r in refusals:
        if r.id in seen:
            report.errors.append(f"{r.id}: duplicate id")
        seen.add(r.id)
        if r.refusal_category not in REFUSAL_CATEGORIES:
            report.errors.append(f"{r.id}: unknown refusal_category {r.refusal_category!r}")
        for path in r.distractor_paths:
            if path not in known:
                report.errors.append(f"{r.id}: distractor path not in corpus: {path}")

    return report
