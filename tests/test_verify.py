"""Tests for the reproducibility check itself.

`make verify` is the repo's strongest claim -- that a reader can re-derive
every committed number. If the checker silently passed on a mismatch, the
claim would be worse than absent, so its failure path is tested explicitly.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.verify import comparable, verify

ROOT = Path(__file__).resolve().parent.parent


def test_committed_runs_reproduce():
    passed, failed, _skipped = verify(ROOT / "results")
    assert failed == 0, "a committed run no longer reproduces from its own config"
    assert passed >= 4, "expected at least the four BM25 runs to be checked"


def test_comparable_excludes_machine_dependent_fields():
    """Latency must not be compared: it would fail on any other hardware."""
    metrics = json.loads((ROOT / "results" / "2026-09-06-sectionaware-bm25" / "metrics.json").read_text())
    fields = comparable(metrics)
    assert "cost" not in fields
    assert "recall_strict@3" in fields
    assert "threshold" not in fields["refusal"], "a config choice is not a result"


def test_verify_detects_a_mismatch(tmp_path):
    """The failure path, which is the only part that matters."""
    source = ROOT / "results" / "2026-09-06-sectionaware-bm25"
    target = tmp_path / source.name
    target.mkdir(parents=True)
    (target / "config.json").write_text((source / "config.json").read_text())

    tampered = json.loads((source / "metrics.json").read_text())
    tampered["recall_strict@3"] = 0.99
    (target / "metrics.json").write_text(json.dumps(tampered))

    passed, failed, _skipped = verify(tmp_path)
    assert failed == 1, "a tampered metric must not pass verification"
    assert passed == 0
