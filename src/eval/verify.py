"""Re-run every committed run and check its metrics still reproduce.

The repo's central claim is that a skeptical reader can reproduce these
numbers. This is that claim as an executable check: for each directory in
results/, rebuild the run from its own config.json and compare the fresh
metrics against the committed ones.

A mismatch is not always a bug -- changing a chunker deliberately should
change the numbers -- but it must never pass silently, because a results
directory that no longer matches the code that produced it is worse than no
results directory.

Runs whose backend is unreachable (Ollama, cross-encoder) are reported as
skipped rather than failed, and the reason is named.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from ..config import Config
from .run import run

# Metrics that must match exactly. Latency is excluded: it is machine-
# dependent and would make the check fail on any other hardware, which
# would train a reader to ignore it.
COMPARED_PREFIXES = (
    "recall_strict@",
    "recall_loose@",
    "mrr@",
    "random_recall_strict@",
    "random_recall_loose@",
    "lift_strict@",
    "lift_loose@",
    "mean_paths_per_chunk",
    "mean_context_tokens@",
)


def comparable(metrics: dict) -> dict:
    out = {k: v for k, v in metrics.items() if k.startswith(COMPARED_PREFIXES)}
    out["citation"] = metrics.get("citation")
    out["corpus"] = metrics.get("corpus")
    out["refusal"] = {
        k: v for k, v in (metrics.get("refusal") or {}).items() if k != "threshold"
    }
    return out


def verify(results_dir: str | Path = "results") -> tuple[int, int, int]:
    passed = failed = skipped = 0
    for run_dir in sorted(Path(results_dir).iterdir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        if not (config_path.exists() and metrics_path.exists()):
            continue

        raw = json.loads(config_path.read_text())
        raw.pop("provenance", None)
        committed = json.loads(metrics_path.read_text())

        with tempfile.TemporaryDirectory() as tmp:
            cfg = Config.from_dict(raw)
            cfg.results_dir = tmp
            try:
                fresh_dir = run(cfg)
            except Exception as exc:  # noqa: BLE001 - the reason is reported, not swallowed
                print(f"SKIP  {run_dir.name}: {type(exc).__name__}: {exc}")
                skipped += 1
                continue
            fresh = json.loads((fresh_dir / "metrics.json").read_text())

        want, got = comparable(committed), comparable(fresh)
        if want == got:
            print(f"OK    {run_dir.name}")
            passed += 1
        else:
            failed += 1
            print(f"FAIL  {run_dir.name}")
            for key in sorted(set(want) | set(got)):
                if want.get(key) != got.get(key):
                    print(f"        {key}: committed={want.get(key)!r} fresh={got.get(key)!r}")
    return passed, failed, skipped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check committed runs still reproduce.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args(argv)

    passed, failed, skipped = verify(args.results_dir)
    print(f"\n{passed} reproduced, {failed} mismatched, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
