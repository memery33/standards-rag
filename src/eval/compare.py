"""Build the README comparison table from committed runs.

Reads results/ rather than accepting hand-typed numbers, so the table cannot
drift from the runs it claims to summarise.
"""

from __future__ import annotations

import json
from pathlib import Path

COLUMNS = [
    ("run", None),
    ("chunker", None),
    ("retriever", None),
    ("rerank", None),
    ("R@1 strict", "recall_strict@1"),
    ("R@3 strict", "recall_strict@3"),
    ("R@3 loose", "recall_loose@3"),
    ("lift@3 strict", "lift_strict@3"),
    ("MRR@3", "mrr@3"),
    ("cite F1", None),
    ("paths/chunk", "mean_paths_per_chunk"),
]


def rows(results_dir: str | Path = "results") -> list[dict]:
    out = []
    for run_dir in sorted(Path(results_dir).iterdir()):
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "config.json"
        if not (metrics_path.exists() and config_path.exists()):
            continue
        m = json.loads(metrics_path.read_text())
        c = json.loads(config_path.read_text())
        out.append(
            {
                "run": run_dir.name,
                "chunker": c["chunker"]["name"],
                "retriever": c["retriever"]["name"],
                "rerank": c["reranker"]["name"],
                "cite F1": m["citation"]["f1"],
                **{label: m.get(key) for label, key in COLUMNS if key},
            }
        )
    return out


def render(results_dir: str | Path = "results") -> str:
    data = rows(results_dir)
    if not data:
        return "_No runs committed yet._"
    headers = [label for label, _ in COLUMNS]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for row in data:
        lines.append(
            "| " + " | ".join(f"[{row['run']}](results/{row['run']}/)" if h == "run" else str(row.get(h, "")) for h in headers) + " |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print(render())
