"""Run configuration.

Everything an experiment varies lives here and nothing else hardcodes it.
Switching from BM25 to dense, or from extractive to Ollama generation, is a
config change and no code change -- which is what lets runs produced on
different machines land in the same results/ format and stay comparable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Component:
    """A named strategy plus its parameters."""

    name: str
    params: dict = field(default_factory=dict)

    @classmethod
    def of(cls, raw: dict | str | None, default: str) -> Component:
        if raw is None:
            return cls(default)
        if isinstance(raw, str):
            return cls(raw)
        return cls(raw.get("name", default), dict(raw.get("params", {})))


@dataclass
class Config:
    """One eval run's full configuration. Serialised verbatim to config.json."""

    run_name: str = "unnamed"
    doc_id: str = "itcp-fundamentals"
    corpus_path: str = "data/raw/itcp-fundamentals.md"
    questions_path: str = "data/golden/questions.jsonl"
    refusals_path: str = "data/golden/refusals.jsonl"
    results_dir: str = "results"

    chunker: Component = field(
        default_factory=lambda: Component("section_aware", {"max_tokens": 220, "min_tokens": 60})
    )
    retriever: Component = field(default_factory=lambda: Component("bm25"))
    embedder: Component = field(default_factory=lambda: Component("none"))
    reranker: Component = field(default_factory=lambda: Component("none"))
    generator: Component = field(default_factory=lambda: Component("extractive"))

    ks: tuple[int, ...] = (1, 3, 5, 10)
    top_k: int = 10
    # 0.70 is the balanced-accuracy optimum from the sweep in
    # results/*/metrics.json under refusal_sweep. It was selected on the eval
    # set, which inflates it -- see "What did not work" in the README. The
    # default matches the committed runs so `make eval` with no config
    # reproduces documented behaviour rather than never refusing.
    refusal_threshold: float = 0.70
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: dict) -> Config:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise KeyError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(
            **{
                k: v
                for k, v in raw.items()
                if k not in {"chunker", "retriever", "embedder", "reranker", "generator", "ks"}
            }
        )
        cfg.chunker = Component.of(raw.get("chunker"), "section_aware")
        cfg.retriever = Component.of(raw.get("retriever"), "bm25")
        cfg.embedder = Component.of(raw.get("embedder"), "none")
        cfg.reranker = Component.of(raw.get("reranker"), "none")
        cfg.generator = Component.of(raw.get("generator"), "extractive")
        cfg.ks = tuple(raw.get("ks", cfg.ks))
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> Config:
        return cls.from_dict(json.loads(Path(path).read_text()))

    def to_dict(self) -> dict:
        out = asdict(self)
        out["ks"] = list(self.ks)
        return out

    def fingerprint(self) -> str:
        """The parts that must match for two runs to be comparable."""
        return json.dumps(
            {
                "chunker": asdict(self.chunker),
                "retriever": asdict(self.retriever),
                "embedder": asdict(self.embedder),
                "reranker": asdict(self.reranker),
                "generator": asdict(self.generator),
                "top_k": self.top_k,
            },
            sort_keys=True,
        )
