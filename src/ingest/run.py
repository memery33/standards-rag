"""Parse and chunk the corpus, printing the chunk inventory.

Useful on its own: it shows what a chunking configuration actually produced
before any retrieval is involved, which is where most surprises live.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import Config
from .chunkers import build_chunker
from .models import render_path
from .parser import parse_document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse and chunk data/raw.")
    parser.add_argument("--config", help="path to a run config JSON file")
    parser.add_argument("--json", action="store_true", help="emit the inventory as JSON")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    doc = parse_document(Path(cfg.corpus_path).read_text(), cfg.doc_id)
    chunker = build_chunker(cfg.chunker.name, **cfg.chunker.params)
    chunks = chunker.chunk(doc)

    if args.json:
        print(
            json.dumps(
                [
                    {
                        "chunk_id": c.chunk_id,
                        "primary": render_path(c.primary),
                        "covers": list(c.covers_str),
                        "n_tokens": c.n_tokens,
                        "contains_table": c.contains_table,
                        "oversize": c.oversize,
                    }
                    for c in chunks
                ],
                indent=2,
            )
        )
        return 0

    print(f"corpus   {cfg.corpus_path}")
    print(f"words    {len(doc.source.split())}")
    print(f"sections {len(doc.nodes())}")
    print(f"chunker  {chunker.name} {cfg.chunker.params or ''}")
    print(f"chunks   {len(chunks)}\n")
    for c in chunks:
        flags = "".join(["T" if c.contains_table else "-", "O" if c.oversize else "-"])
        print(f"  {c.chunk_id}  {flags}  {c.n_tokens:4d}tok  {len(c.covers)} path(s)")
        print(f"      primary: {render_path(c.primary)}")
        if len(c.covers) > 1:
            for path in c.covers_str:
                if path != render_path(c.primary):
                    print(f"      also:    {path}")
    tokens = [c.n_tokens for c in chunks]
    print(
        f"\ntokens/chunk  min={min(tokens)} mean={sum(tokens) / len(tokens):.1f} max={max(tokens)}"
    )
    print(
        f"paths/chunk   mean={sum(len(c.covers) for c in chunks) / len(chunks):.2f}  "
        f"(higher means chunks straddle more sections, which inflates loose recall)"
    )
    print(f"tables intact {sum(c.contains_table for c in chunks)}  oversize {sum(c.oversize for c in chunks)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
