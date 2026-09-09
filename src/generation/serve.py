"""Minimal query interface.

Deliberately small. The evaluation harness is the deliverable; this exists so
the citation and refusal behaviour can be inspected by hand on an arbitrary
question, not as a product surface.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..config import Config
from ..ingest.chunkers import build_chunker
from ..ingest.models import render_path
from ..ingest.parser import parse_document
from ..retrieval.registry import build_reranker, build_retriever
from .answer import build_generator


def answer_question(cfg: Config, question: str) -> None:
    doc = parse_document(Path(cfg.corpus_path).read_text(), cfg.doc_id)
    chunks = build_chunker(cfg.chunker.name, **cfg.chunker.params).chunk(doc)
    retriever = build_retriever(cfg.retriever, cfg.embedder)
    retriever.index(chunks)
    reranker = build_reranker(cfg.reranker)
    generator = build_generator(cfg.generator)
    if hasattr(generator, "fit"):
        generator.fit(chunks)

    ranked = reranker.rerank(question, retriever.search(question, cfg.top_k), cfg.top_k)
    result = generator.generate(question, ranked[: max(cfg.ks)], cfg.refusal_threshold)

    print(f"\nQ: {question}\n")
    print(result.text)
    print(
        f"\n[refused={result.refused} support={result.support_score:.3f} "
        f"threshold={cfg.refusal_threshold}]"
    )
    print("\nRetrieved:")
    for i, scored in enumerate(ranked[:5], start=1):
        print(f"  {i}. {scored.score:6.3f}  {render_path(scored.chunk.primary)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask one question against the corpus.")
    parser.add_argument("--config", help="path to a run config JSON file")
    parser.add_argument("--question", "-q", help="the question; omit for an interactive prompt")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    if args.question:
        answer_question(cfg, args.question)
        return 0

    print("Ask a question about the ITCP corpus. Ctrl-D to exit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if question:
            answer_question(cfg, question)


if __name__ == "__main__":
    raise SystemExit(main())
