"""The eval harness. `make eval` runs this.

Writes one timestamped directory per run into results/, in the format
results/README.md specifies: config.json, metrics.json, per_question.jsonl,
notes.md. Per-question output is not optional -- aggregates hide which
questions failed and why, and on a 25-question set the individual failures
are the whole story.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from ..config import Config
from ..generation.answer import build_generator
from ..ingest.chunkers import build_chunker
from ..ingest.parser import parse_document
from ..retrieval.registry import build_reranker, build_retriever
from .golden import load_questions, load_refusals, validate
from .quality import (
    citation_accuracy,
    groundedness,
    modality_fidelity,
    refusal_metrics,
    refusal_sweep,
)
from .scoring import aggregate, score_question


def git_sha(path: str = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def golden_sha(path: str) -> str:
    """The golden set's blob hash.

    results/README.md rule 1: a run is only comparable to another run on the
    same golden set version, so the version is recorded rather than assumed.
    """
    try:
        return subprocess.check_output(
            ["git", "hash-object", path], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, OSError):
        return "unknown"


def run(cfg: Config) -> Path:
    doc = parse_document(Path(cfg.corpus_path).read_text(), cfg.doc_id)
    questions = load_questions(cfg.questions_path)
    refusals = load_refusals(cfg.refusals_path)

    # A drifted golden set produces confidently wrong numbers, so the run
    # refuses to start rather than producing them.
    validate(questions, refusals, doc).raise_if_invalid()

    chunker = build_chunker(cfg.chunker.name, **cfg.chunker.params)
    chunks = chunker.chunk(doc)

    retriever = build_retriever(cfg.retriever, cfg.embedder)
    retriever.index(chunks)
    reranker = build_reranker(cfg.reranker)
    generator = build_generator(cfg.generator)
    if hasattr(generator, "fit"):
        generator.fit(chunks)

    rows: list[dict] = []
    scores = []
    golds = []
    ans_support: list[float] = []
    ans_refused: list[bool] = []
    citations = []
    grounded: list[float] = []
    modality: list[float] = []
    latencies: list[float] = []
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []

    for q in questions:
        started = time.perf_counter()
        candidates = retriever.search(q.question, cfg.top_k)
        ranked = reranker.rerank(q.question, candidates, cfg.top_k)
        answer = generator.generate(q.question, ranked[: max(cfg.ks)], cfg.refusal_threshold)
        elapsed = time.perf_counter() - started

        retrieval = score_question(q.id, [s.chunk for s in ranked], q.gold, cfg.ks)
        cite = citation_accuracy(answer.citations, q.gold)
        ground = groundedness(answer, ranked)
        mod = modality_fidelity(answer, ranked)

        scores.append(retrieval)
        golds.append(q.gold)
        citations.append(cite)
        grounded.append(ground)
        if mod is not None:
            modality.append(mod)
        ans_support.append(answer.support_score)
        ans_refused.append(answer.refused)
        latencies.append(elapsed)
        prompt_tokens.append(answer.prompt_tokens)
        completion_tokens.append(answer.completion_tokens)

        row = retrieval.to_json()
        row.update(
            {
                "kind": "answerable",
                "question": q.question,
                "answer_type": q.answer_type,
                "tests": list(q.tests),
                "difficulty": q.difficulty,
                "answer": answer.text,
                "refused": answer.refused,
                "support_score": round(answer.support_score, 4),
                "citations": list(answer.citation_strs),
                "citation": cite.to_json(),
                "groundedness": round(ground, 4),
                "modality_fidelity": None if mod is None else round(mod, 4),
                "latency_s": round(elapsed, 5),
                "prompt_tokens": answer.prompt_tokens,
                "completion_tokens": answer.completion_tokens,
                "notes": q.notes,
            }
        )
        rows.append(row)

    un_support: list[float] = []
    un_refused: list[bool] = []
    for r in refusals:
        started = time.perf_counter()
        candidates = retriever.search(r.question, cfg.top_k)
        ranked = reranker.rerank(r.question, candidates, cfg.top_k)
        answer = generator.generate(r.question, ranked[: max(cfg.ks)], cfg.refusal_threshold)
        elapsed = time.perf_counter() - started

        un_support.append(answer.support_score)
        un_refused.append(answer.refused)
        latencies.append(elapsed)
        prompt_tokens.append(answer.prompt_tokens)
        completion_tokens.append(answer.completion_tokens)

        rows.append(
            {
                "kind": "refusal",
                "question_id": r.id,
                "question": r.question,
                "refusal_category": r.refusal_category,
                "expected": "refuse",
                "refused": answer.refused,
                "support_score": round(answer.support_score, 4),
                # A wrong answer here is worse than a wrong answer elsewhere:
                # it is a fabricated citation on a safety question.
                "fabricated_citations": [] if answer.refused else list(answer.citation_strs),
                "answer": answer.text,
                "retrieved_ids": [s.chunk.chunk_id for s in ranked[: max(cfg.ks)]],
                "latency_s": round(elapsed, 5),
                "notes": r.notes,
            }
        )

    metrics = aggregate(scores, chunks, golds, cfg.ks)
    metrics["corpus"] = {
        "words": len(doc.source.split()),
        "sections": len(doc.nodes()),
        "chunks": len(chunks),
        "mean_chunk_tokens": round(sum(c.n_tokens for c in chunks) / len(chunks), 1),
        "oversize_chunks": sum(c.oversize for c in chunks),
    }
    metrics["citation"] = {
        "precision": round(sum(c.precision for c in citations) / len(citations), 4),
        "recall": round(sum(c.recall for c in citations) / len(citations), 4),
        "f1": round(sum(c.f1 for c in citations) / len(citations), 4),
        "exact_match_rate": round(sum(c.exact for c in citations) / len(citations), 4),
    }
    metrics["groundedness"] = {
        "mean": round(sum(grounded) / len(grounded), 4),
        "caveat": (
            "Near-trivially high for the extractive generator, which lifts "
            "sentences verbatim from the retrieved context. Discriminates "
            "between configurations only on the LLM generation path."
        ),
    }
    metrics["modality_fidelity"] = {
        "mean": round(sum(modality) / len(modality), 4) if modality else None,
        "n_scored": len(modality),
    }
    metrics["refusal"] = refusal_metrics(ans_refused, un_refused)
    metrics["refusal"]["threshold"] = cfg.refusal_threshold
    metrics["refusal_sweep"] = refusal_sweep(ans_support, un_support)
    metrics["cost"] = {
        "mean_latency_s": round(sum(latencies) / len(latencies), 5),
        "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95)], 5),
        "total_prompt_tokens": sum(prompt_tokens),
        "total_completion_tokens": sum(completion_tokens),
        "mean_tokens_per_query": round(
            (sum(prompt_tokens) + sum(completion_tokens)) / len(latencies), 1
        ),
    }
    metrics["by_answer_type"] = {}
    for kind in sorted({q.answer_type for q in questions}):
        subset = [s for s, q in zip(scores, questions) if q.answer_type == kind]
        metrics["by_answer_type"][kind] = {
            "n": len(subset),
            **{
                f"recall_strict@{k}": round(sum(s.strict[k] for s in subset) / len(subset), 4)
                for k in cfg.ks
            },
            **{f"mrr@{k}": round(sum(s.rr[k] for s in subset) / len(subset), 4) for k in cfg.ks},
        }
    metrics["by_tag"] = {}
    for tag in sorted({t for q in questions for t in q.tests}):
        subset = [s for s, q in zip(scores, questions) if tag in q.tests]
        metrics["by_tag"][tag] = {
            "n": len(subset),
            "recall_strict@3": round(sum(s.strict[3] for s in subset) / len(subset), 4),
            "recall_loose@3": round(sum(s.loose[3] for s in subset) / len(subset), 4),
        }

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out_dir = Path(cfg.results_dir) / f"{stamp}-{cfg.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    config_payload = cfg.to_dict()
    config_payload["provenance"] = {
        "git_sha": git_sha(),
        "golden_set_sha": golden_sha(cfg.questions_path),
        "refusal_set_sha": golden_sha(cfg.refusals_path),
        "corpus_sha": golden_sha(cfg.corpus_path),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "fingerprint": cfg.fingerprint(),
    }
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    with (out_dir / "per_question.jsonl").open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    (out_dir / "notes.md").write_text(render_notes(cfg, metrics, rows))
    return out_dir


def render_notes(cfg: Config, metrics: dict, rows: list[dict]) -> str:
    """A notes.md stub with the numbers already filled in.

    The prose is for a human to complete; the figures are generated so that
    a run's notes can never disagree with its own metrics.json.
    """
    ks = cfg.ks
    failures = [
        r
        for r in rows
        if r.get("kind") == "answerable" and not r.get("recall_strict", {}).get("3", False)
    ]
    fabricated = [r for r in rows if r.get("kind") == "refusal" and r.get("fabricated_citations")]

    lines = [
        f"# {cfg.run_name}",
        "",
        f"{cfg.notes or '_What changed since the last run, and what it did to the numbers._'}",
        "",
        "## Configuration",
        "",
        f"- chunker: `{cfg.chunker.name}` {cfg.chunker.params or ''}",
        f"- retriever: `{cfg.retriever.name}` {cfg.retriever.params or ''}",
        f"- embedder: `{cfg.embedder.name}`",
        f"- reranker: `{cfg.reranker.name}`",
        f"- generator: `{cfg.generator.name}`",
        (
            f"- corpus: {metrics['corpus']['words']} words, "
            f"{metrics['corpus']['sections']} sections, "
            f"{metrics['corpus']['chunks']} chunks"
        ),
        "",
        "## Retrieval",
        "",
        "| k | recall_strict | recall_loose | MRR | random_loose | beats random |",
        "|---|---|---|---|---|---|",
    ]
    for k in ks:
        lines.append(
            f"| {k} | {metrics[f'recall_strict@{k}']} | {metrics[f'recall_loose@{k}']} | "
            f"{metrics[f'mrr@{k}']} | {metrics[f'random_recall_loose@{k}']} | "
            f"{'yes' if metrics[f'beats_random_loose@{k}'] else 'NO'} |"
        )
    lines += [
        "",
        "| k | lift_strict (recall - random) | lift_loose |",
        "|---|---|---|",
    ]
    for k in ks:
        lines.append(f"| {k} | {metrics[f'lift_strict@{k}']} | {metrics[f'lift_loose@{k}']} |")
    lines += [
        "",
        (
            f"Mean heading paths per retrieved chunk: {metrics['mean_paths_per_chunk']}. "
            "A higher value means each chunk covers more of the document, which "
            "inflates loose recall without improving citation precision."
        ),
        "",
        "## Answer quality",
        "",
        (
            f"- citation F1: {metrics['citation']['f1']} "
            f"(precision {metrics['citation']['precision']}, recall {metrics['citation']['recall']})"
        ),
        f"- citation exact match: {metrics['citation']['exact_match_rate']}",
        f"- groundedness: {metrics['groundedness']['mean']} — {metrics['groundedness']['caveat']}",
        (
            f"- modality fidelity: {metrics['modality_fidelity']['mean']} "
            f"over {metrics['modality_fidelity']['n_scored']} answers using a normative verb"
        ),
        "",
        "## Refusal",
        "",
        f"- threshold: {metrics['refusal']['threshold']}",
        f"- correct refusals: {metrics['refusal']['correct_refusals']}",
        f"- wrong refusals on answerable questions: {metrics['refusal']['wrong_refusals']}",
        f"- balanced accuracy: {metrics['refusal']['balanced_accuracy']}",
        f"- fabricated citations on out-of-corpus questions: {len(fabricated)}",
        "",
        "The full threshold sweep is in `metrics.json` under `refusal_sweep`.",
        "",
        "## Cost",
        "",
        (
            f"- mean latency: {metrics['cost']['mean_latency_s']}s, "
            f"p95 {metrics['cost']['p95_latency_s']}s"
        ),
        f"- mean tokens per query: {metrics['cost']['mean_tokens_per_query']}",
        "",
        f"## Questions failing recall_strict@3 ({len(failures)})",
        "",
    ]
    for r in failures:
        lines.append(f"- `{r['question_id']}` [{r['answer_type']}] {r['question']}")
        lines.append(f"  gold: {'; '.join(r['gold'])}")
    if fabricated:
        lines += ["", "## Fabricated citations (should be empty)", ""]
        for r in fabricated:
            lines.append(f"- `{r['question_id']}` cited {r['fabricated_citations']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a configuration against the golden set.")
    parser.add_argument("--config", help="path to a run config JSON file")
    parser.add_argument("--name", help="override run_name")
    args = parser.parse_args(argv)

    cfg = Config.load(args.config) if args.config else Config()
    if args.name:
        cfg.run_name = args.name
    out_dir = run(cfg)
    metrics = json.loads((out_dir / "metrics.json").read_text())
    print(f"wrote {out_dir}")
    print(
        f"  recall_strict@3={metrics['recall_strict@3']} "
        f"loose@3={metrics['recall_loose@3']} (random {metrics['random_recall_loose@3']}) "
        f"mrr@3={metrics['mrr@3']} citation_f1={metrics['citation']['f1']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
