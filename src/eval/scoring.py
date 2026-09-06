"""Retrieval scoring.

Two decisions here carry most of the weight:

1. Recall is reported strict AND loose. Strict requires every gold section in
   the top k; loose requires one. Multi-section questions look far better
   under loose, so reporting only loose would flatter exactly the questions
   the golden set was weighted toward.

2. Every metric is reported beside a random-retrieval baseline. On a small
   corpus, recall@10 over ~15 chunks means retrieving two-thirds of the
   document -- random scores ~0.67 and a system scoring 0.71 has measured
   nothing. The baseline makes that visible instead of letting a large
   number pass for a good one.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from math import comb
from typing import Iterable, Sequence

from ..ingest.models import Chunk, HeadingPath

DEFAULT_KS = (1, 3, 5, 10)


def covered_paths(chunks: Iterable[Chunk]) -> set[HeadingPath]:
    """Union of the heading paths a set of chunks covers."""
    covered: set[HeadingPath] = set()
    for chunk in chunks:
        covered.update(chunk.covers)
    return covered


def recall_at_k(
    retrieved: Sequence[Chunk], gold: set[HeadingPath], k: int
) -> tuple[bool, bool]:
    """(strict, loose) recall at k.

    strict: every gold section appears somewhere in the top k.
    loose:  at least one does.
    """
    if not gold:
        raise ValueError("recall is undefined for an empty gold set")
    covered = covered_paths(retrieved[:k])
    return gold <= covered, bool(gold & covered)


def reciprocal_rank(
    retrieved: Sequence[Chunk], gold: set[HeadingPath], k: int | None = None
) -> float:
    """1/rank of the first chunk covering any gold section, else 0."""
    if not gold:
        raise ValueError("reciprocal rank is undefined for an empty gold set")
    window = retrieved if k is None else retrieved[:k]
    for rank, chunk in enumerate(window, start=1):
        if gold & set(chunk.covers):
            return 1.0 / rank
    return 0.0


def _hit_chunks(corpus: Sequence[Chunk], gold: set[HeadingPath]) -> int:
    """How many chunks in the corpus cover at least one gold section."""
    return sum(1 for chunk in corpus if gold & set(chunk.covers))


def random_loose_recall(corpus_size: int, hit_count: int, k: int) -> float:
    """Exact probability that k chunks drawn at random include a hit.

    Closed form: 1 - C(N-m, k) / C(N, k). No sampling noise, so the baseline
    printed next to a real score is itself reproducible.
    """
    if corpus_size <= 0 or hit_count <= 0:
        return 0.0
    k = min(k, corpus_size)
    misses = corpus_size - hit_count
    if misses < k:
        return 1.0
    return 1.0 - comb(misses, k) / comb(corpus_size, k)


def random_strict_recall(
    corpus: Sequence[Chunk], gold: set[HeadingPath], k: int, trials: int = 2000, seed: int = 0
) -> float:
    """Monte Carlo estimate for strict recall under random retrieval.

    Strict recall has no clean closed form once chunks cover overlapping sets
    of sections, so this samples. The seed is fixed, so the number is stable
    across runs and can be diffed like any other metric.
    """
    if not corpus or not gold:
        return 0.0
    k = min(k, len(corpus))
    rng = random.Random(seed)
    indices = range(len(corpus))
    hits = 0
    for _ in range(trials):
        sample = rng.sample(indices, k)
        if gold <= covered_paths(corpus[i] for i in sample):
            hits += 1
    return hits / trials


@dataclass
class QuestionScore:
    """Per-question retrieval outcome. Written to per_question.jsonl."""

    question_id: str
    gold: tuple[HeadingPath, ...]
    retrieved_ids: tuple[str, ...]
    strict: dict[int, bool] = field(default_factory=dict)
    loose: dict[int, bool] = field(default_factory=dict)
    rr: dict[int, float] = field(default_factory=dict)
    first_hit_rank: int | None = None
    context_tokens: dict[int, int] = field(default_factory=dict)
    paths_per_chunk: float = 0.0

    def to_json(self) -> dict:
        return {
            "question_id": self.question_id,
            "gold": [" > ".join(p) for p in self.gold],
            "retrieved_ids": list(self.retrieved_ids),
            "recall_strict": {str(k): v for k, v in sorted(self.strict.items())},
            "recall_loose": {str(k): v for k, v in sorted(self.loose.items())},
            "mrr": {str(k): round(v, 6) for k, v in sorted(self.rr.items())},
            "first_hit_rank": self.first_hit_rank,
            "context_tokens": {str(k): v for k, v in sorted(self.context_tokens.items())},
            "paths_per_chunk": round(self.paths_per_chunk, 4),
        }


def score_question(
    question_id: str,
    retrieved: Sequence[Chunk],
    gold: set[HeadingPath],
    ks: Sequence[int] = DEFAULT_KS,
) -> QuestionScore:
    """Score one question's retrieval at every k.

    context_tokens and paths_per_chunk are recorded alongside recall on
    purpose. A fixed window straddling three sections is credited for all
    three, so it can win loose recall simply by dragging in more text. These
    two numbers make that trade visible in the same row.
    """
    score = QuestionScore(
        question_id=question_id,
        gold=tuple(sorted(gold)),
        retrieved_ids=tuple(c.chunk_id for c in retrieved),
    )
    for k in ks:
        strict, loose = recall_at_k(retrieved, gold, k)
        score.strict[k] = strict
        score.loose[k] = loose
        score.rr[k] = reciprocal_rank(retrieved, gold, k)
        score.context_tokens[k] = sum(c.n_tokens for c in retrieved[:k])
    for rank, chunk in enumerate(retrieved, start=1):
        if gold & set(chunk.covers):
            score.first_hit_rank = rank
            break
    if retrieved:
        score.paths_per_chunk = sum(len(c.covers) for c in retrieved) / len(retrieved)
    return score


def aggregate(
    scores: Sequence[QuestionScore],
    corpus: Sequence[Chunk],
    golds: Sequence[set[HeadingPath]],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict:
    """Aggregate per-question scores, each beside its random baseline.

    `beats_random` is the field worth reading first: a metric that does not
    clear its own baseline is not evidence, however large it looks.
    """
    if not scores:
        return {"n_questions": 0}

    n = len(scores)
    out: dict = {
        "n_questions": n,
        "corpus_chunks": len(corpus),
        "mean_paths_per_chunk": round(
            sum(s.paths_per_chunk for s in scores) / n, 4
        ),
    }
    for k in ks:
        strict = sum(s.strict[k] for s in scores) / n
        loose = sum(s.loose[k] for s in scores) / n
        mrr = sum(s.rr[k] for s in scores) / n
        rand_loose = (
            sum(
                random_loose_recall(len(corpus), _hit_chunks(corpus, g), k)
                for g in golds
            )
            / len(golds)
            if golds
            else 0.0
        )
        rand_strict = (
            sum(random_strict_recall(corpus, g, k) for g in golds) / len(golds)
            if golds
            else 0.0
        )
        out[f"recall_strict@{k}"] = round(strict, 4)
        out[f"recall_loose@{k}"] = round(loose, 4)
        out[f"mrr@{k}"] = round(mrr, 4)
        out[f"random_recall_strict@{k}"] = round(rand_strict, 4)
        out[f"random_recall_loose@{k}"] = round(rand_loose, 4)
        out[f"beats_random_loose@{k}"] = bool(loose > rand_loose)
        out[f"beats_random_strict@{k}"] = bool(strict > rand_strict)
        out[f"mean_context_tokens@{k}"] = round(
            sum(s.context_tokens[k] for s in scores) / n, 1
        )
    return out
