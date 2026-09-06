"""Answer-side metrics: groundedness, citation accuracy, refusal behaviour.

Citation accuracy is the metric that discriminates between configurations
here. Groundedness is reported but is near-trivially high for the extractive
generator, which lifts its sentences verbatim from the retrieved context --
that is stated wherever the number appears rather than left for a reader to
discover.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..generation.answer import Answer
from ..ingest.models import HeadingPath
from ..retrieval.base import Scored
from ..retrieval.lexical import tokenize


def groundedness(answer: Answer, context: Sequence[Scored], threshold: float = 0.6) -> float:
    """Fraction of answer sentences whose terms appear in the retrieved context.

    A blunt instrument: it detects wholesale fabrication, not subtle
    misstatement. An answer that flips "shall" to "should" while reusing
    every other word scores a perfect 1.0 here, which is why modality
    fidelity is scored separately.
    """
    from ..generation.answer import _split_sentences

    if answer.refused:
        return 1.0
    sentences = [s for s in _split_sentences(answer.text) if not s.lower().startswith("sources")]
    if not sentences:
        return 0.0
    context_terms: set[str] = set()
    for scored in context:
        context_terms |= set(tokenize(scored.chunk.text))

    supported = 0
    for sentence in sentences:
        terms = set(tokenize(sentence))
        if not terms:
            continue
        if len(terms & context_terms) / len(terms) >= threshold:
            supported += 1
    return supported / len(sentences)


@dataclass(frozen=True)
class CitationScore:
    precision: float
    recall: float
    f1: float
    exact: bool

    def to_json(self) -> dict:
        return {
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "exact": self.exact,
        }


def citation_accuracy(cited: Sequence[HeadingPath], gold: set[HeadingPath]) -> CitationScore:
    """Compare cited sections against the golden set's source_ids.

    Precision matters as much as recall in this domain: citing four sections
    when one is correct is not a better answer than citing the one, because
    a reader checking the citation is sent to text that does not support the
    claim.
    """
    cited_set = set(cited)
    if not cited_set and not gold:
        return CitationScore(1.0, 1.0, 1.0, True)
    if not cited_set:
        return CitationScore(0.0, 0.0, 0.0, False)
    hits = len(cited_set & gold)
    precision = hits / len(cited_set)
    recall = hits / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return CitationScore(precision, recall, f1, cited_set == gold)


def modality_fidelity(answer: Answer, context: Sequence[Scored]) -> float | None:
    """Whether normative verbs in the answer also appear in the context.

    Returns None when the answer uses no normative verb, so questions that
    do not turn on modality are excluded from the average rather than
    scoring a free 1.0.

    This is the check groundedness cannot make: swapping "shall" for
    "should" preserves nearly every token but changes the legal weight of
    the requirement, which in a standards document is the whole point.
    """
    from ..ingest.parser import MODALITY

    if answer.refused:
        return None
    answer_forms = {
        (m.group(1).lower(), m.group(2) is not None) for m in MODALITY.finditer(answer.text)
    }
    if not answer_forms:
        return None
    context_forms = {
        (m.group(1).lower(), m.group(2) is not None)
        for scored in context
        for m in MODALITY.finditer(scored.chunk.text)
    }
    return len(answer_forms & context_forms) / len(answer_forms)


def refusal_metrics(
    answerable_refused: Sequence[bool], unanswerable_refused: Sequence[bool]
) -> dict:
    """Both directions of the refusal decision.

    A system that refuses everything scores a perfect refusal rate, so the
    refusal rate alone is not a safety metric. Over-refusal is reported
    beside it and balanced accuracy summarises the trade.
    """
    n_ans = len(answerable_refused)
    n_un = len(unanswerable_refused)
    correct_refusals = sum(unanswerable_refused)
    over_refusals = sum(answerable_refused)
    refusal_rate = correct_refusals / n_un if n_un else 0.0
    over_rate = over_refusals / n_ans if n_ans else 0.0
    return {
        "refusal_rate": round(refusal_rate, 4),
        "over_refusal_rate": round(over_rate, 4),
        "balanced_accuracy": round((refusal_rate + (1 - over_rate)) / 2, 4),
        "correct_refusals": f"{correct_refusals}/{n_un}",
        "wrong_refusals": f"{over_refusals}/{n_ans}",
    }


def refusal_sweep(
    answerable_scores: Sequence[float],
    unanswerable_scores: Sequence[float],
    thresholds: Sequence[float] = tuple(i / 20 for i in range(4, 20)),
) -> list[dict]:
    """The whole trade-off curve, not one chosen point.

    Committed because a single refusal number invites picking the threshold
    that flatters the run. The curve shows what any other choice would have
    cost.
    """
    out = []
    for t in thresholds:
        ans_refused = [s < t for s in answerable_scores]
        un_refused = [s < t for s in unanswerable_scores]
        row = {"threshold": round(t, 3)}
        row.update(refusal_metrics(ans_refused, un_refused))
        out.append(row)
    return out
