"""Answer synthesis with citations, and the refusal path.

The refusal path matters as much as the answer path, so it is a first-class
decision made before any answer is composed rather than a fallback when
composition fails.

Two generators. ExtractiveGenerator runs anywhere and composes an answer
from sentences actually present in the retrieved chunks. OllamaGenerator is
implemented but unrun in the committed results.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from ..ingest.models import HeadingPath, render_path
from ..retrieval.base import Scored
from ..retrieval.lexical import tokenize

SENTENCE = re.compile(r"(?<=[.!?])\s+")

REFUSAL_TEXT = (
    "The retrieved sections do not support an answer to this question. "
    "No citation is given because none would be accurate."
)


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[HeadingPath, ...]
    refused: bool
    support_score: float
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def citation_strs(self) -> tuple[str, ...]:
        return tuple(render_path(p) for p in self.citations)


class Generator(Protocol):
    name: str

    def generate(self, question: str, context: Sequence[Scored], threshold: float) -> Answer: ...


def build_idf(chunks: Sequence) -> dict[str, float]:
    """Inverse document frequency over the chunk corpus.

    Needed because unweighted term coverage cannot separate an answerable
    question from an out-of-corpus one. Every refusal question in the golden
    set is lexically adjacent by design -- "What does OSHA 29 CFR 1926
    require for internal traffic control plans?" shares five common words
    with the corpus and scores 0.55 unweighted, indistinguishable from a real
    question. Weighting by IDF makes the terms that matter the rare ones, and
    a term the corpus has never seen carries the maximum weight while being
    permanently uncovered.
    """
    from math import log

    n = len(chunks)
    df: dict[str, int] = {}
    for chunk in chunks:
        for term in set(tokenize(chunk.text)):
            df[term] = df.get(term, 0) + 1
    return {term: log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def max_idf(idf: dict[str, float], n_chunks: int) -> float:
    """Weight for a term absent from the corpus entirely."""
    from math import log

    return log(n_chunks + 1) + 1.0


def support_score(
    question: str,
    context: Sequence[Scored],
    idf: dict[str, float] | None = None,
    n_chunks: int = 0,
) -> float:
    """How much of the question's content the retrieved context accounts for.

    Measured against the question rather than the top retrieval score: BM25
    scores are unbounded and corpus-relative, so thresholding on them would
    move the refusal boundary whenever the corpus changed. This is bounded in
    [0, 1] and means the same thing across runs and across retrievers.

    With an IDF map, coverage is weighted so rare terms dominate. Without
    one it degrades to plain term coverage, which is retained only so the
    function is usable before the corpus is indexed.
    """
    terms = set(tokenize(question))
    if not terms or not context:
        return 0.0
    covered: set[str] = set()
    for scored in context:
        covered |= terms & set(tokenize(scored.chunk.text))

    if not idf:
        return len(covered) / len(terms)

    ceiling = max_idf(idf, n_chunks or len(idf))
    weights = {term: idf.get(term, ceiling) for term in terms}
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return sum(weights[term] for term in covered) / total


def _split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping table rows whole.

    A markdown row like "| Spotters | Receive training. Stay visible. |" is
    one citable unit. Splitting it on periods strands half a duty from the
    role it belongs to, which in this corpus is a wrong citation rather than
    an untidy one.
    """
    out: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("|"):
            out.append(stripped)
        else:
            out.extend(s.strip() for s in SENTENCE.split(stripped) if s.strip())
    return out


@dataclass
class ExtractiveGenerator:
    """Compose an answer from sentences present verbatim in the context.

    A real limitation, stated plainly: because every sentence is lifted from
    the retrieved chunks, groundedness is near 1.0 by construction. That
    metric therefore says little about this generator and is reported mainly
    as a control for the LLM path, where it can actually fail. Citation
    accuracy is the metric that discriminates here.
    """

    max_sentences: int = 4
    name: str = "extractive"
    idf: dict[str, float] = field(default_factory=dict, repr=False)
    n_chunks: int = 0

    def fit(self, chunks: Sequence) -> None:
        """Learn corpus term statistics used by the refusal decision."""
        self.idf = build_idf(chunks)
        self.n_chunks = len(chunks)

    def generate(self, question: str, context: Sequence[Scored], threshold: float) -> Answer:
        score = support_score(question, context, self.idf, self.n_chunks)
        if score < threshold or not context:
            return Answer(REFUSAL_TEXT, (), True, score)

        terms = set(tokenize(question))
        candidates: list[tuple[float, int, str, HeadingPath]] = []
        for rank, scored in enumerate(context):
            for sentence in _split_sentences(scored.chunk.body):
                sentence_terms = set(tokenize(sentence))
                if not sentence_terms:
                    continue
                overlap = len(terms & sentence_terms)
                if not overlap:
                    continue
                # Rank position breaks ties toward the better-retrieved chunk.
                candidates.append(
                    (overlap / len(terms), rank, sentence, scored.chunk.primary)
                )

        if not candidates:
            return Answer(REFUSAL_TEXT, (), True, score)

        candidates.sort(key=lambda c: (-c[0], c[1]))
        chosen = candidates[: self.max_sentences]

        cited: list[HeadingPath] = []
        for _, _, _, path in chosen:
            if path not in cited:
                cited.append(path)

        # Table rows go on their own line. Space-joining them runs one row
        # into the next, which visually reattaches a duty to the wrong role
        # even though each row was selected whole.
        parts: list[str] = []
        for _, _, sentence, _ in chosen:
            if sentence.startswith("|"):
                parts.append("\n" + sentence + "\n")
            else:
                parts.append(sentence)
        body = " ".join(parts).replace("\n ", "\n").replace(" \n", "\n").strip()
        body = re.sub(r"\n{2,}", "\n", body)
        cites = "; ".join(render_path(p) for p in cited)
        text = f"{body}\n\nSources: {cites}"
        return Answer(
            text=text,
            citations=tuple(cited),
            refused=False,
            support_score=score,
            prompt_tokens=sum(len(s.chunk.text.split()) for s in context),
            completion_tokens=len(text.split()),
        )


PROMPT = """You are answering questions about an Internal Traffic Control Plan \
standards reference. Use ONLY the numbered sections below.

Rules:
- Cite the exact section heading paths you used, under a final "Sources:" line.
- Preserve normative force exactly: shall, should, must and may are not \
interchangeable, and a condition attached to a requirement must be kept.
- If the sections do not support an answer, reply with exactly: {refusal}
  Do not cite anything in that case.

Sections:
{context}

Question: {question}
Answer:"""


@dataclass
class OllamaGenerator:
    """Answer synthesis via a local Ollama model.

    NOT EXERCISED IN THIS REPO'S COMMITTED RUNS. No Ollama server was
    reachable from the environment that produced them, so this path is
    unverified against a live model.
    """

    model: str = "llama3.1:8b"
    host: str = "http://localhost:11434"
    timeout: float = 300.0
    temperature: float = 0.0
    name: str = "ollama"
    idf: dict[str, float] = field(default_factory=dict, repr=False)
    n_chunks: int = 0

    def fit(self, chunks: Sequence) -> None:
        self.idf = build_idf(chunks)
        self.n_chunks = len(chunks)

    def generate(self, question: str, context: Sequence[Scored], threshold: float) -> Answer:
        import httpx

        score = support_score(question, context, self.idf, self.n_chunks)
        blocks = "\n\n".join(
            f"[{i}] {render_path(s.chunk.primary)}\n{s.chunk.body}"
            for i, s in enumerate(context, start=1)
        )
        prompt = PROMPT.format(context=blocks, question=question, refusal=REFUSAL_TEXT)

        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                },
            )
            response.raise_for_status()
            payload = response.json()

        text = payload.get("response", "").strip()
        refused = REFUSAL_TEXT[:60].lower() in text.lower()
        return Answer(
            text=text,
            citations=() if refused else parse_citations(text),
            refused=refused,
            support_score=score,
            prompt_tokens=payload.get("prompt_eval_count", 0),
            completion_tokens=payload.get("eval_count", 0),
        )


def parse_citations(text: str) -> tuple[HeadingPath, ...]:
    """Pull heading paths out of a generated answer's Sources line."""
    from ..ingest.models import parse_path

    match = re.search(r"^sources?\s*:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if not match:
        return ()
    out: list[HeadingPath] = []
    for raw in re.split(r"[;\n]", match.group(1)):
        cleaned = raw.strip().strip(".").lstrip("-* ").strip()
        if not cleaned:
            continue
        path = parse_path(cleaned)
        if path and path not in out:
            out.append(path)
    return tuple(out)


GENERATORS = {"extractive": ExtractiveGenerator, "ollama": OllamaGenerator}


def build_generator(spec) -> Generator:
    if spec.name not in GENERATORS:
        raise KeyError(f"unknown generator {spec.name!r}; have {sorted(GENERATORS)}")
    return GENERATORS[spec.name](**spec.params)
