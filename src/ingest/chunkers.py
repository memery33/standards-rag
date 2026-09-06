"""Chunking strategies behind one interface.

Both chunkers emit Chunks whose ids encode the strategy but whose `covers`
field does not. Scoring matches gold heading paths against `covers`, so the
two strategies are measured against identical ground truth no matter where
they cut.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from .models import Chunk, Document, HeadingNode, HeadingPath, render_path

TokenCounter = Callable[[str], int]


def word_tokens(text: str) -> int:
    """Whitespace token count.

    Not a model tokenizer. It is deterministic and dependency-free, which is
    what chunk-boundary decisions need; real token cost is measured separately
    against the actual model at eval time.
    """
    return len(text.split())


class Chunker(Protocol):
    name: str

    def chunk(self, doc: Document) -> list[Chunk]: ...


def _own_spans(doc: Document) -> list[tuple[int, int, HeadingPath]]:
    """Each node's own span. These partition the document."""
    spans = [(n.start, n.end, n.heading_path) for n in doc.nodes()]
    spans.sort()
    return spans


def _coverage(
    doc: Document, start: int, end: int, spans: Sequence[tuple[int, int, HeadingPath]]
) -> tuple[tuple[HeadingPath, ...], HeadingPath]:
    """Which heading paths a character range covers, and which dominates.

    A path counts as covered only if the overlap contains a non-whitespace
    character, so a window that spills one newline into the next section is
    not credited with retrieving it.
    """
    contributions: list[tuple[int, HeadingPath]] = []
    for s_start, s_end, path in spans:
        lo, hi = max(start, s_start), min(end, s_end)
        if lo >= hi:
            continue
        if not doc.source[lo:hi].strip():
            continue
        contributions.append((hi - lo, path))
    if not contributions:
        return (), ()
    contributions.sort(key=lambda c: (-c[0], c[1]))
    covers = tuple(path for _, path in sorted(contributions, key=lambda c: c[1]))
    return covers, contributions[0][1]


@dataclass
class FixedWindowChunker:
    """A sliding window over the raw source. The baseline that should lose.

    Deliberately naive: no awareness of headings, tables or steps. It will cut
    tables in half and strand "shall be worn" from the section that gives it a
    subject. Kept honest rather than strawmanned -- it windows the real source
    text, the same thing an off-the-shelf character splitter would see.
    """

    size: int = 180
    overlap: int = 40
    count_tokens: TokenCounter = word_tokens
    name: str = "fixed_window"

    def chunk(self, doc: Document) -> list[Chunk]:
        if self.overlap >= self.size:
            raise ValueError("overlap must be smaller than size")
        spans = _own_spans(doc)
        words: list[tuple[int, int]] = []
        cursor = 0
        for word in doc.source.split():
            idx = doc.source.find(word, cursor)
            words.append((idx, idx + len(word)))
            cursor = idx + len(word)

        chunks: list[Chunk] = []
        step = self.size - self.overlap
        for ordinal, lo in enumerate(range(0, max(len(words), 1), step)):
            window = words[lo : lo + self.size]
            if not window:
                break
            start, end = window[0][0], window[-1][1]
            text = doc.source[start:end]
            covers, primary = _coverage(doc, start, end, spans)
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}/{self.name}/{ordinal:04d}",
                    doc_id=doc.doc_id,
                    strategy=self.name,
                    ordinal=ordinal,
                    text=text,
                    body=text,
                    covers=covers,
                    primary=primary,
                    char_start=start,
                    char_end=end,
                    n_tokens=self.count_tokens(text),
                    contains_table="|" in text,
                )
            )
            if lo + self.size >= len(words):
                break
        return chunks


@dataclass
class _Block:
    """An atomic unit of a section: a table, a step, or a run of prose."""

    text: str
    start: int
    end: int
    atomic: bool = False
    is_table: bool = False


def _blocks(doc: Document, node: HeadingNode) -> list[_Block]:
    """Split a node's own content into blocks that may never be broken.

    Tables and numbered steps are atomic. Prose between them is splittable at
    sentence boundaries.
    """
    from .parser import split_sentences

    fixed = [(t.start, t.end, True, True) for t in node.tables]
    fixed += [(s.start, s.end, True, False) for s in node.steps]
    fixed.sort()

    body_start = node.start + len(node.title) + node.level + 1
    blocks: list[_Block] = []
    cursor = min(body_start, node.end)
    for start, end, atomic, is_table in fixed:
        if start > cursor:
            gap = doc.source[cursor:start]
            offset = cursor
            for sentence in split_sentences(gap):
                idx = doc.source.find(sentence, offset, start)
                if idx == -1:
                    continue
                blocks.append(_Block(sentence, idx, idx + len(sentence)))
                offset = idx + len(sentence)
        blocks.append(_Block(doc.source[start:end], start, end, atomic, is_table))
        cursor = max(cursor, end)
    if cursor < node.end:
        tail = doc.source[cursor : node.end]
        offset = cursor
        for sentence in split_sentences(tail):
            idx = doc.source.find(sentence, offset, node.end)
            if idx == -1:
                continue
            blocks.append(_Block(sentence, idx, idx + len(sentence)))
            offset = idx + len(sentence)
    return [b for b in blocks if b.text.strip()]


@dataclass
class SectionAwareChunker:
    """One chunk per leaf section, respecting the document's own boundaries.

    Undersized sections merge with adjacent siblings; oversized ones split at
    sentence boundaries, never inside a table or a numbered step. The heading
    path is prepended so a chunk reading "shall be worn" still knows it sits
    under "Worker visibility".
    """

    max_tokens: int = 220
    min_tokens: int = 60
    count_tokens: TokenCounter = word_tokens
    prepend_breadcrumb: bool = True
    name: str = "section_aware"

    def _emit(
        self,
        doc: Document,
        ordinal: int,
        blocks: list[_Block],
        covers: tuple[HeadingPath, ...],
        primary: HeadingPath,
        oversize: bool,
    ) -> Chunk:
        body = "\n\n".join(b.text.strip() for b in blocks).strip()
        prefix = f"{render_path(primary)}\n\n" if self.prepend_breadcrumb else ""
        text = prefix + body
        return Chunk(
            chunk_id=f"{doc.doc_id}/{self.name}/{ordinal:04d}",
            doc_id=doc.doc_id,
            strategy=self.name,
            ordinal=ordinal,
            text=text,
            body=body,
            covers=covers,
            primary=primary,
            char_start=blocks[0].start,
            char_end=blocks[-1].end,
            n_tokens=self.count_tokens(text),
            contains_table=any(b.is_table for b in blocks),
            oversize=oversize,
        )

    def chunk(self, doc: Document) -> list[Chunk]:
        if self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must not exceed max_tokens")

        units: list[tuple[HeadingPath, list[_Block]]] = []
        for node in doc.nodes():
            blocks = _blocks(doc, node)
            if blocks:
                units.append((node.heading_path, blocks))

        chunks: list[Chunk] = []
        pending: list[tuple[HeadingPath, list[_Block]]] = []

        def flush() -> None:
            if not pending:
                return
            blocks = [b for _, bs in pending for b in bs]
            sizes: dict[HeadingPath, int] = {}
            for path, bs in pending:
                sizes[path] = sizes.get(path, 0) + sum(len(b.text) for b in bs)
            primary = max(sorted(sizes), key=lambda p: sizes[p])
            covers = tuple(sorted(sizes))
            chunks.append(self._emit(doc, len(chunks), blocks, covers, primary, False))
            pending.clear()

        for path, blocks in units:
            size = self.count_tokens("\n\n".join(b.text for b in blocks))

            if size > self.max_tokens:
                flush()
                current: list[_Block] = []
                for block in blocks:
                    block_size = self.count_tokens(block.text)
                    current_size = self.count_tokens("\n\n".join(b.text for b in current))
                    if current and current_size + block_size > self.max_tokens:
                        chunks.append(
                            self._emit(doc, len(chunks), current, (path,), path, False)
                        )
                        current = []
                    if block_size > self.max_tokens:
                        # An atomic block bigger than the budget stays whole.
                        # Flagged, never split -- a halved table is worse than
                        # an oversized chunk.
                        chunks.append(
                            self._emit(doc, len(chunks), [block], (path,), path, True)
                        )
                        continue
                    current.append(block)
                if current:
                    chunks.append(self._emit(doc, len(chunks), current, (path,), path, False))
                continue

            pending.append((path, blocks))
            pending_size = self.count_tokens(
                "\n\n".join(b.text for _, bs in pending for b in bs)
            )
            if pending_size >= self.min_tokens:
                flush()
        flush()
        return chunks


CHUNKERS: dict[str, type] = {
    FixedWindowChunker.name: FixedWindowChunker,
    SectionAwareChunker.name: SectionAwareChunker,
}


def build_chunker(name: str, **kwargs) -> Chunker:
    """Config-driven construction. Nothing an experiment varies is hardcoded."""
    if name not in CHUNKERS:
        raise KeyError(f"unknown chunker {name!r}; have {sorted(CHUNKERS)}")
    return CHUNKERS[name](**kwargs)
