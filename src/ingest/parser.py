"""Markdown -> heading tree.

Deliberately not a clause parser. The corpus has no numbered clauses, so the
structure that matters is: heading nesting, markdown tables, one numbered
process whose steps reference each other, and normative verbs.

All offsets index into `Document.source` so any chunk can be traced back to
the exact bytes it came from.
"""

from __future__ import annotations

import re

from .models import (
    Document,
    HeadingNode,
    ModalitySpan,
    ProcedureStep,
    Table,
)

ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
FENCE = re.compile(r"^\s*(```|~~~)")
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
LIST_ITEM = re.compile(r"^(\s*)(\d+)[.)][ \t]+(.*)$")
STEP_REF = re.compile(r"\bstep\s+(\d+)\b", re.IGNORECASE)

# "must" is included deliberately: the corpus uses shall/should/must with
# different weight, and conflating them is exactly the failure being tested.
MODALITY = re.compile(
    r"\b(shall|should|must|may)\b(?:[ \t]+(not|never))?",
    re.IGNORECASE,
)

# Good enough for prose that ends sentences with a period, question mark or
# colon-introduced list. Abbreviations are not handled; the corpus has few.
SENTENCE_END = re.compile(r"(?<=[.!?])[ \t]+(?=[A-Z(])")


def _line_offsets(text: str) -> list[tuple[int, int, str]]:
    """(start, end, line_without_newline) for each line."""
    out: list[tuple[int, int, str]] = []
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n").rstrip("\r")
        out.append((pos, pos + len(stripped), stripped))
        pos += len(line)
    return out


def _fenced_lines(lines: list[tuple[int, int, str]]) -> set[int]:
    """Indices of lines inside fenced code blocks, so we don't parse them."""
    inside: set[int] = set()
    fence: str | None = None
    for i, (_, _, line) in enumerate(lines):
        m = FENCE.match(line)
        if m and fence is None:
            fence = m.group(1)
            inside.add(i)
        elif m and fence and line.strip().startswith(fence):
            inside.add(i)
            fence = None
        elif fence is not None:
            inside.add(i)
    return inside


def find_tables(lines, lo: int, hi: int, skip: set[int]) -> list[Table]:
    """Markdown tables in lines[lo:hi]: a header row, a separator, then rows."""
    tables: list[Table] = []
    i = lo
    while i < hi - 1:
        if i in skip or not TABLE_ROW.match(lines[i][2]):
            i += 1
            continue
        if not TABLE_SEP.match(lines[i + 1][2]):
            i += 1
            continue
        start_line = i
        j = i + 2
        while j < hi and j not in skip and TABLE_ROW.match(lines[j][2]):
            j += 1
        header = lines[start_line][2]
        n_cols = len([c for c in header.strip().strip("|").split("|")])
        caption = ""
        k = start_line - 1
        while k >= lo and not lines[k][2].strip():
            k -= 1
        if k >= lo and not TABLE_ROW.match(lines[k][2]):
            caption = lines[k][2].strip()
        raw_start, raw_end = lines[start_line][0], lines[j - 1][1]
        tables.append(
            Table(
                caption=caption,
                raw="\n".join(lines[n][2] for n in range(start_line, j)),
                start=raw_start,
                end=raw_end,
                n_rows=j - start_line - 2,
                n_cols=n_cols,
            )
        )
        i = j
    return tables


def find_steps(lines, lo: int, hi: int, skip: set[int]) -> list[ProcedureStep]:
    """Ordered-list items in lines[lo:hi], with their cross-references.

    Continuation lines (indented, non-blank) are folded into the step above so
    a step that wraps stays one unit.
    """
    steps: list[ProcedureStep] = []
    current: dict | None = None

    def flush() -> None:
        if current is None:
            return
        text = "\n".join(current["lines"]).strip()
        refs = tuple(
            sorted({int(m.group(1)) for m in STEP_REF.finditer(text)} - {current["index"]})
        )
        steps.append(
            ProcedureStep(
                index=current["index"],
                text=text,
                start=current["start"],
                end=current["end"],
                step_refs=refs,
            )
        )

    for i in range(lo, hi):
        if i in skip:
            continue
        start, end, line = lines[i]
        m = LIST_ITEM.match(line)
        if m:
            flush()
            current = {
                "index": int(m.group(2)),
                "lines": [m.group(3)],
                "start": start,
                "end": end,
            }
        elif current is not None:
            if not line.strip():
                flush()
                current = None
            elif line.startswith((" ", "\t")):
                current["lines"].append(line.strip())
                current["end"] = end
            else:
                flush()
                current = None
    flush()
    return steps


def find_modality(text: str, offset: int) -> list[ModalitySpan]:
    """Normative verbs in `text`, with the sentence each sits in."""
    spans: list[ModalitySpan] = []
    for m in MODALITY.finditer(text):
        left = text.rfind(".", 0, m.start())
        right = text.find(".", m.end())
        sentence = text[left + 1 : right if right != -1 else len(text)].strip()
        spans.append(
            ModalitySpan(
                verb=m.group(1).lower(),
                negated=m.group(2) is not None,
                sentence=" ".join(sentence.split()),
                start=offset + m.start(),
                end=offset + m.end(),
            )
        )
    return spans


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences. Used only to break oversized sections."""
    parts = [p.strip() for p in SENTENCE_END.split(text)]
    return [p for p in parts if p]


def parse_document(source: str, doc_id: str) -> Document:
    """Build the heading tree.

    A node's own span runs from its heading line to the next heading line of
    *any* level, so own spans partition the document exactly and every
    character belongs to exactly one node. That is what makes chunk coverage
    unambiguous later.
    """
    lines = _line_offsets(source)
    skip = _fenced_lines(lines)

    heads: list[tuple[int, int, str]] = []  # (line index, level, title)
    for i, (_, _, line) in enumerate(lines):
        if i in skip:
            continue
        m = ATX_HEADING.match(line)
        if m:
            heads.append((i, len(m.group(1)), m.group(2).strip()))

    doc = Document(doc_id=doc_id, source=source)
    if not heads:
        doc.preamble = source
        return doc
    doc.preamble = source[: lines[heads[0][0]][0]]

    stack: list[HeadingNode] = []
    for n, (line_i, level, title) in enumerate(heads):
        body_lo = line_i + 1
        body_hi = heads[n + 1][0] if n + 1 < len(heads) else len(lines)
        start = lines[line_i][0]
        end = lines[body_hi][0] if body_hi < len(lines) else len(source)

        while stack and stack[-1].level >= level:
            stack.pop()
        path = tuple(node.title for node in stack) + (title,)

        tables = find_tables(lines, body_lo, body_hi, skip)
        steps = find_steps(lines, body_lo, body_hi, skip)

        body_start = lines[body_lo][0] if body_lo < len(lines) else end
        body = source[body_start:end]
        table_spans = [(t.start, t.end) for t in tables]
        prose_parts, cursor = [], body_start
        for t_start, t_end in table_spans:
            prose_parts.append(source[cursor:t_start])
            cursor = t_end
        prose_parts.append(source[cursor:end])
        prose = "".join(prose_parts)

        node = HeadingNode(
            heading_path=path,
            level=level,
            title=title,
            text=body.strip(),
            start=start,
            end=end,
            tables=tables,
            steps=steps,
            modality_spans=find_modality(prose, body_start),
        )
        if stack:
            stack[-1].children.append(node)
        else:
            doc.roots.append(node)
        stack.append(node)

    return doc
