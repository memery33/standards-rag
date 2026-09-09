"""Core document and chunk types.

The corpus this targets is markdown with no clause numbers, so the stable
anchor is the heading path: ("Operation templates", "Asphalt milling").
Heading paths are the ground truth the golden set records and the only thing
both chunkers are scored against, no matter where they cut.
"""

from __future__ import annotations

from dataclasses import dataclass, field

HeadingPath = tuple[str, ...]

PATH_SEP = " > "


def render_path(path: HeadingPath) -> str:
    """Render a heading path the way the golden set and citations write it."""
    return PATH_SEP.join(path)


def parse_path(rendered: str) -> HeadingPath:
    """Inverse of render_path, for reading heading paths out of JSONL."""
    return tuple(part.strip() for part in rendered.split(PATH_SEP) if part.strip())


@dataclass(frozen=True)
class ModalitySpan:
    """One normative verb occurrence.

    Retrieval does not use these, but citation scoring does: an answer that
    says "should" where the source says "shall" is a citation failure, not a
    near miss. That distinction only exists if it is recorded at parse time.
    """

    verb: str  # normalised: shall | should | must | may
    negated: bool
    sentence: str
    start: int  # char offset into the source document
    end: int

    @property
    def normative(self) -> str:
        """The full normative form, e.g. 'shall not'."""
        return f"{self.verb} not" if self.negated else self.verb


@dataclass(frozen=True)
class Table:
    """A markdown table. Atomic: never split across chunks."""

    caption: str  # nearest preceding non-empty line, often the binding prose
    raw: str
    start: int
    end: int
    n_rows: int  # body rows, excluding the header and separator
    n_cols: int


@dataclass(frozen=True)
class ProcedureStep:
    """One step of a numbered process.

    step_refs carries the cross-references between steps ("as identified in
    Step 3"). In a document with no clause numbers these are the only
    intra-section cross-references that exist, so they are worth keeping.
    """

    index: int  # 1-based, as written in the document
    text: str
    start: int
    end: int
    step_refs: tuple[int, ...] = ()


@dataclass
class HeadingNode:
    """A node in the heading tree.

    `text`, `tables` and `steps` cover only what this node owns directly --
    content appearing before any child heading. Descendant content belongs to
    the descendant. This keeps own-spans a true partition of the document,
    which is what makes chunk coverage unambiguous.
    """

    heading_path: HeadingPath
    level: int
    title: str
    text: str
    start: int  # start of this node's heading line
    end: int  # start of the next heading line of any level
    tables: list[Table] = field(default_factory=list)
    steps: list[ProcedureStep] = field(default_factory=list)
    modality_spans: list[ModalitySpan] = field(default_factory=list)
    children: list[HeadingNode] = field(default_factory=list)

    @property
    def path_str(self) -> str:
        return render_path(self.heading_path)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def walk(self):
        """Depth-first, document order, including self."""
        yield self
        for child in self.children:
            yield from child.walk()


@dataclass
class Document:
    doc_id: str
    source: str  # the raw markdown, verbatim; all offsets index into this
    roots: list[HeadingNode] = field(default_factory=list)
    preamble: str = ""  # any text before the first heading

    def nodes(self) -> list[HeadingNode]:
        return [node for root in self.roots for node in root.walk()]

    def by_path(self) -> dict[HeadingPath, HeadingNode]:
        return {node.heading_path: node for node in self.nodes()}


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit.

    The id encodes the strategy that produced it; the ground truth does not.
    `covers` is what scoring matches gold heading paths against, so a fixed
    window that straddles three sections is credited for all three -- and
    `covers` length is reported per run so that credit stays visible rather
    than quietly inflating recall.
    """

    chunk_id: str  # {doc_id}/{strategy}/{ordinal:04d}
    doc_id: str
    strategy: str
    ordinal: int
    text: str  # what gets embedded, including any breadcrumb prefix
    body: str  # the same content without the breadcrumb prefix
    covers: tuple[HeadingPath, ...]
    primary: HeadingPath
    char_start: int
    char_end: int
    n_tokens: int
    contains_table: bool = False
    oversize: bool = False  # exceeded max_tokens but was not splittable

    @property
    def covers_str(self) -> tuple[str, ...]:
        return tuple(render_path(p) for p in self.covers)
