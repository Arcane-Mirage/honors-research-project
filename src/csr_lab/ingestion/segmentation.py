"""Deterministic segmentation of document text into candidate requirement blocks.

The segmenter only *proposes* blocks. Deciding which blocks are requirements is
done later (by the LLM in the staged workflows). Headings, front matter and
document identifiers are never emitted as requirement candidates.

Recognised layouts:
* numbered paragraphs:            ``4.2.1 The supplier shall ...``
* prefixed chapter labels:        ``TSA 2.1 The supplier shall ...`` (like ``VDA 6.5``)
* headings followed by paragraphs: ``3.1 Control Plans`` / ``The supplier shall ...``
* multi-line paragraphs, lettered sub-items and pipe tables (continuation lines)

Every segment records exact character offsets into the document text, so the
segment text is always an exact slice of the source.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field

from csr_lab.schemas import InputDocument, PageSpan

SegmentKind = Literal["front_matter", "heading", "numbered_paragraph", "paragraph"]

# "4.2.1 Text", "TSA 2.1 Text", "7. Text". The label must be followed by an
# upper-case letter or "(" so wrapped lines such as "10 years after ..." are not
# mistaken for new sections.
_LABEL_RE = re.compile(r"^(?P<label>(?:[A-Z]{2,6}[ \t])?\d{1,3}(?:\.\d{1,3}){0,5})\.?[ \t]+(?P<rest>[A-Z(\"'].*)$")
MODAL_RE = re.compile(r"\b(shall|must|is required to|are required to|is required|are required)\b", re.IGNORECASE)
_TABLE_RE = re.compile(r"^\s*\|")
_LONG_PARAGRAPH_CHARS = 2500


class Segment(BaseModel):
    segment_id: str
    document_id: str
    kind: SegmentKind
    chapter: str | None = None
    label: str | None = Field(default=None, description="Label printed on this segment's own first line")
    title: str | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    start_offset: int
    end_offset: int
    line_start_offset: int = Field(description="Start of the first line, including any label")
    page_start: int | None = None
    page_end: int | None = None
    has_modal: bool = False
    confidence: Literal["high", "low"] = "low"
    notes: list[str] = Field(default_factory=list)

    @property
    def is_candidate(self) -> bool:
        return self.kind in ("numbered_paragraph", "paragraph")


def _label_depth(label: str) -> int:
    return label.split()[-1].count(".") + 1


def _looks_like_heading(rest: str) -> bool:
    stripped = rest.strip()
    words = stripped.split()
    if not words or len(words) > 10:
        return False
    if stripped.endswith((".", ":", ";", ",")) or MODAL_RE.search(stripped):
        return False
    capitalised = sum(1 for w in words if w[:1].isupper() or w.lower() in {"and", "of", "for", "the", "to", "in"})
    return capitalised / len(words) >= 0.6


def page_range(start: int, end: int, pages: list[PageSpan]) -> tuple[int | None, int | None]:
    if not pages:
        return None, None
    first = last = None
    for page in pages:
        if page.start_offset <= start <= page.end_offset and first is None:
            first = page.page_number
        if page.start_offset <= max(start, end - 1) <= page.end_offset:
            last = page.page_number
    return first, last or first


def _iter_lines(text: str) -> list[tuple[int, int, str]]:
    """(start, end, content) for each line; end excludes the newline."""
    lines = []
    pos = 0
    for raw in text.split("\n"):
        lines.append((pos, pos + len(raw), raw))
        pos += len(raw) + 1
    return lines


class _Block:
    def __init__(self, kind: SegmentKind, line_start: int, text_start: int, label: str | None, chapter: str | None):
        self.kind = kind
        self.line_start = line_start
        self.text_start = text_start
        self.text_end = text_start
        self.label = label
        self.chapter = chapter
        self.title: str | None = None
        self.section_path: list[str] = []
        self.notes: list[str] = []


def segment_document(doc: InputDocument) -> list[Segment]:
    text = doc.text or ""
    lines = _iter_lines(text)
    has_labels = any(_LABEL_RE.match(content) for _, _, content in lines)

    segments: list[Segment] = []
    heading_stack: list[tuple[int, str, str]] = []  # (depth, label, "label title")
    current: _Block | None = None
    front_matter: _Block | None = None
    seen_first_label = False
    last_top_level: int | None = None

    def ancestors(depth: int) -> list[str]:
        return [name for d, _, name in heading_stack if d < depth]

    def current_heading_label() -> str | None:
        return heading_stack[-1][1] if heading_stack else None

    def emit(block: _Block | None) -> None:
        if block is None or block.text_end <= block.text_start:
            return
        body = text[block.text_start : block.text_end]
        stripped_end = block.text_start + len(body.rstrip())
        body = text[block.text_start : stripped_end]
        if not body.strip():
            return
        has_modal = bool(MODAL_RE.search(body))
        notes = list(block.notes)
        confidence: Literal["high", "low"] = "low"
        if block.kind in ("numbered_paragraph", "paragraph"):
            if has_modal and block.chapter is not None:
                confidence = "high"
            if not has_modal:
                notes.append("no modal verb (shall/must/required); may be informational")
            if len(body) > _LONG_PARAGRAPH_CHARS:
                confidence = "low"
                notes.append("long block; may contain several requirements")
            if block.chapter is None:
                notes.append("no chapter label could be associated")
        p_start, p_end = page_range(block.text_start, stripped_end, doc.pages)
        segments.append(
            Segment(
                segment_id=f"SEG-{doc.document_id[4:]}-{len(segments) + 1:04d}",
                document_id=doc.document_id,
                kind=block.kind,
                chapter=block.chapter,
                label=block.label,
                title=block.title,
                section_path=block.section_path,
                text=body,
                start_offset=block.text_start,
                end_offset=stripped_end,
                line_start_offset=block.line_start,
                page_start=p_start,
                page_end=p_end,
                has_modal=has_modal,
                confidence=confidence,
                notes=notes,
            )
        )

    for start, end, content in lines:
        is_blank = not content.strip()
        match = _LABEL_RE.match(content) if not is_blank else None

        if not seen_first_label and has_labels and not match:
            # Everything before the first numbered line is front matter.
            if front_matter is None and not is_blank:
                front_matter = _Block("front_matter", start, start, None, None)
            if front_matter is not None and not is_blank:
                front_matter.text_end = end
            continue

        if is_blank:
            emit(current)
            current = None
            continue

        if match:
            if not seen_first_label:
                seen_first_label = True
                emit(front_matter)
            emit(current)
            current = None
            label = match.group("label")
            rest = match.group("rest")
            depth = _label_depth(label)
            rest_start = start + match.start("rest")
            notes: list[str] = []
            top = int(label.split()[-1].split(".")[0])
            if last_top_level is not None and top > last_top_level + 1:
                notes.append(f"numbering jumps from {last_top_level} to {top}")
            last_top_level = top if last_top_level is None else max(last_top_level, top)

            if _looks_like_heading(rest):
                while heading_stack and heading_stack[-1][0] >= depth:
                    heading_stack.pop()
                block = _Block("heading", start, rest_start, label, label)
                block.title = rest.strip()
                block.section_path = ancestors(depth)
                block.text_end = end
                block.notes = notes
                emit(block)
                heading_stack.append((depth, label, f"{label} {rest.strip()}"))
            else:
                current = _Block("numbered_paragraph", start, rest_start, label, label)
                current.section_path = ancestors(depth)
                current.text_end = end
                current.notes = notes
            continue

        if current is not None and current.kind in ("numbered_paragraph", "paragraph"):
            current.text_end = end  # continuation line (wrapped text, list item, table row)
            continue

        emit(current)
        current = _Block("paragraph", start, start, None, current_heading_label())
        current.section_path = [name for _, _, name in heading_stack]
        current.text_end = end
        if _TABLE_RE.match(content):
            current.notes.append("table rows without an introducing paragraph")

    emit(current)
    if not seen_first_label:
        emit(front_matter)
    return segments


def requirement_candidates(segments: list[Segment]) -> list[Segment]:
    return [s for s in segments if s.is_candidate]
