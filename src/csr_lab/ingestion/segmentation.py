"""src/csr_lab/ingestion/segmentation.py

Find the labelled lines of a document: numbered headings and numbered paragraphs.

``ingestion/chapters.py`` uses this when a document has no table of contents: every
labelled line starts a chapter. ``Segment`` is also the unit record every stage works on
(one per chapter or model-made chunk).

Recognised layouts:
* numbered paragraphs:            ``4.2.1 The supplier shall ...``
* prefixed chapter labels:        ``HVG 2.1 The supplier shall ...`` (like ``VDA 6.5``)
* headings followed by paragraphs: ``3.1 Control Plans`` / ``The supplier shall ...``
* multi-line paragraphs, lettered sub-items and pipe tables (continuation lines)

Every segment records exact character offsets into the document text, so the
segment text is always an exact slice of the source.

The second kind of labelled line is a customer label inside a chapter ("Navistar: ...",
"MAN/Scania: ..."); see "Customer scoping" at the end of this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from csr_lab.extraction.fidelity import page_range  # re-exported for the unit builders
from csr_lab.extraction.normalization import name_match_key
from csr_lab.schemas import CustomerBlock, InputDocument

SegmentKind = Literal["front_matter", "heading", "numbered_paragraph", "paragraph", "passage"]

# "4.2.1 Text", "HVG 2.1 Text", "7. Text". The label must be followed by an
# upper-case letter or "(" so wrapped lines such as "10 years after ..." are not
# mistaken for new sections.
_LABEL_RE = re.compile(r"^(?P<label>(?:[A-Z]{2,6}[ \t])?\d{1,3}(?:\.\d{1,3}){0,5})\.?[ \t]+(?P<rest>[A-Z(\"'].*)$")
LABEL_RE = _LABEL_RE
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
    origin: Literal["deterministic", "llm"] = Field(
        default="deterministic", description="Who made this unit: csr-lab (chapters) or the model (semantic chunks)."
    )
    text_is_source_slice: bool = Field(
        default=True, description="False when the unit's text is a chunk whose wording differs from the source."
    )

    @property
    def is_candidate(self) -> bool:
        return self.kind in ("numbered_paragraph", "paragraph", "passage")


def line_spans(text: str) -> list[tuple[int, int, str]]:
    """(start, end, content) of every line of ``text``; offsets exclude the newline."""
    out, pos = [], 0
    for raw in text.split("\n"):
        out.append((pos, pos + len(raw), raw))
        pos += len(raw) + 1
    return out


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


# ---------------------------------------------------------------------------
# Customer scoping
# ---------------------------------------------------------------------------
# Customer scoping: which parts of a chapter apply to which customer.
#
# Standards shared by several customers label paragraphs with the customers they apply to
# ("Navistar: ...", "MAN: ...", "Scania/Navistar: ...", "MAN Truck & BUS / Scania / VW T&B /
# International Motors LLC:"), and some chapters are titled with a customer ("3.3.3 MAN").
# Unlabelled text applies to every customer (CVS10 chapter 0: "When no brands are stated, it
# is a common requirement").
#
# ``find_blocks`` cuts a unit's text into *scope blocks*. A block starts at a label line (or is
# the whole chapter, when the chapter title is a label) and ends at the next label line, a
# numbered heading ('4.4.1.3.3 Disposal ...'), a blank line (unless the label stands alone on
# its line), or a paragraph end: a line ending a sentence followed by a line that starts with a
# capital letter or a bullet. Ending a block early is the safe error: the rest is treated as
# applying to everyone, so in ``filter`` mode it is kept.
#
# A label line is recognised only when *every* name before the colon is a known customer
# (``system_prompt.customer`` / ``customer_aliases`` and ``system_prompt.other_customers_names``),
# matched on whole words from the start ("MAN Truck & BUS" is MAN, "International Motors LLC."
# is the customer when "International" is one of its names). Words after a known name must be
# capitalised name words ("Volkswagen (VW) Group After Sales"), at most five, so "International
# Motors IMDS account: 186555" is not a label.
#
# Modes (``PipelineOptions.customer_scoping``):
#
# * ``off``: nothing is computed;
# * ``tag``: blocks are found and recorded (``Requirement.customer_scope``); every reference
#   says which customers it applies to (``ReferenceMention.applies_to``); the model sees the
#   full text. ``coverage.references.from_other_customer_blocks`` counts references the model
#   took from blocks that apply only to other customers;
# * ``filter``: as ``tag``, and blocks that apply only to other customers are replaced by a short
#   marker in the text the model sees (``RequirementDraft.model_text``). The exported requirement
#   text stays verbatim, and quotes are still located in the full source text.

_LABEL_LINE_RE = re.compile(r"^\s*(?P<prefix>[^:\n]{1,120}?)\s*:(?!//)")
_PART_SPLIT_RE = re.compile(r"\s*(?:/|,|\band\b|;)\s*")
_PARENS_RE = re.compile(r"\([^)]*\)")
_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")
_NEW_PARAGRAPH_RE = re.compile(r"^\s*(?:[A-Z(\"'“„•▪·◦\-–]|o\s)")
MAX_LABEL_WORDS = 16
MAX_EXTRA_NAME_WORDS = 5
_NAME_JOINERS = {"and", "of", "for", "&", "de", "e"}


@dataclass
class CustomerMatcher:
    """Known customer names -> canonical customer name; the target is the customer read for."""

    target: str
    names: dict[str, str] = field(default_factory=dict)  # match key -> canonical customer

    @classmethod
    def from_settings(cls, settings) -> CustomerMatcher | None:
        """From ``SystemPromptSettings``; None when no other customer is configured."""
        others = settings.other_customers_names
        if not others:
            return None
        matcher = cls(target=settings.customer)
        for name in [settings.customer, *settings.customer_aliases]:
            matcher._add(name, settings.customer)
        for customer, aliases in others.items():
            for name in [customer, *(aliases or [])]:
                matcher._add(name, customer)
        return matcher

    def _add(self, name: str, customer: str) -> None:
        key = name_match_key(_PARENS_RE.sub(" ", name))
        if key:
            self.names.setdefault(key, customer)

    def customer_of(self, part: str) -> str | None:
        """'MAN Truck & BUS' -> 'MAN'; the longest known name that starts the part wins. Words
        after the name must be capitalised name words ('Group After Sales'), not 'IMDS account'."""
        cleaned = _PARENS_RE.sub(" ", part)
        key = name_match_key(cleaned)
        if not key:
            return None
        best = None
        for name, customer in self.names.items():
            if (key == name or key.startswith(name + " ")) and (best is None or len(name) > len(best[0])):
                best = (name, customer)
        if best is None:
            return None
        n = len(best[0].split())
        extra = re.findall(r"[^\W_]+|&", cleaned)[n:]
        if len(extra) > MAX_EXTRA_NAME_WORDS or any(not (w[:1].isupper() or w.lower() in _NAME_JOINERS) for w in extra):
            return None
        return best[1]

    def label_customers(self, prefix: str) -> list[str] | None:
        """Customers named by a label ('Scania/Navistar' -> ['Scania', 'Navistar']); None if not a label."""
        if len(prefix.split()) > MAX_LABEL_WORDS or not prefix.strip():
            return None
        parts = [p for p in _PART_SPLIT_RE.split(prefix.strip()) if p.strip()]
        customers = []
        for p in parts:
            c = self.customer_of(p)
            if c is None:
                return None
            customers.append(c)
        return list(dict.fromkeys(customers)) or None


def find_blocks(text: str, offset: int, matcher: CustomerMatcher, title: str | None = None) -> list[CustomerBlock]:
    """Scope blocks of one unit's text. ``offset``: where the text starts in its document."""
    end_of_text = len(text.rstrip())
    if title and (whole := matcher.label_customers(title)):
        return [CustomerBlock(start_offset=offset, end_offset=offset + end_of_text, customers=whole, label=title, applies_to_target=matcher.target in whole)]
    lines = line_spans(text)
    blocks: list[CustomerBlock] = []
    current: tuple[int, list[str], str] | None = None  # (start, customers, label)
    bare_label = False  # the current block's label stands alone on its line; its text follows

    def close(end: int) -> None:
        nonlocal current
        if current is not None:
            start, customers, label = current
            stop = len(text[:end].rstrip())
            if stop > start:
                blocks.append(CustomerBlock(start_offset=offset + start, end_offset=offset + stop, customers=customers, label=label, applies_to_target=matcher.target in customers))
        current = None

    for i, (start, end, content) in enumerate(lines):
        if not content.strip():
            if not bare_label:
                close(start)
            continue
        if (m := _LABEL_LINE_RE.match(content)) and (customers := matcher.label_customers(m.group("prefix"))):
            close(start)
            current = (start + len(content) - len(content.lstrip()), customers, m.group("prefix").strip())
            bare_label = not content[m.end() :].strip()
            continue
        bare_label = False
        if current is not None and LABEL_RE.match(content.strip()):
            close(start)
            continue
        if current is not None and i > 0:
            prev = lines[i - 1][2]
            if _SENTENCE_END_RE.search(prev) and _NEW_PARAGRAPH_RE.match(content):
                close(start)
    close(len(text))
    return blocks


def hide_other_customers(text: str, offset: int, blocks: list[CustomerBlock]) -> str:
    """The text with blocks for other customers replaced by a marker."""
    out, pos = [], 0
    for b in sorted(blocks, key=lambda b: b.start_offset):
        if b.applies_to_target:
            continue
        s, e = b.start_offset - offset, b.end_offset - offset
        if s < pos or e > len(text):
            continue
        out.append(text[pos:s])
        out.append(f"[text for {', '.join(b.customers)} only omitted]")
        pos = e
    out.append(text[pos:])
    return "".join(out)


def block_at(blocks: list[CustomerBlock], position: int | None) -> CustomerBlock | None:
    if position is None:
        return None
    return next((b for b in blocks if b.start_offset <= position < b.end_offset), None)
