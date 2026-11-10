"""src/csr_lab/ingestion/chapters.py

Cut a document into whole chapters: one unit (and one requirement) per chapter.

A chapter runs from its heading line to the next chapter heading. Its text is an
exact slice of the source *without* the heading line; the label goes to
``Segment.chapter`` and the heading words to ``Segment.title``. A heading
directly followed by a sub-heading ('1 Request ...' then '1.1 Offer ...') has no
text of its own and is kept as a non-candidate ``heading`` segment.

Where the chapter list comes from:

1. **Table of contents** (``chapter_use_toc``): lines that end in a dot leader and a
   page number ('1.1.4 Compliance with ... ....... 7'). Each entry is searched for
   in the body, in order, outside the contents pages. Unnumbered entries ('Scope',
   'Annex A (informative) Change history') are found by their title. Headings that
   are not in the contents ('3.3.3.1 ...') stay inside their parent chapter, so
   numbered list items such as '1. TRATON Customer Requirements CVS10' never cut a
   chapter. The contents pages themselves are not analysed.
2. **No usable contents**: every labelled line found by the paragraph segmenter
   (a numbered heading or a numbered paragraph) starts a chapter; unlabelled
   paragraphs stay in the chapter above them.

``chapter_max_depth`` (0 = no limit) keeps deeper labels inside their parent:
with 2, '4.4.1' stays in chapter '4.4'. Text before the first chapter is front
matter (a cover page), analysed only with ``chapter_include_front_matter``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from csr_lab.config import ChunkingSettings
from csr_lab.ingestion.segmentation import MODAL_RE, Segment, page_range, segment_document
from csr_lab.schemas import InputDocument

# '1.1.4 Compliance with ... regulations......... 7', 'Scope ......... 4'
_TOC_LINE_RE = re.compile(r"^(?P<entry>\S.*?)\s*(?:\.\s*){4,}(?P<page>\d{1,4})\s*$")
_TOC_LABEL_RE = re.compile(r"^(?P<label>(?:[A-Z]{2,6}[ \t])?\d{1,3}(?:\.\d{1,3}){0,6})\.?\s+(?P<title>\S.*)$")
_MIN_TOC_ENTRIES = 3
_TITLE_MATCH_CHARS = 40
_FRONT_MATTER = "Front matter"


@dataclass
class _Start:
    """A chapter heading found in the body."""

    line_start: int  # offset of the heading line
    body_start: int  # offset after the heading line
    label: str | None
    title: str | None

    @property
    def chapter(self) -> str:
        return self.label or self.title or "?"

    def depth(self) -> int:
        return self.label.split()[-1].count(".") + 1 if self.label else 1


def _norm(text: str) -> str:
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(text.split()).lower()


def _lines(text: str) -> list[tuple[int, int, str]]:
    out, pos = [], 0
    for raw in text.split("\n"):
        out.append((pos, pos + len(raw), raw))
        pos += len(raw) + 1
    return out


def _parse_toc(lines: list[tuple[int, int, str]]) -> tuple[list[tuple[str | None, str]], tuple[int, int] | None]:
    """Contents entries as (label, title), and the character span of the contents lines."""
    entries: list[tuple[str | None, str]] = []
    first = last = None
    pending = ""  # a long entry wrapped over two lines
    for i, (start, end, content) in enumerate(lines):
        m = _TOC_LINE_RE.match(content.strip())
        if not m:
            # Only lines between two contents lines can be the first half of a wrapped entry.
            pending = content.strip() if first is not None and i + 1 < len(lines) and _TOC_LINE_RE.match(lines[i + 1][2].strip()) else ""
            continue
        entry = m.group("entry").strip()
        if pending and not _TOC_LABEL_RE.match(entry):
            entry = f"{pending} {entry}"
        pending = ""
        lm = _TOC_LABEL_RE.match(entry)
        entries.append((lm.group("label"), lm.group("title").strip()) if lm else (None, entry))
        first = start if first is None else first
        last = end
    if len(entries) < _MIN_TOC_ENTRIES or first is None or last is None:
        return [], None
    return entries, (first, last)


def _matches(content: str, label: str | None, title: str) -> bool:
    line = _norm(content)
    want = _norm(title)[:_TITLE_MATCH_CHARS]
    if label:
        lab = _norm(label)
        if not (line.startswith(lab + " ") or line.startswith(lab + ". ")):
            return False
        rest = line[len(lab) :].lstrip(". ")
        return rest.startswith(want) or (len(rest) >= 3 and want.startswith(rest))
    return line == _norm(title) or (line.startswith(want) and len(line) <= len(_norm(title)) + 3)


def _starts_from_toc(text: str, lines, entries, toc_span) -> tuple[list[_Start], list[str]]:
    starts: list[_Start] = []
    notes: list[str] = []
    i = 0
    for label, title in entries:
        found = None
        for j in range(i, len(lines)):
            start, end, content = lines[j]
            if toc_span[0] <= start <= toc_span[1]:
                continue
            if content.strip() and _matches(content, label, title):
                found = j
                break
        if found is None:
            notes.append(f"contents entry '{(label + ' ') if label else ''}{title}' not found in the text; its text stays in the chapter before it")
            continue
        start, end, _ = lines[found]
        starts.append(_Start(line_start=start, body_start=min(end + 1, len(text)), label=label, title=title))
        i = found + 1
    return starts, notes


def _starts_from_labels(doc: InputDocument) -> list[_Start]:
    starts = []
    for seg in segment_document(doc):
        if seg.label is None:
            continue
        if seg.kind == "heading":
            body = seg.end_offset + 1
            starts.append(_Start(seg.line_start_offset, min(body, len(doc.text or "")), seg.label, seg.title))
        else:  # '1.2 The supplier shall ...': the chapter text starts right after the label
            starts.append(_Start(seg.line_start_offset, seg.start_offset, seg.label, None))
    return starts


def split_chapters(doc: InputDocument, settings: ChunkingSettings) -> tuple[list[Segment], list[str]]:
    """Segments for ``doc``, one per chapter, and notes on how the chapters were found."""
    text = doc.text or ""
    lines = _lines(text)
    notes: list[str] = []
    toc_span = None
    starts: list[_Start] = []
    if settings.chapter_use_toc:
        entries, toc_span = _parse_toc(lines)
        if entries:
            starts, notes = _starts_from_toc(text, lines, entries, toc_span)
            notes.insert(0, f"chapters from the table of contents: {len(starts)} of {len(entries)} entries found")
    if not starts:
        toc_span = None
        starts = _starts_from_labels(doc)
        notes.insert(0, f"no usable table of contents; {len(starts)} chapters from numbered headings")

    if settings.chapter_max_depth:
        starts = [s for s in starts if s.depth() <= settings.chapter_max_depth]

    segments: list[Segment] = []

    def add(kind, start: int, end: int, line_start: int, chapter: str | None, title: str | None, path: list[str], extra: list[str]) -> None:
        body = text[start:end]
        end = start + len(body.rstrip())
        lead = len(body) - len(body.lstrip())
        if start + lead < end:
            start += lead
        body = text[start:end]
        if kind != "heading" and not body.strip():
            kind = "heading"
        p_start, p_end = page_range(start, max(start, end), doc.pages)
        has_modal = bool(MODAL_RE.search(body))
        note_list = list(extra)
        if kind in ("numbered_paragraph", "paragraph") and not has_modal:
            note_list.append("no modal verb (shall/must/required); may be informational")
        segments.append(
            Segment(
                segment_id=f"SEG-{doc.document_id[4:]}-{len(segments) + 1:04d}",
                document_id=doc.document_id,
                kind=kind,
                chapter=chapter,
                label=chapter,
                title=title,
                section_path=path,
                text=body if kind != "heading" else (title or chapter or ""),
                start_offset=start if kind != "heading" else line_start,
                end_offset=end if kind != "heading" else max(line_start, end),
                line_start_offset=line_start,
                page_start=p_start,
                page_end=p_end,
                has_modal=has_modal,
                confidence="high" if chapter else "low",
                notes=note_list,
            )
        )

    first = starts[0].line_start if starts else len(text)
    front_end = first
    if toc_span is not None and toc_span[0] < first:
        front_end = toc_span[0]
    if text[:front_end].strip():
        kind = "paragraph" if settings.chapter_include_front_matter else "front_matter"
        add(kind, 0, front_end, 0, _FRONT_MATTER if kind == "paragraph" else None, None, [], ["text before the first chapter"])

    stack: list[tuple[int, str]] = []  # (depth, "label title") of the open parent chapters
    for k, s in enumerate(starts):
        end = starts[k + 1].line_start if k + 1 < len(starts) else len(text)
        extra: list[str] = []
        if toc_span is not None and s.body_start <= toc_span[0] < end:
            end = toc_span[0]  # the contents pages are never part of a chapter
        while stack and s.label and stack[-1][0] >= s.depth():
            stack.pop()
        path = [name for _, name in stack]
        kind = "numbered_paragraph" if s.label else "paragraph"
        add(kind, s.body_start, max(s.body_start, end), s.line_start, s.chapter, s.title, path, extra)
        if s.label:
            stack.append((s.depth(), f"{s.label} {s.title}" if s.title else s.label))
    return segments, notes
