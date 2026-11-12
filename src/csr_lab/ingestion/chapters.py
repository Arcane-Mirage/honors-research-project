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
   An entry whose heading is worded differently in the body ('0 General provisions' in the
   contents, '0 General Regulations' in the text) is found by its number alone, or by its
   title under another number (a contents typo: '3.1 Components ...' for '3.3.1 Components
   ...'), with ``chapter_toc_fallback``. The search stops at the next entry that matches
   normally, and every such match is noted. A heading printed as a bare number line followed
   by its title line ('1' / 'Scope', as PyMuPDF extracts some PDFs) is matched too.
2. **No usable contents**: every labelled line found by the paragraph segmenter
   (a numbered heading or a numbered paragraph) starts a chapter; unlabelled
   paragraphs stay in the chapter above them.

``chapter_max_depth`` (0 = no limit) keeps deeper labels inside their parent:
with 2, '4.4.1' stays in chapter '4.4'. Text before the first chapter is front
matter (a cover page), analysed only with ``chapter_include_front_matter``.

**Documents without chapters** (fewer than ``chapter_min_count``: a contract, a form ...)
are cut into passages of about ``passage_words`` words at blank lines, else at line ends
(``split_passages``). Passages are analysed for references and conflicts but are not
requirements (``Segment.kind == "passage"``, IDs ``PSG-<document>-P001``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from csr_lab.config import ChunkingSettings
from csr_lab.ingestion.segmentation import MODAL_RE, Segment, line_spans, page_range, segment_document
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
    text = re.sub(r"(?<=\w)[-‐‑–](?=\w)", " ", text)  # 'Self-Declaration' == 'Self Declaration'
    text = re.sub(r"(?<=\w)[-‐‑–]\s*$", "", text)  # a heading wrapped after 'Require-'
    return " ".join(text.split()).lower()


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
        if not line.startswith((lab + " ", lab + ". ")):
            return False
        rest = line[len(lab) :].lstrip(". ")
        return rest.startswith(want) or (len(rest) >= 3 and want.startswith(rest))
    return line == _norm(title) or (line.startswith(want) and len(line) <= len(_norm(title)) + 3)


_HEADING_REST_RE = re.compile(r"^[A-Z(\"'„“]")
_MAX_HEADING_WORDS = 14


def _number_heading(content: str, label: str) -> bool:
    """'0  General Regulations' for contents entry '0': the label, then a short capitalised title."""
    m = _TOC_LABEL_RE.match(content.strip())
    if not m or _norm(m.group("label")) != _norm(label):
        return False
    rest = m.group("title").strip()
    return bool(_HEADING_REST_RE.match(rest)) and len(rest.split()) <= _MAX_HEADING_WORDS and not rest.endswith((".", ";", ","))


def _title_heading(content: str, title: str) -> str | None:
    """The body label when the line is '<another label> <title>' (a contents numbering typo)."""
    m = _TOC_LABEL_RE.match(content.strip())
    if m and _norm(m.group("title")) == _norm(title):
        return m.group("label")
    return None


def _starts_from_toc(text: str, lines, entries, toc_span, fallback: bool = True) -> tuple[list[_Start], list[str]]:
    starts: list[_Start] = []
    notes: list[str] = []

    def body_lines(lo: int, hi: int):
        for j in range(lo, hi):
            start, _end, content = lines[j]
            if not (toc_span[0] <= start <= toc_span[1]) and content.strip():
                yield j, content

    def exact(entry, lo: int) -> int | None:
        label, title = entry
        return next((j for j, content in body_lines(lo, len(lines)) if _matches(content, label, title) or split_heading(j, label, title)), None)

    def split_heading(j: int, label: str | None, title: str) -> bool:
        """'1' on one line and 'Scope' on the next."""
        if not label or _norm(lines[j][2]).rstrip(".") != _norm(label):
            return False
        k = next((k for k in range(j + 1, min(j + 3, len(lines))) if lines[k][2].strip()), None)
        return k is not None and _matches(f"{label} {lines[k][2].strip()}", label, title)

    def heading_end(j: int, label: str | None) -> int:
        """Index of the heading's last line (the title line of a split heading)."""
        if label and _norm(lines[j][2]).rstrip(".") == _norm(label):
            return next((k for k in range(j + 1, min(j + 3, len(lines))) if lines[k][2].strip()), j)
        return j

    i = 0
    for k, (label, title) in enumerate(entries):
        name = f"{(label + ' ') if label else ''}{title}"
        found, body_label = exact((label, title), i), label
        if found is None and fallback:
            # Search only up to the next entry that matches normally, so the order is kept.
            nxt = next((j for e in entries[k + 1 :] if (j := exact(e, i)) is not None), len(lines))
            if label:
                found = next((j for j, content in body_lines(i, nxt) if _number_heading(content, label)), None)
                if found is not None:
                    notes.append(f"contents entry '{name}' matched by its number only (body heading: '{lines[found][2].strip()}')")
            if found is None:
                hit = next(((j, lab) for j, content in body_lines(i, nxt) if (lab := _title_heading(content, title))), None)
                if hit is not None:
                    found, body_label = hit
                    notes.append(f"contents entry '{name}' matched by its title under number {body_label} (contents numbering differs)")
        if found is None:
            notes.append(f"contents entry '{name}' not found in the text; its text stays in the chapter before it")
            continue
        last = heading_end(found, label)
        start = lines[found][0]
        end = lines[last][1]
        starts.append(_Start(line_start=start, body_start=min(end + 1, len(text)), label=body_label, title=title))
        i = last + 1
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
    lines = line_spans(text)
    notes: list[str] = []
    toc_span = None
    starts: list[_Start] = []
    if settings.chapter_use_toc:
        entries, toc_span = _parse_toc(lines)
        if entries:
            starts, notes = _starts_from_toc(text, lines, entries, toc_span, settings.chapter_toc_fallback)
            notes.insert(0, f"chapters from the table of contents: {len(starts)} of {len(entries)} entries found")
    if not starts:
        toc_span = None
        starts = _starts_from_labels(doc)
        notes.insert(0, f"no usable table of contents; {len(starts)} chapters from numbered headings")

    if settings.chapter_max_depth:
        starts = [s for s in starts if s.depth() <= settings.chapter_max_depth]
    if len(starts) < settings.chapter_min_count:
        passages = split_passages(doc, settings.passage_words)
        return passages, [
            (f"no chapters ({len(starts)} found, chapter_min_count={settings.chapter_min_count}): "
            f"{len(passages)} passages of about {settings.passage_words} words, analysed for references and "
            "conflicts only (no requirements)")
        ]

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


_BLANK_RE = re.compile(r"\n[ \t]*\n")


def split_passages(doc: InputDocument, words: int) -> list[Segment]:
    """Cut a document without chapters into verbatim passages of about ``words`` words.

    Paragraphs (blank-line separated) are packed up to the budget; a paragraph longer than
    the budget is split at line ends. Every passage is an exact slice of the source.
    """
    text = doc.text or ""
    pieces: list[tuple[int, int]] = []  # paragraph or line spans, in order
    pos = 0
    for m in [*_BLANK_RE.finditer(text), None]:
        end = m.start() if m else len(text)
        para = text[pos:end]
        if para.strip():
            if len(para.split()) <= words:
                pieces.append((pos, end))
            else:
                for start, line_end, content in line_spans(para):
                    if content.strip():
                        pieces.append((pos + start, pos + line_end))
        pos = m.end() if m else len(text)

    groups: list[list[tuple[int, int]]] = []
    count = 0
    for span in pieces:
        n = len(text[span[0] : span[1]].split())
        if groups and count + n > words and count >= words // 2:
            groups.append([])
            count = 0
        if not groups:
            groups.append([])
        groups[-1].append(span)
        count += n

    segments: list[Segment] = []
    for i, group in enumerate(groups, start=1):
        raw_start, end = group[0][0], group[-1][1]
        body = text[raw_start:end]
        start = raw_start + len(body) - len(body.lstrip())
        end = start + len(text[start:end].rstrip())
        p_start, p_end = page_range(start, max(start, end), doc.pages)
        segments.append(
            Segment(
                segment_id=f"SEG-{doc.document_id[4:]}-{i:04d}",
                document_id=doc.document_id,
                kind="passage",
                chapter=None,
                label=f"P{i:03d}",
                title=f"Passage {i}",
                text=text[start:end],
                start_offset=start,
                end_offset=end,
                line_start_offset=start,
                page_start=p_start,
                page_end=p_end,
                has_modal=bool(MODAL_RE.search(text[start:end])),
                confidence="low",
                notes=["document without chapters: passage, not a requirement"],
            )
        )
    return segments
