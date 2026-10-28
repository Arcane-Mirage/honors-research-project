"""src/csr_lab/extraction/grounding.py

Locate quoted text in source documents and build ``Evidence`` records.

Matching is exact first, then whitespace-insensitive (runs of spaces and
newlines compare equal). Nothing else is normalised: a quote with changed
words, case or punctuation is reported as ``not_found``.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

from csr_lab.schemas import Evidence, InputDocument, SourceLocation, TextMatch

_WS = re.compile(r"\s+")


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


@dataclass
class TextIndex:
    """Whitespace-normalised view of a text with a map back to original offsets."""

    text: str
    _norm: str = field(init=False, repr=False)
    _orig_pos: list[int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        chars: list[str] = []
        positions: list[int] = []
        in_ws = False
        for i, ch in enumerate(self.text):
            if ch.isspace():
                if not in_ws:
                    chars.append(" ")
                    positions.append(i)
                in_ws = True
            else:
                chars.append(ch)
                positions.append(i)
                in_ws = False
        self._norm = "".join(chars)
        self._orig_pos = positions

    def _norm_index(self, orig: int) -> int:
        return bisect.bisect_left(self._orig_pos, orig)

    def find(self, needle: str, start: int = 0, end: int | None = None) -> tuple[int, int, TextMatch] | None:
        """Find ``needle`` within ``text[start:end]``. Returns (start, end, match kind)."""
        end = len(self.text) if end is None else end
        if not needle or not needle.strip():
            return None
        pos = self.text.find(needle, start, end)
        if pos != -1:
            return pos, pos + len(needle), "exact"
        norm_needle = collapse_ws(needle)
        if not norm_needle:
            return None
        n_start, n_end = self._norm_index(start), self._norm_index(end)
        pos = self._norm.find(norm_needle, n_start, n_end)
        if pos == -1:
            return None
        orig_start = self._orig_pos[pos]
        last = pos + len(norm_needle) - 1
        orig_end = self._orig_pos[last] + 1
        return orig_start, orig_end, "whitespace_normalized"


class SourceIndex:
    """TextIndex per input document, built once per analysis."""

    def __init__(self, documents: list[InputDocument]):
        self.documents = {d.document_id: d for d in documents}
        self._indexes: dict[str, TextIndex] = {}

    def index(self, document_id: str) -> TextIndex:
        if document_id not in self._indexes:
            self._indexes[document_id] = TextIndex(self.documents[document_id].text or "")
        return self._indexes[document_id]

    def text(self, document_id: str) -> str:
        return self.documents[document_id].text or ""

    def locate(
        self, document_id: str, needle: str, start: int = 0, end: int | None = None
    ) -> tuple[int, int, TextMatch] | None:
        if document_id not in self.documents:
            return None
        return self.index(document_id).find(needle, start, end)


def build_evidence(
    sources: SourceIndex,
    *,
    document_id: str,
    quote: str,
    requirement_id: str | None,
    span: tuple[int, int] | None,
    chapter: str | None = None,
    check: bool = True,
) -> Evidence:
    """Evidence for ``quote``, searched inside ``span`` (the requirement) when given.

    With ``check=False`` the quote is recorded as given and marked not_checked.
    """
    quote = quote.strip() or quote
    if not check:
        return Evidence(
            source_document_id=document_id,
            requirement_id=requirement_id,
            quoted_text=quote,
            verified=None,
            verification="not_checked",
        )
    found = None
    if span is not None:
        found = sources.locate(document_id, quote, span[0], span[1])
    if found is None:
        found = sources.locate(document_id, quote)
    if found is None:
        return Evidence(
            source_document_id=document_id,
            requirement_id=requirement_id,
            quoted_text=quote,
            verified=False,
            verification="not_found",
        )
    start, end, kind = found
    exact_text = sources.text(document_id)[start:end]
    return Evidence(
        source_document_id=document_id,
        requirement_id=requirement_id,
        quoted_text=exact_text,
        source_location=SourceLocation(
            source_document_id=document_id, chapter=chapter, start_offset=start, end_offset=end, match=kind
        ),
        start_offset=start,
        end_offset=end,
        verified=True,
        verification=kind,
    )


def evidence_inside(evidence: Evidence, span: tuple[int, int] | None) -> bool:
    """True when verified evidence lies inside the given requirement span."""
    if not evidence.verified or span is None or evidence.start_offset is None:
        return False
    return span[0] <= evidence.start_offset and evidence.end_offset <= span[1]  # type: ignore[operator]
