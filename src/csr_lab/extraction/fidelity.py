"""src/csr_lab/extraction/fidelity.py

Check whether text fragments (chunks, quotes, requirements) really occur in a source text.

Each fragment is located in three steps:

1. **exact**: the fragment occurs character for character;
2. **whitespace_normalized**: it occurs once runs of spaces/newlines are treated as equal;
3. **fuzzy**: otherwise the fragment's words are aligned to the source words
   (case- and punctuation-insensitive) around the best anchor. ``fragment_ratio``
   is the share of the fragment's words that are source words in source order;
   ``span_ratio`` is the share of the aligned source span that the fragment kept.
   Below ``min_ratio`` the fragment is reported as ``not_found``.

A fuzzy match is *not* verbatim. ``differences`` lists what changed (replaced,
inserted and deleted words) so a reviewer can see whether meaning changed.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:  # imported lazily at run time so csr_lab.schemas can import this module
    from csr_lab.extraction.grounding import TextIndex

FragmentMatch = Literal["exact", "whitespace_normalized", "fuzzy", "not_found"]

_WORD_RE = re.compile(r"\S+")
_EDGE_PUNCT = "\"'`.,;:!?()[]{}|*_"
_MODAL_RE = re.compile(r"\b(shall|must|is required|are required)\b", re.IGNORECASE)
_ANCHOR_N = 3
_MAX_DIFFERENCES = 12


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextDifference(_Model):
    """One edit between the fragment and the source span it was aligned to."""

    op: Literal["replace", "insert", "delete"] = Field(
        description="replace: words changed; insert: words only in the fragment; delete: source words the fragment dropped"
    )
    fragment_text: str = ""
    source_text: str = ""


class FragmentCheck(_Model):
    fragment_id: str
    document_id: str | None = None
    title: str | None = None
    match: FragmentMatch
    fragment_ratio: float = Field(description="Share of the fragment's words that are source words, in source order (1.0 = verbatim).")
    span_ratio: float = Field(description="Share of the aligned source span's words that the fragment kept (1.0 = nothing dropped).")
    start_offset: int | None = None
    end_offset: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    reported_pages: list[int] = Field(default_factory=list, description="Pages the producer of the fragment claimed, if any.")
    pages_agree: bool | None = Field(default=None, description="Reported pages equal the located pages (None when either is unknown).")
    fragment_words: int = 0
    differences: list[TextDifference] = Field(default_factory=list)


class SourceGap(_Model):
    """Source text that no fragment covered."""

    document_id: str | None = None
    start_offset: int
    end_offset: int
    words: int
    looks_like_requirement: bool = Field(description="Contains shall/must/required: possibly lost requirement text.")
    preview: str


class FidelitySummary(_Model):
    fragments: int = 0
    exact: int = 0
    whitespace_normalized: int = 0
    fuzzy: int = 0
    not_found: int = 0
    verbatim_rate: float | None = Field(default=None, description="(exact + whitespace_normalized) / fragments")
    mean_fragment_ratio: float | None = None
    source_coverage: float | None = Field(default=None, description="Share of the source's words inside some located fragment.")
    gaps: int = 0
    requirement_like_gaps: int = 0
    pages_checked: int = 0
    pages_agree: int = 0


class FidelityReport(_Model):
    origin: str = Field(default="external", description="Who produced the fragments: provided, llm, external ...")
    fragments: list[FragmentCheck] = Field(default_factory=list)
    gaps: list[SourceGap] = Field(default_factory=list)
    summary: FidelitySummary = Field(default_factory=FidelitySummary)


# ---------------------------------------------------------------------------
# Word-level view of a source text
# ---------------------------------------------------------------------------


def _norm_word(word: str) -> str:
    return word.strip(_EDGE_PUNCT).lower()


class WordIndex:
    """Source words with offsets, plus an n-gram index for anchoring fuzzy matches."""

    def __init__(self, text: str):
        self.text = text
        self.spans: list[tuple[int, int]] = []
        self.words: list[str] = []
        for m in _WORD_RE.finditer(text):
            w = _norm_word(m.group())
            if not w:
                continue
            self.spans.append((m.start(), m.end()))
            self.words.append(w)
        self._ngrams: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i in range(len(self.words) - _ANCHOR_N + 1):
            self._ngrams[tuple(self.words[i : i + _ANCHOR_N])].append(i)

    def word_at_or_after(self, offset: int) -> int:
        lo, hi = 0, len(self.spans)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.spans[mid][1] <= offset:
                lo = mid + 1
            else:
                hi = mid
        return lo

    def best_window(self, fwords: list[str], lo: int = 0, hi: int | None = None) -> tuple[int, int] | None:
        """Source word window most likely to contain the fragment (vote over shared n-grams)."""
        hi = len(self.words) if hi is None else hi
        votes: Counter[int] = Counter()
        n = _ANCHOR_N if len(fwords) >= _ANCHOR_N else 1
        for i in range(len(fwords) - n + 1):
            gram = tuple(fwords[i : i + n])
            positions = self._ngrams.get(gram, []) if n == _ANCHOR_N else [j for j, w in enumerate(self.words) if w == gram[0]]
            if len(positions) > 50:  # very common phrase: carries no location information
                continue
            for pos in positions:
                if lo <= pos < hi:
                    votes[pos - i] += 1
        if not votes:
            return None
        # Neighbouring offsets (a few words inserted or dropped) vote together.
        best_offset, best_score = 0, -1
        for offset in votes:
            score = sum(votes.get(offset + d, 0) for d in range(-8, 9))
            if score > best_score or (score == best_score and offset < best_offset):
                best_offset, best_score = offset, score
        slack = max(10, len(fwords) // 4)
        start = max(lo, best_offset - slack)
        end = min(hi, best_offset + len(fwords) + slack)
        return start, end


# ---------------------------------------------------------------------------
# Checking fragments
# ---------------------------------------------------------------------------


def _pages_for(start: int | None, end: int | None, pages: list | None) -> tuple[int | None, int | None]:
    if start is None or end is None or not pages:
        return None, None
    first = last = None
    for page in pages:
        if page.start_offset <= start <= page.end_offset and first is None:
            first = page.page_number
        if page.start_offset <= max(start, end - 1) <= page.end_offset:
            last = page.page_number
    return first, last or first


def check_fragment(
    fragment: str,
    *,
    text_index: TextIndex,
    word_index: WordIndex,
    fragment_id: str = "F1",
    document_id: str | None = None,
    title: str | None = None,
    min_ratio: float = 0.6,
    search_span: tuple[int, int] | None = None,
    pages: list | None = None,
    reported_pages: Iterable[int] = (),
) -> FragmentCheck:
    """Locate one fragment. ``search_span`` (character offsets) is tried first when given."""
    reported = sorted({int(p) for p in reported_pages})
    fwords_raw = [m.group() for m in _WORD_RE.finditer(fragment)]
    fwords = [w for w in (_norm_word(x) for x in fwords_raw) if w]
    base = dict(fragment_id=fragment_id, document_id=document_id, title=title, reported_pages=reported, fragment_words=len(fwords))

    def finish(check: FragmentCheck) -> FragmentCheck:
        check.page_start, check.page_end = _pages_for(check.start_offset, check.end_offset, pages)
        if reported and check.page_start is not None:
            check.pages_agree = reported == list(range(check.page_start, (check.page_end or check.page_start) + 1))
        return check

    if not fwords:
        return finish(FragmentCheck(**base, match="not_found", fragment_ratio=0.0, span_ratio=0.0))

    spans = [search_span, None] if search_span else [None]
    for span in spans:
        found = text_index.find(fragment.strip(), *(span or (0, None)))
        if found is not None:
            start, end, kind = found
            return finish(FragmentCheck(**base, match=kind, fragment_ratio=1.0, span_ratio=1.0, start_offset=start, end_offset=end))

    # Fuzzy: align words inside the best window (first inside the hinted span, then anywhere).
    best: FragmentCheck | None = None
    for span in spans:
        lo = hi = None
        if span:
            lo, hi = word_index.word_at_or_after(span[0]), word_index.word_at_or_after(span[1]) + 1
        window = word_index.best_window(fwords, lo or 0, hi)
        if window is None:
            continue
        w0, w1 = window
        swords = word_index.words[w0:w1]
        matcher = SequenceMatcher(None, fwords, swords, autojunk=False)
        blocks = [b for b in matcher.get_matching_blocks() if b.size]
        if not blocks:
            continue
        matched = sum(b.size for b in blocks)
        s_first = w0 + blocks[0].b
        s_last = w0 + blocks[-1].b + blocks[-1].size - 1
        span_words = s_last - s_first + 1
        fragment_ratio = matched / len(fwords)
        span_ratio = matched / span_words if span_words else 0.0
        diffs: list[TextDifference] = []
        b_lo, b_hi = blocks[0].b, blocks[-1].b + blocks[-1].size
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag == "delete":  # words only in the fragment
                op, ftxt, stxt = "insert", " ".join(fwords_raw_for(fwords_raw, fwords, i1, i2)), ""
            elif tag == "insert":  # source words missing from the fragment
                if j2 <= b_lo or j1 >= b_hi:
                    continue  # outside the aligned span: just window slack
                op, ftxt, stxt = "delete", "", _source_text(word_index, w0 + j1, w0 + j2)
            else:
                op, ftxt, stxt = "replace", " ".join(fwords_raw_for(fwords_raw, fwords, i1, i2)), _source_text(word_index, w0 + j1, w0 + j2)
            diffs.append(TextDifference(op=op, fragment_text=ftxt[:200], source_text=stxt[:200]))
        candidate = FragmentCheck(
            **base,
            match="fuzzy" if fragment_ratio >= min_ratio else "not_found",
            fragment_ratio=round(fragment_ratio, 4),
            span_ratio=round(span_ratio, 4),
            start_offset=word_index.spans[s_first][0] if fragment_ratio >= min_ratio else None,
            end_offset=word_index.spans[s_last][1] if fragment_ratio >= min_ratio else None,
            differences=diffs[:_MAX_DIFFERENCES],
        )
        if best is None or candidate.fragment_ratio > best.fragment_ratio:
            best = candidate
        if candidate.match == "fuzzy":
            break
    if best is None:
        best = FragmentCheck(**base, match="not_found", fragment_ratio=0.0, span_ratio=0.0)
    return finish(best)


def fwords_raw_for(raw: list[str], normed: list[str], i1: int, i2: int) -> list[str]:
    """Original spelling of normalised fragment words i1..i2 (punctuation-only words were dropped)."""
    kept = [w for w in raw if _norm_word(w)]
    return kept[i1:i2] if len(kept) == len(normed) else normed[i1:i2]


def _source_text(index: WordIndex, i: int, j: int) -> str:
    if i >= j or i >= len(index.spans):
        return ""
    return index.text[index.spans[i][0] : index.spans[min(j, len(index.spans)) - 1][1]]


def source_gaps(
    text: str, located: list[tuple[int, int]], *, document_id: str | None = None, min_words: int = 6
) -> tuple[list[SourceGap], float | None]:
    """Uncovered stretches of ``text`` and the share of words covered by ``located`` spans."""
    index = WordIndex(text)
    if not index.spans:
        return [], None
    covered = [False] * len(index.spans)
    for start, end in located:
        i = index.word_at_or_after(start)
        while i < len(index.spans) and index.spans[i][0] < end:
            covered[i] = True
            i += 1
    gaps: list[SourceGap] = []
    i = 0
    while i < len(covered):
        if covered[i]:
            i += 1
            continue
        j = i
        while j < len(covered) and not covered[j]:
            j += 1
        if j - i >= min_words:
            start, end = index.spans[i][0], index.spans[j - 1][1]
            body = text[start:end]
            gaps.append(
                SourceGap(
                    document_id=document_id,
                    start_offset=start,
                    end_offset=end,
                    words=j - i,
                    looks_like_requirement=bool(_MODAL_RE.search(body)),
                    preview=" ".join(body.split())[:160],
                )
            )
        i = j
    return gaps, round(sum(covered) / len(covered), 4)


def summarize(checks: list[FragmentCheck], gaps: list[SourceGap], coverage: float | None) -> FidelitySummary:
    n = len(checks)
    counts = Counter(c.match for c in checks)
    page_checks = [c for c in checks if c.pages_agree is not None]
    return FidelitySummary(
        fragments=n,
        exact=counts["exact"],
        whitespace_normalized=counts["whitespace_normalized"],
        fuzzy=counts["fuzzy"],
        not_found=counts["not_found"],
        verbatim_rate=round((counts["exact"] + counts["whitespace_normalized"]) / n, 4) if n else None,
        mean_fragment_ratio=round(sum(c.fragment_ratio for c in checks) / n, 4) if n else None,
        source_coverage=coverage,
        gaps=len(gaps),
        requirement_like_gaps=sum(1 for g in gaps if g.looks_like_requirement),
        pages_checked=len(page_checks),
        pages_agree=sum(1 for c in page_checks if c.pages_agree),
    )


def check_fragments(
    source_text: str,
    fragments: list[str] | list[dict],
    *,
    document_id: str | None = None,
    pages: list | None = None,
    min_ratio: float = 0.6,
    origin: str = "external",
    gap_min_words: int = 6,
) -> FidelityReport:
    """Check every fragment against one source text and report coverage gaps.

    ``fragments`` are strings, or dicts with ``text``/``content`` and optional
    ``id``/``chunk_id``, ``title`` and ``source_pages``/``pages``.
    """
    from csr_lab.extraction.grounding import TextIndex

    text_index, word_index = TextIndex(source_text), WordIndex(source_text)
    checks: list[FragmentCheck] = []
    for i, frag in enumerate(fragments, start=1):
        if isinstance(frag, str):
            frag = {"text": frag}
        body = frag.get("text") if frag.get("text") is not None else frag.get("content", "")
        checks.append(
            check_fragment(
                str(body),
                text_index=text_index,
                word_index=word_index,
                fragment_id=str(frag.get("id") or frag.get("chunk_id") or f"F{i}"),
                document_id=document_id,
                title=frag.get("title"),
                min_ratio=min_ratio,
                pages=pages,
                reported_pages=frag.get("source_pages") or frag.get("pages") or [],
            )
        )
    located = [(c.start_offset, c.end_offset) for c in checks if c.start_offset is not None and c.end_offset is not None]
    gaps, coverage = source_gaps(source_text, located, document_id=document_id, min_words=gap_min_words)  # type: ignore[arg-type]
    return FidelityReport(origin=origin, fragments=checks, gaps=gaps, summary=summarize(checks, gaps, coverage))


def format_report(report: FidelityReport, *, max_rows: int = 60) -> str:
    """Plain-text table for the command line."""
    s = report.summary
    lines = [
        f"fragments: {s.fragments}  exact: {s.exact}  whitespace-only: {s.whitespace_normalized}  "
        f"fuzzy (rewritten): {s.fuzzy}  not found: {s.not_found}",
        f"verbatim rate: {_pct(s.verbatim_rate)}  mean word ratio: {_pct(s.mean_fragment_ratio)}  "
        f"source coverage: {_pct(s.source_coverage)}  gaps: {s.gaps} ({s.requirement_like_gaps} look like requirements)",
    ]
    if s.pages_checked:
        lines.append(f"reported pages agree with located pages: {s.pages_agree}/{s.pages_checked}")
    lines.append("")
    lines.append(f"{'id':<14} {'match':<22} {'words':>5} {'ratio':>6} {'kept':>6}  pages  first difference")
    for c in report.fragments[:max_rows]:
        pages = "-" if c.page_start is None else (f"{c.page_start}" if c.page_end in (None, c.page_start) else f"{c.page_start}-{c.page_end}")
        if c.pages_agree is False:
            pages += f" (said {','.join(map(str, c.reported_pages))})"
        diff = ""
        if c.differences:
            d = c.differences[0]
            diff = f"{d.op}: '{d.source_text[:40]}' -> '{d.fragment_text[:40]}'"
        lines.append(f"{c.fragment_id[:14]:<14} {c.match:<22} {c.fragment_words:>5} {c.fragment_ratio:>6.2f} {c.span_ratio:>6.2f}  {pages:<6} {diff}")
    if len(report.fragments) > max_rows:
        lines.append(f"... {len(report.fragments) - max_rows} more")
    if report.gaps:
        lines += ["", "uncovered source text:"]
        for g in report.gaps[:20]:
            flag = "REQUIREMENT? " if g.looks_like_requirement else ""
            lines.append(f"  [{g.start_offset}-{g.end_offset}] {flag}{g.preview}")
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"
