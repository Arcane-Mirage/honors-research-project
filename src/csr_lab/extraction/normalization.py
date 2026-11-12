"""src/csr_lab/extraction/normalization.py

Conservative, deterministic canonicalisation of document identifiers.

The same function is used to build document IDs in every strategy and by the
evaluator, so formatting differences ("Specification ABC-100" vs "ABC-100")
never count as extraction errors.

What it does:
* upper-cases, unifies Unicode dashes and underscores to "-", and repairs a hyphen with a
  space on one side only, a PDF extraction artifact ("MPAPS B -50" -> "MPAPS-B-50");
* removes leading articles and document-type words ("the", "Specification");
* splits off a *location* inside the referenced document ("CVS10, Chapter 4.6",
  "Chapter 3.3 of Formel Q konkret", "VDA 6.3 question 5.1") into ``location``;
* splits off an explicit revision ("Rev. B", "Revision D", "rev.3"), an edition
  ("4th edition", "Edition 9" -> "EDITION 4") or a year ("VDA 6.3 (2023)", "ISO 9001:2015");
* splits "IDENTIFIER - Title" at a spaced dash ("CVS10 - Commercial Vehicle Standard 10");
* joins spaced identifiers whose parts are upper-case codes ("STD 3868" -> "STD-3868"),
  skipping volume words ("VDA Volume 6.3" -> "VDA-6.3");
* prefers an identifier with letters over a bare number ("CVS10 ... Standard 10" -> "CVS10").

``ParsedIdentifier.compact`` is the identifier with the spaces between its parts removed
("CVS 10" -> "CVS10"). Identifiers whose compact forms are equal differ only by spaces;
``merging.merge_spacing_variants`` uses that to merge "CVS 10" with "CVS10".

What it deliberately does not do:
* drop leading zeros ("QP-012" and "QP-12" stay distinct; flagged as near duplicates);
* merge identifiers that merely look similar;
* infer a revision, name or type that the text does not state.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from csr_lab.schemas import DocumentType

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−_"), "-")
_REVISION_RE = re.compile(
    r"(?:[,;]?\s*\(?\s*|\s*)\b(?:rev(?:ision)?|issue|version|ver)\b\.?\s*[:#]?\s*(?P<rev>[A-Za-z0-9]{1,4}(?:\.\d{1,3})?)\s*\)?\s*$",
    re.IGNORECASE,
)
_ID_TOKEN_RE = re.compile(r"^[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*$")
# 'CVS10, Chapter 4.6 "Change Management"', 'VDA 6.3 question 5.1', 'ISO 9001 clause 8.3'
_LOCATOR_WORD = r"(?:chapters?|chap\.|sections?|sect\.|clauses?|paragraphs?|para\.|questions?|items?|§)"
_LOCATION = r"(?P<loc>\d{1,3}(?:\.\d{1,3}){0,6})(?![\d.]*\d)\.?"
_LOCATOR_AFTER_RE = re.compile(r"(?:^|[\s,;(])" + _LOCATOR_WORD + r"\s*" + _LOCATION, re.IGNORECASE)
_LOCATOR_FIRST_RE = re.compile(r"^" + _LOCATOR_WORD + r"\s*" + _LOCATION + r"\s+(?:of|in)\s+(?P<rest>\S.*)$", re.IGNORECASE)
# '4th edition', '2nd Edition', 'Edition 9', '5. Edition'
_EDITION_RE = re.compile(
    r"[,;]?\s*\(?\b(?:(?P<n1>\d{1,2})(?:st|nd|rd|th|\.)?\s*edition|edition\s*(?P<n2>\d{1,2}))\b\)?", re.IGNORECASE
)
_PAREN_YEAR_RE = re.compile(r"\s*\(\s*(?P<year>(?:19|20)\d{2})\s*\)")
_COLON_YEAR_RE = re.compile(r"(?<=\d):(?P<year>(?:19|20)\d{2})\b")
_TITLE_DASH = " - "  # 'CVS10 - Commercial Vehicle Standard 10' (dashes are unified first)
_VOLUME_WORDS = {"volume", "vol", "vol.", "band", "bd."}  # 'VDA Volume 6.3' == 'VDA 6.3'
_ONE_SIDED_HYPHEN_RE = re.compile(r"(?<=[A-Za-z0-9]) -(?=[A-Za-z0-9])|(?<=[A-Za-z0-9])- (?=[A-Za-z0-9])")
_CODE_TOKEN_RE = re.compile(r"^[A-Z]{1,8}$")
_PAREN_NAME_RE = re.compile(r"\(([^()]{3,80})\)")

ARTICLES = {"the", "a", "an", "per", "see"}
TYPE_WORDS: dict[str, DocumentType] = {
    "drawing": DocumentType.ENGINEERING_DRAWING,
    "dwg": DocumentType.ENGINEERING_DRAWING,
    "procedure": DocumentType.PROCEDURE,
    "standard": DocumentType.STANDARD,
    "manual": DocumentType.QUALITY_MANUAL,
    "specification": DocumentType.TECHNICAL_SPECIFICATION,
    "spec": DocumentType.TECHNICAL_SPECIFICATION,
    "spec.": DocumentType.TECHNICAL_SPECIFICATION,
    "method": DocumentType.TECHNICAL_SPECIFICATION,
    "form": DocumentType.FORM,
    "template": DocumentType.FORM,
    "worksheet": DocumentType.FORM,
}
_TYPE_PHRASES: list[tuple[str, DocumentType]] = [
    ("customer-specific requirements", DocumentType.CUSTOMER_SPECIFICATION),
    ("customer specific requirements", DocumentType.CUSTOMER_SPECIFICATION),
    ("customer specification", DocumentType.CUSTOMER_SPECIFICATION),
    ("quality manual", DocumentType.QUALITY_MANUAL),
    ("test method", DocumentType.TECHNICAL_SPECIFICATION),
]

_SECTION_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)*$")
_DATE_RES = [re.compile(r"^\d{4}-\d{1,2}-\d{1,2}$"), re.compile(r"^\d{1,2}[./]\d{1,2}[./]\d{2,4}$")]
_NON_DOC_CUES = re.compile(
    r"(?:part\s+(?:number|no\.?)|p/n|\bpn|section|clause|chapter|sheet|lot|serial(?:\s+number)?|item)\s*[:#]?\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedIdentifier:
    raw: str
    canonical: str | None
    revision: str | None
    name: str | None
    type_hint: DocumentType
    location: str | None = None  # chapter / section / question inside the referenced document
    compact: str | None = None  # identifier parts joined without separators ('CVS 10' -> 'CVS10')

    @property
    def key(self) -> str:
        return reference_key(self.canonical, self.revision, self.name)


def _clean(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw).translate(_DASHES)
    text = re.sub(r"\s+", " ", text).strip()
    text = _ONE_SIDED_HYPHEN_RE.sub("-", text)
    return text.strip(" \t\"'`.,;:")


def normalize_revision(revision: str | None) -> str | None:
    """'Rev. B' -> 'B'; '4th edition' / 'Edition 4' -> 'EDITION 4'; '(2023)' -> '2023'."""
    if revision is None:
        return None
    rev = _clean(revision)
    if m := _EDITION_RE.fullmatch(rev) or _EDITION_RE.fullmatch(" " + rev):
        return f"EDITION {int(m.group('n1') or m.group('n2'))}"
    rev = re.sub(r"^(?:rev(?:ision)?|issue|version|ver)\.?\s*[:#]?\s*", "", rev, flags=re.IGNORECASE)
    rev = rev.strip(" .()").upper()
    return rev or None


def normalize_location(location: str | None) -> str | None:
    """A chapter label from a stated location: 'Chapter 4.6' -> '4.6', '§ 3' -> '3'."""
    if not location or not location.strip():
        return None
    if m := re.search(r"\d{1,3}(?:\.\d{1,3}){0,6}", location):
        return m.group(0)
    return _clean(location) or None


def normalize_name(name: str) -> str:
    words = _clean(name).split()
    while words and words[0].lower() in ARTICLES:
        words = words[1:]
    return " ".join(words).upper()


def reference_key(canonical: str | None, revision: str | None = None, name: str | None = None) -> str:
    """Matching key: 'IW-001-SWR@REV-B', 'QP-12', or 'NAME:SUPPLIER QUALITY MANUAL'."""
    if canonical:
        base = re.sub(r"\s+", "-", canonical.strip().upper())
        base = re.sub(r"-{2,}", "-", base)
        rev = normalize_revision(revision)
        return f"{base}@REV-{rev}" if rev else base
    if name:
        return f"NAME:{normalize_name(name)}"
    raise ValueError("reference_key needs a canonical identifier or a name")


def type_hint_from_text(text: str) -> DocumentType:
    lowered = text.lower()
    for phrase, doc_type in _TYPE_PHRASES:
        if phrase in lowered:
            return doc_type
    found = [TYPE_WORDS[w] for w in re.findall(r"[a-z.]+", lowered) if w in TYPE_WORDS]
    return found[-1] if found else DocumentType.UNKNOWN


def _split_location(text: str) -> tuple[str, str | None]:
    """('CVS10, Chapter 4.6 "Change Management"') -> ('CVS10', '4.6'). Nothing is split off
    when no document would remain ('Section 4.12' stays as it is)."""
    if m := _LOCATOR_FIRST_RE.match(text):
        return m.group("rest").strip(" ,;"), m.group("loc")
    if (m := _LOCATOR_AFTER_RE.search(text)) and text[: m.start()].strip(" ,;("):
        return text[: m.start()].strip(" ,;("), m.group("loc")
    return text, None


def _split_revision_words(text: str) -> tuple[str, str | None]:
    """Edition and year revisions: ('AIAG PPAP manual 4th edition') -> ('AIAG PPAP manual', 'EDITION 4')."""
    if m := _EDITION_RE.search(text):
        rest = (text[: m.start()] + text[m.end() :]).strip(" ,;")
        if rest:
            return rest, f"EDITION {int(m.group('n1') or m.group('n2'))}"
    if m := _PAREN_YEAR_RE.search(text):
        rest = (text[: m.start()] + text[m.end() :]).strip(" ,;")
        if rest and any(c.isdigit() for c in rest):  # 'VDA 6.3 (2023)'; not a lone year
            return rest, m.group("year")
    if m := _COLON_YEAR_RE.search(text):
        return (text[: m.start()] + text[m.end() :]).strip(" ,;"), m.group("year")
    return text, None


def _id_span(tokens: list[str], index: int) -> tuple[int, list[str]]:
    """The identifier ending at ``tokens[index]``, extended leftwards by upper-case code tokens
    ('STD 3868'), skipping a volume word between them ('VDA Volume 6.3'). Returns (start, parts)."""
    start = index
    parts = [tokens[index].strip(",;:()")]
    while start > 0:
        prev = tokens[start - 1]
        if _CODE_TOKEN_RE.match(prev) and prev.lower() not in TYPE_WORDS:
            parts.insert(0, prev)
            start -= 1
        elif prev.lower() in _VOLUME_WORDS and start > 1 and _CODE_TOKEN_RE.match(tokens[start - 2]):
            start -= 1  # the volume word is dropped; the code before it is taken next
        else:
            break
    return start, parts


def parse_identifier(raw: str, revision_hint: str | None = None) -> ParsedIdentifier:
    """Split a raw mention into canonical identifier, revision, location and (optional) name."""
    text = _clean(raw)
    text, location = _split_location(text)
    text, word_revision = _split_revision_words(text)
    name_from_title = None
    if _TITLE_DASH in text:
        left, right = text.split(_TITLE_DASH, 1)
        if any(_ID_TOKEN_RE.match(t.strip(",;:()")) and any(c.isdigit() for c in t) for t in left.split()):
            text, name_from_title = left.strip(), right.strip(" ,;") or None
    name_from_parens = None
    if m := _PAREN_NAME_RE.search(text):
        inner = m.group(1).strip()
        if not _REVISION_RE.fullmatch(" " + inner) and not any(ch.isdigit() for ch in inner):
            name_from_parens = inner
            text = (text[: m.start()] + text[m.end() :]).strip()

    revision = normalize_revision(revision_hint)
    if m := _REVISION_RE.search(text):
        revision = revision or normalize_revision(m.group("rev"))
        text = text[: m.start()].strip(" ,;(")
    revision = revision or word_revision

    type_hint = type_hint_from_text(raw)
    tokens = text.split()
    # Candidate identifiers: every token that looks like a code and contains a digit, extended
    # leftwards by upper-case code tokens ("ES 4.12", "STD 3868"). The last candidate with a
    # letter wins ("CVS10 Commercial Vehicle Standard 10" -> CVS10); else the last one.
    candidates = []
    for i, tok in enumerate(tokens):
        tok = tok.strip(",;:()")
        if _ID_TOKEN_RE.match(tok) and any(c.isdigit() for c in tok):
            start, parts = _id_span(tokens, i)
            candidates.append((start, parts))
    if not candidates:
        words = [w for w in tokens]
        while words and words[0].lower() in ARTICLES:
            words = words[1:]
        name = " ".join(words) or None
        return ParsedIdentifier(raw=raw, canonical=None, revision=revision, name=name, type_hint=type_hint, location=location)

    with_letters = [c for c in candidates if any(ch.isalpha() for ch in "".join(c[1]))]
    start, parts = (with_letters or candidates)[-1]
    canonical = re.sub(r"-{2,}", "-", "-".join(parts).upper())
    compact = "".join(parts).upper()

    before = [w for w in tokens[:start] if w.lower() not in ARTICLES]
    name = name_from_parens or name_from_title
    if name is None:
        descriptive = [w for w in before if w.lower().strip(".") not in TYPE_WORDS]
        capitalised = [w for w in descriptive if w[:1].isupper()]
        if len(capitalised) >= 2:
            name = " ".join(before)
    return ParsedIdentifier(
        raw=raw, canonical=canonical, revision=revision, name=name, type_hint=type_hint, location=location, compact=compact
    )


def canonical_input_identifier(identifier: str) -> str:
    parsed = parse_identifier(identifier)
    return parsed.canonical or re.sub(r"\s+", "-", _clean(identifier).upper())


def non_document_reason(canonical: str | None, preceding_text: str = "") -> str | None:
    """Return why an identifier looks like an ordinary number, or None if it may be a document."""
    if canonical is None:
        return None
    if _SECTION_NUMBER_RE.match(canonical):
        return "looks like a section number or quantity, not a document identifier"
    if any(r.match(canonical) for r in _DATE_RES):
        return "looks like a date"
    if not any(c.isalpha() for c in canonical):
        return "contains no letters; likely a part number or quantity"
    if _NON_DOC_CUES.search(preceding_text[-30:]):
        return "preceded by a part-number/section cue"
    return None


def squash(identifier: str) -> str:
    """Comparison form used only to *flag* near duplicates, never to merge them."""
    collapsed = re.sub(r"[^A-Z0-9]", "", identifier.upper())
    return re.sub(r"(?<![0-9])0+(?=[0-9])", "", collapsed)


def near_duplicates(identifiers: list[str]) -> list[tuple[str, str]]:
    pairs = []
    unique = sorted(set(identifiers))
    for i, a in enumerate(unique):
        for b in unique[i + 1 :]:
            if squash(a) == squash(b):
                pairs.append((a, b))
    return pairs


# ---------------------------------------------------------------------------
# Matching document names (documents.yaml names, titles, customer names)
# ---------------------------------------------------------------------------
# A name matches when its words are equal (case, punctuation and spaces ignored, so "Form el
# Q" matches too) or when the shorter of the two (at least three words) is a word-suffix of the
# longer ("VW Formel Q Konkret"). A name that matches several identifiers stays unresolved.

_MIN_SUFFIX_WORDS = 3


def name_match_key(name: str) -> str:
    """Words only, upper-case: 'Formel Q – konkret' == 'FORMEL Q KONKRET' == 'formel q konkret'."""
    return " ".join(re.findall(r"[^\W_]+", name.upper()))


def is_word_suffix(a: str, b: str, min_words: int = _MIN_SUFFIX_WORDS) -> bool:
    """The shorter of two names (at least ``min_words`` words) ends the longer one, word by word."""
    short, long_ = (a, b) if len(a) <= len(b) else (b, a)
    return len(short.split()) >= min_words and long_.endswith(" " + short)


@dataclass
class NameIndex:
    """Names -> identifier. ``resolve`` returns an identifier only when the match is unambiguous."""

    by_key: dict[str, set[str]] = field(default_factory=dict)
    by_bare: dict[str, set[str]] = field(default_factory=dict)

    def add(self, name: str, identifier: str) -> None:
        key = name_match_key(name)
        if not key:
            return
        self.by_key.setdefault(key, set()).add(identifier)
        self.by_bare.setdefault(key.replace(" ", ""), set()).add(identifier)

    def __bool__(self) -> bool:
        return bool(self.by_key)

    def resolve(self, name: str) -> str | None:
        key = name_match_key(name)
        if not key:
            return None
        hits = self.by_key.get(key) or self.by_bare.get(key.replace(" ", ""))
        if not hits:
            hits = {i for other, idents in self.by_key.items() if is_word_suffix(key, other) for i in idents}
        return next(iter(hits)) if len(hits) == 1 else None
