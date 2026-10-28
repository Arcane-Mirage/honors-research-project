"""src/csr_lab/extraction/normalization.py

Conservative, deterministic canonicalisation of document identifiers.

The same function is used to build document IDs in every strategy and by the
evaluator, so formatting differences ("Specification ABC-100" vs "ABC-100")
never count as extraction errors.

What it does:
* upper-cases, unifies Unicode dashes and underscores to "-";
* removes leading articles and document-type words ("the", "Specification");
* splits off an explicit revision ("Rev. B", "Revision D", "rev.3");
* joins spaced identifiers whose parts are upper-case codes ("TSA MP 215" -> "TSA-MP-215").

What it deliberately does not do:
* drop leading zeros ("QP-012" and "QP-12" stay distinct; flagged as near duplicates);
* merge identifiers that merely look similar;
* infer a revision, name or type that the text does not state.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from csr_lab.schemas import DocumentType

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—―−_"), "-")
_REVISION_RE = re.compile(
    r"(?:[,;]?\s*\(?\s*|\s*)\b(?:rev(?:ision)?|issue|version|ver)\b\.?\s*[:#]?\s*(?P<rev>[A-Za-z0-9]{1,4}(?:\.\d{1,3})?)\s*\)?\s*$",
    re.IGNORECASE,
)
_ID_TOKEN_RE = re.compile(r"^[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*$")
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

    @property
    def key(self) -> str:
        return reference_key(self.canonical, self.revision, self.name)


def _clean(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw).translate(_DASHES)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip(" \t\"'`.,;:")


def normalize_revision(revision: str | None) -> str | None:
    if revision is None:
        return None
    rev = _clean(revision)
    rev = re.sub(r"^(?:rev(?:ision)?|issue|version|ver)\.?\s*[:#]?\s*", "", rev, flags=re.IGNORECASE)
    rev = rev.strip(" .()").upper()
    return rev or None


def normalize_name(name: str) -> str:
    words = _clean(name).split()
    while words and words[0].lower() in ARTICLES:
        words = words[1:]
    return " ".join(words).upper()


def reference_key(canonical: str | None, revision: str | None = None, name: str | None = None) -> str:
    """Matching key: 'CVM-ES-210@REV-B', 'QP-12', or 'NAME:SUPPLIER QUALITY MANUAL'."""
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


def parse_identifier(raw: str, revision_hint: str | None = None) -> ParsedIdentifier:
    """Split a raw mention into canonical identifier, revision and (optional) name."""
    text = _clean(raw)
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

    type_hint = type_hint_from_text(raw)
    tokens = text.split()
    # The identifier is the last token that looks like a code and contains a digit,
    # extended leftwards by upper-case code tokens ("ES 4.12", "TSA MP 215").
    id_index = None
    for i in range(len(tokens) - 1, -1, -1):
        tok = tokens[i].strip(",;:()")
        if _ID_TOKEN_RE.match(tok) and any(c.isdigit() for c in tok):
            id_index = i
            break

    if id_index is None:
        words = [w for w in tokens]
        while words and words[0].lower() in ARTICLES:
            words = words[1:]
        name = " ".join(words) or None
        return ParsedIdentifier(raw=raw, canonical=None, revision=revision, name=name, type_hint=type_hint)

    start = id_index
    while start > 0 and _CODE_TOKEN_RE.match(tokens[start - 1]) and tokens[start - 1].lower() not in TYPE_WORDS:
        start -= 1
    parts = [t.strip(",;:()") for t in tokens[start : id_index + 1]]
    canonical = re.sub(r"-{2,}", "-", "-".join(parts).upper())

    before = [w for w in tokens[:start] if w.lower() not in ARTICLES]
    name = name_from_parens
    if name is None:
        descriptive = [w for w in before if w.lower().strip(".") not in TYPE_WORDS]
        capitalised = [w for w in descriptive if w[:1].isupper()]
        if len(capitalised) >= 2:
            name = " ".join(before)
    return ParsedIdentifier(raw=raw, canonical=canonical, revision=revision, name=name, type_hint=type_hint)


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
