"""Heuristic rules that stand in for an LLM inside the offline mock provider.

These rules exist so the full pipeline can run without an API. They are
simple and deliberately imperfect, they are *not* a proposed extraction
method, and results produced with them are not evidence of LLM performance.
They never read gold annotations: they only see the prompt's input JSON.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from csr_lab.ingestion.segmentation import MODAL_RE

_TYPE = r"(?:Specification|Standard|Procedure|Drawing|Method|Manual|Policy)"
HYPHEN_ID_RE = re.compile(r"(?<![A-Za-z0-9-])(?P<id>(?:[A-Z]{2,6}-){1,3}[A-Z]{0,4}\d{1,5}[A-Z]?(?:\.\d{1,3})?)(?![A-Za-z0-9-])")
SPACED_ID_RE = re.compile(r"(?P<type>Specification|Standard|Procedure|Drawing|Method)\s+(?P<id>(?:[A-Z]{2,6}\s){1,2}\d{1,4}(?:\.\d{1,3})?)(?![\d.]*\d)")
REV_AFTER_RE = re.compile(r"^,?\s*(?:Rev(?:ision)?\.?|rev\.?)\s*(?P<rev>[A-Z0-9]{1,3})\b")
NAME_BEFORE_RE = re.compile(r"(?P<name>[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|and|of|for))*\s+" + _TYPE + r")\s+$")
NAME_ONLY_RE = re.compile(
    r"\bthe\s+(?P<name>[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|and|of|for))*\s+" + _TYPE + r")\b(?!\s*\(?\s*[A-Z]{2,6}[-\s])"
)
PAREN_NAME_AFTER_RE = re.compile(r"^\s*\((?P<name>[A-Z][A-Za-z ]{3,60})\)")
TYPE_BEFORE_RE = re.compile(r"\b(specification|standard|procedure|drawing|method|manual|form|policy)\s+$", re.IGNORECASE)

EXCLUDE_RE = re.compile(
    r"does not apply|do not apply|is not applicable|are not applicable|shall not be used|shall not be applied|must not be used",
    re.IGNORECASE,
)
SUPERSEDE_RE = re.compile(r"\bsupersed(?:es|e|ed)\b|\breplaces\b|\bin place of\b", re.IGNORECASE)
PRECEDENCE_RE = re.compile(r"take[s]? precedence|shall govern|shall prevail|\bgoverns\b", re.IGNORECASE)
_ABBREVIATIONS = {"rev", "no", "fig", "e.g", "i.e", "approx", "min", "max"}

STOPWORDS = set(
    """the a an and or of to in on for with by be is are shall must not no any all each this that these those as at
    from per its it which when where than then into under over such other only also may can will been being has
    have had their there was were within before after least more most less according accordance document
    supplier suppliers""".split()
)


@dataclass
class MockMention:
    identifier: str
    revision: str | None
    name: str | None
    relation: str
    quote: str
    start: int
    end: int
    doc_type: str | None


def has_modal(text: str) -> bool:
    return bool(MODAL_RE.search(text))


def clause_bounds(text: str, pos: int) -> tuple[int, int]:
    """Sentence/clause around ``pos`` (boundaries are '.' or ';' followed by whitespace)."""
    start = 0
    for m in re.finditer(r"[.;](?=\s|$)", text):
        word = re.findall(r"[A-Za-z.]+$", text[: m.start()])
        if word and word[0].lower().strip(".") in _ABBREVIATIONS:
            continue
        if m.end() <= pos:
            start = m.end()
        elif m.start() >= pos:
            return start, m.end()
    return start, len(text)


def _quote(text: str, start: int, end: int) -> str:
    c_start, c_end = clause_bounds(text, start)
    clause = text[c_start:c_end].strip()
    if len(clause) <= 180:
        return clause
    lo = max(c_start, start - 60)
    hi = min(c_end, end + 60)
    while lo > c_start and not text[lo - 1].isspace():
        lo -= 1
    while hi < c_end and not text[hi].isspace():
        hi += 1
    return text[lo:hi].strip()


def _relation(text: str, pos: int) -> str:
    c_start, c_end = clause_bounds(text, pos)
    clause = text[c_start:c_end]
    if EXCLUDE_RE.search(clause):
        return "excludes"
    if SUPERSEDE_RE.search(clause):
        return "supersedes"
    if PRECEDENCE_RE.search(clause):
        return "takes_precedence_over"
    return "references"


def _doc_type(word: str | None) -> str | None:
    if not word:
        return None
    return {
        "specification": "technical_specification",
        "method": "technical_specification",
        "standard": "standard",
        "procedure": "procedure",
        "drawing": "engineering_drawing",
        "manual": "quality_manual",
        "form": "form",
    }.get(word.lower())


def find_references(text: str) -> list[MockMention]:
    mentions: list[MockMention] = []
    taken: list[tuple[int, int]] = []

    def overlaps(s: int, e: int) -> bool:
        return any(s < te and ts < e for ts, te in taken)

    for regex in (HYPHEN_ID_RE, SPACED_ID_RE):
        for m in regex.finditer(text):
            s, e = m.start("id"), m.end("id")
            if overlaps(s, e):
                continue
            before = text[:s]
            if re.search(r"(?:part\s+number|p/n)\s*$", before, re.IGNORECASE):
                continue
            revision = None
            if rm := REV_AFTER_RE.match(text[e:]):
                revision = rm.group("rev")
            name = None
            if nm := NAME_BEFORE_RE.search(before):
                words = [w for w in nm.group("name").split() if w[:1].isupper()]
                if len(words) >= 3:  # at least two capitalised words besides the type word
                    name = nm.group("name")
            if name is None and (pm := PAREN_NAME_AFTER_RE.match(text[e:])):
                name = pm.group("name")
            type_word = m.groupdict().get("type")
            if type_word is None and (tm := TYPE_BEFORE_RE.search(before[-25:])):
                type_word = tm.group(1)
            if type_word is None and (tm := re.match(r"\s+(specification|standard|procedure)\b", text[e:], re.IGNORECASE)):
                type_word = tm.group(1)
            taken.append((s, e))
            mentions.append(
                MockMention(
                    identifier=m.group("id"),
                    revision=revision,
                    name=name,
                    relation=_relation(text, s),
                    quote=_quote(text, s, e),
                    start=s,
                    end=e,
                    doc_type=_doc_type(type_word) or (_doc_type(name.split()[-1]) if name else None),
                )
            )
    for m in NAME_ONLY_RE.finditer(text):
        s, e = m.start("name"), m.end("name")
        if overlaps(s, e):
            continue
        words = [w for w in m.group("name").split() if w[:1].isupper()]
        if len(words) < 3:
            continue
        taken.append((s, e))
        mentions.append(
            MockMention(
                identifier=m.group("name"),
                revision=None,
                name=m.group("name"),
                relation=_relation(text, s),
                quote=_quote(text, s, e),
                start=s,
                end=e,
                doc_type=_doc_type(m.group("name").split()[-1]),
            )
        )
    mentions.sort(key=lambda x: x.start)
    return mentions


def distractor_tokens(text: str) -> list[str]:
    """Tokens a careless model might mistake for document identifiers."""
    tokens = re.findall(r"(?<![\w-])(\d+(?:[.-]\d+)+|[A-Z]{1,3}\d{1,4})(?![\w-])", text)
    return [t for t in tokens if not HYPHEN_ID_RE.fullmatch(t)]


# ---------------------------------------------------------------------------
# Conflict heuristics
# ---------------------------------------------------------------------------


def content_words(text: str) -> set[str]:
    words = set()
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in STOPWORDS or len(w) < 3:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                w = w[: -len(suffix)]
                break
        words.add(w)
    return words


def overlap(a: str, b: str) -> float:
    wa, wb = content_words(a), content_words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


_UNIT = r"(?:°C|°F|mm|cm|kg|g|N·m|Nm|%|µg/cm²|µg/m³|lux|kPa|MPa)"
_BOUND_RE = re.compile(
    r"(?P<kw>maximum|minimum|max\.|min\.|no more than|not more than|no less than|not less than|at least|at most|"
    r"shall not exceed|not exceed|up to)\s+(?:(?:[a-z]+\s+){0,3}?(?:of\s+)?)?(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>" + _UNIT + r")",
    re.IGNORECASE,
)
_EXACT_RE = re.compile(
    r"\bat\s+(?:a\s+)?(?:(?:[a-z]+\s+){0,3}?of\s+)?(?P<val>\d+(?:\.\d+)?)\s*(?P<unit>°C|°F)", re.IGNORECASE
)
_UPPER_KW = {"maximum", "max.", "no more than", "not more than", "at most", "shall not exceed", "not exceed", "up to"}
_DURATION_RE = re.compile(
    r"(?P<kw>within|no later than|not later than|no earlier than|not earlier than|at least|for a minimum of|"
    r"a minimum of|minimum of|at most)\s+(?P<val>\d+)\s+(?P<unit>(?:working\s+)?(?:hours?|days?|weeks?|months?|years?|minutes?))"
    r"(?:\s+(?P<rel>before|after|of|from|following))?",
    re.IGNORECASE,
)
_APPROVAL_REQ = re.compile(r"require[sd]?\s+(?:prior\s+|written\s+)*approval|approval\s+(?:is|shall be)\s+required", re.I)
_APPROVAL_NEG = re.compile(r"without\s+(?:prior\s+|written\s+)*approval|do(?:es)?\s+not\s+require\s+(?:\w+\s+)?approval", re.I)
_OPPOSITES = [("100%", "sampling"), ("returnable", "expendable"), ("returnable", "one-way"), ("reusable", "single-use")]


@dataclass
class Constraint:
    direction: str  # upper | lower | exact
    value: float
    unit: str
    start: int
    end: int


def numeric_constraints(text: str) -> list[Constraint]:
    out: list[Constraint] = []
    spans: list[tuple[int, int]] = []
    for m in _BOUND_RE.finditer(text):
        kw = m.group("kw").lower()
        out.append(Constraint("upper" if kw in _UPPER_KW else "lower", float(m.group("val")), m.group("unit"), m.start(), m.end()))
        spans.append((m.start(), m.end()))
    for m in _EXACT_RE.finditer(text):
        if any(m.start() < e and s < m.end() for s, e in spans):
            continue
        out.append(Constraint("exact", float(m.group("val")), m.group("unit"), m.start(), m.end()))
    return out


def duration_constraints(text: str) -> list[Constraint]:
    out = []
    for m in _DURATION_RE.finditer(text):
        kw, rel = m.group("kw").lower(), (m.group("rel") or "").lower()
        if kw in ("within", "at most"):
            direction = "upper"
        elif kw in ("no later than", "not later than"):
            direction = "lower" if rel == "before" else "upper"
        elif kw in ("no earlier than", "not earlier than"):
            direction = "upper" if rel == "before" else "lower"
        else:
            direction = "lower"
        unit = re.sub(r"s$", "", m.group("unit").lower())
        out.append(Constraint(direction, float(m.group("val")), unit, m.start(), m.end()))
    return out


def _incompatible(ca: Constraint, cb: Constraint) -> bool:
    if ca.unit != cb.unit:
        return False
    pairs = [(ca, cb), (cb, ca)]
    for x, y in pairs:
        if x.direction == "lower" and y.direction == "upper" and x.value > y.value:
            return True
        if x.direction == "exact" and y.direction == "upper" and x.value > y.value:
            return True
        if x.direction == "exact" and y.direction == "lower" and x.value < y.value:
            return True
    return False


@dataclass
class PairJudgement:
    conflict_type: str
    description: str
    quote_a: str
    quote_b: str
    ambiguous: bool = False


def _clause(text: str, pos: int) -> str:
    s, e = clause_bounds(text, pos)
    return text[s:e].strip()


def assess_pair(text_a: str, doc_a: str, text_b: str, doc_b: str) -> PairJudgement | None:
    refs_a, refs_b = find_references(text_a), find_references(text_b)

    # Mutual precedence claims between two documents.
    if PRECEDENCE_RE.search(text_a) and PRECEDENCE_RE.search(text_b) and doc_a != doc_b:
        if doc_b in text_a and doc_a in text_b:
            return PairJudgement(
                "potential_precedence_conflict",
                "Each document states that it takes precedence over the other.",
                _clause(text_a, PRECEDENCE_RE.search(text_a).start()),  # type: ignore[union-attr]
                _clause(text_b, PRECEDENCE_RE.search(text_b).start()),  # type: ignore[union-attr]
            )

    # A document excluded in one requirement and required in the other.
    for x, y, flip in ((refs_a, refs_b, False), (refs_b, refs_a, True)):
        for ex in (m for m in x if m.relation == "excludes"):
            for ref in (m for m in y if m.relation != "excludes" and m.identifier == ex.identifier):
                qa, qb = (ex.quote, ref.quote) if not flip else (ref.quote, ex.quote)
                return PairJudgement(
                    "applicability_conflict",
                    f"One requirement states that {ex.identifier} does not apply; the other requires it.",
                    qa,
                    qb,
                )

    subject = overlap(text_a, text_b)

    for ca in numeric_constraints(text_a):
        for cb in numeric_constraints(text_b):
            if _incompatible(ca, cb) and subject >= 0.08:
                return PairJudgement(
                    "numeric_incompatibility",
                    f"Limits {ca.direction} {ca.value:g} {ca.unit} and {cb.direction} {cb.value:g} {cb.unit} "
                    "cannot both be met for what appears to be the same characteristic.",
                    _clause(text_a, ca.start),
                    _clause(text_b, cb.start),
                )

    for ca in duration_constraints(text_a):
        for cb in duration_constraints(text_b):
            if _incompatible(ca, cb) and subject >= 0.08:
                return PairJudgement(
                    "timing_incompatibility",
                    f"Time limits ({ca.direction} {ca.value:g} {ca.unit}, {cb.direction} {cb.value:g} {cb.unit}) "
                    "appear incompatible for the same activity.",
                    _clause(text_a, ca.start),
                    _clause(text_b, cb.start),
                )

    for req_text, neg_text, flip in ((text_a, text_b, False), (text_b, text_a, True)):
        if (mr := _APPROVAL_REQ.search(req_text)) and (mn := _APPROVAL_NEG.search(neg_text)) and subject >= 0.08:
            qa, qb = _clause(req_text, mr.start()), _clause(neg_text, mn.start())
            if flip:
                qa, qb = qb, qa
            return PairJudgement(
                "contradictory_obligation", "One requirement demands approval that the other waives.", qa, qb
            )

    la, lb = text_a.lower(), text_b.lower()
    for x, y in _OPPOSITES:
        for p, q, flip in ((la, lb, False), (lb, la, True)):
            if x in p and y in q and subject >= 0.08:
                qa = _clause(text_a if not flip else text_b, p.index(x))
                qb = _clause(text_b if not flip else text_a, q.index(y))
                if flip:
                    qa, qb = qb, qa
                return PairJudgement(
                    "process_incompatibility", f"'{x}' and '{y}' requirements appear to be incompatible.", qa, qb
                )

    for ma in refs_a:
        for mb in refs_b:
            if ma.identifier == mb.identifier and ma.revision and mb.revision and ma.revision != mb.revision:
                return PairJudgement(
                    "other_or_uncertain",
                    f"Different revisions of {ma.identifier} are required ({ma.revision} vs {mb.revision}).",
                    ma.quote,
                    mb.quote,
                    ambiguous=True,
                )
    return None


def first_clause(text: str) -> str:
    return _clause(text, 0)
