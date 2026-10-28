"""src/csr_lab/evaluation/matching.py

Deterministic matching of predictions to gold annotations.

Matching rules (applied identically to every strategy):

1. **Requirements** are aligned one-to-one within each source document.
   Score = max(text similarity, character-span overlap). Text similarity is
   ``difflib`` ratio on lower-cased, whitespace-collapsed text; span overlap is
   intersection-over-union of character offsets. Pairs scoring at least the
   threshold (default 0.75) are matched greedily, best first.
2. **Documents** are compared by key: canonical identifier (via the shared
   canonicalisation) plus revision when one is stated, e.g. ``CVM-ES-210@REV-B``.
   A revision-qualified reference and an unqualified one are different keys.
   A name-only prediction has key ``NAME:<name>`` and never matches an identifier.
3. **References** are scored as (gold requirement, document key) pairs. A
   correct document attached to the wrong requirement is one false positive
   and one false negative. Repeated predictions of the same pair count once and
   are reported separately as duplicates.
4. **Conflicts** are unordered pairs of aligned requirement IDs.

Before matching, ``reanchor_gold`` moves gold requirement offsets of PDF inputs onto
the text the PDF parser actually produced (spacing differs, running headers can sit
inside a requirement). Each gold requirement is located whitespace-insensitively,
or else by its first and last eight words. Only offsets change; gold text, references
and conflicts are untouched. A requirement that cannot be located keeps its original
offsets (span overlap then fails for it; text similarity still works).
"""

from __future__ import annotations

from difflib import SequenceMatcher

from csr_lab.extraction.grounding import TextIndex
from csr_lab.extraction.normalization import reference_key
from csr_lab.schemas import (
    CaseGold,
    GoldReference,
    GoldRequirement,
    InputDocument,
    ReferencedDocument,
    Requirement,
)

UNMATCHED = "UNMATCHED:"


def norm_text(text: str) -> str:
    return " ".join(text.split()).lower()


def text_similarity(a: str, b: str) -> float:
    na, nb = norm_text(a), norm_text(b)
    if na == nb:
        return 1.0
    matcher = SequenceMatcher(None, na, nb, autojunk=False)
    if matcher.real_quick_ratio() < 0.5 or matcher.quick_ratio() < 0.5:
        return 0.0
    return matcher.ratio()


def span_iou(a: tuple[int | None, int | None], b: tuple[int, int]) -> float:
    if a[0] is None or a[1] is None:
        return 0.0
    inter = min(a[1], b[1]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union


def align_requirements(
    predicted: list[Requirement], gold: list[GoldRequirement], threshold: float
) -> dict[str, str]:
    """predicted requirement_id -> gold_id."""
    scored = []
    for p in predicted:
        for g in gold:
            if p.source_document_id != g.source_document_id:
                continue
            score = max(
                text_similarity(p.requirement_text, g.requirement_text),
                span_iou((p.source_location.start_offset, p.source_location.end_offset), (g.start_offset, g.end_offset)),
            )
            if score >= threshold:
                scored.append((score, p.requirement_id, g.gold_id))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    mapping: dict[str, str] = {}
    used: set[str] = set()
    for _, pid, gid in scored:
        if pid in mapping or gid in used:
            continue
        mapping[pid] = gid
        used.add(gid)
    return mapping


def mapped_id(pred_id: str, alignment: dict[str, str]) -> str:
    return alignment.get(pred_id, f"{UNMATCHED}{pred_id}")


def gold_key(ref: GoldReference) -> str:
    return reference_key(ref.canonical_identifier, ref.revision)


def predicted_key(doc: ReferencedDocument) -> str:
    if doc.canonical_identifier:
        return reference_key(doc.canonical_identifier, doc.revision)
    name = (doc.metadata.get("names") or [None])[0] or doc.document_name or doc.document_id
    return reference_key(None, None, name)


def unordered(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


# ---------------------------------------------------------------------------
# Granularity-independent matching (units that may hold several requirements)
# ---------------------------------------------------------------------------


def _words(text: str) -> list[str]:
    return [w.strip("\"'.,;:()[]|").lower() for w in text.split() if w.strip("\"'.,;:()[]|")]


def contains_requirement(unit: Requirement, gold: GoldRequirement, min_share: float = 0.8) -> bool:
    """True when a predicted unit contains a gold requirement.

    By offsets when both are known (2-character tolerance), otherwise when at least
    ``min_share`` of the gold requirement's words occur in the unit's text. The word
    test also accepts units whose wording was changed a little (a reworded chunk).
    """
    if unit.source_document_id != gold.source_document_id:
        return False
    start, end = unit.source_location.start_offset, unit.source_location.end_offset
    if start is not None and end is not None and start <= gold.start_offset + 2 and gold.end_offset <= end + 2:
        return True
    gold_words = _words(gold.requirement_text)
    if not gold_words:
        return False
    unit_counts: dict[str, int] = {}
    for w in _words(unit.requirement_text):
        unit_counts[w] = unit_counts.get(w, 0) + 1
    hits = 0
    for w in gold_words:
        if unit_counts.get(w, 0) > 0:
            unit_counts[w] -= 1
            hits += 1
    return hits / len(gold_words) >= min_share


def family_key(key: str) -> str:
    """'CVM-TM-311@REV-B' -> 'CVM-TM-311'."""
    return key.split("@", 1)[0]


_EDGE_WORDS = 8


def reanchor_gold(gold: CaseGold, documents: list[InputDocument]) -> CaseGold:
    pdf = {d.document_id: d for d in documents if d.format == "pdf" and d.text}
    if not pdf:
        return gold
    indexes: dict[str, TextIndex] = {}
    requirements = []
    changed = False
    for g in gold.requirements:
        doc = pdf.get(g.source_document_id)
        if doc is None:
            requirements.append(g)
            continue
        index = indexes.setdefault(doc.document_id, TextIndex(doc.text or ""))
        span = None
        if found := index.find(g.requirement_text):
            span = found[0], found[1]
        else:
            words = g.requirement_text.split()
            head = index.find(" ".join(words[:_EDGE_WORDS]))
            if head is not None:
                tail = index.find(" ".join(words[-_EDGE_WORDS:]), head[0])
                if tail is not None and tail[1] > head[0]:
                    span = head[0], tail[1]
        if span and span != (g.start_offset, g.end_offset):
            requirements.append(g.model_copy(update={"start_offset": span[0], "end_offset": span[1]}))
            changed = True
        else:
            requirements.append(g)
    return gold.model_copy(update={"requirements": requirements}) if changed else gold
