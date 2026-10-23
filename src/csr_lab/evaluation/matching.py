"""Deterministic matching of predictions to gold annotations.

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
"""

from __future__ import annotations

from difflib import SequenceMatcher

from csr_lab.extraction.normalization import reference_key
from csr_lab.schemas import GoldReference, GoldRequirement, ReferencedDocument, Requirement

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
