"""Turn model-reported requirements into grounded ``RequirementDraft`` records.

Grounding is applied identically in every strategy and only *annotates*:
* the reported text is located in the source (exact, then whitespace-insensitive);
* when found, the stored text is the exact source slice, so wording is verbatim;
* when not found, the model's text is kept and flagged ``not_found`` with no offsets.

Requirements found twice (for example in overlapping chunks) are merged.
IDs are assigned per document in source order, so they are stable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from csr_lab.drafts import RequirementDraft
from csr_lab.extraction.grounding import SourceIndex
from csr_lab.ids import requirement_id
from csr_lab.ingestion.segmentation import Segment


@dataclass
class ReportedRequirement:
    """A requirement as reported by a model, before grounding."""

    document_id: str | None
    chapter: str | None
    text: str
    chunk_id: str
    segment_id: str | None = None
    local_key: str | None = None  # single-pass keys (R1, R2 ...)


def _segment_for_span(segments: list[Segment], document_id: str, start: int, end: int) -> Segment | None:
    best, best_overlap = None, 0
    for seg in segments:
        if seg.document_id != document_id or not seg.is_candidate:
            continue
        ov = min(end, seg.end_offset) - max(start, seg.start_offset)
        if ov > best_overlap:
            best, best_overlap = seg, ov
    return best


def ground_requirements(
    reported: list[ReportedRequirement],
    *,
    sources: SourceIndex,
    segments: list[Segment],
    chunk_documents: dict[str, list[str]],
) -> tuple[list[RequirementDraft], dict[str, str], list[str]]:
    """Ground, deduplicate and number requirements.

    Returns (drafts, local_key -> requirement_id, warnings).
    """
    seg_by_id = {s.segment_id: s for s in segments}
    warnings: list[str] = []
    grounded: list[tuple[RequirementDraft, ReportedRequirement]] = []

    for i, rep in enumerate(reported):
        text = rep.text.strip()
        if not text:
            warnings.append(f"empty requirement text reported in {rep.chunk_id}; ignored")
            continue
        seg = seg_by_id.get(rep.segment_id) if rep.segment_id else None
        doc_candidates: list[str] = []
        if seg:
            doc_candidates.append(seg.document_id)
        if rep.document_id and rep.document_id in sources.documents and rep.document_id not in doc_candidates:
            doc_candidates.append(rep.document_id)
        doc_candidates += [d for d in chunk_documents.get(rep.chunk_id, []) if d not in doc_candidates]

        found = None
        found_doc = None
        for doc_id in doc_candidates:
            if seg and seg.document_id == doc_id:
                found = sources.locate(doc_id, text, seg.start_offset, seg.end_offset)
            if found is None:
                found = sources.locate(doc_id, text)
            if found is not None:
                found_doc = doc_id
                break

        if found is not None and found_doc is not None:
            start, end, match = found
            exact = sources.text(found_doc)[start:end]
            seg = seg if seg and seg.document_id == found_doc else _segment_for_span(segments, found_doc, start, end)
            draft = RequirementDraft(
                requirement_id=f"TMP-{i}",
                document_id=found_doc,
                chapter=rep.chapter if rep.chapter else (seg.chapter if seg else None),
                text=exact,
                start_offset=start,
                end_offset=end,
                match=match,
                section_path=seg.section_path if seg else [],
                segment_ids=[seg.segment_id] if seg else [],
                chunk_ids=[rep.chunk_id],
                segmentation_confidence=seg.confidence if seg else "not_segmented",
                page_start=seg.page_start if seg else None,
                page_end=seg.page_end if seg else None,
            )
        else:
            doc_id = doc_candidates[0] if doc_candidates else (rep.document_id or "")
            if doc_id not in sources.documents:
                warnings.append(f"requirement from {rep.chunk_id} names an unknown document; ignored")
                continue
            draft = RequirementDraft(
                requirement_id=f"TMP-{i}",
                document_id=doc_id,
                chapter=rep.chapter or (seg.chapter if seg else None),
                text=text,
                match="not_found",
                segment_ids=[seg.segment_id] if seg else [],
                chunk_ids=[rep.chunk_id],
                segmentation_confidence=seg.confidence if seg else "not_segmented",
                notes=["requirement text not found verbatim in the source; kept as reported"],
            )
        grounded.append((draft, rep))

    # Merge duplicates (same document, overlapping spans or identical text).
    merged: list[tuple[RequirementDraft, list[ReportedRequirement]]] = []
    for draft, rep in grounded:
        target = None
        for existing, reps in merged:
            if existing.document_id != draft.document_id:
                continue
            if draft.span and existing.span:
                ov = min(draft.span[1], existing.span[1]) - max(draft.span[0], existing.span[0])
                shorter = min(draft.span[1] - draft.span[0], existing.span[1] - existing.span[0])
                if shorter > 0 and ov / shorter >= 0.8:
                    target = (existing, reps)
                    break
            elif " ".join(draft.text.split()) == " ".join(existing.text.split()):
                target = (existing, reps)
                break
        if target is None:
            merged.append((draft, [rep]))
        else:
            existing, reps = target
            existing.chunk_ids = list(dict.fromkeys([*existing.chunk_ids, *draft.chunk_ids]))
            existing.segment_ids = list(dict.fromkeys([*existing.segment_ids, *draft.segment_ids]))
            reps.append(rep)

    # Stable numbering per document in source order.
    doc_order = {doc_id: i for i, doc_id in enumerate(sources.documents)}
    merged.sort(
        key=lambda item: (
            doc_order.get(item[0].document_id, 10**6),
            item[0].start_offset if item[0].start_offset is not None else 10**12,
            item[0].text,
        )
    )
    counters: dict[str, int] = {}
    key_map: dict[str, str] = {}
    drafts: list[RequirementDraft] = []
    for draft, reps in merged:
        counters[draft.document_id] = counters.get(draft.document_id, 0) + 1
        draft.requirement_id = requirement_id(draft.document_id, counters[draft.document_id])
        drafts.append(draft)
        for rep in reps:
            if rep.local_key:
                key_map[f"{rep.chunk_id}:{rep.local_key}"] = draft.requirement_id
    return drafts, key_map, warnings
