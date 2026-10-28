"""src/csr_lab/ingestion/chunking.py

Section-aware chunking shared by every strategy.

All strategies in an experiment receive the same chunk plan, so differences in
results cannot come from different input slicing.

Rules:
* Segments are never split. A segment larger than ``chunk_max_chars`` becomes
  its own (oversized) chunk and the limitation is recorded.
* A chunk preferably ends at a top-level heading once it is reasonably full.
* ``chunk_overlap_segments`` trailing candidates of the previous chunk are
  repeated as read-only context in the next chunk of the same document.
* Text beyond ``max_document_chars`` is not processed; this is recorded as a
  coverage limitation, never silently dropped.
* Chunks may span several documents of a bundle when they fit, which lets
  single-pass prompts see cross-document content.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from csr_lab.config import ProcessingSettings
from csr_lab.ingestion.segmentation import Segment
from csr_lab.schemas import CoverageLimitation, InputDocument

_PREFERRED_BREAK_FILL = 0.6


class Chunk(BaseModel):
    chunk_id: str
    document_ids: list[str]
    segment_ids: list[str] = Field(description="Segments this chunk is responsible for")
    context_segment_ids: list[str] = Field(default_factory=list, description="Overlap: shown, not extracted")
    char_count: int
    oversized: bool = False


class ChunkPlan(BaseModel):
    chunks: list[Chunk]
    processed_segment_ids: list[str]
    skipped_segment_ids: list[str]
    limitations: list[CoverageLimitation]


def _segment_size(seg: Segment) -> int:
    # Chunk units whose text differs from the source have only approximate offsets.
    return max(seg.end_offset - seg.line_start_offset, len(seg.text)) + 1


def plan_chunks(
    documents: list[InputDocument], segments: list[Segment], settings: ProcessingSettings
) -> ChunkPlan:
    by_doc: dict[str, list[Segment]] = {}
    for seg in segments:
        by_doc.setdefault(seg.document_id, []).append(seg)

    limitations: list[CoverageLimitation] = []
    processed: list[Segment] = []
    skipped: list[str] = []
    for doc in documents:
        used = 0
        for seg in by_doc.get(doc.document_id, []):
            size = _segment_size(seg)
            if used + size > settings.max_document_chars:
                skipped.append(seg.segment_id)
                continue
            used += size
            processed.append(seg)
        doc_skipped = [s for s in by_doc.get(doc.document_id, []) if s.segment_id in set(skipped)]
        if doc_skipped:
            limitations.append(
                CoverageLimitation(
                    kind="truncated_input",
                    document_id=doc.document_id,
                    message=(
                        f"document exceeds max_document_chars={settings.max_document_chars}; "
                        f"{len(doc_skipped)} segments starting at offset {doc_skipped[0].start_offset} "
                        "were not analysed"
                    ),
                )
            )

    chunks: list[Chunk] = []
    current: list[Segment] = []
    current_size = 0
    previous_tail: list[Segment] = []

    def close() -> None:
        nonlocal current, current_size, previous_tail
        if not current:
            return
        first_doc = current[0].document_id
        context = [s for s in previous_tail if s.document_id == first_doc]
        doc_ids = list(dict.fromkeys(s.document_id for s in [*context, *current]))
        chunk_id = f"CHUNK-{len(chunks) + 1:03d}"
        oversized = current_size > settings.chunk_max_chars
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_ids=doc_ids,
                segment_ids=[s.segment_id for s in current],
                context_segment_ids=[s.segment_id for s in context],
                char_count=current_size + sum(_segment_size(s) for s in context),
                oversized=oversized,
            )
        )
        if oversized:
            limitations.append(
                CoverageLimitation(
                    kind="oversized_segment",
                    chunk_id=chunk_id,
                    document_id=first_doc,
                    message=f"a single segment exceeds chunk_max_chars={settings.chunk_max_chars}; sent unsplit",
                )
            )
        candidates = [s for s in current if s.is_candidate]
        n = settings.chunk_overlap_segments
        previous_tail = candidates[-n:] if n > 0 else []
        current, current_size = [], 0

    for seg in processed:
        size = _segment_size(seg)
        is_top_heading = seg.kind == "heading" and seg.label is not None and "." not in seg.label.split()[-1]
        if current and current_size + size > settings.chunk_max_chars:
            close()
        elif current and is_top_heading and current_size >= _PREFERRED_BREAK_FILL * settings.chunk_max_chars:
            close()
        current.append(seg)
        current_size += size
    close()

    return ChunkPlan(
        chunks=chunks,
        processed_segment_ids=[s.segment_id for s in processed],
        skipped_segment_ids=skipped,
        limitations=limitations,
    )


def chunk_raw_text(chunk: Chunk, segments: dict[str, Segment], documents: dict[str, InputDocument]) -> list[dict]:
    """Raw source text per document covered by a chunk (labels and headings included).

    Used by the single-pass baseline, which receives text rather than segments.
    """
    out = []
    ids = [*chunk.context_segment_ids, *chunk.segment_ids]
    for doc_id in chunk.document_ids:
        segs = [segments[i] for i in ids if segments[i].document_id == doc_id]
        if not segs:
            continue
        doc = documents[doc_id]
        start = min(s.line_start_offset for s in segs)
        end = max(s.end_offset for s in segs)
        if any(s.origin != "deterministic" for s in segs):
            # Chunk units: send each unit's own text (it may differ from the source).
            out.append(
                {
                    "document_id": doc_id,
                    "identifier": doc.canonical_identifier,
                    "title": doc.title,
                    "text_start_offset": start,
                    "text": "\n\n".join(s.text for s in segs),
                }
            )
            continue
        out.append(
            {
                "document_id": doc_id,
                "identifier": doc.canonical_identifier,
                "title": doc.title,
                "text_start_offset": start,
                "text": (doc.text or "")[start:end],
            }
        )
    return out
