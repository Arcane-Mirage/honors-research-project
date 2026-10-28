"""src/csr_lab/ingestion/units.py

Chunks from a chunk file or from the model, turned into analysable units.

**Model chunking** (``chunking.method: llm``, also usable on its own via ``csr-lab chunk``)::

    batches = paragraph_batches(text, batch_words)        # fill to N words at blank lines
    for each batch: model -> [{title, text}, ...]          # task "semantic_chunking"

A failed or empty model reply keeps the whole batch as one chunk (recorded), so text is
never lost silently. The chunks are returned as ``ChunkInput`` records that remember
their batch span.

**Checking chunks against the source** (``segments_from_chunks``). Every chunk becomes
one ``Segment`` (``origin`` = ``provided`` or ``llm``). The chunk is first located in the
source text (``extraction.fidelity.check_fragment``):

* found verbatim (exact or whitespace-only differences): the unit's text is the exact
  source slice, so later stages work on source text and offsets are real;
* found only approximately (the chunk was reworded, or text was dropped):
  - ``repair: none``: the unit's text is the chunk as given, offsets are approximate and
    ``text_is_source_slice`` is false. Later stages can then only quote the chunk;
  - ``repair: source_slice``: the unit's text is replaced by the source span the chunk was
    aligned to (a deterministic repair; the chunk's own wording is discarded);
* not found: the chunk is analysed as given, without offsets.

The returned ``FragmentCheck`` records are what ``metadata.input_fidelity`` reports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from csr_lab.config import ChunkingSettings
from csr_lab.extraction.fidelity import FragmentCheck, WordIndex, check_fragment
from csr_lab.extraction.grounding import TextIndex
from csr_lab.ingestion.segmentation import LABEL_RE, MODAL_RE, Segment, page_range
from csr_lab.llm.interface import StructuredLLM
from csr_lab.prompts import chunking as chunking_prompt
from csr_lab.schemas import InputDocument

# ---------------------------------------------------------------------------
# Chunks -> units, checked against the source
# ---------------------------------------------------------------------------

@dataclass
class ChunkInput:
    chunk_id: str
    content: str
    title: str | None = None
    reported_pages: list[int] = field(default_factory=list)
    search_span: tuple[int, int] | None = None  # where the chunk is expected (e.g. its batch)


def _first_label(text: str) -> str | None:
    for line in text.split("\n"):
        if m := LABEL_RE.match(line.strip()):
            return m.group("label")
    return None


def segments_from_chunks(
    doc: InputDocument,
    chunks: list[ChunkInput],
    *,
    origin: str,
    settings: ChunkingSettings,
    indexes: tuple[TextIndex, WordIndex] | None = None,
) -> tuple[list[Segment], list[FragmentCheck]]:
    text = doc.text or ""
    text_index, word_index = indexes or (TextIndex(text), WordIndex(text))
    prefix = {"provided": "P", "llm": "L"}.get(origin, "X")
    segments: list[Segment] = []
    checks: list[FragmentCheck] = []
    for n, chunk in enumerate(chunks, start=1):
        check = check_fragment(
            chunk.content,
            text_index=text_index,
            word_index=word_index,
            fragment_id=chunk.chunk_id,
            document_id=doc.document_id,
            title=chunk.title,
            min_ratio=settings.min_match_ratio,
            search_span=chunk.search_span,
            pages=doc.pages,
            reported_pages=chunk.reported_pages,
        )
        checks.append(check)
        located = check.start_offset is not None and check.end_offset is not None
        notes = [f"{origin} chunk {chunk.chunk_id}: {check.match} (word ratio {check.fragment_ratio:.2f})"]
        if check.match in ("exact", "whitespace_normalized") or (located and settings.repair == "source_slice"):
            start, end = check.start_offset, check.end_offset
            body = text[start:end]  # type: ignore[index]
            from_source = True
            if check.match == "fuzzy":
                notes.append("chunk wording differed from the source; the aligned source text is analysed instead")
        else:
            body = chunk.content.strip()
            start, end = (check.start_offset or 0), (check.end_offset or 0)
            from_source = False
            if check.match == "fuzzy":
                notes.append("chunk wording differs from the source; analysed as given (not verbatim)")
            else:
                notes.append("chunk text was not found in the source; analysed as given")
        if located:
            p_start, p_end = page_range(start, end, doc.pages)  # type: ignore[arg-type]
        else:
            p_start = min(chunk.reported_pages) if chunk.reported_pages else None
            p_end = max(chunk.reported_pages) if chunk.reported_pages else None
        has_modal = bool(MODAL_RE.search(body))
        segments.append(
            Segment(
                segment_id=f"SEG-{doc.document_id[4:]}-{prefix}{n:03d}",
                document_id=doc.document_id,
                kind="paragraph",
                chapter=_first_label(body),
                title=chunk.title,
                text=body,
                start_offset=start,  # type: ignore[arg-type]
                end_offset=end,  # type: ignore[arg-type]
                line_start_offset=start,  # type: ignore[arg-type]
                page_start=p_start,
                page_end=p_end,
                has_modal=has_modal,
                confidence="low",
                notes=notes,
                origin=origin,  # type: ignore[arg-type]
                text_is_source_slice=from_source,
            )
        )
    return segments, checks


# ---------------------------------------------------------------------------
# Model chunking (chunking.method: llm)
# ---------------------------------------------------------------------------

_PARA_BREAK = re.compile(r"\n[ \t]*\n")


def paragraph_batches(text: str, batch_words: int) -> list[tuple[int, int]]:
    """Character spans of word-count batches cut at blank lines (not at sections).

    This is the generic "fill a batch to N words" step that precedes model-based
    chunking; a single paragraph longer than the budget becomes its own batch.
    """
    paragraphs: list[tuple[int, int]] = []
    pos = 0
    for m in _PARA_BREAK.finditer(text):
        if text[pos : m.start()].strip():
            paragraphs.append((pos, m.start()))
        pos = m.end()
    if text[pos:].strip():
        paragraphs.append((pos, len(text.rstrip())))
    batches: list[tuple[int, int]] = []
    start = end = None
    words = 0
    for p_start, p_end in paragraphs:
        n = len(text[p_start:p_end].split())
        if start is not None and words + n > batch_words:
            batches.append((start, end))  # type: ignore[arg-type]
            start, words = None, 0
        if start is None:
            start = p_start
        end = p_end
        words += n
    if start is not None:
        batches.append((start, end))  # type: ignore[arg-type]
    return batches


@dataclass
class ModelChunks:
    chunks: dict[str, list[ChunkInput]] = field(default_factory=dict)  # document_id -> chunks
    failed_batches: list[tuple[str, int, str]] = field(default_factory=list)  # (document_id, batch, error)
    truncated: list[str] = field(default_factory=list)  # documents cut at max_chars


def model_chunks(llm: StructuredLLM, documents: list[InputDocument], settings: ChunkingSettings, max_chars: int) -> ModelChunks:
    out = ModelChunks()
    jobs = []
    for d in documents:
        text = d.text or ""
        if len(text) > max_chars:
            out.truncated.append(d.document_id)
        out.chunks.setdefault(d.document_id, [])
        for b, span in enumerate(paragraph_batches(text[:max_chars], settings.batch_words), start=1):
            jobs.append((d, b, span))

    def run(job):
        d, b, (start, end) = job
        payload = {"document_id": d.document_id, "target_chunk_words": settings.target_chunk_words, "text": (d.text or "")[start:end]}
        system, user = chunking_prompt.SPEC.render(payload)
        return job, llm.call(
            task="semantic_chunking",
            stage="llm_chunk",
            system=system,
            user=user,
            response_model=chunking_prompt.SemanticChunkingResponse,
            prompt_version=chunking_prompt.SPEC.fingerprint,
        )

    for (d, b, (start, end)), result in llm.map(run, jobs):
        short = d.document_id[4:]
        inputs = out.chunks[d.document_id]
        if not result.ok or result.value is None or not any(c.text.strip() for c in result.value.chunks):
            out.failed_batches.append((d.document_id, b, "; ".join(result.errors)[:200] or "no chunks returned"))
            inputs.append(ChunkInput(f"{short}-B{b:02d}-FALLBACK", (d.text or "")[start:end], "fallback", search_span=(start, end)))
            continue
        for k, chunk in enumerate(result.value.chunks, start=1):
            if chunk.text.strip():
                inputs.append(ChunkInput(f"{short}-B{b:02d}-C{k:02d}", chunk.text, chunk.title or None, search_span=(start, end)))
    return out
