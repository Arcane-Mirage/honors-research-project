"""tests/test_segmentation.py

Segmentation into exact source slices; chunking limits, overlap and truncation.
"""

from csr_lab.config import ProcessingSettings
from csr_lab.ingestion.chunking import chunk_raw_text, plan_chunks
from csr_lab.ingestion.segmentation import requirement_candidates, segment_document
from csr_lab.schemas import InputDocument

TEXT = """\
ACME
Example CSR
Document ID: EX-1

1 Scope
1.1 This document applies to suppliers.

4 Quality Planning
4.2 Control Plans
The supplier shall prepare control plans and keep them for a period of
10 years after the end of production.
4.2.1 The supplier shall perform the tests listed below:
| Test | Method |
| Salt spray | EX-TM-1 |
TSA 2.1 The supplier shall follow prefixed chapter labels.

5 Records
The supplier shall retain records.

The supplier shall also archive drawings.
"""


def _doc(text=TEXT):
    return InputDocument(
        document_id="DOC-EX-1",
        canonical_identifier="EX-1",
        identifier_source="header",
        filename="x.txt",
        format="txt",
        sha256="0",
        char_count=len(text),
        parse_status="ok",
        text=text,
    )


def test_segments_are_exact_slices_with_chapters():
    doc = _doc()
    segs = segment_document(doc)
    for s in segs:
        assert doc.text[s.start_offset : s.end_offset] == s.text
    cands = requirement_candidates(segs)
    chapters = [s.chapter for s in cands]
    assert chapters == ["1.1", "4.2", "4.2.1", "TSA 2.1", "5", "5"]
    assert all(s.kind != "heading" for s in cands)
    assert segs[0].kind == "front_matter"


def test_wrapped_number_line_and_tables_are_continuations():
    cands = requirement_candidates(segment_document(_doc()))
    control = next(s for s in cands if s.chapter == "4.2")
    assert "10 years after the end of production." in control.text
    table = next(s for s in cands if s.chapter == "4.2.1")
    assert "EX-TM-1" in table.text
    assert table.section_path == ["4 Quality Planning", "4.2 Control Plans"]


def test_headings_and_informational_text_flags():
    segs = segment_document(_doc())
    headings = [s for s in segs if s.kind == "heading"]
    assert [h.label for h in headings] == ["1", "4", "4.2", "5"]
    scope = next(s for s in segs if s.chapter == "1.1")
    assert not scope.has_modal and scope.confidence == "low"


def test_chunking_limits_overlap_and_truncation():
    body = "\n".join(f"1.{i} The supplier shall do item {i} with enough words to take space." for i in range(1, 40))
    doc = _doc("Document ID: EX-1\n\n1 Scope\n" + body + "\n")
    segs = segment_document(doc)
    plan = plan_chunks([doc], segs, ProcessingSettings(chunk_max_chars=600, chunk_overlap_segments=1))
    assert len(plan.chunks) > 2
    assert all(c.char_count <= 600 + 200 for c in plan.chunks)
    assert plan.chunks[1].context_segment_ids, "overlap segment expected"
    assert set(plan.chunks[1].context_segment_ids) <= set(plan.chunks[0].segment_ids)
    covered = [sid for c in plan.chunks for sid in c.segment_ids]
    assert sorted(covered) == sorted(s.segment_id for s in segs)

    truncated = plan_chunks([doc], segs, ProcessingSettings(max_document_chars=1000, chunk_max_chars=600))
    assert truncated.skipped_segment_ids
    assert any(lim.kind == "truncated_input" for lim in truncated.limitations)


def test_oversized_segment_is_kept_whole():
    long_req = "1.1 The supplier shall " + "comply with everything " * 80 + "."
    doc = _doc("Document ID: EX-1\n\n1 Scope\n" + long_req + "\n")
    segs = segment_document(doc)
    plan = plan_chunks([doc], segs, ProcessingSettings(chunk_max_chars=500))
    assert any(c.oversized for c in plan.chunks)
    assert any(lim.kind == "oversized_segment" for lim in plan.limitations)


def test_chunk_raw_text_includes_labels():
    doc = _doc()
    segs = segment_document(doc)
    plan = plan_chunks([doc], segs, ProcessingSettings())
    raw = chunk_raw_text(plan.chunks[0], {s.segment_id: s for s in segs}, {doc.document_id: doc})
    assert "4.2.1 The supplier shall perform" in raw[0]["text"]
