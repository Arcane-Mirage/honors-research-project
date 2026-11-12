"""tests/test_segmentation.py

Labelled-line segmentation (the chapter fallback) into exact source slices; chunking limits, overlap and truncation.
"""

import pytest

from csr_lab.config import ProcessingSettings, SystemPromptSettings
from csr_lab.ingestion.chunking import plan_chunks
from csr_lab.ingestion.segmentation import CustomerMatcher, find_blocks, hide_other_customers, segment_document
from csr_lab.pipelines.registry import derive_strategy
from csr_lab.pipelines.run import run_analysis
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
HVG 2.1 The supplier shall follow prefixed chapter labels.

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
    cands = [s for s in segs if s.is_candidate]
    chapters = [s.chapter for s in cands]
    assert chapters == ["1.1", "4.2", "4.2.1", "HVG 2.1", "5", "5"]
    assert all(s.kind != "heading" for s in cands)
    assert segs[0].kind == "front_matter"


def test_wrapped_number_line_and_tables_are_continuations():
    cands = [s for s in segment_document(_doc()) if s.is_candidate]
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


# ---------------------------------------------------------------------------
# Customer-labelled blocks ('MAN: ...', 'Scania/Navistar: ...'): finding them, tagging references
# with the customers they apply to, and hiding other customers' blocks from the model.
# ---------------------------------------------------------------------------

SETTINGS = SystemPromptSettings(
    customer="Navistar",
    customer_aliases=["International", "International Motors", "International Motors LLC"],
    other_customers_names={"MAN": ["MAN Truck & Bus"], "Scania": ["Scania CV AB"], "VWTB": ["VW T&B"]},
)

CHAPTER = """\
Suppliers shall keep records according to HVG-QM-001.
MAN: The relevant MAN standard is MAN-ST-3431.
Scania/Navistar: PPAP submission level 4 shall be used according to AIAG-PP-004.
Information arising out of the business relationship may be shared.
MAN Truck & BUS / Scania / VW T&B:

ePPAP is used according to MAN-ST-239.
Navistar: The routine ISQ-FO-018 shall be filled and documented.
International Motors IMDS account: 186555
"""


@pytest.fixture
def matcher():
    return CustomerMatcher.from_settings(SETTINGS)


def test_label_parts_must_all_be_customers(matcher):
    assert matcher.label_customers("Scania/Navistar") == ["Scania", "Navistar"]
    assert matcher.label_customers("MAN Truck & BUS / Scania / VW T&B / International Motors LLC.") == ["MAN", "Scania", "VWTB", "Navistar"]
    assert matcher.label_customers("International Motors IMDS account") is None  # not a customer name
    assert matcher.label_customers("Field claims") is None
    assert CustomerMatcher.from_settings(SystemPromptSettings(customer="Navistar")) is None  # no other customers: scoping off


def test_blocks_and_their_extent(matcher):
    blocks = find_blocks(CHAPTER, 100, matcher)
    got = [(b.customers, CHAPTER[b.start_offset - 100 : b.end_offset - 100]) for b in blocks]
    assert got == [
        (["MAN"], "MAN: The relevant MAN standard is MAN-ST-3431."),
        (["Scania", "Navistar"], "Scania/Navistar: PPAP submission level 4 shall be used according to AIAG-PP-004."),
        (["MAN", "Scania", "VWTB"], "MAN Truck & BUS / Scania / VW T&B:\n\nePPAP is used according to MAN-ST-239."),  # a bare label keeps its text
        (["Navistar"], "Navistar: The routine ISQ-FO-018 shall be filled and documented."),
    ]
    assert [b.applies_to_target for b in blocks] == [False, True, False, True]


def test_chapter_titled_with_a_customer_is_one_block(matcher):
    (block,) = find_blocks("Proof of inspection results shall be sufficient.\nMore text.", 0, matcher, title="MAN")
    assert block.customers == ["MAN"] and not block.applies_to_target


def test_hiding_replaces_other_customers_blocks(matcher):
    blocks = find_blocks(CHAPTER, 0, matcher)
    shown = hide_other_customers(CHAPTER, 0, blocks)
    assert "MAN-ST-3431" not in shown and "MAN-ST-239" not in shown and "[text for MAN only omitted]" in shown
    assert "AIAG-PP-004" in shown and "ISQ-FO-018" in shown and "HVG-QM-001" in shown


def _scoping_doc(text: str) -> InputDocument:
    return InputDocument(document_id="DOC-CVS-T", canonical_identifier="CVS-T", identifier_source="filename", filename="cvs_t.txt",
                         format="txt", sha256="0", char_count=len(text), parse_status="ok", text=text)


@pytest.mark.parametrize("mode", ["off", "tag", "filter"])
def test_modes_end_to_end(config, mode):
    text = "Contents\n1 General .... 1\n2 Records .... 2\n3 Approval .... 3\n1 General\n" + CHAPTER + "2 Records\nRecords shall be kept.\n3 Approval\nApproval is needed.\n"
    cfg = config.model_copy(update={"system_prompt": SETTINGS})
    cfg.llm.mock.reference_miss_rate = 0.0
    cfg.llm.mock.hallucination_rate = 0.0
    spec = derive_strategy(f"staged__scope_{mode}", "staged", {"customer_scoping": mode})
    result = run_analysis([_scoping_doc(text)], strategy=spec, config=cfg)
    chapter = next(r for r in result.requirements if r.chapter == "1")
    assert chapter.requirement_text == CHAPTER.rstrip()  # the export is always verbatim
    found = {m.raw_identifier: m.applies_to for m in chapter.reference_mentions}
    scope = result.coverage.customer_scope
    other = result.coverage.references.from_other_customer_blocks
    if mode == "off":
        assert other is None and scope.blocks == 0 and "MAN-ST-3431" in found
    elif mode == "tag":
        assert found["MAN-ST-3431"] == ["MAN"] and found["ISQ-FO-018"] == ["Navistar"] and found["HVG-QM-001"] == []
        assert other == 2 and scope.other_customer_blocks == 2 and scope.chars_hidden == 0  # MAN-ST-3431 and MAN-ST-239
    else:
        assert "MAN-ST-3431" not in found and "MAN-ST-239" not in found and found["ISQ-FO-018"] == ["Navistar"]
        assert other == 0 and scope.chars_hidden > 0
