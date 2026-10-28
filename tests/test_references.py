"""tests/test_references.py

Reference identifiers: normalisation, distractors, verification, evidence repair and re-extraction.
"""

import pytest

from csr_lab.config import VerificationSettings
from csr_lab.drafts import RequirementDraft
from csr_lab.extraction.grounding import SourceIndex, TextIndex
from csr_lab.extraction.normalization import near_duplicates, non_document_reason, parse_identifier
from csr_lab.extraction.references import MentionCounter, mentions_from_model
from csr_lab.extraction.verification import scan_candidates, verify_pass
from csr_lab.pipelines.run import run_analysis
from csr_lab.schemas import InputDocument, MentionStatus
from helpers import ScriptedProvider


@pytest.mark.parametrize(
    "raw, canonical, revision",
    [
        ("Specification CVM-QS-110", "CVM-QS-110", None),
        ("the TSA-MP-210 specification", "TSA-MP-210", None),
        ("Specification TSA MP 215", "TSA-MP-215", None),
        ("Engineering Standard ES 4.12", "ES-4.12", None),
        ("CVM-ES-210 Rev. B", "CVM-ES-210", "B"),
        ("CVM-ES-210 Revision D", "CVM-ES-210", "D"),
        ("TSA-SW-500 rev.3", "TSA-SW-500", "3"),
        ("CVM–TM–311 Rev B", "CVM-TM-311", "B"),
        ("Procedure QP-12", "QP-12", None),
    ],
)
def test_parse_identifier(raw, canonical, revision):
    parsed = parse_identifier(raw)
    assert (parsed.canonical, parsed.revision) == (canonical, revision)


def test_names_and_revisions_are_distinct_keys():
    assert parse_identifier("CVM-ES-210 Rev B").key != parse_identifier("CVM-ES-210 Rev D").key
    assert parse_identifier("CVM-ES-210 Rev B").key != parse_identifier("CVM-ES-210").key
    name_only = parse_identifier("the Corvane Supplier Quality Manual")
    assert name_only.canonical is None and name_only.key == "NAME:CORVANE SUPPLIER QUALITY MANUAL"
    named = parse_identifier("Corvane Supplier Quality Manual CVM-SQM-001")
    assert named.canonical == "CVM-SQM-001" and named.name == "Corvane Supplier Quality Manual"


def test_conservative_normalisation_flags_but_does_not_merge():
    assert parse_identifier("QP-012").key != parse_identifier("QP-12").key
    assert near_duplicates(["QP-012", "QP-12", "QP-13"]) == [("QP-012", "QP-12")]


@pytest.mark.parametrize("token, preceding", [("4.12", ""), ("2026-02-15", ""), ("55-1020", ""), ("A-12", "part number ")])
def test_distractors(token, preceding):
    assert non_document_reason(parse_identifier(token).canonical, preceding)


def test_text_index_whitespace_matching():
    idx = TextIndex("The supplier shall\nfollow   QP-12.")
    start, end, kind = idx.find("supplier shall follow QP-12")
    assert kind == "whitespace_normalized"
    assert idx.text[start:end] == "supplier shall\nfollow   QP-12"
    assert idx.find("supplier must follow") is None


def _setup(text: str):
    doc = InputDocument(
        document_id="DOC-T", canonical_identifier="T", identifier_source="header", filename="t.txt",
        format="txt", sha256="0", char_count=len(text), parse_status="ok", text=text,
    )
    sources = SourceIndex([doc])
    req = RequirementDraft(requirement_id="REQ-T-001", document_id="DOC-T", chapter="1", text=text, start_offset=0, end_offset=len(text), match="exact")
    return sources, req


def test_verify_pass_rejects_merges_repairs_and_audits():
    text = "The supplier shall follow CVM-QS-110 and see section 4.2; drawing DRW-045 also applies."
    sources, req = _setup(text)
    refs = [
        {"identifier": "CVM-QS-110", "quoted_text": "follow CVM-QS-110"},
        {"identifier": "CVM-QS-110", "quoted_text": "follow CVM-QS-110"},  # duplicate
        {"identifier": "GEN-SPEC-001", "quoted_text": "according to GEN-SPEC-001"},  # hallucinated
        {"identifier": "4.2", "quoted_text": "section 4.2"},  # section number
        {"identifier": "CVM-QS-110 ", "quoted_text": "paraphrased quote"},  # duplicate with bad quote
    ]
    mentions = mentions_from_model(refs, req, sources=sources, counter=MentionCounter())
    vp = verify_pass([req], mentions, VerificationSettings(enabled=True), sources=sources, attempt=0, examined=set())
    actions = [e.action for e in vp.events]
    assert actions.count("rejected") == 2
    assert "merged_duplicate" in actions
    active = [m for m in mentions if m.active]
    assert [m.canonical for m in active] == ["CVM-QS-110"]
    assert [c.text for c in vp.candidates] == ["DRW-045"]
    assert vp.candidates[0].kind == "possible_missed"


def test_evidence_repair():
    text = "Parts shall conform to CVM-TM-320 in all cases."
    sources, req = _setup(text)
    mentions = mentions_from_model([{"identifier": "CVM-TM-320", "quoted_text": "not in the text"}], req, sources=sources, counter=MentionCounter())
    assert mentions[0].evidence.verified is False
    vp = verify_pass([req], mentions, VerificationSettings(enabled=True), sources=sources, attempt=0, examined=set())
    assert any(e.action == "evidence_repaired" for e in vp.events)
    assert mentions[0].evidence.verified and mentions[0].evidence.repaired


def test_scan_candidates_finds_ids_and_names():
    found = [t for _, _, t in scan_candidates("Use TSA MP 215 and the Corvane Supplier Quality Manual, not D65 light.")]
    assert "TSA MP 215" in found and "Corvane Supplier Quality Manual" in found and "D65" in found
    attached = [t for _, _, t in scan_candidates("Follow the Corvane Supplier Quality Manual CVM-SQM-001.")]
    assert attached == ["CVM-SQM-001"]


def _scripted_missing_drawing(always_reject: bool = False):
    def respond(task, payload, request):
        if task == "requirement_identification":
            return {"requirements": [{"segment_id": s["segment_id"], "chapter": s["chapter"], "requirement_text": s["text"]} for s in payload["segments"] if "shall" in s["text"] and not s["context_only"]]}
        if task == "reference_extraction":
            results = []
            for r in payload["requirements"]:
                refs = [{"identifier": "CVM-QS-110", "quoted_text": "Specification CVM-QS-110"}] if "CVM-QS-110" in r["text"] else []
                results.append({"requirement_id": r["requirement_id"], "references": refs})
            return {"results": results}
        if task == "reference_verification":
            return {"decisions": [{"candidate_id": c["candidate_id"], "is_document_reference": not always_reject, "identifier": c["text"], "quoted_text": c["text"]} for c in payload["candidates"]]}
        if task == "conflict_analysis":
            return {"assessments": [{"pair_id": p["pair_id"], "potential_conflict": False} for p in payload["pairs"]]}
        raise AssertionError(task)

    return respond


def test_targeted_reextraction_adds_missed_reference_and_keeps_initial(simple_docs, config):
    provider = ScriptedProvider(_scripted_missing_drawing())
    result = run_analysis(simple_docs, strategy="staged_verified", config=config, provider=provider)
    req = next(r for r in result.requirements if r.chapter == "2.1")
    assert set(req.referenced_document_ids) == {"DOC-CVM-QS-110", "DOC-DRW-045"}
    added = next(m for m in req.reference_mentions if m.document_id == "DOC-DRW-045")
    assert added.status == MentionStatus.ADDED_DURING_VERIFICATION and added.verified
    report = result.metadata.reference_verification
    assert report.enabled and report.counts["added_during_verification"] >= 1
    assert all(m.status == MentionStatus.FOUND_INITIALLY for m in report.initial_mentions)
    assert not any(m.document_id == "DOC-DRW-045" for m in report.initial_mentions)


def test_verification_loop_is_bounded(simple_docs, config):
    provider = ScriptedProvider(_scripted_missing_drawing(always_reject=True))
    result = run_analysis(simple_docs, strategy="staged_verified", config=config, provider=provider)
    report = result.metadata.reference_verification
    assert report.attempts_used <= 2
    assert any(e.action == "candidate_rejected" for e in report.events)
    verification_calls = [c for c in result.metadata.llm_calls if c.task == "reference_verification"]
    assert 1 <= len(verification_calls) <= 2 * len(result.requirements)
