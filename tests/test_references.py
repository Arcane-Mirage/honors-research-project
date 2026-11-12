"""tests/test_references.py

Reference identifiers: normalisation, distractors, verification, evidence repair and re-extraction.
"""

import pytest
from helpers import ScriptedProvider

from csr_lab.config import VerificationSettings
from csr_lab.drafts import RequirementDraft
from csr_lab.extraction.grounding import SourceIndex, TextIndex
from csr_lab.extraction.normalization import near_duplicates, non_document_reason, parse_identifier, reference_key
from csr_lab.extraction.references import MentionCounter, mentions_from_model
from csr_lab.extraction.verification import scan_candidates, verify_pass
from csr_lab.pipelines.run import run_analysis
from csr_lab.schemas import InputDocument, MentionStatus


@pytest.mark.parametrize(
    "raw, canonical, revision",
    [
        ("Specification HVG-QS-110", "HVG-QS-110", None),
        ("the HVG-MP-210 specification", "HVG-MP-210", None),
        ("Specification HVG MP 215", "HVG-MP-215", None),
        ("Engineering Standard ES 4.12", "ES-4.12", None),
        ("HVG-ES-210 Rev. B", "HVG-ES-210", "B"),
        ("HVG-ES-210 Revision D", "HVG-ES-210", "D"),
        ("HVG-SW-500 rev.3", "HVG-SW-500", "3"),
        ("HVG–TM–311 Rev B", "HVG-TM-311", "B"),
        ("Procedure QP-12", "QP-12", None),
    ],
)
def test_parse_identifier(raw, canonical, revision):
    parsed = parse_identifier(raw)
    assert (parsed.canonical, parsed.revision) == (canonical, revision)


def test_names_and_revisions_are_distinct_keys():
    assert parse_identifier("HVG-ES-210 Rev B").key != parse_identifier("HVG-ES-210 Rev D").key
    assert parse_identifier("HVG-ES-210 Rev B").key != parse_identifier("HVG-ES-210").key
    name_only = parse_identifier("the Halvard Supplier Quality Manual")
    assert name_only.canonical is None and name_only.key == "NAME:HALVARD SUPPLIER QUALITY MANUAL"
    named = parse_identifier("Halvard Supplier Quality Manual HVG-SQM-001")
    assert named.canonical == "HVG-SQM-001" and named.name == "Halvard Supplier Quality Manual"


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
    text = "The supplier shall follow HVG-QS-110 and see section 4.2; drawing DRW-045 also applies."
    sources, req = _setup(text)
    refs = [
        {"identifier": "HVG-QS-110", "quoted_text": "follow HVG-QS-110"},
        {"identifier": "HVG-QS-110", "quoted_text": "follow HVG-QS-110"},  # duplicate
        {"identifier": "GEN-SPEC-001", "quoted_text": "according to GEN-SPEC-001"},  # hallucinated
        {"identifier": "4.2", "quoted_text": "section 4.2"},  # section number
        {"identifier": "HVG-QS-110 ", "quoted_text": "paraphrased quote"},  # duplicate with bad quote
    ]
    mentions = mentions_from_model(refs, req, sources=sources, counter=MentionCounter())
    vp = verify_pass([req], mentions, VerificationSettings(enabled=True), sources=sources, attempt=0, examined=set())
    actions = [e.action for e in vp.events]
    assert actions.count("rejected") == 2
    assert "merged_duplicate" in actions
    active = [m for m in mentions if m.active]
    assert [m.canonical for m in active] == ["HVG-QS-110"]
    assert [c.text for c in vp.candidates] == ["DRW-045"]
    assert vp.candidates[0].kind == "possible_missed"


def test_evidence_repair():
    text = "Parts shall conform to HVG-TM-320 in all cases."
    sources, req = _setup(text)
    mentions = mentions_from_model([{"identifier": "HVG-TM-320", "quoted_text": "not in the text"}], req, sources=sources, counter=MentionCounter())
    assert mentions[0].evidence.verified is False
    vp = verify_pass([req], mentions, VerificationSettings(enabled=True), sources=sources, attempt=0, examined=set())
    assert any(e.action == "evidence_repaired" for e in vp.events)
    assert mentions[0].evidence.verified and mentions[0].evidence.repaired


def test_scan_candidates_finds_ids_and_names():
    found = [t for _, _, t in scan_candidates("Use HVG MP 215 and the Halvard Supplier Quality Manual, not D65 light.")]
    assert "HVG MP 215" in found and "Halvard Supplier Quality Manual" in found and "D65" in found
    attached = [t for _, _, t in scan_candidates("Follow the Halvard Supplier Quality Manual HVG-SQM-001.")]
    assert attached == ["HVG-SQM-001"]


def _scripted_missing_drawing(always_reject: bool = False):
    def respond(task, payload, request):
        if task == "reference_extraction":
            results = []
            for r in payload["requirements"]:
                refs = [{"identifier": "HVG-QS-110", "quoted_text": "Specification HVG-QS-110"}] if "HVG-QS-110" in r["text"] else []
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
    assert set(req.referenced_document_ids) == {"DOC-HVG-QS-110", "DOC-DRW-045"}
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


def test_artifact_normalized_matching_maps_back_to_source_offsets():
    source = "the s upplier must report require-\nments to the cus-\ntomer and MPAPS B -50 applies"
    plain = TextIndex(source)
    assert plain.find("the supplier must report requirements") is None
    idx = TextIndex(source, artifact_match=True)
    start, end, kind = idx.find("the supplier must report requirements")
    assert kind == "artifact_normalized"
    assert source[start:end] == "the s upplier must report require-\nments"
    start, end, kind = idx.find("MPAPS B-50 applies")
    assert (source[start:end], kind) == ("MPAPS B -50 applies", "artifact_normalized")
    assert idx.find("the cus") is not None  # an exact match still wins
    assert idx.find("supplier") is None or idx.find("supplier")[2] == "artifact_normalized"
    assert idx.find("cust") is None  # too short to match without spaces


def test_source_index_artifact_switch():
    doc = InputDocument(document_id="DOC-X", canonical_identifier="X", identifier_source="filename", filename="x.txt",
                        format="txt", sha256="0", char_count=20, parse_status="ok", text="all s upplier records")
    assert SourceIndex([doc]).locate("DOC-X", "all supplier records") is None
    assert SourceIndex([doc], artifact_match=True).locate("DOC-X", "all supplier records")[2] == "artifact_normalized"


# ---------------------------------------------------------------------------
# Identifier normalisation on reference strings taken from real public CSRs (CVS10, CVS168,
# Formel Q Konkret / Capability / Appendix / Brand-specific supplements, ISQ-011).
# 
# Each row: raw mention -> (canonical identifier, revision, location, matching key). A name-only
# mention has no canonical identifier; its key is 'NAME:...'. The revision of a name-only
# mention is parsed but not part of its key (names are never revision-qualified).
# ---------------------------------------------------------------------------

CASES = [
    # Documents known only by name
    ("Formel Q Konkret", None, None, None, "NAME:FORMEL Q KONKRET"),
    ("Formel Q konkret", None, None, None, "NAME:FORMEL Q KONKRET"),
    ("VW Formel Q Konkret", None, None, None, "NAME:VW FORMEL Q KONKRET"),
    ("Formel Q Capability version 9", None, "9", None, "NAME:FORMEL Q CAPABILITY"),
    ("Formel Q Capability, 9th Edition", None, "EDITION 9", None, "NAME:FORMEL Q CAPABILITY"),
    ("Formel Q Neuteile Integral", None, None, None, "NAME:FORMEL Q NEUTEILE INTEGRAL"),
    # Chapter-qualified references: the location is split off
    ("CVS10, Chapter 4.6", "CVS10", None, "4.6", "CVS10"),
    ("CVS10, Chapter 4.6 “Change Management”", "CVS10", None, "4.6", "CVS10"),
    ("Formel Q Konkret, chapter 3.4", None, None, "3.4", "NAME:FORMEL Q KONKRET"),
    ("Chapter 3.3 of Formel Q konkret", None, None, "3.3", "NAME:FORMEL Q KONKRET"),
    ("VDA 6.3 question 5.1", "VDA-6.3", None, "5.1", "VDA-6.3"),
    ("PPAP manual 4th edition, section 2.2.15", None, "EDITION 4", "2.2.15", "NAME:PPAP MANUAL"),
    ("Section 4.12", "4.12", None, None, "4.12"),  # nothing left to be the document: kept (a section number)
    # Identifier followed by its title
    ("CVS10 – Commercial Vehicle Standard 10", "CVS10", None, None, "CVS10"),
    ("CVS10 Commercial Vehicle Standard 10", "CVS10", None, None, "CVS10"),
    ("MPAPS B-50 – Restricted Chemical Substances", "MPAPS-B-50", None, None, "MPAPS-B-50"),
    # PDF artifact: a hyphen with a space on one side
    ("MPAPS B-50", "MPAPS-B-50", None, None, "MPAPS-B-50"),
    ("MPAPS B -50", "MPAPS-B-50", None, None, "MPAPS-B-50"),
    # Years and editions are revisions
    ("VDA 6.3 (2023)", "VDA-6.3", "2023", None, "VDA-6.3@REV-2023"),
    ("IATF 16949:2016", "IATF-16949", "2016", None, "IATF-16949@REV-2016"),
    ("ISO 9001:2015", "ISO-9001", "2015", None, "ISO-9001@REV-2015"),
    ("AIAG PPAP manual 4th edition", None, "EDITION 4", None, "NAME:AIAG PPAP MANUAL"),
    ("PPAP 4th Edition", None, "EDITION 4", None, "NAME:PPAP"),
    ("APQP Manual 2nd edition", None, "EDITION 2", None, "NAME:APQP MANUAL"),
    # Volume words
    ("VDA 6.3", "VDA-6.3", None, None, "VDA-6.3"),
    ("VDA Volume 6.3 Process Audit", "VDA-6.3", None, None, "VDA-6.3"),
    ("VDA Volume 2", "VDA-2", None, None, "VDA-2"),
    ("VDA Vol. 2", "VDA-2", None, None, "VDA-2"),
    # Codes, spaced and unspaced (spacing variants are merged later, see test_spacing_variants)
    ("CVS10", "CVS10", None, None, "CVS10"),
    ("CVS 10", "CVS-10", None, None, "CVS-10"),
    ("IATF 16949", "IATF-16949", None, None, "IATF-16949"),
    ("IATF16949", "IATF16949", None, None, "IATF16949"),
    ("DIN EN ISO 9001", "DIN-EN-ISO-9001", None, None, "DIN-EN-ISO-9001"),
    ("IW-001-SWR", "IW-001-SWR", None, None, "IW-001-SWR"),
    ("Document IW-001-SWR", "IW-001-SWR", None, None, "IW-001-SWR"),
    ("ISQ-018-FO", "ISQ-018-FO", None, None, "ISQ-018-FO"),
    ("MAN 239-1", "MAN-239-1", None, None, "MAN-239-1"),
    ("M 3335", "M-3335", None, None, "M-3335"),
    ("M3335", "M3335", None, None, "M3335"),
    ("STD3868", "STD3868", None, None, "STD3868"),
    ("PRC200-A", "PRC200-A", None, None, "PRC200-A"),
    ("VW 99000", "VW-99000", None, None, "VW-99000"),
    ("VW01155", "VW01155", None, None, "VW01155"),
    ("CVS92 (under development)", "CVS92", None, None, "CVS92"),
]


@pytest.mark.parametrize("raw, canonical, revision, location, key", CASES, ids=[c[0] for c in CASES])
def test_real_reference_strings(raw, canonical, revision, location, key):
    p = parse_identifier(raw)
    assert (p.canonical, p.revision, p.location) == (canonical, revision, location)
    assert p.key == key


def test_titles_are_kept_as_names():
    assert parse_identifier("CVS10 – Commercial Vehicle Standard 10").name == "Commercial Vehicle Standard 10"
    assert parse_identifier("MPAPS B-50 – Restricted Chemical Substances").name == "Restricted Chemical Substances"
    assert parse_identifier("Formel Q – konkret").canonical is None  # a dash inside a name is not a title split


@pytest.mark.parametrize(
    "a, b, same",
    [
        ("CVS 10", "CVS10", True),
        ("IATF 16949", "IATF16949", True),
        ("M 3335", "M3335", True),
        ("MAN 239-1", "MAN-2391", False),  # a real hyphen is not a space
        ("PRC200-A", "PRC-200A", False),
        ("QP-012", "QP-12", False),  # leading zeros are never merged
    ],
)
def test_spacing_variants(a, b, same):
    assert (parse_identifier(a).compact == parse_identifier(b).compact) is same


def test_revision_forms_are_normalised():
    assert reference_key("PPAP", "4th edition") == reference_key("PPAP", "Edition 4") == "PPAP@REV-EDITION 4"
