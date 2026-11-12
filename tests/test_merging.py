"""tests/test_merging.py

Reference merging stage (extraction/merging.py) and the error breakdown that measures it.
"""

from helpers import ScriptedProvider

from csr_lab.evaluation.metrics import reference_errors
from csr_lab.ingestion.loader import load_bundle
from csr_lab.llm.mock_provider import MockProvider
from csr_lab.pipelines.registry import derive_strategy
from csr_lab.pipelines.run import run_analysis

DOC = """\
HALVARD MOTORS
Merge Test
Document ID: HVG-CSR-950
Revision: A

1 Requirements
1.1 The supplier shall follow the Packaging and Labeling Specification HVG-PKG-210 for all parts.
1.2 Damaged packaging shall be reported as described in the Packaging and Labeling Specification.
1.3 Coating adhesion shall be tested according to HVG-TM-311 Rev B.
1.4 Salt spray tests shall follow HVG-TM-311 Rev C.
1.5 Mitigation actions shall be documented according to HVG-QP-031 and reported per HVG-QP-032.
"""


def _docs(tmp_path):
    path = tmp_path / "merge.txt"
    path.write_text(DOC, encoding="utf-8")
    return load_bundle([path])


def _keys(result):
    docs = {d.document_id: d for d in result.referenced_documents}
    return {r.chapter: sorted(docs[i].metadata["key"] for i in r.referenced_document_ids) for r in result.requirements}


def _strategy(method, **extra):
    return derive_strategy(f"t_merge_{method}", "staged_verified", {"reference_merging": {"method": method, **extra}, "conflicts": {"enabled": False}})


def test_deterministic_leaves_name_unresolved(tmp_path, config):
    result = run_analysis(_docs(tmp_path), strategy=_strategy("deterministic"), config=config)
    assert _keys(result)["1.2"] == ["NAME:PACKAGING AND LABELING SPECIFICATION"]
    assert result.metadata.reference_merges == []


def test_hybrid_links_name_and_keeps_revisions_apart(tmp_path, config):
    result = run_analysis(_docs(tmp_path), strategy=_strategy("hybrid"), config=config)
    keys = _keys(result)
    assert keys["1.2"] == ["HVG-PKG-210"]
    assert keys["1.3"] == ["HVG-TM-311@REV-B"] and keys["1.4"] == ["HVG-TM-311@REV-C"]
    applied = [d for d in result.metadata.reference_merges if d.applied]
    assert applied and applied[0].method == "hybrid" and applied[0].canonical == "HVG-PKG-210"
    assert any(c.stage == "merge_references" and c.task == "reference_alias_check" for c in result.metadata.llm_calls)


def test_llm_merge_applies_model_grouping_including_its_mistakes(tmp_path, config):
    cfg = config.model_copy(deep=True)
    cfg.llm.mock.merge_revision_drop_rate = 1.0
    cfg.llm.mock.merge_confusion_rate = 1.0
    cfg.llm.mock.merge_alias_miss_rate = 0.0
    result = run_analysis(_docs(tmp_path), strategy=_strategy("llm"), config=cfg)
    keys = _keys(result)
    assert keys["1.2"] == ["HVG-PKG-210"]  # alias resolved
    assert keys["1.3"] == keys["1.4"] == ["HVG-TM-311"]  # revisions dropped
    assert len(set(keys["1.5"])) == 1  # HVG-QP-031 and HVG-QP-032 merged into one document
    assert all(d.method == "llm" and d.applied for d in result.metadata.reference_merges)


def test_hybrid_revision_guard(tmp_path, config):
    """A name with an explicit revision must not be linked to a different provided revision."""
    provided = tmp_path / "pkg.txt"
    provided.write_text("HALVARD MOTORS\nPackaging and Labeling Specification\nDocument ID: HVG-PKG-210\nRevision: D\n\n1 Scope\n1.1 The supplier shall pack parts.\n", encoding="utf-8")
    citing = tmp_path / "csr.txt"
    citing.write_text("HALVARD MOTORS\nRev Test\nDocument ID: HVG-CSR-951\nRevision: A\n\n1 Requirements\n1.1 Labels shall follow the Packaging and Labeling Specification Rev C.\n", encoding="utf-8")
    docs = load_bundle([provided, citing])
    mock = MockProvider(config.llm.mock)

    def fn(task, payload, request):
        if task == "reference_extraction":  # the name is extracted with its stated revision
            results = []
            for r in payload["requirements"]:
                refs = []
                if "Packaging and Labeling Specification Rev C" in r["text"]:
                    refs = [{"identifier": "Packaging and Labeling Specification", "revision": "C", "quoted_text": "Packaging and Labeling Specification Rev C"}]
                results.append({"requirement_id": r["requirement_id"], "references": refs})
            return {"results": results}
        if task == "reference_alias_check":  # a careless model: link every name to the provided revision D
            decisions = []
            for c in payload["candidates"]:
                opt = next(o for o in c["options"] if o["revision"] == "D")
                decisions.append({"candidate_id": c["candidate_id"], "same_document_as": opt["option_id"]})
            return {"decisions": decisions}
        return mock.generate(request).text

    def keys(protect):
        spec = derive_strategy(f"t_guard_{protect}", "staged_verified", {"reference_merging": {"method": "hybrid", "protect_revisions": protect}, "conflicts": {"enabled": False}})
        result = run_analysis(docs, strategy=spec, config=config, provider=ScriptedProvider(fn))
        req = next(r for r in result.requirements if r.source_document_id == "DOC-HVG-CSR-951-REV-A")
        by_id = {d.document_id: d for d in result.referenced_documents}
        return [by_id[i].metadata["key"] for i in req.referenced_document_ids], result.metadata.reference_merges

    guarded, merges = keys(True)
    assert guarded == ["NAME:PACKAGING AND LABELING SPECIFICATION"]  # left unresolved rather than linked to Rev D
    assert any(not d.applied and "protect_revisions" in d.reason for d in merges)
    unguarded, merges = keys(False)
    assert unguarded == ["HVG-PKG-210@REV-D"]  # the model's (wrong) revision wins without the guard


def test_reference_error_kinds():
    gold = {("R1", "HVG-TM-311@REV-B"), ("R2", "HVG-PKG-210"), ("R3", "QP-012"), ("R4", "HVG-QP-031"), ("R5", "HVG-LBL-220"), ("R6", "X-1")}
    pred = {("R1", "HVG-TM-311"), ("R2", "NAME:PACKAGING SPECIFICATION"), ("R3", "QP-12"), ("R4", "HVG-QP-032"), ("R6", "HVG-LBL-220"), ("R6", "GEN-SPEC-001"), ("R6", "X-1")}
    resolution = {("R2", "HVG-PKG-210"): "name_alias"}
    errors = reference_errors(pred, gold, resolution)
    assert errors["revision"] == 1
    assert errors["unresolved_alias"] == 1
    assert errors["identifier_variant"] == 1
    assert errors["confused_identifier"] == 1
    assert errors["wrong_requirement"] == 1  # HVG-LBL-220 belongs to R5
    assert errors["spurious"] == 1
    assert errors["missed"] == 1  # R5's HVG-LBL-220


def test_spacing_variants_are_merged_and_input_spelling_wins():
    from csr_lab.drafts import MentionDraft
    from csr_lab.extraction.merging import merge_spacing_variants
    from csr_lab.extraction.normalization import parse_identifier
    from csr_lab.schemas import Evidence, InputDocument

    def mention(i: int, raw: str, req: str = "R1") -> MentionDraft:
        p = parse_identifier(raw)
        return MentionDraft(mention_id=f"M{i}", requirement_id=req, raw_identifier=raw, canonical=p.canonical,
                            evidence=Evidence(source_document_id="DOC-A", quoted_text=raw))

    ms = [mention(1, "CVS 10"), mention(2, "CVS10", "R2"), mention(3, "IATF 16949"), mention(4, "IATF16949", "R3"), mention(5, "IATF 16949", "R4"),
          mention(6, "MAN 239-1"), mention(7, "MAN-2391"), mention(8, "QP-012"), mention(9, "QP-12")]
    cvs10 = InputDocument(document_id="DOC-CVS10", canonical_identifier="CVS10", identifier_source="filename", filename="CVS10.pdf",
                          format="pdf", sha256="0", char_count=0, parse_status="ok", text="")
    decisions = merge_spacing_variants(ms, [cvs10])
    canon = {m.mention_id: m.canonical for m in ms}
    assert canon["M1"] == canon["M2"] == "CVS10"  # the input document's spelling
    assert canon["M3"] == canon["M4"] == canon["M5"] == "IATF-16949"  # the more frequent spelling
    assert (canon["M6"], canon["M7"], canon["M8"], canon["M9"]) == ("MAN-239-1", "MAN-2391", "QP-012", "QP-12")
    assert {d.method for d in decisions} == {"spacing"} and len(decisions) == 2
    assert merge_spacing_variants(ms, [cvs10]) == []  # idempotent
