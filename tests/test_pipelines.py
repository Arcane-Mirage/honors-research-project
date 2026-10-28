"""tests/test_pipelines.py

Every strategy end to end: valid output, determinism, workflow shapes and failure handling.
"""

import json

import pytest

from csr_lab.pipelines.registry import build_workflow, derive_strategy, get_strategy, register_variants
from csr_lab.pipelines.run import run_analysis
from csr_lab.schemas import AnalysisResult
from helpers import ScriptedProvider, case_documents

STRATEGIES = ["single_pass", "staged", "staged_verified", "graph_assisted"]


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("case_id", ["dev_corvane_production", "dev_corvane_packaging", "heldout_tessaly_battery"])
def test_every_strategy_produces_valid_output(dataset, noisy_config, strategy, case_id):
    _, docs, gold = case_documents(dataset, case_id)
    result = run_analysis(docs, strategy=strategy, config=noisy_config, case_id=case_id, run_seed=7)
    AnalysisResult.model_validate_json(result.model_dump_json())
    assert result.status == "completed"
    assert result.metadata.strategy == strategy and result.metadata.provider_is_mock
    assert len(result.requirements) == len(gold.requirements)
    assert all(r.department is None and r.department_status == "deferred" for r in result.requirements)
    assert all(c.review_status == "requires_human_review" for c in result.potential_conflicts)
    graph_calls = result.metadata.operational.graph_informed_calls
    assert (graph_calls > 0) == (strategy == "graph_assisted")
    if strategy == "single_pass":
        assert result.metadata.operational.llm_calls == 1


def test_runs_are_deterministic(dataset, noisy_config):
    _, docs, _ = case_documents(dataset, "dev_corvane_packaging")

    def strip(r):
        d = r.model_dump(mode="json")
        d["metadata"].pop("created_at")
        d["metadata"]["operational"] = None
        for c in d["metadata"]["llm_calls"]:
            c["latency_seconds"] = 0
        return json.dumps(d, sort_keys=True)

    a = run_analysis(docs, strategy="graph_assisted", config=noisy_config, run_seed=3)
    b = run_analysis(docs, strategy="graph_assisted", config=noisy_config, run_seed=3)
    c = run_analysis(docs, strategy="graph_assisted", config=noisy_config, run_seed=4)
    assert strip(a) == strip(b)
    assert a.analysis_id != c.analysis_id


def test_workflow_shapes_follow_options():
    nodes = {name: set(build_workflow(get_strategy(name)).get_graph().nodes) for name in STRATEGIES}
    assert "single_pass_extract" in nodes["single_pass"]
    assert "verify_references" not in nodes["staged"]
    assert {"verify_references", "targeted_reextraction"} <= nodes["staged_verified"]
    assert "build_graph" not in nodes["staged_verified"]
    assert {"build_graph", "graph_reference_audit"} <= nodes["graph_assisted"]


def test_variants_do_not_change_their_base():
    base_before = get_strategy("graph_assisted").options.model_dump()
    register_variants({"test_variant_no_rules": {"base": "graph_assisted", "options": {"graph": {"applicability_rules": False}}}})
    variant = get_strategy("test_variant_no_rules")
    assert variant.base == "graph_assisted" and not variant.options.graph.applicability_rules
    assert get_strategy("graph_assisted").options.model_dump() == base_before
    assert derive_strategy("x", "staged", {"conflicts": {"enabled": False}}).options.conflicts.enabled is False
    with pytest.raises(ValueError):
        register_variants({"orphan": {"base": "does_not_exist_either"}, "does_not_exist_either": {"base": "orphan"}})


def test_conflicts_disabled_reports_not_examined(dataset, config):
    _, docs, _ = case_documents(dataset, "dev_corvane_production")
    spec = derive_strategy("no_conflicts", "staged", {"conflicts": {"enabled": False}})
    result = run_analysis(docs, strategy=spec, config=config)
    assert result.potential_conflicts == []
    assert result.coverage.conflict_analysis.status == "disabled"
    assert result.coverage.conflict_analysis.outcome == "not_examined"


def test_coverage_marks_unavailable_documents(dataset, config):
    _, docs, _ = case_documents(dataset, "heldout_tessaly_battery")
    result = run_analysis(docs, strategy="staged_verified", config=config)
    status = {d.document_id: d.conflict_analysis_status for d in result.coverage.referenced_documents}
    assert status["DOC-TSA-MAT-420"] == "examined"
    assert status["DOC-TSA-MAT-425"] == "not_examined_source_unavailable"
    assert any(lim.kind == "unavailable_referenced_document" and lim.document_id == "DOC-TSA-MAT-425" for lim in result.coverage.limitations)
    assert result.coverage.conflict_analysis.outcome == "potential_conflicts_found"


def test_malformed_output_is_retried_then_recorded(simple_docs, config):
    calls = {"n": 0}

    def respond(task, payload, request):
        calls["n"] += 1
        if calls["n"] == 1:
            return "Sure! Here is the JSON: {not valid"
        return {"requirements": [], "potential_conflicts": []}

    result = run_analysis(simple_docs, strategy="single_pass", config=config, provider=ScriptedProvider(respond))
    call = result.metadata.llm_calls[0]
    assert call.success and call.attempts == 2
    assert result.metadata.operational.json_validation_failures == 1
    assert result.requirements == []

    always_bad = ScriptedProvider(lambda *a: "no json at all")
    failed = run_analysis(simple_docs, strategy="staged", config=config, provider=always_bad)
    assert failed.status == "completed_with_errors"
    assert not failed.metadata.llm_calls[0].success
    assert failed.metadata.operational.json_validation_failures == 1 + config.llm.max_validation_retries
    assert any(lim.kind == "llm_call_failed" for lim in failed.coverage.limitations)


def test_missing_requirement_in_reference_response_is_failed_not_empty(simple_docs, config):
    def respond(task, payload, request):
        if task == "requirement_identification":
            return {"requirements": [{"segment_id": s["segment_id"], "requirement_text": s["text"]} for s in payload["segments"] if "shall" in s["text"]]}
        if task == "reference_extraction":
            first = payload["requirements"][0]["requirement_id"]
            return {"results": [{"requirement_id": first, "references": []}]}
        return {"assessments": []}

    result = run_analysis(simple_docs, strategy="staged", config=config, provider=ScriptedProvider(respond))
    statuses = [r.extraction_metadata.reference_analysis_status for r in result.requirements]
    assert statuses[0] == "unverified" and set(statuses[1:]) == {"failed"}


def test_unusable_input_gives_failed_result(tmp_path, config):
    from csr_lab.ingestion.loader import load_bundle

    path = tmp_path / "empty.txt"
    path.write_text("   \n", encoding="utf-8")
    result = run_analysis(load_bundle([path]), strategy="graph_assisted", config=config)
    assert result.status == "failed"
    assert result.coverage.documents[0].status == "unsupported"
    assert any(lim.kind == "unsupported_input" for lim in result.coverage.limitations)
    assert result.metadata.operational.llm_calls == 0
