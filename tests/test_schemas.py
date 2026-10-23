import pytest
from pydantic import ValidationError

from csr_lab.pipelines.run import run_analysis
from csr_lab.schemas import (
    AnalysisResult,
    Evidence,
    PotentialConflict,
    Requirement,
    RequirementExtractionMetadata,
    SourceLocation,
)


def _requirement(**kw):
    base = dict(
        requirement_id="REQ-X-001",
        source_document_id="DOC-X",
        chapter="1.1",
        requirement_text="The supplier shall comply.",
        source_location=SourceLocation(source_document_id="DOC-X"),
        extraction_metadata=RequirementExtractionMetadata(strategy="t", reference_analysis_status="not_run"),
    )
    base.update(kw)
    return Requirement(**base)


def test_department_is_null_while_deferred():
    assert _requirement().department is None
    with pytest.raises(ValidationError):
        _requirement(department="Quality")


def test_referenced_ids_must_match_mentions():
    with pytest.raises(ValidationError):
        _requirement(referenced_document_ids=["DOC-A"])


def test_conflict_cannot_be_marked_resolved():
    with pytest.raises(ValidationError):
        PotentialConflict(
            conflict_id="CON-1",
            related_requirement_ids=["A", "B"],
            conflict_type="numeric_incompatibility",
            description="x",
            reason_for_review="x",
            review_status="resolved",
        )


def test_verified_evidence_needs_offsets():
    with pytest.raises(ValidationError):
        Evidence(source_document_id="DOC-X", quoted_text="abc", verified=True)
    with pytest.raises(ValidationError):
        SourceLocation(source_document_id="DOC-X", start_offset=5, end_offset=2)


def test_result_round_trip_and_integrity(simple_docs, config):
    result = run_analysis(simple_docs, strategy="graph_assisted", config=config)
    again = AnalysisResult.model_validate_json(result.model_dump_json())
    assert again == result
    assert {"schema_version", "analysis_id", "input_documents", "requirements", "referenced_documents", "relationships", "potential_conflicts", "coverage", "metadata"} <= set(result.model_dump())

    data = result.model_dump(mode="json")
    data["requirements"][0]["referenced_document_ids"] = ["DOC-MISSING"]
    data["requirements"][0]["reference_mentions"][0]["document_id"] = "DOC-MISSING"
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(data)

    data = result.model_dump(mode="json")
    data["relationships"][0]["target_node_id"] = "DOC-NOPE"
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate(data)


def test_json_schema_exports():
    schema = AnalysisResult.model_json_schema()
    assert "requirements" in schema["properties"]
    assert "PotentialConflict" in schema["$defs"]
