"""tests/test_conflicts.py

Conflict candidate selection, evidence validation and the graph flagging rules.
"""

from csr_lab.config import ConflictSettings, GraphSettings
from csr_lab.conflicts.candidate_selection import attribute_score, select_candidates
from csr_lab.conflicts.detector import graph_rule_conflicts, merge_conflicts, validate_conflict_evidence
from csr_lab.drafts import ConflictDraft, RequirementDraft
from csr_lab.extraction.grounding import SourceIndex
from csr_lab.pipelines.assembly import provisional_graph
from csr_lab.pipelines.run import run_analysis
from csr_lab.schemas import ConflictType, InputDocument
from helpers import case_documents


def _reqs(texts):
    return [RequirementDraft(requirement_id=f"REQ-X-{i:03d}", document_id="DOC-X", chapter=str(i), text=t) for i, t in enumerate(texts, 1)]


TEXTS = [
    "Parts shall be validated at a maximum test temperature of 85 °C.",
    "Thermal validation shall be performed at a minimum test temperature of 105 °C.",
    "Labels shall be printed in black ink.",
    "The supplier shall notify the buyer of delays.",
]


def test_all_pairs_when_small():
    cands, total, notes, decisions = select_candidates(_reqs(TEXTS), ConflictSettings())
    assert total == 6 and len(cands) == 6
    assert all(c.selectors == ["all_pairs"] for c in cands) and not decisions


def test_topic_selection_when_large():
    cands, total, notes, _ = select_candidates(_reqs(TEXTS), ConflictSettings(all_pairs_max=0, attribute_min_score=0.15))
    assert total == 6
    assert [c.requirement_ids for c in cands] == [("REQ-X-001", "REQ-X-002")]
    assert cands[0].selectors[0].startswith("attribute_overlap") and notes
    assert attribute_score(TEXTS[0], TEXTS[1]) > attribute_score(TEXTS[0], TEXTS[2])


def test_graph_candidates_are_added_with_provenance(dataset, config):
    _, docs, _ = case_documents(dataset, "dev_corvane_packaging")
    result = run_analysis(docs, strategy="graph_assisted", config=config.model_copy(deep=True))
    from csr_lab.relationships.builder import graph_from_result

    graph = graph_from_result(result)
    reqs = [
        RequirementDraft(requirement_id=r.requirement_id, document_id=r.source_document_id, chapter=r.chapter, text=r.requirement_text)
        for r in result.requirements
    ]
    base, _, _, _ = select_candidates(reqs, ConflictSettings(all_pairs_max=0, attribute_min_score=0.9))
    cands, _, _, decisions = select_candidates(reqs, ConflictSettings(all_pairs_max=0, attribute_min_score=0.9), graph=graph, graph_settings=GraphSettings(enabled=True))
    assert len(cands) > len(base)
    graph_only = [c for c in cands if c.graph_derived]
    assert graph_only and decisions
    assert any(s.startswith("shared_reference:") for c in graph_only for s in c.selectors)
    assert any(s == "relation_statement:applicability" for c in graph_only for s in c.selectors)
    # A disabled graph is ignored even if one is passed.
    off, _, _, _ = select_candidates(reqs, ConflictSettings(all_pairs_max=0, attribute_min_score=0.9), graph=graph, graph_settings=GraphSettings(enabled=False))
    assert len(off) == len(base)


def test_evidence_validation_repairs_unlocatable_quotes():
    text = "Parts shall be validated at a maximum test temperature of 85 °C."
    doc = InputDocument(document_id="DOC-X", canonical_identifier="X", identifier_source="header", filename="x", format="txt", sha256="0", char_count=len(text), parse_status="ok", text=text + "\n" + TEXTS[1])
    sources = SourceIndex([doc])
    a = RequirementDraft(requirement_id="REQ-X-001", document_id="DOC-X", chapter="1", text=text, start_offset=0, end_offset=len(text))
    b_start = len(text) + 1
    b = RequirementDraft(requirement_id="REQ-X-002", document_id="DOC-X", chapter="2", text=TEXTS[1], start_offset=b_start, end_offset=b_start + len(TEXTS[1]))
    conflict = ConflictDraft(requirement_ids=["REQ-X-001", "REQ-X-002"], conflict_type=ConflictType.NUMERIC_INCOMPATIBILITY, description="x", evidence=[], evidence_valid=False)
    repairs = validate_conflict_evidence([conflict], {a.requirement_id: a, b.requirement_id: b}, sources)
    assert repairs == 2 and conflict.evidence_valid
    assert all(e.repaired and e.verified for e in conflict.evidence)


def test_graph_rules_and_merge(dataset, config):
    _, docs, _ = case_documents(dataset, "heldout_tessaly_battery")
    result = run_analysis(docs, strategy="graph_assisted", config=config)
    applicability = [c for c in result.potential_conflicts if c.conflict_type == ConflictType.APPLICABILITY_CONFLICT]
    assert applicability and "graph_applicability_rule" in applicability[0].detection_methods
    assert all(c.review_status == "requires_human_review" for c in result.potential_conflicts)

    reqs = [RequirementDraft(requirement_id=r.requirement_id, document_id=r.source_document_id, chapter=r.chapter, text=r.requirement_text) for r in result.requirements]
    from csr_lab.relationships.builder import graph_from_result

    rules = graph_rule_conflicts(graph_from_result(result), GraphSettings(enabled=True), {r.requirement_id for r in reqs})
    kinds = {c.conflict_type for c in rules}
    assert kinds == {ConflictType.APPLICABILITY_CONFLICT, ConflictType.POTENTIAL_PRECEDENCE_CONFLICT}
    merged = merge_conflicts(rules + [c.model_copy(update={"detection_methods": ["llm_pairwise"]}) for c in rules])
    assert len(merged) == len(rules)
    assert all("llm_pairwise" in c.detection_methods and any(m.startswith("graph_") for m in c.detection_methods) for c in merged)


def test_non_graph_strategies_have_no_graph_decisions(dataset, config):
    _, docs, _ = case_documents(dataset, "heldout_tessaly_battery")
    for strategy in ("single_pass", "staged", "staged_verified"):
        result = run_analysis(docs, strategy=strategy, config=config)
        assert result.metadata.graph_decisions == []
        assert result.metadata.operational.graph_informed_calls == 0
        assert all("graph" not in m for c in result.potential_conflicts for m in c.detection_methods)


def test_provisional_graph_matches_final_export(dataset, config):
    _, docs, _ = case_documents(dataset, "dev_corvane_production")
    result = run_analysis(docs, strategy="staged", config=config)
    assert {e.relationship_type.value for e in result.relationships} >= {"contains", "references"}
    # graph building on drafts works without a full result
    assert provisional_graph(docs, [], []).nodes
