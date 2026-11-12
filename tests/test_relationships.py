"""tests/test_relationships.py

Relationship graph structure, queries, flagging rules and validation.
"""

import json

import pytest

from csr_lab.conflicts.detector import _reaches, classify_conflicts
from csr_lab.pipelines.run import analyze_files, run_analysis
from csr_lab.relationships import queries
from csr_lab.relationships.builder import RelationshipGraph, graph_from_result, is_pointer
from csr_lab.relationships.validation import validate_graph
from csr_lab.schemas import ConflictType, PotentialConflict, Relationship, RelationshipType


@pytest.fixture
def packaging_graph(rules_docs, config):
    docs = rules_docs
    result = run_analysis(docs, strategy="graph_assisted", config=config)
    return result, graph_from_result(result)


CSR200 = "DOC-HVG-CSR-200-REV-A"  # input documents are revisions
PKG210_D = "DOC-HVG-PKG-210-REV-D"


def _req(result, doc, chapter):
    return next(r.requirement_id for r in result.requirements if r.source_document_id == doc and r.chapter == chapter)


def test_graph_structure(packaging_graph):
    result, graph = packaging_graph
    types = {e.relationship_type for e in graph.edges}
    assert {RelationshipType.CONTAINS, RelationshipType.REFERENCES, RelationshipType.EXPLICITLY_EXCLUDES,
            RelationshipType.EXPLICITLY_TAKES_PRECEDENCE_OVER, RelationshipType.POTENTIALLY_CONFLICTS_WITH} <= types
    report = validate_graph(graph)
    assert report.valid, report
    # The provided packaging specification is a revision node under its document node.
    # Unrevisioned references point at the document node, which resolves to Rev D.
    pkg = graph.node("DOC-HVG-PKG-210")
    assert pkg.attributes["content_available"] and pkg.attributes["is_input_document"]
    assert graph.revisions_of("DOC-HVG-PKG-210") == [PKG210_D]
    assert graph.current_revision("DOC-HVG-PKG-210") == PKG210_D
    assert graph.same_document("DOC-HVG-PKG-210", PKG210_D) and graph.family_of(PKG210_D) == "DOC-HVG-PKG-210"
    assert all(graph.container_of(r.requirement_id) in (CSR200, PKG210_D) for r in result.requirements)
    doc_edges = [e for e in graph.edges if e.metadata.get("origin") == "aggregation"]
    assert any(e.source_node_id == CSR200 and e.target_node_id == "DOC-HVG-PKG-210" for e in doc_edges)
    assert all(e.evidence for e in graph.edges if e.metadata.get("origin") == "extraction")


def test_queries(packaging_graph):
    result, graph = packaging_graph
    r12 = _req(result, CSR200, "1.2")
    assert queries.documents_referenced_by(graph, r12) == ["DOC-HVG-PKG-210"]
    referencing = queries.requirements_referencing(graph, "DOC-HVG-PKG-210")
    assert r12 in referencing and len(referencing) >= 2
    shared = queries.shared_documents(graph)
    assert "DOC-HVG-LBL-220" in shared and len(shared["DOC-HVG-LBL-220"]) == 2
    pairs = queries.requirements_sharing_references(graph, r12)
    assert pairs and all(r12 in (a, b) for a, b, _ in pairs)
    assert queries.unresolved_documents(graph) == []
    unavailable = {n.node_id for n in queries.unavailable_documents(graph)}
    assert "DOC-HVG-PKG-050" in unavailable and "DOC-HVG-PKG-210" not in unavailable
    assert queries.potentially_conflicting_requirements(graph)
    edge = next(e for e in graph.edges if e.relationship_type == RelationshipType.EXPLICITLY_EXCLUDES)
    assert "does not apply" in queries.evidence_for(graph, edge.relationship_id)[0].quoted_text
    exported = json.loads(json.dumps(queries.export_graph(graph)))
    assert RelationshipGraph.model_validate(exported).edges[0].relationship_id == graph.edges[0].relationship_id
    assert queries.to_mermaid(graph).startswith("flowchart LR")


def test_applicability_and_precedence_rules(packaging_graph):
    result, graph = packaging_graph
    flags = queries.find_applicability_conflicts(graph)
    assert [f.requirement_ids for f in flags] == [tuple(sorted((_req(result, CSR200, "1.3"), _req(result, PKG210_D, "2.1"))))]
    assert flags[0].evidence and flags[0].relationship_ids
    prec = queries.find_precedence_conflicts(graph)
    assert [set(f.requirement_ids) for f in prec] == [{_req(result, CSR200, "5.1"), _req(result, PKG210_D, "4.1")}]


def test_shared_reference_alone_is_not_flagged(packaging_graph):
    _, graph = packaging_graph
    flagged = {f.requirement_ids for f in queries.find_applicability_conflicts(graph) + queries.find_precedence_conflicts(graph)}
    for a, b, _ in queries.requirements_sharing_references(graph):
        if (a, b) in flagged:
            # only allowed when an explicit exclusion/precedence statement is involved
            assert any(e.relationship_type != RelationshipType.REFERENCES for e in graph.out_edges(a) + graph.out_edges(b))


def test_applicability_rule_needs_related_document(packaging_graph):
    _, graph = packaging_graph
    assert queries.find_applicability_conflicts(graph, hops=0) == []  # the reference is in a different document


def test_validation_detects_problems(packaging_graph):
    _, graph = packaging_graph
    edge = graph.edges[1]
    broken = RelationshipGraph(
        nodes=graph.nodes,
        edges=[
            *graph.edges,
            edge,  # duplicate
            Relationship(relationship_id="REL-bad", source_node_id="REQ-NOPE", target_node_id=edge.target_node_id, relationship_type=RelationshipType.REFERENCES),
        ],
    )
    report = validate_graph(broken)
    assert not report.valid
    assert report.duplicate_edges and "REL-bad" in report.invalid_endpoints and "REL-bad" in report.inconsistent_edge_ids


# ---------------------------------------------------------------------------
# Delta documents (inherits_from / amends), chapter references and precedence classification.
# ---------------------------------------------------------------------------

BASE = """\
Halvard Q Agreement
Contents
1 General .......... 1
1.1 Offer prerequisites .......... 1
1.2 Records .......... 2
2 Approval .......... 2
1 General
1.1 Offer prerequisites
The supplier shall submit a complete offer.
1.2 Records
The supplier shall retain quality records for a minimum of 15 years after the end of series production.
2 Approval
Approval shall be requested in writing.
"""

DELTA = """\
HALVARD VEHICLE GROUP
Document ID: HVG-CVS-20
Revision: 2025-04
Contents
Scope .......... 1
1 General .......... 1
1.1 Offer prerequisites .......... 1
1.2 Records .......... 2
2 Approval .......... 2
Scope
This standard follows the document structure of the Halvard Q Agreement.
1 General
1.1 Offer prerequisites
See Halvard Q Agreement. No additional requirements.
1.2 Records
Quality records shall be destroyed no later than 10 years after the end of series production.
2 Approval
Approval requests shall follow the Halvard Q Agreement, chapter 1.1 and be sent by e-mail.
"""


def test_pointer_chapters():
    assert is_pointer("See Formel Q Konkret. No additional Q requirements in regard to Formel Q.")
    assert is_pointer("See Formel Q Konkret.")
    assert not is_pointer("See CVS92. Other additional brand specific requirements may apply.")
    assert not is_pointer("The supplier shall retain records. " * 10)


def _conflict(a: str, b: str) -> PotentialConflict:
    return PotentialConflict(conflict_id="CON-1", related_requirement_ids=[a, b], conflict_type=ConflictType.OTHER_OR_UNCERTAIN,
                             description="x", reason_for_review="x")


@pytest.mark.parametrize(
    "facts, pair, status, prevailing",
    [
        ({("DOC-A", "DOC-B"): ["A over B"]}, ("RA", "RB"), "resolved_by_precedence", "DOC-A"),
        ({("DOC-B", "DOC-A"): ["B over A"]}, ("RA", "RB"), "resolved_by_precedence", "DOC-B"),
        ({("DOC-A", "DOC-C"): ["A over C"], ("DOC-C", "DOC-B"): ["C over B"]}, ("RA", "RB"), "resolved_by_precedence", "DOC-A"),  # transitive
        ({("DOC-A", "DOC-B"): ["A over B"], ("DOC-B", "DOC-A"): ["B over A"]}, ("RA", "RB"), "mutual_precedence_claims", None),
        ({}, ("RA", "RB"), "unresolved", None),
        ({("DOC-A", "DOC-B"): ["A over B"]}, ("RA", "RA2"), "unresolved", None),  # one document
    ],
)
def test_precedence_classification(facts, pair, status, prevailing):
    conflict = _conflict(*pair)
    classify_conflicts([conflict], {"RA": "DOC-A", "RA2": "DOC-A", "RB": "DOC-B"}, facts)
    assert (conflict.precedence_status, conflict.prevailing_document_id) == (status, prevailing)


def test_transitive_basis_lists_each_step():
    assert _reaches({("A", "C"): ["A>C"], ("C", "B"): ["C>B"]}, "A", "B") == ["A>C", "C>B"]


def _bundle(tmp_path, registry: str | None):
    folder = tmp_path / "b"
    folder.mkdir()
    (folder / "halvard_q_agreement.txt").write_text(BASE, encoding="utf-8")
    (folder / "hvg_cvs_20.txt").write_text(DELTA, encoding="utf-8")
    if registry:
        (folder / "documents.yaml").write_text(registry, encoding="utf-8")
    return folder


REGISTRY = """\
documents:
  - file: halvard_q_agreement.txt
    title: Halvard Q Agreement
  - file: hvg_cvs_20.txt
    takes_precedence_over: [HALVARD-Q-AGREEMENT]
"""


def _run(folder, config, strategy="graph_assisted", **options):
    config.llm.mock.reference_miss_rate = 0.0
    config.llm.mock.hallucination_rate = 0.0
    config.llm.mock.conflict_miss_rate = 0.0
    config.llm.mock.conflict_false_positive_rate = 0.0
    return analyze_files([folder], strategy=strategy, config=config)


def test_delta_document_end_to_end(tmp_path, config):
    result = _run(_bundle(tmp_path, REGISTRY), config)
    delta = next(d for d in result.input_documents if d.canonical_identifier == "HVG-CVS-20")
    assert (delta.follows_structure_of, delta.structure_source) == ("HALVARD-Q-AGREEMENT", "text")
    edges = {(e.relationship_type, e.source_node_id.rsplit("-", 1)[-1], e.target_node_id.rsplit("-", 1)[-1]) for e in result.relationships}
    assert (RelationshipType.INHERITS_FROM, "1.1", "1.1") in edges  # 'See ... No additional requirements.'
    assert (RelationshipType.AMENDS, "1.2", "1.2") in edges
    assert (RelationshipType.REFERENCES_CHAPTER, "2", "1.1") in edges  # 'the Halvard Q Agreement, chapter 1.1'
    records = [c for c in result.potential_conflicts if {r.rsplit("-", 1)[-1] for r in c.related_requirement_ids} == {"1.2"}]
    assert len(records) == 1, [c.related_requirement_ids for c in result.potential_conflicts]
    assert records[0].precedence_status == "resolved_by_precedence"
    assert records[0].prevailing_document_id == "DOC-HVG-CVS-20"
    assert any("documents.yaml" in b for b in records[0].precedence_basis)


def test_structure_and_precedence_can_be_switched_off(tmp_path, config):
    from csr_lab.pipelines.registry import derive_strategy

    spec = derive_strategy("graph_assisted__nostructure", "graph_assisted", {"structure_links": "off", "conflicts": {"precedence_classification": False}})
    result = _run(_bundle(tmp_path, REGISTRY), config, strategy=spec)
    assert not any(e.relationship_type in (RelationshipType.AMENDS, RelationshipType.INHERITS_FROM) for e in result.relationships)
    assert all(c.precedence_status == "not_assessed" for c in result.potential_conflicts)
