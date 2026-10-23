import json

import pytest

from csr_lab.pipelines.run import run_analysis
from csr_lab.relationships import queries
from csr_lab.relationships.builder import RelationshipGraph, graph_from_result
from csr_lab.relationships.validation import validate_graph
from csr_lab.schemas import Relationship, RelationshipType
from helpers import case_documents


@pytest.fixture
def packaging_graph(dataset, config):
    _, docs, _ = case_documents(dataset, "dev_corvane_packaging")
    result = run_analysis(docs, strategy="graph_assisted", config=config)
    return result, graph_from_result(result)


def _req(result, doc, chapter):
    return next(r.requirement_id for r in result.requirements if r.source_document_id == doc and r.chapter == chapter)


def test_graph_structure(packaging_graph):
    result, graph = packaging_graph
    types = {e.relationship_type for e in graph.edges}
    assert {RelationshipType.CONTAINS, RelationshipType.REFERENCES, RelationshipType.EXPLICITLY_EXCLUDES,
            RelationshipType.EXPLICITLY_TAKES_PRECEDENCE_OVER, RelationshipType.POTENTIALLY_CONFLICTS_WITH} <= types
    report = validate_graph(graph)
    assert report.valid, report
    # The provided packaging specification is one node, referenced and available.
    pkg = graph.node("DOC-CVM-PKG-210")
    assert pkg.attributes["content_available"] and pkg.attributes["is_input_document"]
    doc_edges = [e for e in graph.edges if e.metadata.get("origin") == "aggregation"]
    assert any(e.source_node_id == "DOC-CVM-CSR-200" and e.target_node_id == "DOC-CVM-PKG-210" for e in doc_edges)
    assert all(e.evidence for e in graph.edges if e.metadata.get("origin") == "extraction")


def test_queries(packaging_graph):
    result, graph = packaging_graph
    r12 = _req(result, "DOC-CVM-CSR-200", "1.2")
    assert queries.documents_referenced_by(graph, r12) == ["DOC-CVM-PKG-210"]
    referencing = queries.requirements_referencing(graph, "DOC-CVM-PKG-210")
    assert r12 in referencing and len(referencing) >= 3
    shared = queries.shared_documents(graph)
    assert "DOC-CVM-LBL-220" in shared and len(shared["DOC-CVM-LBL-220"]) == 2
    pairs = queries.requirements_sharing_references(graph, r12)
    assert pairs and all(r12 in (a, b) for a, b, _ in pairs)
    assert queries.unresolved_documents(graph) == []
    unavailable = {n.node_id for n in queries.unavailable_documents(graph)}
    assert "DOC-CVM-PKG-050" in unavailable and "DOC-CVM-PKG-210" not in unavailable
    assert queries.potentially_conflicting_requirements(graph)
    edge = next(e for e in graph.edges if e.relationship_type == RelationshipType.EXPLICITLY_EXCLUDES)
    assert "does not apply" in queries.evidence_for(graph, edge.relationship_id)[0].quoted_text
    exported = json.loads(json.dumps(queries.export_graph(graph)))
    assert RelationshipGraph.model_validate(exported).edges[0].relationship_id == graph.edges[0].relationship_id
    assert queries.to_mermaid(graph).startswith("flowchart LR")


def test_applicability_and_precedence_rules(packaging_graph):
    result, graph = packaging_graph
    flags = queries.find_applicability_conflicts(graph)
    assert [f.requirement_ids for f in flags] == [tuple(sorted((_req(result, "DOC-CVM-CSR-200", "1.3"), _req(result, "DOC-CVM-PKG-210", "2.1"))))]
    assert flags[0].evidence and flags[0].relationship_ids
    prec = queries.find_precedence_conflicts(graph)
    assert [set(f.requirement_ids) for f in prec] == [{_req(result, "DOC-CVM-CSR-200", "5.1"), _req(result, "DOC-CVM-PKG-210", "4.1")}]


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
