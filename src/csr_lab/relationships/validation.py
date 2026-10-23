"""Structural validation of a relationship graph.

Structural validity (every edge points at a real node, no duplicates, IDs are
consistent) says nothing about whether the extracted relationships are
*correct*; that is measured separately by the evaluator against gold data.
"""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, Field

from csr_lab.ids import document_node_id, relationship_id
from csr_lab.relationships.builder import RelationshipGraph

_HYPOTHESIS_TYPES = {"potentially_conflicts_with"}


class GraphValidationReport(BaseModel):
    valid: bool
    node_count: int
    edge_count: int
    invalid_endpoints: list[str] = Field(default_factory=list)
    duplicate_node_ids: list[str] = Field(default_factory=list)
    duplicate_edges: list[str] = Field(default_factory=list)
    inconsistent_node_ids: list[str] = Field(default_factory=list)
    inconsistent_edge_ids: list[str] = Field(default_factory=list)
    edges_without_evidence: list[str] = Field(default_factory=list)
    unlabelled_hypotheses: list[str] = Field(default_factory=list)


def validate_graph(graph: RelationshipGraph) -> GraphValidationReport:
    node_ids = [n.node_id for n in graph.nodes]
    known = set(node_ids)
    dup_nodes = sorted(k for k, v in Counter(node_ids).items() if v > 1)

    invalid = [
        e.relationship_id for e in graph.edges if e.source_node_id not in known or e.target_node_id not in known
    ]
    triples = Counter((e.source_node_id, e.relationship_type.value, e.target_node_id) for e in graph.edges)
    dup_edges = sorted(f"{s} -{t}-> {d}" for (s, t, d), n in triples.items() if n > 1)

    inconsistent_nodes = []
    for n in graph.nodes:
        if n.node_type == "document":
            canonical = n.attributes.get("canonical_identifier")
            if canonical and not n.attributes.get("is_input_document"):
                expected = document_node_id(canonical, n.attributes.get("revision"))
                if expected != n.node_id:
                    inconsistent_nodes.append(n.node_id)
        elif not n.node_id.startswith("REQ-"):
            inconsistent_nodes.append(n.node_id)

    inconsistent_edges = [
        e.relationship_id
        for e in graph.edges
        if e.relationship_id != relationship_id(e.source_node_id, e.relationship_type.value, e.target_node_id)
    ]
    no_evidence = [
        e.relationship_id
        for e in graph.edges
        if e.metadata.get("origin") in ("extraction", "aggregation") and not e.evidence
    ]
    unlabelled = [
        e.relationship_id
        for e in graph.edges
        if e.relationship_type.value in _HYPOTHESIS_TYPES and not e.metadata.get("analytical_hypothesis")
    ]
    report = GraphValidationReport(
        valid=not (invalid or dup_nodes or dup_edges or inconsistent_nodes or inconsistent_edges or no_evidence or unlabelled),
        node_count=len(graph.nodes),
        edge_count=len(graph.edges),
        invalid_endpoints=invalid,
        duplicate_node_ids=dup_nodes,
        duplicate_edges=dup_edges,
        inconsistent_node_ids=inconsistent_nodes,
        inconsistent_edge_ids=inconsistent_edges,
        edges_without_evidence=no_evidence,
        unlabelled_hypotheses=unlabelled,
    )
    return report
