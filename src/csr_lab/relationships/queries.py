"""Deterministic queries over the relationship graph.

Includes two flagging rules (Section 10.1 of the project brief). They identify
*where* potentially conflicting relationships occur and attach the edges and
evidence involved. They do not explain, rank or resolve conflicts:

* applicability: a requirement explicitly excludes document X, while a
  requirement in the same or a related (reference-linked) document references X;
* precedence: two documents each explicitly claim precedence over the other.

Note: two requirements that merely reference the same document are never
flagged by these rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from csr_lab.relationships.builder import GraphNode, RelationshipGraph
from csr_lab.schemas import Evidence, RelationshipType

REF = RelationshipType.REFERENCES


def _requirement_ref_edges(graph: RelationshipGraph, node_id: str):
    return [e for e in graph.out_edges(node_id, REF) if e.metadata.get("origin") == "extraction"]


def documents_referenced_by(graph: RelationshipGraph, requirement_id: str) -> list[str]:
    return [e.target_node_id for e in _requirement_ref_edges(graph, requirement_id)]


def requirements_referencing(graph: RelationshipGraph, document_id: str) -> list[str]:
    return [
        e.source_node_id
        for e in graph.in_edges(document_id, REF)
        if e.metadata.get("origin") == "extraction"
    ]


def shared_documents(graph: RelationshipGraph, min_requirements: int = 2) -> dict[str, list[str]]:
    """Documents referenced by at least ``min_requirements`` requirements."""
    out = {}
    for node in graph.nodes_of_type("document"):
        reqs = requirements_referencing(graph, node.node_id)
        if len(reqs) >= min_requirements:
            out[node.node_id] = sorted(reqs)
    return out


def requirements_sharing_references(
    graph: RelationshipGraph, requirement_id: str | None = None, max_fanout: int | None = None
) -> list[tuple[str, str, list[str]]]:
    """Requirement pairs connected through at least one shared referenced document."""
    shared: dict[tuple[str, str], list[str]] = {}
    for doc_id, reqs in shared_documents(graph).items():
        if max_fanout is not None and len(reqs) > max_fanout:
            continue
        for a, b in combinations(reqs, 2):
            if requirement_id is None or requirement_id in (a, b):
                shared.setdefault((a, b), []).append(doc_id)
    return [(a, b, docs) for (a, b), docs in sorted(shared.items())]


def unresolved_documents(graph: RelationshipGraph) -> list[GraphNode]:
    """Document nodes known only by a name, with no identifier."""
    return [n for n in graph.nodes_of_type("document") if not n.attributes.get("canonical_identifier")]


def unavailable_documents(graph: RelationshipGraph) -> list[GraphNode]:
    return [n for n in graph.nodes_of_type("document") if not n.attributes.get("content_available")]


def potentially_conflicting_requirements(
    graph: RelationshipGraph, requirement_id: str | None = None
) -> list[tuple[str, str, str | None]]:
    out = []
    for e in graph.edges:
        if e.relationship_type != RelationshipType.POTENTIALLY_CONFLICTS_WITH:
            continue
        if requirement_id is None or requirement_id in (e.source_node_id, e.target_node_id):
            out.append((e.source_node_id, e.target_node_id, e.metadata.get("conflict_id")))
    return out


def evidence_for(graph: RelationshipGraph, relationship_id: str) -> list[Evidence]:
    for e in graph.edges:
        if e.relationship_id == relationship_id:
            return list(e.evidence)
    raise KeyError(relationship_id)


def related_documents(graph: RelationshipGraph, document_id: str, hops: int = 1) -> set[str]:
    """Documents within ``hops`` document-level reference links, in either direction."""
    frontier, seen = {document_id}, {document_id}
    for _ in range(hops):
        nxt = set()
        for doc in frontier:
            for e in graph.out_edges(doc, REF) + graph.in_edges(doc, REF):
                if e.metadata.get("origin") != "aggregation":
                    continue
                other = e.target_node_id if e.source_node_id == doc else e.source_node_id
                if other not in seen:
                    nxt.add(other)
        seen |= nxt
        frontier = nxt
    return seen


def requirements_in_linked_documents(graph: RelationshipGraph) -> list[tuple[str, str, str, str]]:
    """(req_a, req_b, doc_a, doc_b) for requirements in two documents linked by a reference."""
    pairs = []
    for e in graph.edges:
        if e.relationship_type != REF or e.metadata.get("origin") != "aggregation":
            continue
        src_reqs = [x.target_node_id for x in graph.out_edges(e.source_node_id, RelationshipType.CONTAINS)]
        tgt_reqs = [x.target_node_id for x in graph.out_edges(e.target_node_id, RelationshipType.CONTAINS)]
        for a in src_reqs:
            for b in tgt_reqs:
                pairs.append((a, b, e.source_node_id, e.target_node_id))
    return pairs


@dataclass
class GraphFlag:
    rule: str
    requirement_ids: tuple[str, str]
    document_ids: list[str]
    relationship_ids: list[str]
    evidence: list[Evidence] = field(default_factory=list)
    detail: str = ""


def find_applicability_conflicts(graph: RelationshipGraph, hops: int = 1) -> list[GraphFlag]:
    flags: dict[tuple[str, str], GraphFlag] = {}
    for ex in [e for e in graph.edges if e.relationship_type == RelationshipType.EXPLICITLY_EXCLUDES]:
        req_a, excluded = ex.source_node_id, ex.target_node_id
        doc_a = graph.container_of(req_a)
        if doc_a is None:
            continue
        related = related_documents(graph, doc_a, hops)
        for ref in graph.in_edges(excluded, REF):
            req_b = ref.source_node_id
            if ref.metadata.get("origin") != "extraction" or req_b == req_a:
                continue
            if any(e.target_node_id == excluded for e in graph.out_edges(req_b, RelationshipType.EXPLICITLY_EXCLUDES)):
                continue  # both exclude it: consistent
            doc_b = graph.container_of(req_b)
            if doc_b not in related:
                continue
            path = [
                e.relationship_id
                for e in graph.edges
                if e.metadata.get("origin") == "aggregation"
                and {e.source_node_id, e.target_node_id} == {doc_a, doc_b}
            ]
            pair = tuple(sorted((req_a, req_b)))
            flags.setdefault(
                pair,  # type: ignore[arg-type]
                GraphFlag(
                    rule="applicability",
                    requirement_ids=pair,  # type: ignore[arg-type]
                    document_ids=sorted({doc_a, doc_b, excluded}),
                    relationship_ids=[ex.relationship_id, ref.relationship_id, *path],
                    evidence=[*ex.evidence, *ref.evidence],
                    detail=f"{req_a} states {excluded} does not apply; {req_b} ({doc_b}) references {excluded}.",
                ),
            )
    return list(flags.values())


def find_precedence_conflicts(graph: RelationshipGraph) -> list[GraphFlag]:
    flags: dict[tuple[str, str], GraphFlag] = {}
    prec = [e for e in graph.edges if e.relationship_type == RelationshipType.EXPLICITLY_TAKES_PRECEDENCE_OVER]
    for e1 in prec:
        doc1 = graph.container_of(e1.source_node_id)
        for e2 in prec:
            doc2 = graph.container_of(e2.source_node_id)
            if e1 is e2 or doc1 is None or doc2 is None or doc1 == doc2:
                continue
            if e1.target_node_id == doc2 and e2.target_node_id == doc1:
                pair = tuple(sorted((e1.source_node_id, e2.source_node_id)))
                flags.setdefault(
                    pair,  # type: ignore[arg-type]
                    GraphFlag(
                        rule="precedence",
                        requirement_ids=pair,  # type: ignore[arg-type]
                        document_ids=sorted({doc1, doc2}),
                        relationship_ids=sorted({e1.relationship_id, e2.relationship_id}),
                        evidence=[*e1.evidence, *e2.evidence],
                        detail=f"{doc1} and {doc2} each state that they take precedence over the other.",
                    ),
                )
    return list(flags.values())


def export_graph(graph: RelationshipGraph) -> dict:
    return graph.model_dump(mode="json")


def to_mermaid(graph: RelationshipGraph, include_contains: bool = False) -> str:
    """Lightweight Mermaid flowchart for inspection (the JSON graph is the real artifact)."""

    def nid(x: str) -> str:
        return "n_" + "".join(c if c.isalnum() else "_" for c in x)

    lines = ["flowchart LR"]
    for n in graph.nodes:
        label = n.label.replace('"', "'")[:50]
        shape = f'["{label}"]' if n.node_type == "requirement" else f'[("{label}")]'
        lines.append(f"  {nid(n.node_id)}{shape}")
    styles = {
        RelationshipType.REFERENCES: "-->",
        RelationshipType.CONTAINS: "-.->",
        RelationshipType.POTENTIALLY_CONFLICTS_WITH: "<-.->",
    }
    for e in graph.edges:
        if e.relationship_type == RelationshipType.CONTAINS and not include_contains:
            continue
        arrow = styles.get(e.relationship_type, "==>")
        lines.append(f"  {nid(e.source_node_id)} {arrow}|{e.relationship_type.value}| {nid(e.target_node_id)}")
    return "\n".join(lines) + "\n"
