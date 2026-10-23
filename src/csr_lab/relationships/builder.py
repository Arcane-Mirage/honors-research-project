"""Build the JSON relationship graph from requirements, documents and conflicts.

Nodes: requirements and documents (input documents and referenced documents
share one ID space, so a provided document that is also referenced is a single
node with ``content_available: true``).

Edges, each with ``metadata.origin``:
* ``contains``        input document -> requirement           (document_structure)
* ``references``      requirement -> document, with evidence  (extraction)
* ``explicitly_*``    requirement -> document, only when the text states it (extraction)
* ``references``      document -> document, derived from its requirements (aggregation)
* ``potentially_conflicts_with`` requirement -> requirement (analysis; analytical_hypothesis: true)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from csr_lab.ids import relationship_id
from csr_lab.schemas import (
    RELATION_EDGE_TYPES,
    SCHEMA_VERSION,
    AnalysisResult,
    Evidence,
    InputDocument,
    PotentialConflict,
    ReferencedDocument,
    Relationship,
    RelationshipType,
    Requirement,
)

MAX_AGGREGATED_EVIDENCE = 5


class GraphNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    node_type: Literal["requirement", "document"]
    label: str
    attributes: dict[str, Any] = Field(default_factory=dict)


class RelationshipGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    nodes: list[GraphNode] = Field(default_factory=list)
    edges: list[Relationship] = Field(default_factory=list)

    _nodes: dict[str, GraphNode] = PrivateAttr(default_factory=dict)
    _out: dict[str, list[Relationship]] = PrivateAttr(default_factory=dict)
    _in: dict[str, list[Relationship]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self.reindex()

    def reindex(self) -> None:
        self._nodes = {n.node_id: n for n in self.nodes}
        self._out, self._in = {}, {}
        for e in self.edges:
            self._out.setdefault(e.source_node_id, []).append(e)
            self._in.setdefault(e.target_node_id, []).append(e)

    def node(self, node_id: str) -> GraphNode | None:
        return self._nodes.get(node_id)

    def out_edges(self, node_id: str, edge_type: RelationshipType | None = None) -> list[Relationship]:
        return [e for e in self._out.get(node_id, []) if edge_type is None or e.relationship_type == edge_type]

    def in_edges(self, node_id: str, edge_type: RelationshipType | None = None) -> list[Relationship]:
        return [e for e in self._in.get(node_id, []) if edge_type is None or e.relationship_type == edge_type]

    def nodes_of_type(self, node_type: str) -> list[GraphNode]:
        return [n for n in self.nodes if n.node_type == node_type]

    def container_of(self, requirement_id: str) -> str | None:
        edges = self.in_edges(requirement_id, RelationshipType.CONTAINS)
        return edges[0].source_node_id if edges else None


def _edge(source: str, edge_type: RelationshipType, target: str, evidence: list[Evidence], **metadata: Any) -> Relationship:
    return Relationship(
        relationship_id=relationship_id(source, edge_type.value, target),
        source_node_id=source,
        target_node_id=target,
        relationship_type=edge_type,
        evidence=evidence,
        metadata=metadata,
    )


def build_graph(
    input_documents: list[InputDocument],
    requirements: list[Requirement],
    referenced_documents: list[ReferencedDocument],
    conflicts: list[PotentialConflict] | None = None,
) -> RelationshipGraph:
    nodes: dict[str, GraphNode] = {}
    for doc in input_documents:
        nodes[doc.document_id] = GraphNode(
            node_id=doc.document_id,
            node_type="document",
            label=doc.canonical_identifier or doc.filename,
            attributes={
                "canonical_identifier": doc.canonical_identifier,
                "title": doc.title,
                "revision": doc.revision,
                "content_available": doc.parse_status in ("ok", "partial"),
                "is_input_document": True,
                "names": [doc.title] if doc.title else [],
            },
        )
    for ref in referenced_documents:
        attrs = {
            "canonical_identifier": ref.canonical_identifier,
            "revision": ref.revision,
            "document_name": ref.document_name,
            "document_type": ref.document_type.value,
            "content_available": ref.content_available,
            "is_input_document": ref.document_id in nodes,
            "names": [n for n in ref.metadata.get("names", []) if n],
        }
        if ref.document_id in nodes:
            existing = nodes[ref.document_id].attributes
            existing["names"] = list(dict.fromkeys([*existing.get("names", []), *attrs["names"]]))
            existing["document_type"] = attrs["document_type"]
            existing["document_name"] = ref.document_name
        else:
            nodes[ref.document_id] = GraphNode(
                node_id=ref.document_id,
                node_type="document",
                label=ref.canonical_identifier or ref.document_name or ref.document_id,
                attributes=attrs,
            )
    for req in requirements:
        nodes[req.requirement_id] = GraphNode(
            node_id=req.requirement_id,
            node_type="requirement",
            label=f"{req.chapter or '?'}: {req.requirement_text[:60]}",
            attributes={"chapter": req.chapter, "source_document_id": req.source_document_id},
        )

    edges: dict[str, Relationship] = {}

    def add(edge: Relationship) -> None:
        edges.setdefault(edge.relationship_id, edge)

    doc_level: dict[tuple[str, str], list[tuple[str, Evidence]]] = {}
    for req in requirements:
        add(_edge(req.source_document_id, RelationshipType.CONTAINS, req.requirement_id, [], origin="document_structure"))
        by_doc: dict[str, list] = {}
        for m in req.reference_mentions:
            by_doc.setdefault(m.document_id, []).append(m)
        for doc_id, mentions in by_doc.items():
            add(
                _edge(
                    req.requirement_id,
                    RelationshipType.REFERENCES,
                    doc_id,
                    [m.evidence for m in mentions],
                    origin="extraction",
                    mention_ids=[m.mention_id for m in mentions],
                    statuses=sorted({m.status.value for m in mentions}),
                    verified=all(m.verified for m in mentions),
                )
            )
            for m in mentions:
                if m.relation in RELATION_EDGE_TYPES:
                    add(
                        _edge(
                            req.requirement_id,
                            RELATION_EDGE_TYPES[m.relation],
                            doc_id,
                            [m.evidence],
                            origin="extraction",
                            mention_ids=[m.mention_id],
                        )
                    )
            if doc_id != req.source_document_id:
                doc_level.setdefault((req.source_document_id, doc_id), []).extend(
                    (req.requirement_id, m.evidence) for m in mentions
                )

    for (src, tgt), items in sorted(doc_level.items()):
        add(
            _edge(
                src,
                RelationshipType.REFERENCES,
                tgt,
                [ev for _, ev in items[:MAX_AGGREGATED_EVIDENCE]],
                origin="aggregation",
                derived=True,
                via_requirement_ids=sorted({rid for rid, _ in items}),
            )
        )

    for conflict in conflicts or []:
        a, b = sorted(conflict.related_requirement_ids)[:2]
        add(
            _edge(
                a,
                RelationshipType.POTENTIALLY_CONFLICTS_WITH,
                b,
                conflict.evidence,
                origin="analysis",
                analytical_hypothesis=True,
                conflict_id=conflict.conflict_id,
                conflict_type=conflict.conflict_type.value,
                review_status=conflict.review_status,
            )
        )

    return RelationshipGraph(nodes=list(nodes.values()), edges=list(edges.values()))


def graph_from_result(result: AnalysisResult) -> RelationshipGraph:
    """Graph object for an exported result: nodes rebuilt from its lists, edges taken as exported."""
    nodes = build_graph(result.input_documents, result.requirements, result.referenced_documents, []).nodes
    return RelationshipGraph(nodes=nodes, edges=list(result.relationships))
