"""src/csr_lab/relationships/builder.py

Build the JSON relationship graph from requirements, documents and conflicts.

Nodes: requirements and documents. Documents exist at two levels:

* the **document** ``DOC-CVM-PKG-210`` (any revision), and
* a **revision** ``DOC-CVM-PKG-210-REV-D``. Every input document is a revision
  when its header states one; requirements belong to that revision.

A reference points at whatever the text names: "CVM-PKG-210" links to the
document node, "CVM-PKG-210 Rev C" to that revision. Which revision an
unrevisioned reference means is answered at query time
(``RelationshipGraph.current_revision``), never stored on the reference, so a
newer revision does not leave old links stale.

Edges, each with ``metadata.origin``:
* ``contains``        input document -> requirement           (document_structure)
* ``has_revision``    document -> revision                    (document_structure)
* ``references``      requirement -> document, with evidence  (extraction)
* ``explicitly_*``    requirement -> document, only when the text states it (extraction)
* ``references``      document -> document, derived from its requirements (aggregation)
* ``potentially_conflicts_with`` requirement -> requirement (analysis; analytical_hypothesis: true)
* ``superseded_by``   older -> newer revision or requirement version (revision_tracking;
  added by the saved graph store, see ``csr_lab.relationships.store``)
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from csr_lab.ids import document_family_id, relationship_id
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

    # -- revisions ---------------------------------------------------------

    def family_of(self, node_id: str) -> str:
        """The document-level node of a revision node; any other node maps to itself."""
        node = self.node(node_id)
        return (node.attributes.get("revision_of") if node else None) or node_id

    def revisions_of(self, document_id: str) -> list[str]:
        """Revision nodes of a document node, oldest first where the order is known."""
        edges = self.out_edges(document_id, RelationshipType.HAS_REVISION)
        return [e.target_node_id for e in sorted(edges, key=lambda e: (e.metadata.get("sequence") is None, e.metadata.get("sequence") or 0, e.target_node_id))]

    def current_revision(self, document_id: str) -> str | None:
        """The revision an unrevisioned reference to ``document_id`` means, if known."""
        node = self.node(document_id)
        return node.attributes.get("current_revision") if node else None

    def same_document(self, a: str, b: str) -> bool:
        """True for the same node, or a document node and one of its revisions.

        Two different revisions of one document are *not* the same document.
        """
        return a == b or self.family_of(a) == b or self.family_of(b) == a

    def equivalent_documents(self, node_id: str) -> set[str]:
        """``node_id`` plus every node for which ``same_document`` holds."""
        family = self.family_of(node_id)
        out = {node_id, family}
        if family == node_id:
            out.update(self.revisions_of(node_id))
        return out

    def requirements_in(self, document_id: str) -> list[str]:
        """Requirements contained in a document node. For a document-level node, those of
        its current revision (or of all revisions when no current revision is known)."""
        direct = [e.target_node_id for e in self.out_edges(document_id, RelationshipType.CONTAINS)]
        if direct or self.family_of(document_id) != document_id:
            return direct
        current = self.current_revision(document_id)
        revisions = [current] if current else self.revisions_of(document_id)
        return [e.target_node_id for r in revisions for e in self.out_edges(r, RelationshipType.CONTAINS)]


def _edge(source: str, edge_type: RelationshipType, target: str, evidence: list[Evidence], **metadata: Any) -> Relationship:
    return Relationship(
        relationship_id=relationship_id(source, edge_type.value, target),
        source_node_id=source,
        target_node_id=target,
        relationship_type=edge_type,
        evidence=evidence,
        metadata=metadata,
    )


make_edge = _edge  # public name for other modules (the saved graph store)


def _iso_date(value: str | None) -> date | None:
    try:
        return date.fromisoformat(value.strip()) if value else None
    except ValueError:
        return None


def order_revisions(revisions: list[GraphNode], declared: list[str] | None = None) -> tuple[list[GraphNode], str]:
    """Oldest-first order of one document's revisions and how it was decided.

    Revision labels are never sorted as strings (A..Z, AA; 1, 2; '-' all occur).
    Order of preference: a declared order (labels, oldest first) covering every
    given revision > distinct ISO effective dates on every revision > unknown
    (input order kept).
    """
    if declared:
        rank = {label.upper(): i for i, label in enumerate(declared)}
        if all((n.attributes.get("revision") or "").upper() in rank for n in revisions):
            return sorted(revisions, key=lambda n: rank[(n.attributes.get("revision") or "").upper()]), "declared"
    dates = [_iso_date(n.attributes.get("effective_date")) for n in revisions]
    if all(dates) and len(set(dates)) == len(dates):
        return [n for _, n in sorted(zip(dates, revisions), key=lambda x: x[0])], "effective_date"
    return list(revisions), "unknown"


def add_revision_structure(
    nodes: dict[str, GraphNode], edges: dict[str, Relationship], declared_orders: dict[str, list[str]] | None = None
) -> None:
    """Add ``has_revision`` edges, revision sequence numbers and each document's current revision.

    Only revisions whose content was analysed (``analyzed: true``) are ordered by date
    and can be current; revisions that are merely referenced get no sequence unless a
    declared order covers them. With several analysed revisions in unknown order, no
    revision is current.
    """
    by_family: dict[str, list[GraphNode]] = {}
    for node in nodes.values():
        family = node.attributes.get("revision_of")
        if node.node_type == "document" and family and family in nodes:
            by_family.setdefault(family, []).append(node)
    for family, revisions in by_family.items():
        revisions = sorted(revisions, key=lambda n: n.node_id)
        analysed = [n for n in revisions if n.attributes.get("analyzed")]
        rank = {label.upper(): i for i, label in enumerate((declared_orders or {}).get(family) or [])}

        def label(n: GraphNode) -> str:
            return (n.attributes.get("revision") or "").upper()

        if rank and all(label(n) in rank for n in analysed):
            # A declared order must cover every analysed revision; it may also place referenced ones.
            named = sorted([n for n in revisions if label(n) in rank], key=lambda n: rank[label(n)])
            ordered, method = named + [n for n in revisions if label(n) not in rank], "declared"
            placed = {n.node_id for n in named}
        else:
            ordered, method = order_revisions(analysed, None)
            ordered += [n for n in revisions if n not in analysed]
            placed = {n.node_id for n in analysed}
        known = method != "unknown"
        sequence = {n.node_id: i for i, n in enumerate(ordered) if known and n.node_id in placed}
        for rev in ordered:
            seq = sequence.get(rev.node_id)
            rev.attributes["revision_sequence"] = seq
            edge = _edge(family, RelationshipType.HAS_REVISION, rev.node_id, [], origin="document_structure", sequence=seq)
            edges[edge.relationship_id] = edge
        analysed_in_order = [n for n in ordered if n.attributes.get("analyzed")]
        current = analysed_in_order[-1].node_id if analysed_in_order and (known or len(analysed_in_order) == 1) else None
        nodes[family].attributes["current_revision"] = current
        nodes[family].attributes["revision_order"] = method


def build_graph(
    input_documents: list[InputDocument],
    requirements: list[Requirement],
    referenced_documents: list[ReferencedDocument],
    conflicts: list[PotentialConflict] | None = None,
) -> RelationshipGraph:
    nodes: dict[str, GraphNode] = {}
    for doc in input_documents:
        available = doc.parse_status in ("ok", "partial")
        names = [doc.title] if doc.title else []
        family = document_family_id(doc.canonical_identifier) if doc.canonical_identifier else None
        is_revision = bool(family and family != doc.document_id and doc.revision)
        nodes[doc.document_id] = GraphNode(
            node_id=doc.document_id,
            node_type="document",
            label=f"{doc.canonical_identifier} Rev {doc.revision}" if is_revision else (doc.canonical_identifier or doc.filename),
            attributes={
                "canonical_identifier": doc.canonical_identifier,
                "title": doc.title,
                "revision": doc.revision,
                "effective_date": doc.effective_date,
                "content_available": available,
                "analyzed": available,
                "is_input_document": True,
                "names": names,
                **({"revision_of": family} if is_revision else {}),
            },
        )
        if is_revision and family:
            # The document-level node: what unrevisioned references and names point at.
            fam = nodes.setdefault(
                family,
                GraphNode(
                    node_id=family,
                    node_type="document",
                    label=doc.canonical_identifier or family,
                    attributes={"canonical_identifier": doc.canonical_identifier, "revision": None, "names": []},
                ),
            )
            fam.attributes["title"] = fam.attributes.get("title") or doc.title
            fam.attributes["names"] = list(dict.fromkeys([*fam.attributes.get("names", []), *names]))
            fam.attributes["is_input_document"] = True
            fam.attributes["content_available"] = fam.attributes.get("content_available") or available
            fam.attributes["provided_revisions"] = [*fam.attributes.get("provided_revisions", []), doc.revision]
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
            existing["referenced"] = True
        else:
            if ref.canonical_identifier and ref.revision:
                family = document_family_id(ref.canonical_identifier)
                if family != ref.document_id:
                    attrs["revision_of"] = family
            nodes[ref.document_id] = GraphNode(
                node_id=ref.document_id,
                node_type="document",
                label=(f"{ref.canonical_identifier} Rev {ref.revision}" if attrs.get("revision_of") else ref.canonical_identifier)
                or ref.document_name
                or ref.document_id,
                attributes={**attrs, "referenced": True},
            )
    for req in requirements:
        nodes[req.requirement_id] = GraphNode(
            node_id=req.requirement_id,
            node_type="requirement",
            label=f"{req.chapter or '?'}: {req.requirement_text[:60]}",
            attributes={"chapter": req.chapter, "source_document_id": req.source_document_id},
        )

    def family_of(node_id: str) -> str:
        node = nodes.get(node_id)
        return (node.attributes.get("revision_of") if node else None) or node_id

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
            if family_of(doc_id) != family_of(req.source_document_id):  # not a self-reference
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

    add_revision_structure(nodes, edges)
    return RelationshipGraph(nodes=list(nodes.values()), edges=list(edges.values()))


def graph_from_result(result: AnalysisResult) -> RelationshipGraph:
    """Graph object for an exported result: nodes rebuilt from its lists, edges taken as exported."""
    nodes = build_graph(result.input_documents, result.requirements, result.referenced_documents, []).nodes
    return RelationshipGraph(nodes=nodes, edges=list(result.relationships))
