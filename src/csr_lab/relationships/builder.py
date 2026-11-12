"""src/csr_lab/relationships/builder.py

Build the JSON relationship graph from requirements, documents and conflicts.

Nodes: requirements and documents. Documents exist at two levels:

* the **document** ``DOC-HVG-CVS-20`` (any revision), and
* a **revision** ``DOC-HVG-CVS-20-REV-D``. Every input document is a revision
  when its header states one; requirements belong to that revision.

A reference points at whatever the text names: "HVG-CVS-20" links to the
document node, "HVG-CVS-20 Rev C" to that revision. Which revision an
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

Documents without chapters have no requirement nodes. Their passages are stored on the
document node (``attributes.passages``: ID, pages, text), and what the passages state is
drawn from the document node itself, with ``metadata.passage_ids`` naming the passages:
* ``references`` / ``explicitly_*``  document -> document, with evidence      (extraction)
* ``potentially_conflicts_with``     requirement or document -> document       (analysis)
* ``superseded_by``   older -> newer revision or requirement version (revision_tracking;
  added by the saved graph store, see ``csr_lab.relationships.store``)
* ``amends`` / ``inherits_from``  delta-document chapter -> same-numbered base chapter (document_structure)
* ``references_chapter``  requirement -> the chapter a reference names, with evidence (extraction)
(the last three: see "Document structure" at the end of this module)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from csr_lab.drafts import MentionDraft, RequirementDraft
from csr_lab.extraction.normalization import NameIndex, canonical_input_identifier, name_match_key
from csr_lab.ids import document_family_id, relationship_id
from csr_lab.schemas import (
    RELATION_EDGE_TYPES,
    SCHEMA_VERSION,
    AnalysisResult,
    DocumentPassage,
    Evidence,
    InputDocument,
    PotentialConflict,
    ReferencedDocument,
    Relationship,
    RelationshipType,
    Requirement,
)

MAX_AGGREGATED_EVIDENCE = 5


def revision_family(nodes: dict, node_id: str) -> str:
    """The document-level node of a revision node (``attributes.revision_of``); any other node maps to itself."""
    node = nodes.get(node_id)
    return (node.attributes.get("revision_of") if node else None) or node_id


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

    def model_post_init(self, context: Any, /) -> None:
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
        return revision_family(self._nodes, node_id)

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


def make_edge(source: str, edge_type: RelationshipType, target: str, evidence: list[Evidence], **metadata: Any) -> Relationship:
    return Relationship(
        relationship_id=relationship_id(source, edge_type.value, target),
        source_node_id=source,
        target_node_id=target,
        relationship_type=edge_type,
        evidence=evidence,
        metadata=metadata,
    )




def aggregated_reference_edge(source: str, target: str, items: list[tuple[str, Evidence]]) -> Relationship:
    """Document -> document ``references``, derived from its requirements' references
    (``items``: (requirement ID, evidence)); also rebuilt by the saved graph store."""
    return make_edge(
        source,
        RelationshipType.REFERENCES,
        target,
        [ev for _, ev in items[:MAX_AGGREGATED_EVIDENCE]],
        origin="aggregation",
        derived=True,
        via_requirement_ids=sorted({rid for rid, _ in items}),
    )


def conflict_edge(a: str, b: str, conflict: PotentialConflict, passage_ids: list[str] | tuple = (), **extra: Any) -> Relationship:
    """A ``potentially_conflicts_with`` hypothesis between two nodes (requirements or documents)."""
    a, b = sorted((a, b))
    return make_edge(
        a,
        RelationshipType.POTENTIALLY_CONFLICTS_WITH,
        b,
        list(conflict.evidence),
        origin="analysis",
        analytical_hypothesis=True,
        conflict_id=conflict.conflict_id,
        conflict_ids=[conflict.conflict_id],
        conflict_type=conflict.conflict_type.value,
        review_status=conflict.review_status,
        **({"passage_ids": sorted(passage_ids)} if passage_ids else {}),
        **extra,
    )


def add_conflict_edge(edges: dict[str, Relationship], edge: Relationship) -> bool:
    """Add ``edge``; several conflicts between the same two nodes (passages of one document)
    share one edge listing them all. Returns True when the edge is new."""
    existing = edges.get(edge.relationship_id)
    if existing is None:
        edges[edge.relationship_id] = edge
        return True
    existing.evidence = [*existing.evidence, *edge.evidence]
    for key in ("conflict_ids", "passage_ids", "sources"):
        if edge.metadata.get(key):
            existing.metadata[key] = list(dict.fromkeys([*existing.metadata.get(key, []), *edge.metadata[key]]))
    return False


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
            edge = make_edge(family, RelationshipType.HAS_REVISION, rev.node_id, [], origin="document_structure", sequence=seq)
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
    passages: list[DocumentPassage] | None = None,
) -> RelationshipGraph:
    passages = passages or []
    nodes: dict[str, GraphNode] = {}
    for doc in input_documents:
        available = doc.parse_status in ("ok", "partial")
        names = [n for n in dict.fromkeys([doc.title, *doc.aliases]) if n]
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
                **({"follows_structure_of": doc.follows_structure_of, "structure_source": doc.structure_source} if doc.follows_structure_of else {}),
                **({"takes_precedence_over": doc.takes_precedence_over} if doc.takes_precedence_over else {}),
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

    for p in passages:
        doc = nodes.get(p.source_document_id)
        if doc is not None:
            doc.attributes["has_chapters"] = False
            doc.attributes.setdefault("passages", []).append(
                {
                    "passage_id": p.passage_id,
                    "page_start": p.source_location.page_start,
                    "page_end": p.source_location.page_end,
                    "start_offset": p.source_location.start_offset,
                    "text": p.text,
                }
            )

    def family_of(node_id: str) -> str:
        return revision_family(nodes, node_id)

    edges: dict[str, Relationship] = {}

    def add(edge: Relationship) -> None:
        edges.setdefault(edge.relationship_id, edge)

    doc_level: dict[tuple[str, str], list[tuple[str, Evidence]]] = {}
    for req in requirements:
        add(make_edge(req.source_document_id, RelationshipType.CONTAINS, req.requirement_id, [], origin="document_structure"))
        by_doc: dict[str, list] = {}
        for m in req.reference_mentions:
            by_doc.setdefault(m.document_id, []).append(m)
        for doc_id, mentions in by_doc.items():
            add(
                make_edge(
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
                        make_edge(
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
        add(aggregated_reference_edge(src, tgt, items))

    # What passages state, drawn from their document: one edge per (document, target, type).
    stated: dict[tuple[str, RelationshipType, str], list[tuple[str, object]]] = {}
    for p in passages:
        for m in p.reference_mentions:
            if family_of(m.document_id) == family_of(p.source_document_id):
                continue  # a self-reference
            stated.setdefault((p.source_document_id, RelationshipType.REFERENCES, m.document_id), []).append((p.passage_id, m))
            if m.relation in RELATION_EDGE_TYPES:
                stated.setdefault((p.source_document_id, RELATION_EDGE_TYPES[m.relation], m.document_id), []).append((p.passage_id, m))
    for (src, edge_type, tgt), items in sorted(stated.items(), key=lambda kv: (kv[0][0], kv[0][1].value, kv[0][2])):
        mentions = [m for _, m in items]
        add(
            make_edge(
                src,
                edge_type,
                tgt,
                [m.evidence for m in mentions],
                origin="extraction",
                passage_ids=sorted({pid for pid, _ in items}),
                mention_ids=[m.mention_id for m in mentions],
                verified=all(m.verified for m in mentions),
            )
        )

    # Delta documents and chapter references (see "Document structure" below).
    mention_evidence = {m.mention_id: m.evidence for r in requirements for m in r.reference_mentions}
    for se in structure_edges(input_documents, requirements):
        kind = {"amends": RelationshipType.AMENDS, "inherits_from": RelationshipType.INHERITS_FROM, "references_chapter": RelationshipType.REFERENCES_CHAPTER}[se.kind]
        if se.kind == "references_chapter":
            add(make_edge(se.source, kind, se.target, [mention_evidence[i] for i in se.mention_ids], origin="extraction", chapter=se.chapter, mention_ids=se.mention_ids))
        else:
            add(make_edge(se.source, kind, se.target, [], origin="document_structure", chapter=se.chapter))

    owner = {p.passage_id: p.source_document_id for p in passages}
    for conflict in conflicts or []:
        if conflict.related_saved_ids:
            continue  # the other side lives in the saved graph; the store adds this edge
        a, b = (owner.get(u, u) for u in conflict.unit_ids[:2])
        add_conflict_edge(edges, conflict_edge(a, b, conflict, [u for u in conflict.unit_ids if u in owner]))

    add_revision_structure(nodes, edges)
    return RelationshipGraph(nodes=list(nodes.values()), edges=list(edges.values()))


def graph_from_result(result: AnalysisResult) -> RelationshipGraph:
    """Graph object for an exported result: nodes rebuilt from its lists, edges taken as exported."""
    nodes = build_graph(result.input_documents, result.requirements, result.referenced_documents, [], result.document_passages).nodes
    return RelationshipGraph(nodes=nodes, edges=list(result.relationships))


# ---------------------------------------------------------------------------
# Document structure: delta documents and chapter references
# ---------------------------------------------------------------------------
# CVS10 "is following the Volkswagen Formel Q Konkret document structure": its chapter 4.4.2 is
# about the same subject as Formel Q Konkret 4.4.2, and many of its chapters only say "See
# Formel Q Konkret. No additional Q requirements". A document follows another's structure when
# documents.yaml says so (``follows_structure_of``) or, with ``structure_links: detect``, when
# its text says so ("follows the ... structure", "structure of ..."); the named document is
# resolved through the unit's references, the registered names or the analysed documents'
# titles (``annotate_structure``). When both are analysed together, every chapter of the delta
# document gets an edge to the base chapter with the same number (``structure_edges``):
#
# * ``inherits_from`` when the chapter only points to the base document (short text with "no
#   additional" or only "See <document>."): the base chapter applies unchanged;
# * ``amends`` otherwise: the chapter adds to or changes the base chapter (a conflict candidate).
#
# A mention with a location ("CVS10, Chapter 4.6") gets a ``references_chapter`` edge to that
# chapter (or the deepest analysed chapter containing it: 4.4.1 for '4.4.1.3') when the
# referenced document is analysed in the same run.

_FOLLOWS_RES = [
    re.compile(r"\bstructure\s+of\s+(?:the\s+)?(?P<name>[^.;:]{3,90}?)(?=[.;:]|\s+(?:and|which|with|is|are)\b|$)", re.IGNORECASE),
    re.compile(r"\bfollow(?:s|ing)?\s+(?:the\s+)?(?P<name>[^.;:]{3,90}?)\s+(?:document\s+|chapter\s+)?structure\b", re.IGNORECASE),
]
_POINTER_MAX_WORDS = 40
_NO_ADDITIONAL_RE = re.compile(r"\bno\s+additional\b", re.IGNORECASE)
_SEE_ONLY_RE = re.compile(r"\s*see\s+[^.]{2,80}\.?\s*", re.IGNORECASE)


def is_pointer(text: str) -> bool:
    """A chapter that only points to the base document ('See Formel Q Konkret. No additional Q requirements')."""
    if len(text.split()) > _POINTER_MAX_WORDS:
        return False
    return bool(_NO_ADDITIONAL_RE.search(text) or _SEE_ONLY_RE.fullmatch(text))


class DocumentIndex:
    """The analysed documents by document-level node ('DOC-CVS10' for 'DOC-CVS10-REV-2024-08'),
    by identifier, and by title/alias (``NameIndex``: the same name rules as documents.yaml)."""

    def __init__(self, documents: list[InputDocument]):
        self.family = {d.document_id: document_family_id(d.canonical_identifier) if d.canonical_identifier else d.document_id for d in documents}
        self.by_family: dict[str, InputDocument] = {}
        self.names = NameIndex()
        for d in documents:
            fam = self.family[d.document_id]
            self.by_family.setdefault(fam, d)
            for name in [d.title, *d.aliases]:
                if name:
                    self.names.add(name, fam)

    def family_of(self, node_id: str) -> str:
        """A document node's document-level node; any other node (a referenced document) maps to itself."""
        return self.family.get(node_id, node_id)

    def by_node(self, node_id: str) -> InputDocument | None:
        return self.by_family.get(self.family_of(node_id))

    def by_identifier(self, identifier: str | None) -> InputDocument | None:
        if not identifier:
            return None
        return self.by_family.get(document_family_id(canonical_input_identifier(identifier)))


def saved_documents(graph: RelationshipGraph | None) -> dict[str, tuple[str, list[str]]]:
    """Documents of a saved graph that a new document may follow: family node -> (identifier, names)."""
    out: dict[str, tuple[str, list[str]]] = {}
    for n in graph.nodes_of_type("document") if graph else []:
        ident = n.attributes.get("canonical_identifier")
        if ident and not n.attributes.get("revision_of"):
            out[n.node_id] = (ident, [x for x in [*n.attributes.get("names", []), n.attributes.get("title")] if x])
    return out


def annotate_structure(
    documents: list[InputDocument],
    units: list[RequirementDraft],
    mentions: list[MentionDraft],
    *,
    mode: str = "detect",
    names: NameIndex | None = None,
    known: dict[str, tuple[str, list[str]]] | None = None,
) -> list[InputDocument]:
    """Copies of ``documents`` with ``follows_structure_of`` resolved (registry or text).

    The base document is one of this run's documents or, with ``known`` (``saved_documents``),
    a document of the saved graph: a delta document can be added after its base agreement.
    """
    if mode == "off":
        return [d.model_copy(update={"follows_structure_of": None, "structure_source": None, "structure_evidence": None}) for d in documents]
    docs = DocumentIndex(documents)
    targets = {fam: ident for fam, (ident, _) in (known or {}).items()} | {fam: d.canonical_identifier for fam, d in docs.by_family.items()}
    titles = docs.names  # this run's titles and aliases, plus the saved graph's document names
    for fam, (_ident, doc_names) in (known or {}).items():
        for n in doc_names:
            titles.add(n, fam)
    by_unit: dict[str, list[MentionDraft]] = {}
    for m in mentions:
        if m.active:
            by_unit.setdefault(m.requirement_id, []).append(m)
    out = []
    for d in documents:
        own = docs.family_of(d.document_id)
        if d.follows_structure_of:  # declared in documents.yaml
            fam = document_family_id(canonical_input_identifier(d.follows_structure_of))
            update = {"structure_source": d.structure_source or "registry"}
            if fam in targets:
                update["follows_structure_of"] = targets[fam]
            out.append(d.model_copy(update=update))
            continue
        found = None
        if mode == "detect":
            for u in (u for u in units if u.document_id == d.document_id and u.unit_kind != "saved"):
                for rx in _FOLLOWS_RES:
                    if not (match := rx.search(u.text)):
                        continue
                    fam = _resolve_named(" ".join(match.group("name").split()), match, u, by_unit.get(u.requirement_id, []), names, titles, set(targets) - {own})
                    if fam is not None:
                        found = (targets[fam], " ".join(match.group(0).split()))
                        break
                if found:
                    break
        if found:
            out.append(d.model_copy(update={"follows_structure_of": found[0], "structure_source": "text", "structure_evidence": found[1]}))
        else:
            out.append(d)
    return out


def _resolve_named(name: str, match: re.Match, unit: RequirementDraft, mentions: list[MentionDraft], names: NameIndex | None, titles: NameIndex, allowed: set[str]) -> str | None:
    """The document (family node) a structure sentence names: a reference the model found in the
    named span, else a registered name, else a document title; only one of ``allowed``."""
    lo, hi = match.start("name"), match.end("name")
    candidates = []
    for m in mentions:
        start = m.evidence.start_offset - (unit.start_offset or 0) if m.evidence.start_offset is not None else None
        raw = name_match_key(m.raw_identifier)
        if (start is not None and lo <= start < hi) or (raw and raw in name_match_key(name)):
            candidates.append(document_family_id(m.canonical) if m.canonical else m.document_id)
    if names and (ident := names.resolve(name)):
        candidates.append(document_family_id(canonical_input_identifier(ident)))
    candidates.append(titles.resolve(name))
    return next((c for c in candidates if c in allowed), None)


def _chapter_target(chapter: str, chapters: dict[str, str]) -> str | None:
    """The unit for a chapter label, or the deepest analysed chapter containing it ('4.4.1.3' -> '4.4.1')."""
    parts = chapter.split(".")
    for n in range(len(parts), 0, -1):
        label = ".".join(parts[:n])
        if label in chapters:
            return chapters[label]
    return None


@dataclass
class StructureEdge:
    source: str  # requirement ID
    target: str  # requirement ID
    kind: str  # inherits_from | amends | references_chapter
    chapter: str
    mention_ids: list[str]


def structure_edges(documents: list[InputDocument], requirements: list) -> list[StructureEdge]:
    """Chapter-alignment edges (delta documents) and chapter-reference edges.

    ``requirements``: exported ``Requirement`` records (with their reference mentions).
    """
    docs = DocumentIndex(documents)
    chapters: dict[str, dict[str, str]] = {}  # family -> chapter -> requirement ID
    for r in requirements:
        if r.chapter:
            chapters.setdefault(docs.family_of(r.source_document_id), {}).setdefault(r.chapter, r.requirement_id)
    edges: list[StructureEdge] = []
    for d in documents:
        base = docs.by_identifier(d.follows_structure_of)
        if base is None:
            continue
        base_chapters = chapters.get(docs.family_of(base.document_id), {})
        for r in requirements:
            if r.source_document_id != d.document_id or not r.chapter or r.chapter not in base_chapters:
                continue
            kind = "inherits_from" if is_pointer(r.requirement_text) else "amends"
            edges.append(StructureEdge(r.requirement_id, base_chapters[r.chapter], kind, r.chapter, []))
    for r in requirements:
        for m in r.reference_mentions:
            if not m.location:
                continue
            target_doc = docs.by_node(m.document_id)
            if target_doc is None:
                continue
            target = _chapter_target(m.location, chapters.get(docs.family_of(target_doc.document_id), {}))
            if target is None or target == r.requirement_id:
                continue
            edges.append(StructureEdge(r.requirement_id, target, "references_chapter", m.location, [m.mention_id]))
    return edges
