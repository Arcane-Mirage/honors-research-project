"""src/csr_lab/relationships/store.py

A saved (persistent) relationship graph that grows as more documents are analysed.

Each analysis produces a graph for its own document bundle. ``GraphStore``
merges those graphs into one JSON file so knowledge accumulates across runs,
and gives curators tools to inspect, add to and prune it.

Typical use::

    store = GraphStore.load_or_new("outputs/graph_store.json")
    store.merge_result(result)          # an AnalysisResult from any strategy
    store.save()

    store.remove_source("ANL-...")      # undo one analysis
    store.prune(drop_unverified=True)   # keep only verified references
    store.revision_changes("DOC-CVM-CSR-100")

Design rules
------------
* **Provenance.** Every stored node and edge lists the analyses (``sources``)
  that produced it, so anything can be traced or removed by source.
* **Re-analysing a document revision replaces its requirements.** IDs come
  from document revision + chapter + text, so the same requirement keeps its ID;
  a run that reads the revision differently replaces the earlier reading rather
  than mixing with it. (Pass ``replace=False`` to accumulate instead.)
* **Derived edges are recomputed, never merged.** ``has_revision``,
  ``superseded_by`` and document-to-document ``references`` (aggregation) are
  rebuilt by ``refresh()`` after every change, so they cannot go stale.
* **Revisions.** Revisions of a document are ordered by a declared order
  (``set_revision_order``), else by ISO effective dates, else left unordered.
  Consecutive analysed revisions are compared chapter by chapter
  (``csr_lab.relationships.revisions``); requirements of older revisions are
  marked ``superseded``, never deleted.
* **Documents without chapters** keep their passages on the document node
  (``attributes.passages``); what the passages state hangs off the document node
  (edges with ``metadata.passage_ids``). Re-analysing the document replaces both.
* **Conflicts with earlier content.** A graph-assisted run with a saved graph also
  checks its new units against the stored ones (``saved_units``); those conflicts
  carry ``related_saved_ids`` and become edges here (``found_by: saved_graph_check``).
* **Pruning on update.** ``prune_superseded_conflicts`` drops conflicts that involve
  requirements of superseded revisions; the newer revision is checked instead.
* **Storage.** One JSON file, kept apart from the experiment database: the
  store is reusable knowledge, while the SQLite file records experiment runs.

The pipeline can start a graph-assisted run from the store's *documents*
(identifiers, names, types, revisions) via ``document_graph()``; see
``csr_lab.pipelines.run.run_analysis(saved_graph=...)``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from csr_lab.ids import stable_hash
from csr_lab.relationships.builder import (
    MAX_AGGREGATED_EVIDENCE,
    GraphNode,
    RelationshipGraph,
    add_conflict_edge,
    add_revision_structure,
    conflict_edge,
    graph_from_result,
    make_edge,
)
from csr_lab.relationships.revisions import RequirementVersion, RevisionChange, compare_revisions
from csr_lab.schemas import SCHEMA_VERSION, AnalysisResult, Relationship, RelationshipType

STORE_FORMAT_VERSION = "1"

# Edges rebuilt by refresh(); anything merged with these is discarded and recomputed.
_DERIVED_TYPES = {RelationshipType.HAS_REVISION, RelationshipType.SUPERSEDED_BY}
_DERIVED_ORIGINS = {"aggregation", "revision_tracking"}
# Per-run attributes that the store recomputes for its own, larger graph.
_RUN_ONLY_ATTRIBUTES = {"current_revision", "revision_order", "revision_sequence", "provided_revisions"}
_OR_ATTRIBUTES = {"content_available", "analyzed", "is_input_document", "referenced"}


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class StoreSource(BaseModel):
    """One contribution to the store: an analysis result or a manual edit."""

    model_config = ConfigDict(extra="forbid")

    source_id: str
    kind: Literal["analysis", "manual"]
    added_at: str
    case_id: str | None = None
    strategy: str | None = None
    provider_is_mock: bool | None = None
    document_ids: list[str] = Field(default_factory=list, description="Input document revisions it analysed.")
    input_files: list[dict[str, str]] = Field(
        default_factory=list, description="Files it read ({name, sha256} of the file bytes); lets build-graph skip known files."
    )
    model: str | None = None
    note: str = ""


@dataclass
class SavedUnit:
    """A requirement or passage already in the store, offered to new runs for conflict checks."""

    unit_id: str  # requirement node ID, or passage ID
    kind: Literal["chapter", "passage"]
    document_id: str  # the document (revision) node that holds it
    family_id: str
    identifier: str | None
    revision: str | None
    chapter: str | None
    text: str
    start_offset: int | None = None  # in its document's extracted text
    referenced: set[str] = field(default_factory=set)  # document nodes it references (and their families)


class StoreChangeReport(BaseModel):
    """What a store operation changed."""

    nodes_added: int = 0
    nodes_removed: int = 0
    edges_added: int = 0
    edges_removed: int = 0
    requirements_replaced: int = 0
    saved_graph_conflicts_added: int = 0
    superseded_conflicts_pruned: int = 0
    notes: list[str] = Field(default_factory=list)


class GraphStore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    format_version: str = STORE_FORMAT_VERSION
    schema_version: str = SCHEMA_VERSION
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    nodes: dict[str, GraphNode] = Field(default_factory=dict)
    edges: dict[str, Relationship] = Field(default_factory=dict)
    sources: dict[str, StoreSource] = Field(default_factory=dict)
    revision_orders: dict[str, list[str]] = Field(
        default_factory=dict, description="Declared revision order per document node, oldest first."
    )

    # -- files -------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> GraphStore:
        store = cls.model_validate_json(Path(path).read_text(encoding="utf-8"))
        if store.format_version != STORE_FORMAT_VERSION:
            raise ValueError(f"{path}: store format {store.format_version}, expected {STORE_FORMAT_VERSION}")
        return store

    @classmethod
    def load_or_new(cls, path: str | Path) -> GraphStore:
        return cls.load(path) if Path(path).is_file() else cls()

    def save(self, path: str | Path) -> Path:
        """Write atomically (a crash never leaves a half-written file). Keys are sorted for clean diffs."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        data = self.model_dump(mode="json")
        data["nodes"] = dict(sorted(data["nodes"].items()))
        data["edges"] = dict(sorted(data["edges"].items()))
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
        return path

    def fingerprint(self) -> str:
        """Content hash of nodes and edges (timestamps excluded)."""
        payload = json.dumps(
            {
                "nodes": {k: v.model_dump(mode="json") for k, v in sorted(self.nodes.items())},
                "edges": sorted(self.edges),
                "orders": self.revision_orders,
            },
            sort_keys=True,
        )
        return stable_hash(payload, length=16)

    # -- reading -----------------------------------------------------------

    def to_graph(self, include_superseded: bool = True) -> RelationshipGraph:
        """The whole store as a ``RelationshipGraph`` (all query helpers work on it)."""
        nodes = [
            n
            for n in self.nodes.values()
            if include_superseded or n.attributes.get("revision_status") != "superseded"
        ]
        keep = {n.node_id for n in nodes}
        edges = [e for e in self.edges.values() if e.source_node_id in keep and e.target_node_id in keep]
        return RelationshipGraph(nodes=nodes, edges=edges)

    def document_graph(self) -> RelationshipGraph:
        """Only document nodes and ``has_revision`` edges: what a new run may reuse.

        Requirements are left out because their text is not part of the new run's
        sources. Content is marked unavailable: it is not provided to the new run.
        """
        nodes = []
        for n in self.nodes.values():
            if n.node_type != "document":
                continue
            attrs = {k: v for k, v in n.attributes.items() if k != "sources"}
            attrs.update(content_available=False, is_input_document=False, analyzed=False, from_saved_graph=True)
            nodes.append(GraphNode(node_id=n.node_id, node_type="document", label=n.label, attributes=attrs))
        keep = {n.node_id for n in nodes}
        edges = [
            e
            for e in self.edges.values()
            if e.relationship_type == RelationshipType.HAS_REVISION and e.source_node_id in keep and e.target_node_id in keep
        ]
        return RelationshipGraph(nodes=nodes, edges=edges)

    def saved_units(self, exclude_families: set[str] | None = None) -> dict[str, SavedUnit]:
        """Current requirements and passages, except those of documents in ``exclude_families``
        (the documents a new run analyses itself)."""
        exclude = exclude_families or set()

        def family(node_id: str) -> str:
            node = self.nodes.get(node_id)
            return (node.attributes.get("revision_of") if node else None) or node_id

        refs: dict[str, set[str]] = {}
        for e in self.edges.values():
            if e.relationship_type != RelationshipType.REFERENCES or e.metadata.get("origin") != "extraction":
                continue
            for unit in e.metadata.get("passage_ids") or [e.source_node_id]:
                refs.setdefault(unit, set()).update({e.target_node_id, family(e.target_node_id)})
        container = {e.target_node_id: e.source_node_id for e in self.edges.values() if e.relationship_type == RelationshipType.CONTAINS}
        out: dict[str, SavedUnit] = {}
        for k, n in self.nodes.items():
            if n.node_type == "requirement":
                doc_id = container.get(k)
                if doc_id is None or family(doc_id) in exclude or n.attributes.get("revision_status") == "superseded":
                    continue
                kind, text, chapter = "chapter", n.attributes.get("requirement_text"), n.attributes.get("chapter")
                units = [(k, None, n.attributes.get("start_offset"))]
            elif n.node_type == "document" and n.attributes.get("passages") and family(k) not in exclude:
                doc_id, kind, chapter = k, "passage", None
                units = [(p["passage_id"], p.get("text"), p.get("start_offset")) for p in n.attributes["passages"]]
                text = None
            else:
                continue
            doc = self.nodes.get(doc_id)
            attrs = doc.attributes if doc else {}
            for unit_id, unit_text, start in units:
                body = text if kind == "chapter" else unit_text
                if not body:
                    continue
                out[unit_id] = SavedUnit(
                    unit_id=unit_id,
                    kind=kind,  # type: ignore[arg-type]
                    document_id=doc_id,
                    family_id=family(doc_id),
                    identifier=attrs.get("canonical_identifier"),
                    revision=attrs.get("revision"),
                    chapter=chapter,
                    text=body,
                    start_offset=start,
                    referenced=refs.get(unit_id, set()),
                )
        return out

    def unit_node(self, unit_id: str) -> str | None:
        """Graph node of a stored unit: the requirement itself, or the document holding a passage."""
        if unit_id in self.nodes:
            return unit_id
        for k, n in self.nodes.items():
            if any(p.get("passage_id") == unit_id for p in n.attributes.get("passages") or []):
                return k
        return None

    def known_file_hashes(self) -> set[str]:
        """sha256 of every file an analysis in the store has read."""
        return {f["sha256"] for s in self.sources.values() for f in s.input_files if f.get("sha256")}

    def summary(self) -> dict[str, Any]:
        docs = [n for n in self.nodes.values() if n.node_type == "document"]
        reqs = [n for n in self.nodes.values() if n.node_type == "requirement"]
        statuses: dict[str, int] = {}
        for r in reqs:
            key = r.attributes.get("revision_status") or "current"
            statuses[key] = statuses.get(key, 0) + 1
        edge_types: dict[str, int] = {}
        for e in self.edges.values():
            edge_types[e.relationship_type.value] = edge_types.get(e.relationship_type.value, 0) + 1
        unordered = sorted(
            n.node_id
            for n in docs
            if n.attributes.get("revision_order") == "unknown"
            and sum(1 for r in self._revisions(n.node_id) if self.nodes[r].attributes.get("analyzed")) > 1
        )
        return {
            "fingerprint": self.fingerprint(),
            "documents": len(docs),
            "analysed_document_revisions": sum(1 for n in docs if n.attributes.get("analyzed")),
            "requirements": len(reqs),
            "requirements_by_revision_status": statuses,
            "edges_by_type": dict(sorted(edge_types.items())),
            "sources": len(self.sources),
            "mock_sources": sum(1 for s in self.sources.values() if s.provider_is_mock),
            "documents_needing_revision_order": unordered,
        }

    def revision_changes(self, document_id: str) -> list[dict[str, Any]]:
        """Changes between each pair of consecutive analysed revisions of a document."""
        out = []
        revisions = [r for r in self._revisions(document_id) if self.nodes[r].attributes.get("analyzed")]
        for older, newer in pairwise(revisions):
            for change in self._compare(older, newer):
                out.append({"from_revision": older, "to_revision": newer, **change.as_dict()})
        return out

    # -- adding ------------------------------------------------------------

    def merge_result(
        self,
        result: AnalysisResult,
        *,
        source_id: str | None = None,
        include_conflicts: bool = True,
        replace: bool = True,
        note: str = "",
        input_files: list[dict[str, str]] | None = None,
        prune_superseded_conflicts: bool = False,
    ) -> StoreChangeReport:
        """Add one analysis result. See the module docstring for the rules.

        ``input_files``: ``{name, sha256}`` of the files read; defaults to the result's input documents.
        ``prune_superseded_conflicts``: afterwards drop conflicts with requirements of superseded revisions.
        """
        sid = source_id or result.analysis_id
        report = StoreChangeReport()
        if sid in self.sources:
            report.notes.append(f"source {sid} was already merged; its contribution was replaced")
            self._drop_source(sid, report)
        graph = graph_from_result(result)
        analysed = [d.document_id for d in result.input_documents if d.parse_status in ("ok", "partial")]

        if replace:
            for doc_id in analysed:
                old_reqs = [e.target_node_id for e in self.edges.values() if e.source_node_id == doc_id and e.relationship_type == RelationshipType.CONTAINS]
                if old_reqs:
                    report.requirements_replaced += len(old_reqs)
                    self._remove_nodes(set(old_reqs), report)
                # An earlier reading of a document without chapters: its passages and what they stated.
                for rid, e in list(self.edges.items()):
                    if e.metadata.get("passage_ids") and doc_id in (e.source_node_id, e.target_node_id):
                        del self.edges[rid]
                        report.edges_removed += 1
                if doc_id in self.nodes:
                    for key in ("passages", "has_chapters"):
                        self.nodes[doc_id].attributes.pop(key, None)

        req_data = {r.requirement_id: r for r in result.requirements}
        for node in graph.nodes:
            attrs = {k: v for k, v in node.attributes.items() if k not in _RUN_ONLY_ATTRIBUTES}
            if node.node_type == "requirement" and node.node_id in req_data:
                r = req_data[node.node_id]
                doc = graph.node(r.source_document_id)
                attrs.update(
                    requirement_text=r.requirement_text,
                    start_offset=r.source_location.start_offset,
                    revision=doc.attributes.get("revision") if doc else None,
                    document_family_id=graph.family_of(r.source_document_id),
                )
            existing = self.nodes.get(node.node_id)
            if existing is None:
                self.nodes[node.node_id] = GraphNode(
                    node_id=node.node_id, node_type=node.node_type, label=node.label, attributes={**attrs, "sources": [sid]}
                )
                report.nodes_added += 1
            else:
                _merge_attributes(existing.attributes, attrs)
                existing.attributes["sources"] = list(dict.fromkeys([*existing.attributes.get("sources", []), sid]))

        for edge in graph.edges:
            if edge.relationship_type in _DERIVED_TYPES or edge.metadata.get("origin") in _DERIVED_ORIGINS:
                continue
            if not include_conflicts and edge.relationship_type == RelationshipType.POTENTIALLY_CONFLICTS_WITH:
                continue
            existing = self.edges.get(edge.relationship_id)
            if existing is None:
                self.edges[edge.relationship_id] = edge.model_copy(deep=True, update={"metadata": {**edge.metadata, "sources": [sid]}})
                report.edges_added += 1
            else:
                existing.metadata["sources"] = list(dict.fromkeys([*existing.metadata.get("sources", []), sid]))

        if include_conflicts:
            report.saved_graph_conflicts_added = self._add_saved_conflicts(result, sid)

        self.sources[sid] = StoreSource(
            source_id=sid,
            kind="analysis",
            added_at=_now(),
            case_id=result.metadata.case_id,
            strategy=result.metadata.strategy,
            provider_is_mock=result.metadata.provider_is_mock,
            document_ids=analysed,
            input_files=input_files
            if input_files is not None
            else [{"name": d.filename, "sha256": d.sha256} for d in result.input_documents],
            model=(result.metadata.provider or {}).get("model"),
            note=note,
        )
        if result.metadata.provider_is_mock:
            report.notes.append("this result came from the offline mock model")
        self.refresh()
        if prune_superseded_conflicts:
            report.superseded_conflicts_pruned = self.prune_superseded_conflicts()
            if report.superseded_conflicts_pruned:
                report.notes.append(f"{report.superseded_conflicts_pruned} conflicts with superseded requirements were dropped")
        return report

    def _add_saved_conflicts(self, result: AnalysisResult, sid: str) -> int:
        """Edges for conflicts between the new units and units already in the store."""
        owner = {p.passage_id: p.source_document_id for p in result.document_passages}
        added = 0
        for c in result.potential_conflicts:
            if not c.related_saved_ids:
                continue
            new_side = [owner.get(u, u) for u in [*c.related_requirement_ids, *c.related_passage_ids]]
            saved_side = [self.unit_node(u) for u in c.related_saved_ids]
            ends = [n for n in [*new_side, *saved_side] if n and n in self.nodes]
            if len(ends) < 2:
                continue
            passages = [u for u in [*c.related_passage_ids, *c.related_saved_ids] if u not in self.nodes]
            edge = conflict_edge(ends[0], ends[1], c, passages, found_by="saved_graph_check", sources=[sid])
            added += add_conflict_edge(self.edges, edge)
        return added

    def prune_superseded_conflicts(self) -> int:
        """Drop conflict hypotheses that involve a requirement of a superseded revision."""
        stale = [
            rid
            for rid, e in self.edges.items()
            if e.relationship_type == RelationshipType.POTENTIALLY_CONFLICTS_WITH
            and any(self.nodes.get(n) and self.nodes[n].attributes.get("revision_status") == "superseded" for n in (e.source_node_id, e.target_node_id))
        ]
        for rid in stale:
            del self.edges[rid]
        return len(stale)

    def add_node(self, node: GraphNode, *, source_id: str = "manual", note: str = "") -> None:
        """Add or update a node by hand (for example a document the pipeline cannot see)."""
        self._manual_source(source_id, note)
        existing = self.nodes.get(node.node_id)
        if existing is None:
            self.nodes[node.node_id] = node.model_copy(deep=True, update={"attributes": {**node.attributes, "sources": [source_id]}})
        else:
            _merge_attributes(existing.attributes, node.attributes)
            existing.attributes["sources"] = list(dict.fromkeys([*existing.attributes.get("sources", []), source_id]))
        self.refresh()

    def add_edge(self, edge: Relationship, *, source_id: str = "manual", note: str = "") -> None:
        """Add an edge by hand. Both endpoints must already exist; ``origin`` defaults to 'manual'."""
        for end in (edge.source_node_id, edge.target_node_id):
            if end not in self.nodes:
                raise KeyError(f"unknown node {end}")
        if edge.relationship_type in _DERIVED_TYPES:
            raise ValueError(f"{edge.relationship_type.value} edges are derived; declare a revision order instead")
        self._manual_source(source_id, note)
        metadata = {"origin": "manual", **edge.metadata, "sources": [source_id]}
        self.edges[edge.relationship_id] = edge.model_copy(deep=True, update={"metadata": metadata})
        self.refresh()

    def set_revision_order(self, document_id: str, revisions: list[str]) -> None:
        """Declare the order of a document's revisions, oldest first (labels such as ['C', 'D'])."""
        if document_id not in self.nodes:
            raise KeyError(f"unknown document {document_id}")
        self.revision_orders[document_id] = [r.strip().upper() for r in revisions if r.strip()]
        self.refresh()

    # -- removing ----------------------------------------------------------

    def remove_source(self, source_id: str) -> StoreChangeReport:
        """Undo one contribution: everything that only it supported is removed."""
        if source_id not in self.sources:
            raise KeyError(f"unknown source {source_id}")
        report = StoreChangeReport()
        self._drop_source(source_id, report)
        self.refresh()
        return report

    def remove_document(self, document_id: str) -> StoreChangeReport:
        """Remove a document revision with its requirements; for a document node, all its revisions too."""
        if document_id not in self.nodes:
            raise KeyError(f"unknown document {document_id}")
        targets = {document_id, *self._revisions(document_id)}
        reqs = {e.target_node_id for e in self.edges.values() if e.source_node_id in targets and e.relationship_type == RelationshipType.CONTAINS}
        report = StoreChangeReport()
        self._remove_nodes(targets | reqs, report)
        self.revision_orders.pop(document_id, None)
        self.refresh()
        return report

    def remove_nodes(self, node_ids: list[str]) -> StoreChangeReport:
        report = StoreChangeReport()
        self._remove_nodes(set(node_ids) & set(self.nodes), report)
        self.refresh()
        return report

    def remove_edges(self, relationship_ids: list[str]) -> StoreChangeReport:
        report = StoreChangeReport()
        for rid in relationship_ids:
            if self.edges.pop(rid, None) is not None:
                report.edges_removed += 1
        self.refresh()
        return report

    def prune(
        self,
        *,
        drop_superseded: bool = False,
        drop_unverified: bool = False,
        drop_conflicts: bool = False,
        drop_mock: bool = False,
        min_sources: int = 1,
    ) -> StoreChangeReport:
        """Remove low-confidence or outdated content. Orphaned referenced documents go too.

        * ``drop_superseded``: requirements of older revisions.
        * ``drop_unverified``: references that did not pass verification.
        * ``drop_conflicts``: all ``potentially_conflicts_with`` hypotheses.
        * ``drop_mock``: every contribution from the offline mock model.
        * ``min_sources``: references and conflicts seen in fewer analyses
          (only meaningful when results were merged with ``replace=False``).
        """
        report = StoreChangeReport()
        if drop_mock:
            for sid in [s for s, src in self.sources.items() if src.provider_is_mock]:
                self._drop_source(sid, report)
        if drop_superseded:
            self._remove_nodes({k for k, n in self.nodes.items() if n.attributes.get("revision_status") == "superseded"}, report)
        dropped_mentions: set[str] = set()
        for rid, e in list(self.edges.items()):
            origin = e.metadata.get("origin")
            remove = (
                (drop_unverified and e.relationship_type == RelationshipType.REFERENCES and origin == "extraction" and not e.metadata.get("verified"))
                or (drop_conflicts and e.relationship_type == RelationshipType.POTENTIALLY_CONFLICTS_WITH)
                or (origin in ("extraction", "analysis") and len(e.metadata.get("sources", [])) < min_sources)
            )
            if remove:
                dropped_mentions.update(e.metadata.get("mention_ids", []))
                del self.edges[rid]
                report.edges_removed += 1
        # Explicit relation edges (excludes, precedence, ...) go with the reference they qualify.
        for rid, e in list(self.edges.items()):
            ids = e.metadata.get("mention_ids", [])
            if e.relationship_type != RelationshipType.REFERENCES and ids and set(ids) <= dropped_mentions:
                del self.edges[rid]
                report.edges_removed += 1
        self.refresh()
        orphans = {
            k
            for k, n in self.nodes.items()
            if n.node_type == "document"
            and not n.attributes.get("analyzed")
            and not any(k in (e.source_node_id, e.target_node_id) for e in self.edges.values() if e.relationship_type not in _DERIVED_TYPES)
            and not self._revisions(k)
        }
        if orphans:
            self._remove_nodes(orphans, report)
            self.refresh()
        return report

    # -- derived structure -------------------------------------------------

    def refresh(self) -> None:
        """Rebuild derived edges and revision status. Called after every change."""
        self.edges = {
            k: e
            for k, e in self.edges.items()
            if e.relationship_type not in _DERIVED_TYPES
            and e.metadata.get("origin") not in _DERIVED_ORIGINS
            and e.source_node_id in self.nodes
            and e.target_node_id in self.nodes
        }
        for node in self.nodes.values():
            for key in (*_RUN_ONLY_ATTRIBUTES, "revision_status"):
                node.attributes.pop(key, None)
        self._rebuild_aggregation()
        add_revision_structure(self.nodes, self.edges, self.revision_orders)
        self._rebuild_supersession()

    def _rebuild_aggregation(self) -> None:
        container = {e.target_node_id: e.source_node_id for e in self.edges.values() if e.relationship_type == RelationshipType.CONTAINS}

        def family(node_id: str) -> str:
            node = self.nodes.get(node_id)
            return (node.attributes.get("revision_of") if node else None) or node_id

        grouped: dict[tuple[str, str], list[tuple[str, Any]]] = {}
        for e in self.edges.values():
            if e.relationship_type != RelationshipType.REFERENCES or e.metadata.get("origin") != "extraction":
                continue
            src = container.get(e.source_node_id)
            if src is None or family(src) == family(e.target_node_id):
                continue
            grouped.setdefault((src, e.target_node_id), []).extend((e.source_node_id, ev) for ev in e.evidence)
        for (src, tgt), items in sorted(grouped.items()):
            edge = make_edge(
                src,
                RelationshipType.REFERENCES,
                tgt,
                [ev for _, ev in items[:MAX_AGGREGATED_EVIDENCE]],
                origin="aggregation",
                derived=True,
                via_requirement_ids=sorted({rid for rid, _ in items}),
            )
            self.edges[edge.relationship_id] = edge

    def _rebuild_supersession(self) -> None:
        in_family: set[str] = set()
        for family, node in [(k, n) for k, n in self.nodes.items() if n.node_type == "document"]:
            revisions = [r for r in self._revisions(family) if self.nodes[r].attributes.get("analyzed")]
            if not revisions:
                continue
            ordered = node.attributes.get("revision_order") != "unknown" or len(revisions) == 1
            current = node.attributes.get("current_revision")
            for rev in revisions:
                status = ("current" if rev == current else "superseded") if ordered else "undetermined"
                for req in self._requirements(rev):
                    self.nodes[req].attributes["revision_status"] = status
                    in_family.add(req)
            if not ordered:
                continue
            for older, newer in pairwise(revisions):
                edge = make_edge(older, RelationshipType.SUPERSEDED_BY, newer, [], origin="revision_tracking")
                self.edges[edge.relationship_id] = edge
                for change in self._compare(older, newer):
                    if change.old and change.new:
                        edge = make_edge(
                            change.old.requirement_id,
                            RelationshipType.SUPERSEDED_BY,
                            change.new.requirement_id,
                            [],
                            origin="revision_tracking",
                            change=change.change,
                        )
                        self.edges[edge.relationship_id] = edge
        for k, n in self.nodes.items():
            if n.node_type == "requirement" and k not in in_family:
                n.attributes["revision_status"] = "current"

    # -- helpers -----------------------------------------------------------

    def _revisions(self, document_id: str) -> list[str]:
        edges = [e for e in self.edges.values() if e.source_node_id == document_id and e.relationship_type == RelationshipType.HAS_REVISION]
        edges.sort(key=lambda e: (e.metadata.get("sequence") is None, e.metadata.get("sequence") or 0, e.target_node_id))
        return [e.target_node_id for e in edges]

    def _requirements(self, document_id: str) -> list[str]:
        reqs = [e.target_node_id for e in self.edges.values() if e.source_node_id == document_id and e.relationship_type == RelationshipType.CONTAINS]

        def order(r: str) -> tuple[bool, int, str]:
            offset = self.nodes[r].attributes.get("start_offset")
            return offset is None, offset or 0, r

        return sorted(reqs, key=order)

    def _versions(self, document_id: str) -> list[RequirementVersion]:
        return [
            RequirementVersion(r, self.nodes[r].attributes.get("chapter"), self.nodes[r].attributes.get("requirement_text") or "")
            for r in self._requirements(document_id)
        ]

    def _compare(self, older: str, newer: str) -> list[RevisionChange]:
        return compare_revisions(self._versions(older), self._versions(newer))

    def _remove_nodes(self, node_ids: set[str], report: StoreChangeReport) -> None:
        for k in node_ids:
            if self.nodes.pop(k, None) is not None:
                report.nodes_removed += 1
        for rid, e in list(self.edges.items()):
            if e.source_node_id in node_ids or e.target_node_id in node_ids:
                del self.edges[rid]
                report.edges_removed += 1

    def _drop_source(self, source_id: str, report: StoreChangeReport) -> None:
        gone: set[str] = set()
        for k, n in self.nodes.items():
            srcs = n.attributes.get("sources", [])
            if source_id in srcs:
                n.attributes["sources"] = [s for s in srcs if s != source_id]
                if not n.attributes["sources"]:
                    gone.add(k)
        for rid, e in list(self.edges.items()):
            srcs = e.metadata.get("sources", [])
            if source_id in srcs:
                e.metadata["sources"] = [s for s in srcs if s != source_id]
                if not e.metadata["sources"]:
                    del self.edges[rid]
                    report.edges_removed += 1
        self._remove_nodes(gone, report)
        self.sources.pop(source_id, None)

    def _manual_source(self, source_id: str, note: str) -> None:
        if source_id not in self.sources:
            self.sources[source_id] = StoreSource(source_id=source_id, kind="manual", added_at=_now(), note=note)


def _merge_attributes(existing: dict[str, Any], new: dict[str, Any]) -> None:
    """Names are unioned, flags OR-ed, and other values filled in when missing."""
    for key, value in new.items():
        if key == "sources":
            continue
        if key == "names":
            existing["names"] = list(dict.fromkeys([*existing.get("names", []), *(value or [])]))
        elif key in _OR_ATTRIBUTES:
            existing[key] = bool(existing.get(key)) or bool(value)
        elif key in ("requirement_text", "start_offset", "chapter", "passages", "has_chapters"):
            existing[key] = value  # a unit's own data: latest reading wins
        elif existing.get(key) in (None, "", "unknown", []):
            existing[key] = value


def with_saved_documents(graph: RelationshipGraph, saved: RelationshipGraph | None) -> RelationshipGraph:
    """The current run's graph plus documents known from a saved graph.

    Saved nodes add names and identifiers only; current-run nodes keep their own
    availability. Used mid-pipeline by graph-assisted runs; the exported result
    is built from the run's own predictions and never contains saved nodes.
    """
    if saved is None:
        return graph
    nodes = {n.node_id: n.model_copy(deep=True) for n in graph.nodes}
    for n in saved.nodes:
        if n.node_id in nodes:
            target = nodes[n.node_id].attributes
            target["names"] = list(dict.fromkeys([*target.get("names", []), *n.attributes.get("names", [])]))
            for key in ("document_type", "document_name", "title"):
                if target.get(key) in (None, "unknown") and n.attributes.get(key):
                    target[key] = n.attributes[key]
        else:
            nodes[n.node_id] = n.model_copy(deep=True)
    edges = {e.relationship_id: e for e in graph.edges if e.relationship_type != RelationshipType.HAS_REVISION}
    add_revision_structure(nodes, edges)
    return RelationshipGraph(nodes=list(nodes.values()), edges=list(edges.values()))
