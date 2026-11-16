"""src/csr_lab/relationships/issuers.py

Issuer groups read off a relationship graph: which documents each issuer published, and which
documents of one issuer refer to documents of another (cross-group references).

Issuers are stored on document nodes (``attributes.issuer``, set by ``extraction/issuers.py`` and
kept by the saved graph store); a revision without one takes its document's. Groups are derived,
never stored, so they work the same on a run's graph, the saved graph and an exported result.
Cross-group links come from document-to-document ``references`` edges (aggregated from
requirements, or stated by passages), counted per distinct (citing, cited) document pair.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from csr_lab.relationships.builder import RelationshipGraph
from csr_lab.schemas import RelationshipType

UNKNOWN_ISSUER = "unknown"
MAX_EXAMPLES = 5


def node_issuer(graph: RelationshipGraph, node_id: str) -> str | None:
    node = graph.node(node_id)
    if node is None or node.node_type != "document":
        return None
    if issuer := node.attributes.get("issuer"):
        return issuer
    family = graph.node(graph.family_of(node_id))
    return family.attributes.get("issuer") if family is not None else None


@dataclass
class IssuerSummary:
    groups: dict[str | None, list[str]] = field(default_factory=dict)  # issuer (None: unknown) -> family node IDs
    analysed: dict[str | None, list[str]] = field(default_factory=dict)  # ... of which have analysed content
    links: dict[tuple[str, str], list[tuple[str, str]]] = field(default_factory=dict)  # (from, to) -> document pairs

    @property
    def cross_links(self) -> dict[tuple[str, str], list[tuple[str, str]]]:
        return {k: v for k, v in self.links.items() if k[0] != k[1]}

    @property
    def has_issuers(self) -> bool:
        return any(k is not None for k in self.groups)


def summarize_issuers(graph: RelationshipGraph) -> IssuerSummary:
    groups: dict[str | None, set[str]] = defaultdict(set)
    analysed: dict[str | None, set[str]] = defaultdict(set)
    for n in graph.nodes:
        if n.node_type != "document":
            continue
        family = graph.family_of(n.node_id)
        issuer = node_issuer(graph, n.node_id)
        groups[issuer].add(family)
        if n.attributes.get("analyzed") or n.attributes.get("is_input_document"):
            analysed[issuer].add(family)
    # A family listed under an issuer is not also unknown (a revision node without the attribute).
    known = set().union(*(v for k, v in groups.items() if k is not None)) if groups else set()
    if None in groups:
        groups[None] -= known
        analysed[None] -= known
        if not groups[None]:
            del groups[None]
    links: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for e in graph.edges:
        if e.relationship_type != RelationshipType.REFERENCES:
            continue
        src, tgt = graph.node(e.source_node_id), graph.node(e.target_node_id)
        if src is None or tgt is None or src.node_type != "document" or tgt.node_type != "document":
            continue
        a, b = graph.family_of(src.node_id), graph.family_of(tgt.node_id)
        if a == b:
            continue
        key = (node_issuer(graph, src.node_id) or UNKNOWN_ISSUER, node_issuer(graph, tgt.node_id) or UNKNOWN_ISSUER)
        links[key].add((a, b))
    return IssuerSummary(
        groups={k: sorted(v) for k, v in groups.items()},
        analysed={k: sorted(v) for k, v in analysed.items() if v},
        links={k: sorted(v) for k, v in sorted(links.items())},
    )
