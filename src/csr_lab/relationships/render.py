"""src/csr_lab/relationships/render.py

Plain-language Markdown rendering of a relationship graph, for an LLM to read.

The JSON graph is the complete record (validated and queryable). As model input it
is wasteful: braces, keys and opaque hash IDs cost tokens, and the model has to match
edges to nodes itself. This view does that matching: it names documents by identifier
and revision, quotes requirement text, groups statements by document, and keeps
model-generated conflict hypotheses apart from what the documents state.

To measure which rendering works best in conflict prompts, run the
``graph.context_format`` experiments (for example ``graph_context_local``).
"""

from __future__ import annotations

from collections import defaultdict

from csr_lab.relationships.builder import GraphNode, RelationshipGraph
from csr_lab.schemas import RelationshipType

_REQUIREMENT_PHRASES = {
    RelationshipType.REFERENCES: "refers to {t}",
    RelationshipType.EXPLICITLY_EXCLUDES: "states that {t} does not apply",
    RelationshipType.EXPLICITLY_SUPERSEDES: "states that its document supersedes {t}",
    RelationshipType.EXPLICITLY_DEPENDS_ON: "is to be used together with {t}",
    RelationshipType.EXPLICITLY_TAKES_PRECEDENCE_OVER: "states that its document takes precedence over {t}",
}
_SKIP = {RelationshipType.CONTAINS, RelationshipType.HAS_REVISION, RelationshipType.SUPERSEDED_BY, RelationshipType.POTENTIALLY_CONFLICTS_WITH}


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _requirement_text(node: GraphNode) -> str:
    text = node.attributes.get("requirement_text")
    if text:
        return text
    chapter = node.attributes.get("chapter")
    label = node.label
    return label.split(": ", 1)[1] if chapter and label.startswith(f"{chapter}: ") else label


def _one_line(text: str | None) -> str:
    return " ".join((text or "").split())


def _doc_line(graph: RelationshipGraph, node: GraphNode, n_requirements: int) -> str:
    a = node.attributes
    family = graph.node(a["revision_of"]) if a.get("revision_of") else None
    fa = family.attributes if family else {}
    parts = []
    doc_type = a.get("document_type") or fa.get("document_type")
    if doc_type and doc_type not in ("unknown", "other"):
        parts.append(doc_type.replace("_", " "))
    title = _one_line(a.get("title") or a.get("document_name") or fa.get("title") or fa.get("document_name"))
    if title:
        parts.append(f'"{title}"')
    if a.get("analyzed"):
        parts.append(f"analysed, {n_requirements} requirements")
    elif a.get("content_available"):
        parts.append("content provided")
    else:
        parts.append("referenced only, content not provided")
    if a.get("effective_date"):
        parts.append(f"effective {a['effective_date']}")
    if a.get("revision_status") == "superseded" or (family and graph.current_revision(family.node_id) not in (None, node.node_id)):
        parts.append("not the current revision")
    names = {_one_line(n) for n in [*(a.get("names") or []), *(fa.get("names") or [])] if n}
    other_names = sorted(n for n in names if n and n != title)
    if other_names:
        parts.append("also called " + ", ".join(f'"{n}"' for n in other_names[:4]))
    return f"- **{node.label}**: " + "; ".join(parts)


def _family_line(graph: RelationshipGraph, node: GraphNode) -> str | None:
    """A document with several known revisions: their order and the current one."""
    revisions = [r for r in (graph.node(r) for r in graph.revisions_of(node.node_id)) if r]
    if len(revisions) < 2:
        return None
    labels = [r.attributes.get("revision") or r.label for r in revisions]
    known = node.attributes.get("revision_order") not in (None, "unknown")
    order = " → ".join(labels) if known else ", ".join(labels) + " (order unknown)"
    current = graph.node(graph.current_revision(node.node_id) or "")
    return f"- **{node.label}**: revisions {order}" + (f", current {current.attributes.get('revision')}" if current else "")


def to_llm_markdown(
    graph: RelationshipGraph,
    *,
    title: str = "Document relationship graph",
    include_ids: bool = False,
    max_text_chars: int = 240,
    max_quote_chars: int = 200,
) -> str:
    """Markdown summary of the graph: documents, requirement statements, conflict hypotheses.

    ``include_ids`` appends node IDs in backticks so a model can cite them back.
    """
    docs = [n for n in graph.nodes if n.node_type == "document"]
    reqs = [n for n in graph.nodes if n.node_type == "requirement"]
    by_doc: dict[str, list[GraphNode]] = defaultdict(list)
    for r in reqs:
        by_doc[graph.container_of(r.node_id) or "(unknown document)"].append(r)

    def name(node_id: str) -> str:
        node = graph.node(node_id)
        return node.label if node else node_id

    def rid(node: GraphNode) -> str:
        return f" `{node.node_id}`" if include_ids else ""

    def req_ref(node_id: str) -> str:
        node = graph.node(node_id)
        if not node:
            return node_id
        chapter = node.attributes.get("chapter") or "?"
        return f"{name(graph.container_of(node_id) or '')} §{chapter}{rid(node)}"

    conflicts = [e for e in graph.edges if e.relationship_type == RelationshipType.POTENTIALLY_CONFLICTS_WITH]
    stated = [e for e in graph.edges if e.relationship_type not in _SKIP]
    lines = [
        f"# {title}",
        "",
        f"{len(docs)} documents, {len(reqs)} requirements, {len(stated)} stated relationships, "
        f"{len(conflicts)} potential conflicts.",
        "",
        "How to read this: everything under *Documents* and *Requirements* is stated in the documents "
        "(quotes are verbatim). *Potential conflicts* are hypotheses produced by a model; each needs "
        "human review. A document marked 'content not provided' was referenced but never analysed.",
        "",
        "## Documents",
        "",
    ]
    # A document known in revisions is listed per revision; the document itself only gets a
    # line when there are several revisions to put in order.
    for node in sorted(docs, key=lambda n: n.label):
        if graph.revisions_of(node.node_id):
            line = _family_line(graph, node)
            if line:
                lines.append(line + rid(node))
            continue
        lines.append(_doc_line(graph, node, len(by_doc.get(node.node_id, []))) + rid(node))

    lines += ["", "## Requirements and the documents they refer to", ""]
    for doc_id in sorted(by_doc, key=name):
        items = []
        for r in sorted(by_doc[doc_id], key=lambda n: (n.attributes.get("start_offset") or 0, n.attributes.get("chapter") or "")):
            edges = [e for e in graph.out_edges(r.node_id) if e.relationship_type not in _SKIP]
            if not edges:
                continue
            # A qualified statement about X says more than a plain "refers to X".
            qualified = {e.target_node_id for e in edges if e.relationship_type != RelationshipType.REFERENCES}
            phrases = [
                _REQUIREMENT_PHRASES.get(e.relationship_type, e.relationship_type.value + " {t}").format(t=name(e.target_node_id))
                for e in edges
                if not (e.relationship_type == RelationshipType.REFERENCES and e.target_node_id in qualified)
            ]
            status = " (superseded by a newer revision)" if r.attributes.get("revision_status") == "superseded" else ""
            items.append(
                f'- §{r.attributes.get("chapter") or "?"}{rid(r)}{status}: "{_clip(_requirement_text(r), max_text_chars)}"\n'
                f"  → {'; '.join(dict.fromkeys(phrases))}"
            )
        silent = len(by_doc[doc_id]) - len(items)
        if not items and not silent:
            continue
        lines.append(f"### {name(doc_id)}")
        lines += items
        if silent:
            lines.append(f"- ({silent} further requirements refer to no other document)")
        lines.append("")

    lines += ["## Potential conflicts (model hypotheses, require human review)", ""]
    if not conflicts:
        lines.append("None recorded.")
    for e in conflicts:
        kind = (e.metadata.get("conflict_type") or "potential_conflict").replace("_", " ")
        lines.append(f"- {kind}: {req_ref(e.source_node_id)} vs {req_ref(e.target_node_id)}")
        for ev in e.evidence[:2]:
            lines.append(f'  - "{_clip(ev.quoted_text, max_quote_chars)}"')
    return "\n".join(lines).rstrip() + "\n"
