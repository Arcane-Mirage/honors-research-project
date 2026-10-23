"""Graph-informed reference audit (used only by graph-assisted strategies).

Two uses of the relationship graph:

1. **Alias resolution.** A mention that names a document without an identifier
   ("the Corvane Supplier Quality Manual") is resolved to a document node when
   that exact name is attached to an identifier elsewhere in the graph (another
   requirement's evidence, or the title of a provided document).
2. **Known-document sweep.** Each requirement is searched for identifiers and
   names of documents already in the graph that it is not linked to. Hits become
   ``known_document`` candidates for a targeted model check, with the graph
   context (which other requirements reference that document) in the prompt.

Each use is recorded as a ``GraphDecision`` so its effect can be analysed.
"""

from __future__ import annotations

import re

from csr_lab.drafts import MentionDraft, RequirementDraft
from csr_lab.extraction.normalization import normalize_name
from csr_lab.extraction.references import identifier_pattern
from csr_lab.extraction.verification import AuditCandidate
from csr_lab.relationships.builder import RelationshipGraph
from csr_lab.relationships.queries import requirements_referencing
from csr_lab.schemas import GraphDecision, ResolutionMethod, VerificationEvent

_MIN_NAME_WORDS = 3


def build_alias_index(graph: RelationshipGraph) -> dict[str, list[tuple[str, str | None, str]]]:
    """normalised name -> [(canonical identifier, revision, node id)] for nodes with identifiers."""
    index: dict[str, list[tuple[str, str | None, str]]] = {}
    for node in graph.nodes_of_type("document"):
        canonical = node.attributes.get("canonical_identifier")
        if not canonical:
            continue
        names = [*node.attributes.get("names", []), node.attributes.get("document_name"), node.attributes.get("title")]
        for name in {n for n in names if n}:
            key = normalize_name(name)
            if len(key.split()) < _MIN_NAME_WORDS:
                continue
            entry = (canonical, node.attributes.get("revision") if not node.attributes.get("is_input_document") else None, node.node_id)
            if entry not in index.setdefault(key, []):
                index[key].append(entry)
    return index


def _lookup(name: str, index: dict[str, list[tuple[str, str | None, str]]]) -> list[tuple[str, str | None, str]]:
    key = normalize_name(name)
    if key in index:
        return index[key]
    # Allow one name to be a word-suffix of the other ("Supplier Quality Manual" vs
    # "Corvane Supplier Quality Manual") when the shorter has enough words.
    hits = []
    for other, entries in index.items():
        short, long_ = (key, other) if len(key) <= len(other) else (other, key)
        if len(short.split()) >= _MIN_NAME_WORDS and long_.endswith(" " + short):
            hits.extend(entries)
    return list(dict.fromkeys(hits))


def resolve_aliases(
    mentions: list[MentionDraft], graph: RelationshipGraph, attempt: int
) -> tuple[list[VerificationEvent], list[GraphDecision]]:
    index = build_alias_index(graph)
    events: list[VerificationEvent] = []
    decisions: list[GraphDecision] = []
    for m in mentions:
        if not m.active or m.canonical is not None or not m.name:
            continue
        hits = _lookup(m.name, index)
        if len(hits) != 1:
            if len(hits) > 1:
                events.append(
                    VerificationEvent(
                        requirement_id=m.requirement_id,
                        raw_identifier=m.raw_identifier,
                        action="candidate_unresolved",
                        reason=f"name matches several documents: {[h[2] for h in hits]}",
                        attempt=attempt,
                        mention_id=m.mention_id,
                    )
                )
            continue
        canonical, revision, node_id = hits[0]
        m.canonical, m.revision, m.resolution = canonical, revision, ResolutionMethod.NAME_ALIAS
        events.append(
            VerificationEvent(
                requirement_id=m.requirement_id,
                raw_identifier=m.raw_identifier,
                action="resolved_by_alias",
                reason=f"name is used for {canonical} elsewhere in the graph",
                attempt=attempt,
                mention_id=m.mention_id,
            )
        )
        decisions.append(
            GraphDecision(
                decision_type="alias_resolved",
                detail=f"'{m.name}' in {m.requirement_id} resolved to {canonical}",
                node_ids=[m.requirement_id, node_id],
            )
        )
    # Resolution can create duplicates inside a requirement; merge them.
    seen: dict[tuple[str, str], MentionDraft] = {}
    for m in mentions:
        if not m.active:
            continue
        key = (m.requirement_id, m.key)
        if key in seen:
            m.active, m.inactive_reason = False, f"duplicate of {seen[key].mention_id} after alias resolution"
            events.append(
                VerificationEvent(
                    requirement_id=m.requirement_id,
                    raw_identifier=m.raw_identifier,
                    action="merged_duplicate",
                    reason=m.inactive_reason,
                    attempt=attempt,
                    mention_id=m.mention_id,
                )
            )
        else:
            seen[key] = m
    return events, decisions


def known_document_candidates(
    requirements: list[RequirementDraft],
    mentions: list[MentionDraft],
    graph: RelationshipGraph,
    *,
    examined: set[str],
    max_per_requirement: int,
    attempt: int,
) -> tuple[list[AuditCandidate], list[GraphDecision]]:
    linked: dict[str, set[str]] = {}
    linked_identifiers: dict[str, set[str]] = {}
    for m in mentions:
        if m.active:
            linked.setdefault(m.requirement_id, set()).add(m.document_id)
            if m.canonical:
                linked_identifiers.setdefault(m.requirement_id, set()).add(m.canonical)
    documents = [n for n in graph.nodes_of_type("document") if n.attributes.get("canonical_identifier")]
    candidates: list[AuditCandidate] = []
    decisions: list[GraphDecision] = []
    for req in requirements:
        per_req = 0
        for node in documents:
            canonical = node.attributes["canonical_identifier"]
            if node.node_id in linked.get(req.requirement_id, set()) or node.node_id == req.document_id:
                continue
            if canonical in linked_identifiers.get(req.requirement_id, set()):
                continue  # already linked to this identifier (possibly another revision)
            hit = identifier_pattern(canonical).search(req.text)
            names = [n for n in {*node.attributes.get("names", []), node.attributes.get("document_name")} if n]
            if hit is None:
                for name in names:
                    if len(name.split()) < _MIN_NAME_WORDS:
                        continue
                    hit = re.search(r"\s+".join(map(re.escape, name.split())), req.text, re.IGNORECASE)
                    if hit:
                        break
            if hit is None:
                continue
            key = f"graph:{req.requirement_id}:{node.node_id}"
            if key in examined or per_req >= max_per_requirement:
                continue
            per_req += 1
            referenced_by = requirements_referencing(graph, node.node_id)[:3]
            candidates.append(
                AuditCandidate(
                    candidate_id=f"G{attempt}-{len(candidates) + 1:03d}",
                    requirement_id=req.requirement_id,
                    text=hit.group(0),
                    kind="known_document",
                    start=hit.start(),
                    end=hit.end(),
                    key=key,
                    related=[
                        {
                            "document_id": node.node_id,
                            "identifier": canonical,
                            "names": names,
                            "referenced_by_requirements": referenced_by,
                        }
                    ],
                )
            )
            decisions.append(
                GraphDecision(
                    decision_type="audit_candidate",
                    detail=f"'{hit.group(0)}' in {req.requirement_id} matches graph document {canonical}",
                    node_ids=[req.requirement_id, node.node_id],
                )
            )
    return candidates, decisions
