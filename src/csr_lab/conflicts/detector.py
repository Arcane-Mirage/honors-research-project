"""src/csr_lab/conflicts/detector.py

Evidence-based potential-conflict detection.

* ``analyze_pairs`` sends candidate pairs to the model in batches and turns
  positive assessments into ``ConflictDraft`` records. Quotes are located in
  the source and annotated (``evidence_valid``) but never altered here.
* ``validate_conflict_evidence`` (verification strategies) replaces quotes that
  cannot be found with the verbatim requirement text and marks them repaired.
* ``graph_rule_conflicts`` (graph strategies) converts the deterministic graph
  flags into drafts.

* ``classify_conflicts`` labels each conflict with what *stated* precedence says about it.
  "X takes precedence over Y" comes from the model's ``takes_precedence_over`` references and
  from documents.yaml, and is transitive (CVS10 > Formel Q > ISO 9001):
  ``resolved_by_precedence`` (exactly one direction holds; ``prevailing_document_id`` is the
  document whose text prevails), ``mutual_precedence_claims``, or ``unresolved`` (no stated
  precedence, or both units in one document).

All conflicts are potential conflicts for human review. Precedence labels say which text a
stated rule makes prevail; they never resolve a conflict.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from csr_lab.config import GraphSettings
from csr_lab.conflicts.candidate_selection import attribute_score
from csr_lab.drafts import CandidatePair, ConflictDraft, RequirementDraft, ordered_pair
from csr_lab.extraction.grounding import SourceIndex, build_evidence, phrase_pattern
from csr_lab.extraction.normalization import canonical_input_identifier
from csr_lab.ids import document_family_id
from csr_lab.prompts import conflicts as conflict_prompt
from csr_lab.relationships.builder import DocumentIndex, RelationshipGraph, is_pointer
from csr_lab.relationships.queries import find_applicability_conflicts, find_precedence_conflicts
from csr_lab.schemas import ConflictType, Evidence, InputDocument, PotentialConflict, ReferenceRelation, RelationshipType

if TYPE_CHECKING:
    from csr_lab.pipelines.base import PipelineContext

REVIEW_REASON = (
    "Automatically flagged potential incompatibility. Requires engineering review and, where "
    "contractual, negotiation with the customer. Not resolved by the system."
)


def _evidence_valid(evidence: list[Evidence], requirement_ids: list[str]) -> bool:
    covered = {e.requirement_id for e in evidence if e.verified}
    return bool(evidence) and all(e.verified for e in evidence) and set(requirement_ids) <= covered


def build_conflict_evidence(
    quotes: list[tuple[str, str]], requirements: dict[str, RequirementDraft], sources: SourceIndex
) -> list[Evidence]:
    out = []
    for req_id, quote in quotes:
        req = requirements.get(req_id)
        if req is None or not quote.strip():
            continue
        out.append(
            build_evidence(
                sources,
                document_id=req.document_id,
                quote=quote,
                requirement_id=req_id,
                span=req.span,
                chapter=req.chapter,
            )
        )
    return out


def analyze_pairs(
    ctx: PipelineContext,
    candidates: list[CandidatePair],
    requirements: dict[str, RequirementDraft],
    *,
    pairs_per_call: int,
    context_for: Callable[[tuple[str, str]], list[str | dict]] | None = None,
    graph_informed: bool = False,
    evidence_for: Callable[[list[tuple[str, str]]], list[Evidence]] | None = None,
    document_info: dict[str, tuple[str | None, str | None]] | None = None,
    stage: str = "analyze_conflicts",
    method: str = "llm_pairwise",
) -> tuple[list[ConflictDraft], set[tuple[str, str]], int, list[str]]:
    """Returns (conflicts, analysed pairs, failed pair count, errors).

    ``evidence_for`` (quotes -> evidence) and ``document_info`` (document ID -> identifier,
    revision) replace the source lookups for units whose document is not in this run
    (the saved-graph conflict check).
    """
    make_evidence = evidence_for or (lambda quotes: build_conflict_evidence(quotes, requirements, ctx.sources))
    numbered = [(f"P{i + 1:03d}", c) for i, c in enumerate(candidates)]
    batches = [numbered[i : i + pairs_per_call] for i in range(0, len(numbered), pairs_per_call)]

    # The model refers to the two requirements of a pair as "A" and "B" rather than
    # copying long requirement IDs back.
    def describe(req_id: str, key: str) -> dict:
        req = requirements[req_id]
        doc = ctx.sources.documents.get(req.document_id)
        identifier, revision = (doc.canonical_identifier, doc.revision) if doc else (document_info or {}).get(req.document_id, (None, None))
        return {
            "requirement_id": key,
            "document_identifier": identifier,
            "document_revision": revision,
            "chapter": req.chapter,
            "text": req.prompt_text,
        }

    def run(batch):
        pairs = []
        for pair_id, cand in batch:
            a, b = cand.requirement_ids
            item = {"pair_id": pair_id, "requirement_a": describe(a, "A"), "requirement_b": describe(b, "B")}
            if context_for is not None:
                context = context_for((a, b))  # already uses A/B for this pair
                if context:
                    item["related_context"] = context
            pairs.append(item)
        system, user = conflict_prompt.SPEC.render({"pairs": pairs})
        result = ctx.llm.call(
            task="conflict_analysis",
            stage=stage,
            system=system,
            user=user,
            response_model=conflict_prompt.ConflictAnalysisResponse,
            prompt_version=conflict_prompt.SPEC.fingerprint,
            graph_informed=graph_informed and context_for is not None,
        )
        return batch, result

    conflicts: list[ConflictDraft] = []
    analysed: set[tuple[str, str]] = set()
    failed = 0
    errors: list[str] = []
    for batch, result in ctx.llm.map(run, batches):
        if not result.ok or result.value is None:
            failed += len(batch)
            errors.append(f"conflict analysis failed for {len(batch)} pairs: {'; '.join(result.errors)[:200]}")
            continue
        by_id = {a.pair_id: a for a in result.value.assessments}
        for pair_id, cand in batch:
            assessment = by_id.get(pair_id)
            if assessment is None:
                failed += 1
                continue
            analysed.add(cand.requirement_ids)
            cand.analyzed = True
            if not assessment.potential_conflict:
                continue
            ids = list(cand.requirement_ids)
            keys = {"A": ids[0], "B": ids[1]}
            quotes = [(keys[k], e.quoted_text) for e in assessment.evidence if (k := e.requirement_id.strip().upper()) in keys]
            evidence = make_evidence(quotes)
            conflicts.append(
                ConflictDraft(
                    requirement_ids=ids,
                    conflict_type=assessment.conflict_type or ConflictType.OTHER_OR_UNCERTAIN,
                    description=assessment.description or "Potential incompatibility reported by the model.",
                    evidence=evidence,
                    assessment="ambiguous" if assessment.ambiguous else "apparent_incompatibility",
                    detection_methods=[method],
                    evidence_valid=_evidence_valid(evidence, ids),
                    reason_for_review=REVIEW_REASON,
                    related_document_ids=sorted({requirements[i].document_id for i in ids}),
                    graph_informed=graph_informed and context_for is not None,
                    metadata={"selectors": cand.selectors},
                )
            )
    return conflicts, analysed, failed, errors


def validate_conflict_evidence(
    conflicts: list[ConflictDraft], requirements: dict[str, RequirementDraft], sources: SourceIndex
) -> int:
    """Replace unlocatable quotes with the verbatim requirement text. Returns repairs made."""
    repairs = 0
    for conflict in conflicts:
        kept = [e for e in conflict.evidence if e.verified]
        for req_id in conflict.requirement_ids:
            if any(e.requirement_id == req_id for e in kept):
                continue
            req = requirements.get(req_id)
            if req is None or req.span is None:
                continue
            ev = build_evidence(
                sources, document_id=req.document_id, quote=req.text, requirement_id=req_id, span=req.span, chapter=req.chapter
            )
            kept.append(ev.model_copy(update={"repaired": True}))
            repairs += 1
        dropped = len(conflict.evidence) - sum(1 for e in conflict.evidence if e.verified)
        if dropped:
            conflict.metadata["unverifiable_quotes_removed"] = dropped
        conflict.evidence = kept
        conflict.evidence_valid = _evidence_valid(kept, conflict.requirement_ids)
    return repairs


def graph_rule_conflicts(graph: RelationshipGraph, settings: GraphSettings, known_requirements: set[str]) -> list[ConflictDraft]:
    flags = []
    if settings.applicability_rules:
        flags += [(f, ConflictType.APPLICABILITY_CONFLICT) for f in find_applicability_conflicts(graph, settings.related_document_hops)]
    if settings.precedence_rules:
        flags += [(f, ConflictType.POTENTIAL_PRECEDENCE_CONFLICT) for f in find_precedence_conflicts(graph)]
    out = []
    for flag, conflict_type in flags:
        if not set(flag.requirement_ids) <= known_requirements:
            continue
        out.append(
            ConflictDraft(
                requirement_ids=list(flag.requirement_ids),
                conflict_type=conflict_type,
                description=flag.detail,
                evidence=list(flag.evidence),
                detection_methods=[f"graph_{flag.rule}_rule"],
                evidence_valid=_evidence_valid(list(flag.evidence), list(flag.requirement_ids)),
                reason_for_review=REVIEW_REASON,
                related_document_ids=flag.document_ids,
                related_relationship_ids=flag.relationship_ids,
                graph_informed=True,
            )
        )
    return out


def merge_conflicts(conflicts: list[ConflictDraft]) -> list[ConflictDraft]:
    """One record per requirement pair; detection methods and evidence are combined."""
    merged: dict[tuple[str, str], ConflictDraft] = {}
    for c in conflicts:
        pair = ordered_pair(*c.requirement_ids)
        if pair not in merged:
            merged[pair] = c.model_copy(deep=True)
            continue
        m = merged[pair]
        m.detection_methods = list(dict.fromkeys([*m.detection_methods, *c.detection_methods]))
        seen = {(e.requirement_id, e.quoted_text) for e in m.evidence}
        m.evidence += [e for e in c.evidence if (e.requirement_id, e.quoted_text) not in seen]
        m.related_document_ids = sorted({*m.related_document_ids, *c.related_document_ids})
        m.related_relationship_ids = list(dict.fromkeys([*m.related_relationship_ids, *c.related_relationship_ids]))
        m.graph_informed = m.graph_informed or c.graph_informed
        if c.conflict_type != m.conflict_type:
            m.metadata.setdefault("other_conflict_types", []).append(c.conflict_type.value)
        m.evidence_valid = _evidence_valid(m.evidence, m.requirement_ids)
    return [merged[p] for p in sorted(merged)]


# ---------------------------------------------------------------------------
# Graph context for a pair (graph_assisted): facts, then one of several renderings
# ---------------------------------------------------------------------------

CONTEXT_FORMATS = ("sentences", "triples", "json", "ids")
_DOCUMENT_STATEMENTS = (
    RelationshipType.EXPLICITLY_EXCLUDES,
    RelationshipType.EXPLICITLY_TAKES_PRECEDENCE_OVER,
    RelationshipType.EXPLICITLY_SUPERSEDES,
)
_PHRASES = {
    RelationshipType.REFERENCES: "refers to",
    RelationshipType.EXPLICITLY_EXCLUDES: "states that this document does not apply:",
    RelationshipType.EXPLICITLY_TAKES_PRECEDENCE_OVER: "states that its own document takes precedence over",
    RelationshipType.EXPLICITLY_SUPERSEDES: "states that its own document supersedes",
    RelationshipType.EXPLICITLY_DEPENDS_ON: "is to be used together with",
    RelationshipType.AMENDS: "amends the same-numbered chapter of the document it follows:",
    RelationshipType.INHERITS_FROM: "adopts unchanged the same-numbered chapter of the document it follows:",
    RelationshipType.REFERENCES_CHAPTER: "refers to chapter",
}


@dataclass(frozen=True)
class ContextFact:
    """One graph statement about a pair, before it is rendered for the prompt.

    ``kind == "pair"``: an edge from one of the two requirements (``subject`` is its ID).
    ``kind == "same_document"``: an exclusion/precedence/supersession statement made by
    another requirement in the document of A and/or B.
    """

    kind: str
    subject: str
    relation: RelationshipType
    target: str
    target_available: bool | None = None
    subject_chapter: str | None = None
    subject_document: str | None = None
    shares_document_with: tuple[str, ...] = ()
    quote: str = ""
    target_id: str = ""  # node ID of the target (a requirement for structure edges)
    target_is_requirement: bool = False


def graph_context_facts(
    graph: RelationshipGraph, pair: tuple[str, str], max_items: int, *, collapse: bool = True
) -> list[ContextFact]:
    """Short, source-backed graph facts that relate to a pair, in a fixed order.

    With ``collapse``, a plain "refers to X" is dropped when the same requirement also has a
    qualified edge (excludes, takes precedence over, ...) to X, which says more in one item.
    """
    facts: list[ContextFact] = []
    for req_id in pair:
        for e in graph.out_edges(req_id):
            if e.relationship_type in (RelationshipType.CONTAINS, RelationshipType.POTENTIALLY_CONFLICTS_WITH):
                continue
            target = graph.node(e.target_node_id)
            is_req = bool(target and target.node_type == "requirement")
            if is_req:
                container = graph.node(graph.container_of(e.target_node_id) or "")
                label = f"chapter {target.attributes.get('chapter')} of {container.label if container else '?'}"  # type: ignore[union-attr]
            else:
                label = target.label if target else e.target_node_id
            facts.append(
                ContextFact(
                    kind="pair",
                    subject=req_id,
                    relation=e.relationship_type,
                    target=label,
                    target_available=True if is_req else bool(target and target.attributes.get("content_available")),
                    target_id=e.target_node_id,
                    target_is_requirement=is_req,
                )
            )
    containers = {r: graph.container_of(r) for r in pair}
    for doc in dict.fromkeys(d for d in containers.values() if d):  # ordered, unlike a set
        doc_node = graph.node(doc)
        for e in graph.edges:
            if e.relationship_type not in _DOCUMENT_STATEMENTS or e.source_node_id in pair:
                continue
            if graph.container_of(e.source_node_id) != doc:
                continue
            source = graph.node(e.source_node_id)
            target = graph.node(e.target_node_id)
            facts.append(
                ContextFact(
                    kind="same_document",
                    subject=e.source_node_id,
                    relation=e.relationship_type,
                    target=target.label if target else e.target_node_id,
                    subject_chapter=(source.attributes.get("chapter") if source else None),
                    subject_document=doc_node.label if doc_node else doc,
                    shares_document_with=tuple(r for r in pair if containers[r] == doc),
                    quote=" ".join(e.evidence[0].quoted_text.split())[:160] if e.evidence else "",
                )
            )
    if collapse:
        qualified = {(f.subject, f.target) for f in facts if f.kind == "pair" and f.relation != RelationshipType.REFERENCES}
        facts = [f for f in facts if not (f.kind == "pair" and f.relation == RelationshipType.REFERENCES and (f.subject, f.target) in qualified)]
    return list(dict.fromkeys(facts))[:max_items]


def _letters(pair: tuple[str, str]) -> dict[str, str]:
    return {pair[0]: "A", pair[1]: "B"}


def _availability(fact: ContextFact) -> str:
    return "content provided in this analysis" if fact.target_available else "content not provided"


def _same_doc_subject(fact: ContextFact, letters: dict[str, str]) -> str:
    who = [letters[r] for r in fact.shares_document_with]
    holder = f"requirements {' and '.join(who)}" if len(who) > 1 else f"requirement {who[0]}"
    chapter = f"Chapter {fact.subject_chapter}" if fact.subject_chapter else "Another requirement"
    return f"{chapter} of {fact.subject_document} (the document containing {holder})"


def render_fact(fact: ContextFact, pair: tuple[str, str], fmt: str) -> str | dict:
    letters = _letters(pair)
    if fmt == "ids":  # the original rendering: raw IDs and relation codes
        if fact.kind == "pair":
            text = f"{fact.subject} {fact.relation.value} {fact.target} ({'available' if fact.target_available else 'not provided'})"
        else:
            text = f'{fact.subject} {fact.relation.value} {fact.target}: "{fact.quote}"'
        return text.replace(pair[0], "A").replace(pair[1], "B")
    if fmt == "triples":
        if fact.kind == "pair":
            return f"{letters[fact.subject]} | {fact.relation.value} | {fact.target} | {_availability(fact)}"
        who = "+".join(letters[r] for r in fact.shares_document_with)
        return f'{fact.subject_document} ch.{fact.subject_chapter or "?"} (same document as {who}) | {fact.relation.value} | {fact.target} | "{fact.quote}"'
    if fmt == "json":
        if fact.kind == "pair":
            return {"subject": f"requirement {letters[fact.subject]}", "relation": fact.relation.value, "object": fact.target, "object_content_provided": bool(fact.target_available)}
        return {
            "subject": f"chapter {fact.subject_chapter or '?'} of {fact.subject_document}",
            "same_document_as": [letters[r] for r in fact.shares_document_with],
            "relation": fact.relation.value,
            "object": fact.target,
            "quote": fact.quote,
        }
    # sentences
    phrase = _PHRASES.get(fact.relation, fact.relation.value.replace("_", " "))
    if fact.kind == "pair" and fact.target_is_requirement:
        target = f"requirement {letters[fact.target_id]} ({fact.target})" if fact.target_id in letters else fact.target
        return f"Requirement {letters[fact.subject]} {phrase} {target}."
    if fact.kind == "pair":
        if fact.relation == RelationshipType.EXPLICITLY_EXCLUDES:
            return f"Requirement {letters[fact.subject]} states that document {fact.target} does not apply ({_availability(fact)})."
        return f"Requirement {letters[fact.subject]} {phrase} document {fact.target} ({_availability(fact)})."
    if fact.relation == RelationshipType.EXPLICITLY_EXCLUDES:
        return f'{_same_doc_subject(fact, letters)} states that {fact.target} does not apply: "{fact.quote}"'
    return f'{_same_doc_subject(fact, letters)} {phrase} {fact.target}: "{fact.quote}"'


def graph_context_for_pair(
    graph: RelationshipGraph, pair: tuple[str, str], max_items: int, context_format: str = "sentences"
) -> list[str | dict]:
    """Graph context for a pair, rendered for the conflict prompt (A/B = the pair's requirements)."""
    if context_format not in CONTEXT_FORMATS:
        raise ValueError(f"unknown graph context_format {context_format!r}; choose from {CONTEXT_FORMATS}")
    facts = graph_context_facts(graph, pair, max_items, collapse=context_format != "ids")
    return [render_fact(f, pair, context_format) for f in facts]


# ---------------------------------------------------------------------------
# Conflicts with the saved graph
# ---------------------------------------------------------------------------


def saved_graph_candidates(
    units: list[RequirementDraft],
    unit_refs: dict[str, set[str]],
    saved: dict,
    *,
    max_pairs: int,
    min_score: float,
    max_fanout: int,
    families: dict[str, str],
    follows: dict[str, str] | None = None,
) -> tuple[list[CandidatePair], list[str]]:
    """Pairs (new unit, saved unit) whose documents are linked by references.

    ``unit_refs``: new unit ID -> document nodes it references (with their families);
    ``families``: new document ID -> its document-level node.
    A pair is linked when the new unit references the saved unit's document, the saved
    unit references the new unit's document, or both reference a document that at most
    ``max_fanout`` saved units reference (common standards would link everything).
    Linked pairs need a topic overlap of at least ``min_score``; the best ``max_pairs`` are kept.

    ``follows``: new document ID -> the saved document (family node) whose chapter structure it
    follows. Each of its chapters is paired with the saved chapter of the same number (a delta
    chapter amends it), whatever the overlap; pointer chapters ('See ... No additional
    requirements') are skipped.
    """
    fanout: dict[str, int] = {}
    for s in saved.values():
        for d in s.referenced:
            fanout[d] = fanout.get(d, 0) + 1
    scored: list[tuple[float, tuple[str, str], list[str]]] = []
    for u in units:
        u_family = families.get(u.document_id, u.document_id)
        refs = unit_refs.get(u.requirement_id, set())
        base = (follows or {}).get(u.document_id)
        for s in saved.values():
            if base and s.family_id == base and u.chapter and s.chapter == u.chapter and not is_pointer(u.text):
                scored.append((1.0 + attribute_score(u.text, s.text), ordered_pair(u.requirement_id, s.unit_id), [f"saved_graph:aligned_chapter:{u.chapter}"]))
                continue
            selectors = []
            if s.document_id in refs or s.family_id in refs:
                selectors.append(f"saved_graph:references:{s.family_id}")
            if u.document_id in s.referenced or u_family in s.referenced:
                selectors.append(f"saved_graph:referenced_by:{s.family_id}")
            shared = sorted(d for d in refs & s.referenced if fanout.get(d, 0) <= max_fanout)
            selectors += [f"saved_graph:shared_reference:{d}" for d in shared[:3]]
            if not selectors:
                continue
            score = attribute_score(u.prompt_text, s.text)
            if score >= min_score:
                scored.append((score, ordered_pair(u.requirement_id, s.unit_id), selectors))
    scored.sort(key=lambda x: (-x[0], x[1]))
    notes = []
    if len(scored) > max_pairs:
        notes.append(f"saved-graph conflict check: {len(scored)} linked pairs, best {max_pairs} analysed (saved_max_pairs)")
    return [CandidatePair(requirement_ids=pair, selectors=sel, graph_derived=True) for _, pair, sel in scored[:max_pairs]], notes


def saved_unit_evidence(quotes: list[tuple[str, str]], units: dict[str, RequirementDraft], sources: SourceIndex, saved: dict) -> list[Evidence]:
    """Evidence for a pair with one saved unit: its quotes are located in the stored text.

    Offsets are the unit's stored start offset plus the quote's position in the unit text.
    """
    out = []
    for unit_id, quote in quotes:
        if unit_id not in saved:
            out += build_conflict_evidence([(unit_id, quote)], units, sources)
            continue
        quote = quote.strip()
        if not quote:
            continue
        unit = saved[unit_id]
        pattern = phrase_pattern(quote)
        m = re.search(pattern, unit.text) or re.search(pattern, unit.text, re.IGNORECASE)
        located = m is not None and unit.start_offset is not None
        out.append(
            Evidence(
                source_document_id=unit.document_id,
                requirement_id=unit_id,
                quoted_text=quote,
                start_offset=unit.start_offset + m.start() if located else None,  # type: ignore[union-attr, operator]
                end_offset=unit.start_offset + m.end() if located else None,  # type: ignore[union-attr, operator]
                verified=True if located else (None if m is not None else False),  # found but offsets unknown: unchecked
                verification=("exact" if m.group(0) == quote else "whitespace_normalized") if m else "not_found",
            )
        )
    return out


# ---------------------------------------------------------------------------
# Precedence status of conflicts
# ---------------------------------------------------------------------------


def precedence_facts(documents: list[InputDocument], requirements: list) -> dict[tuple[str, str], list[str]]:
    """(winning family, losing family) -> basis strings. From documents.yaml and from extracted
    ``takes_precedence_over`` mentions."""
    docs = DocumentIndex(documents)
    facts: dict[tuple[str, str], list[str]] = {}
    for d in documents:
        own = docs.family_of(d.document_id)
        for ident in d.takes_precedence_over:
            other = document_family_id(canonical_input_identifier(ident))
            if other != own:
                facts.setdefault((own, other), []).append(f"documents.yaml: {d.canonical_identifier} takes precedence over {ident}")
    for r in requirements:
        own = docs.family_of(r.source_document_id)
        for m in r.reference_mentions:
            if m.relation != ReferenceRelation.TAKES_PRECEDENCE_OVER:
                continue
            other = docs.family_of(m.document_id)
            if other != own:
                facts.setdefault((own, other), []).append(f"{r.requirement_id} ({r.chapter}): {m.evidence.quoted_text[:160]}")
    return facts


def _reaches(facts: dict[tuple[str, str], list[str]], a: str, b: str) -> list[str] | None:
    """Basis of 'a takes precedence over b', directly or through other documents; None if not stated."""
    graph: dict[str, list[str]] = {}
    for x, y in facts:
        graph.setdefault(x, []).append(y)
    stack, seen, parent = [a], {a}, {}
    while stack:
        node = stack.pop()
        for nxt in graph.get(node, []):
            if nxt in seen:
                continue
            parent[nxt] = node
            if nxt == b:
                path, cur = [], b
                while cur != a:
                    path = facts[(parent[cur], cur)][:1] + path
                    cur = parent[cur]
                return path
            seen.add(nxt)
            stack.append(nxt)
    return None


def classify_conflicts(conflicts: list[PotentialConflict], unit_family: dict[str, str], facts: dict[tuple[str, str], list[str]]) -> None:
    """Set ``precedence_status``, ``prevailing_document_id`` and ``precedence_basis`` in place."""
    for c in conflicts:
        fams = list(dict.fromkeys(unit_family[u] for u in c.unit_ids[:2] if u in unit_family))
        if len(fams) != 2:
            c.precedence_status = "unresolved"
            c.precedence_basis = ["both units are in the same document" if len(fams) == 1 else "document of a unit unknown"]
            continue
        a, b = fams
        ab, ba = _reaches(facts, a, b), _reaches(facts, b, a)
        if ab and ba:
            c.precedence_status, c.precedence_basis = "mutual_precedence_claims", ab + ba
        elif ab or ba:
            c.precedence_status = "resolved_by_precedence"
            c.prevailing_document_id = a if ab else b
            c.precedence_basis = ab or ba  # type: ignore[assignment]
        else:
            c.precedence_status, c.precedence_basis = "unresolved", []
