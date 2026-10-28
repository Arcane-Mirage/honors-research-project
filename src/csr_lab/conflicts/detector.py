"""src/csr_lab/conflicts/detector.py

Evidence-based potential-conflict detection.

* ``analyze_pairs`` sends candidate pairs to the model in batches and turns
  positive assessments into ``ConflictDraft`` records. Quotes are located in
  the source and annotated (``evidence_valid``) but never altered here.
* ``validate_conflict_evidence`` (verification strategies) replaces quotes that
  cannot be found with the verbatim requirement text and marks them repaired.
* ``graph_rule_conflicts`` (graph strategies) converts the deterministic graph
  flags into drafts.

All conflicts are potential conflicts for human review. Nothing here decides
precedence or resolves a conflict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

from csr_lab.config import GraphSettings
from csr_lab.drafts import CandidatePair, ConflictDraft, RequirementDraft, ordered_pair
from csr_lab.extraction.grounding import SourceIndex, build_evidence
from csr_lab.prompts import conflicts as conflict_prompt
from csr_lab.relationships.builder import RelationshipGraph
from csr_lab.relationships.queries import find_applicability_conflicts, find_precedence_conflicts
from csr_lab.schemas import ConflictType, Evidence, RelationshipType

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
) -> tuple[list[ConflictDraft], set[tuple[str, str]], int, list[str]]:
    """Returns (conflicts, analysed pairs, failed pair count, errors)."""
    numbered = [(f"P{i + 1:03d}", c) for i, c in enumerate(candidates)]
    batches = [numbered[i : i + pairs_per_call] for i in range(0, len(numbered), pairs_per_call)]

    # The model refers to the two requirements of a pair as "A" and "B" rather than
    # copying long requirement IDs back.
    def describe(req_id: str, key: str) -> dict:
        req = requirements[req_id]
        doc = ctx.sources.documents.get(req.document_id)
        return {
            "requirement_id": key,
            "document_identifier": doc.canonical_identifier if doc else None,
            "document_revision": doc.revision if doc else None,
            "chapter": req.chapter,
            "text": req.text,
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
            stage="analyze_conflicts",
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
            evidence = build_conflict_evidence(quotes, requirements, ctx.sources)
            conflicts.append(
                ConflictDraft(
                    requirement_ids=ids,
                    conflict_type=assessment.conflict_type or ConflictType.OTHER_OR_UNCERTAIN,
                    description=assessment.description or "Potential incompatibility reported by the model.",
                    evidence=evidence,
                    assessment="ambiguous" if assessment.ambiguous else "apparent_incompatibility",
                    detection_methods=["llm_pairwise"],
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
            facts.append(
                ContextFact(
                    kind="pair",
                    subject=req_id,
                    relation=e.relationship_type,
                    target=target.label if target else e.target_node_id,
                    target_available=bool(target and target.attributes.get("content_available")),
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
