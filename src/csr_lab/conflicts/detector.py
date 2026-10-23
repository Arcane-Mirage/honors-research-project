"""Evidence-based potential-conflict detection.

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
    context_for: Callable[[tuple[str, str]], list[str]] | None = None,
    graph_informed: bool = False,
) -> tuple[list[ConflictDraft], set[tuple[str, str]], int, list[str]]:
    """Returns (conflicts, analysed pairs, failed pair count, errors)."""
    numbered = [(f"P{i + 1:03d}", c) for i, c in enumerate(candidates)]
    batches = [numbered[i : i + pairs_per_call] for i in range(0, len(numbered), pairs_per_call)]

    def describe(req_id: str) -> dict:
        req = requirements[req_id]
        doc = ctx.sources.documents.get(req.document_id)
        return {
            "requirement_id": req_id,
            "document_identifier": doc.canonical_identifier if doc else None,
            "chapter": req.chapter,
            "text": req.text,
        }

    def run(batch):
        pairs = []
        for pair_id, cand in batch:
            a, b = cand.requirement_ids
            item = {"pair_id": pair_id, "requirement_a": describe(a), "requirement_b": describe(b)}
            if context_for is not None:
                context = context_for((a, b))
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
            quotes = [(e.requirement_id, e.quoted_text) for e in assessment.evidence if e.requirement_id in ids]
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


def graph_context_for_pair(graph: RelationshipGraph, pair: tuple[str, str], max_items: int) -> list[str]:
    """Short, source-backed statements from the graph that relate to a pair."""
    items: list[str] = []
    for req_id in pair:
        for e in graph.out_edges(req_id):
            if e.relationship_type == RelationshipType.CONTAINS:
                continue
            if e.relationship_type == RelationshipType.POTENTIALLY_CONFLICTS_WITH:
                continue
            target = graph.node(e.target_node_id)
            label = target.label if target else e.target_node_id
            available = "available" if target and target.attributes.get("content_available") else "not provided"
            items.append(f"{req_id} {e.relationship_type.value} {label} ({available})")
    docs = [graph.container_of(r) for r in pair]
    for doc in {d for d in docs if d}:
        for e in graph.edges:
            if e.relationship_type in (
                RelationshipType.EXPLICITLY_EXCLUDES,
                RelationshipType.EXPLICITLY_TAKES_PRECEDENCE_OVER,
                RelationshipType.EXPLICITLY_SUPERSEDES,
            ) and graph.container_of(e.source_node_id) == doc and e.source_node_id not in pair:
                target = graph.node(e.target_node_id)
                quote = e.evidence[0].quoted_text[:160] if e.evidence else ""
                items.append(f"{e.source_node_id} {e.relationship_type.value} {target.label if target else e.target_node_id}: \"{quote}\"")
    return list(dict.fromkeys(items))[:max_items]
