"""Evaluation metrics, independent of providers and workflows.

``evaluate_case`` scores one ``AnalysisResult`` against one ``CaseGold``.
Categories are kept separate (requirements, references, graph, conflicts,
operations); there is no single combined score.

Undefined ratios (for example precision with no predictions) are ``None``,
not 0 or 1, so they cannot silently inflate or deflate averages.
"""

from __future__ import annotations

import statistics
from typing import Any

from pydantic import BaseModel, Field

from csr_lab.evaluation.matching import align_requirements, gold_key, mapped_id, predicted_key, unordered
from csr_lab.extraction.normalization import reference_key
from csr_lab.relationships.builder import graph_from_result
from csr_lab.relationships.validation import validate_graph
from csr_lab.schemas import AnalysisResult, CaseGold, ReferenceRelation, RelationshipType


def ratio(num: float, den: float) -> float | None:
    return num / den if den else None


def prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    p = ratio(tp, tp + fp)
    r = ratio(tp, tp + fn)
    if p is None or r is None:
        f1 = None
    else:
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return p, r, f1


class RequirementMetrics(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    matched: int = 0
    chapter_correct: int = 0
    verbatim_exact: int = 0
    location_evaluable: int = 0
    location_correct: int = 0
    chapter_accuracy: float | None = None
    verbatim_exact_rate: float | None = None
    location_accuracy: float | None = None


class ReferenceMetrics(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    gold_requirements: int = 0
    complete_set_correct: int = 0
    complete_set_accuracy: float | None = None
    matched_requirements: int = 0
    complete_set_correct_matched: int = 0
    complete_set_accuracy_matched: float | None = None
    duplicate_predictions: int = 0
    relation_correct: int = 0
    relation_accuracy: float | None = None
    availability_correct: int = 0
    availability_accuracy: float | None = None
    document_level_precision: float | None = None
    document_level_recall: float | None = None
    false_positives: list[str] = Field(default_factory=list)
    false_negatives: list[str] = Field(default_factory=list)


class GraphMetrics(BaseModel):
    structurally_valid: bool = True
    correct_reference_edges: int = 0
    missing_reference_edges: int = 0
    unsupported_reference_edges: int = 0
    duplicate_edges: int = 0
    invalid_endpoints: int = 0
    document_nodes: int = 0
    resolved_document_nodes: int = 0
    node_resolution_rate: float | None = None
    unresolved_identifiers: int = 0
    qualified_edge_tp: int = 0
    qualified_edge_fp: int = 0
    qualified_edge_fn: int = 0
    qualified_edge_f1: float | None = None


class ConflictMetrics(BaseModel):
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    gold_pairs: int = 0
    predicted: int = 0
    candidate_pairs_considered: int = 0
    candidate_hits: int = 0
    candidate_pair_recall: float | None = None
    evidence_valid: int = 0
    evidence_validity_rate: float | None = None
    hard_negative_false_positives: int = 0
    ambiguous_gold: int = 0
    ambiguous_found: int = 0
    ambiguous_recall: float | None = None
    type_correct: int = 0
    type_accuracy: float | None = None
    cross_document_gold: int = 0
    cross_document_found: int = 0
    false_positive_pairs: list[str] = Field(default_factory=list)
    false_negative_pairs: list[str] = Field(default_factory=list)


class CaseEvaluation(BaseModel):
    case_id: str
    status: str
    requirements: RequirementMetrics
    references: ReferenceMetrics
    graph: GraphMetrics
    conflicts: ConflictMetrics
    operational: dict[str, Any] = Field(default_factory=dict)


def evaluate_case(result: AnalysisResult | None, gold: CaseGold, *, threshold: float = 0.75, status: str | None = None) -> CaseEvaluation:
    """Score a result. ``None`` (a failed run) is scored as an empty prediction."""
    preds = result.requirements if result else []
    alignment = align_requirements(preds, gold.requirements, threshold)
    gold_by_id = {g.gold_id: g for g in gold.requirements}

    # -- requirements ------------------------------------------------------
    rm = RequirementMetrics(matched=len(alignment))
    rm.tp, rm.fp, rm.fn = len(alignment), len(preds) - len(alignment), len(gold.requirements) - len(alignment)
    rm.precision, rm.recall, rm.f1 = prf(rm.tp, rm.fp, rm.fn)
    for p in preds:
        if p.requirement_id not in alignment:
            continue
        g = gold_by_id[alignment[p.requirement_id]]
        rm.chapter_correct += int((p.chapter or "").strip() == g.chapter)
        rm.verbatim_exact += int(" ".join(p.requirement_text.split()) == " ".join(g.requirement_text.split()))
        if p.source_location.start_offset is not None:
            rm.location_evaluable += 1
            rm.location_correct += int(
                abs(p.source_location.start_offset - g.start_offset) <= 2 and abs((p.source_location.end_offset or 0) - g.end_offset) <= 2
            )
    rm.chapter_accuracy = ratio(rm.chapter_correct, rm.matched)
    rm.verbatim_exact_rate = ratio(rm.verbatim_exact, rm.matched)
    rm.location_accuracy = ratio(rm.location_correct, rm.matched)

    # -- references --------------------------------------------------------
    docs = {d.document_id: d for d in (result.referenced_documents if result else [])}
    gold_pairs = {(g.gold_id, gold_key(ref)) for g in gold.requirements for ref in g.references}
    gold_relation = {(g.gold_id, gold_key(ref)): ref.relation for g in gold.requirements for ref in g.references}
    gold_available = {reference_key(d.canonical_identifier, d.revision): d.content_available for d in gold.referenced_documents}
    pred_pairs: set[tuple[str, str]] = set()
    pred_relation: dict[tuple[str, str], set[ReferenceRelation]] = {}
    pred_doc_for_pair: dict[tuple[str, str], str] = {}
    duplicates = 0
    for p in preds:
        rid = mapped_id(p.requirement_id, alignment)
        seen: set[str] = set()
        for m in p.reference_mentions:
            doc = docs.get(m.document_id)
            if doc is None:
                continue
            key = predicted_key(doc)
            if key in seen:
                duplicates += 1
            seen.add(key)
            pair = (rid, key)
            pred_pairs.add(pair)
            pred_relation.setdefault(pair, set()).add(m.relation)
            pred_doc_for_pair[pair] = m.document_id
    ref = ReferenceMetrics(duplicate_predictions=duplicates, gold_requirements=len(gold.requirements))
    tp_pairs = pred_pairs & gold_pairs
    ref.tp, ref.fp, ref.fn = len(tp_pairs), len(pred_pairs - gold_pairs), len(gold_pairs - pred_pairs)
    ref.precision, ref.recall, ref.f1 = prf(ref.tp, ref.fp, ref.fn)
    ref.false_positives = sorted(f"{r} -> {k}" for r, k in pred_pairs - gold_pairs)
    ref.false_negatives = sorted(f"{r} -> {k}" for r, k in gold_pairs - pred_pairs)

    inverse = {v: k for k, v in alignment.items()}
    for g in gold.requirements:
        expected = {gold_key(r) for r in g.references}
        if g.gold_id in inverse:
            ref.matched_requirements += 1
            got = {k for (rid, k) in pred_pairs if rid == g.gold_id}
            if got == expected:
                ref.complete_set_correct += 1
                ref.complete_set_correct_matched += 1
    ref.complete_set_accuracy = ratio(ref.complete_set_correct, ref.gold_requirements)
    ref.complete_set_accuracy_matched = ratio(ref.complete_set_correct_matched, ref.matched_requirements)

    for pair in tp_pairs:
        rels = pred_relation.get(pair, set())
        qualified = {r for r in rels if r != ReferenceRelation.REFERENCES}
        predicted_rel = next(iter(qualified)) if len(qualified) == 1 else (ReferenceRelation.REFERENCES if not qualified else None)
        ref.relation_correct += int(predicted_rel == gold_relation[pair])
        doc = docs[pred_doc_for_pair[pair]]
        if pair[1] in gold_available:
            ref.availability_correct += int(doc.content_available == gold_available[pair[1]])
    ref.relation_accuracy = ratio(ref.relation_correct, len(tp_pairs))
    ref.availability_accuracy = ratio(ref.availability_correct, len(tp_pairs))
    gold_docs = {k for _, k in gold_pairs}
    pred_docs = {k for _, k in pred_pairs}
    ref.document_level_precision = ratio(len(gold_docs & pred_docs), len(pred_docs))
    ref.document_level_recall = ratio(len(gold_docs & pred_docs), len(gold_docs))

    # -- graph -------------------------------------------------------------
    gm = GraphMetrics()
    if result is not None:
        graph = graph_from_result(result)
        report = validate_graph(graph)
        gm.structurally_valid = report.valid
        gm.duplicate_edges = len(report.duplicate_edges)
        gm.invalid_endpoints = len(report.invalid_endpoints)
        edge_pairs = set()
        qualified_pred = set()
        for e in graph.edges:
            if e.metadata.get("origin") != "extraction" or not e.source_node_id.startswith("REQ-"):
                continue
            doc = docs.get(e.target_node_id)
            if doc is None:
                continue
            item = (mapped_id(e.source_node_id, alignment), predicted_key(doc))
            if e.relationship_type == RelationshipType.REFERENCES:
                edge_pairs.add(item)
            else:
                qualified_pred.add((*item, e.relationship_type.value))
        gm.correct_reference_edges = len(edge_pairs & gold_pairs)
        gm.missing_reference_edges = len(gold_pairs - edge_pairs)
        gm.unsupported_reference_edges = len(edge_pairs - gold_pairs)
        doc_nodes = result.referenced_documents
        gm.document_nodes = len(doc_nodes)
        gm.resolved_document_nodes = sum(1 for d in doc_nodes if d.canonical_identifier)
        gm.node_resolution_rate = ratio(gm.resolved_document_nodes, gm.document_nodes)
        gm.unresolved_identifiers = len(result.coverage.unresolved_identifiers)
    edge_names = {
        ReferenceRelation.EXCLUDES: "explicitly_excludes",
        ReferenceRelation.SUPERSEDES: "explicitly_supersedes",
        ReferenceRelation.DEPENDS_ON: "explicitly_depends_on",
        ReferenceRelation.TAKES_PRECEDENCE_OVER: "explicitly_takes_precedence_over",
    }
    qualified_gold = {(rid, key, edge_names[rel]) for (rid, key), rel in gold_relation.items() if rel in edge_names}
    if result is not None:
        gm.qualified_edge_tp = len(qualified_pred & qualified_gold)
        gm.qualified_edge_fp = len(qualified_pred - qualified_gold)
    gm.qualified_edge_fn = len(qualified_gold) - gm.qualified_edge_tp
    gm.qualified_edge_f1 = prf(gm.qualified_edge_tp, gm.qualified_edge_fp, gm.qualified_edge_fn)[2]

    # -- conflicts ---------------------------------------------------------
    cm = ConflictMetrics()
    gold_conf = {unordered(*c.requirement_ids): c for c in gold.conflicts}
    hard_neg = {unordered(*p.requirement_ids) for p in gold.non_conflicting_pairs}
    pred_conf = {}
    for c in result.potential_conflicts if result else []:
        a, b = (mapped_id(x, alignment) for x in c.related_requirement_ids[:2])
        pred_conf[unordered(a, b)] = c
    cm.gold_pairs, cm.predicted = len(gold_conf), len(pred_conf)
    hits = set(pred_conf) & set(gold_conf)
    cm.tp, cm.fp, cm.fn = len(hits), len(set(pred_conf) - set(gold_conf)), len(set(gold_conf) - set(pred_conf))
    cm.precision, cm.recall, cm.f1 = prf(cm.tp, cm.fp, cm.fn)
    cm.false_positive_pairs = sorted(" <-> ".join(p) for p in set(pred_conf) - set(gold_conf))
    cm.false_negative_pairs = sorted(" <-> ".join(p) for p in set(gold_conf) - set(pred_conf))
    cm.evidence_valid = sum(1 for c in pred_conf.values() if c.evidence_valid)
    cm.evidence_validity_rate = ratio(cm.evidence_valid, len(pred_conf))
    cm.hard_negative_false_positives = len(set(pred_conf) & hard_neg)
    cm.ambiguous_gold = sum(1 for c in gold_conf.values() if c.category == "ambiguous")
    cm.ambiguous_found = sum(1 for p in hits if gold_conf[p].category == "ambiguous")
    cm.ambiguous_recall = ratio(cm.ambiguous_found, cm.ambiguous_gold)
    cm.type_correct = sum(1 for p in hits if pred_conf[p].conflict_type == gold_conf[p].conflict_type)
    cm.type_accuracy = ratio(cm.type_correct, len(hits))
    cm.cross_document_gold = sum(1 for c in gold_conf.values() if c.cross_document)
    cm.cross_document_found = sum(1 for p in hits if gold_conf[p].cross_document)
    if result is not None:
        candidates = {
            unordered(*(mapped_id(x, alignment) for x in c.requirement_ids)) for c in result.metadata.conflict_candidates
        }
        cm.candidate_pairs_considered = len(candidates)
        cm.candidate_hits = len(candidates & set(gold_conf))
    cm.candidate_pair_recall = ratio(cm.candidate_hits, len(gold_conf))

    # -- operations --------------------------------------------------------
    ops: dict[str, Any] = {}
    if result is not None:
        o = result.metadata.operational
        ops = {
            "total_seconds": o.total_seconds,
            "llm_calls": o.llm_calls,
            "llm_failed_calls": o.llm_failed_calls,
            "json_validation_failures": o.json_validation_failures,
            "retries": o.retries,
            "input_tokens": o.input_tokens,
            "output_tokens": o.output_tokens,
            "input_chars": o.input_chars,
            "output_chars": o.output_chars,
            "estimated_cost_usd": o.estimated_cost_usd,
            "graph_informed_calls": o.graph_informed_calls,
            **{f"stage_seconds.{k}": v for k, v in o.stage_seconds.items()},
        }
    return CaseEvaluation(
        case_id=gold.case_id,
        status=status or (result.status if result else "error"),
        requirements=rm,
        references=ref,
        graph=gm,
        conflicts=cm,
        operational=ops,
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_SUMMED_OPS = [
    "llm_calls",
    "llm_failed_calls",
    "json_validation_failures",
    "retries",
    "input_chars",
    "output_chars",
    "graph_informed_calls",
]
_NULLABLE_SUMS = ["total_seconds", "input_tokens", "output_tokens", "estimated_cost_usd"]


def aggregate(evaluations: list[CaseEvaluation]) -> dict[str, float | None]:
    """Micro-averaged metrics over cases (counts are summed, then ratios computed)."""

    def total(getter) -> int:
        return sum(getter(e) for e in evaluations)

    out: dict[str, float | None] = {"cases": len(evaluations)}
    for prefix, get in (("requirements", lambda e: e.requirements), ("references", lambda e: e.references), ("conflicts", lambda e: e.conflicts)):
        tp, fp, fn = total(lambda e, g=get: g(e).tp), total(lambda e, g=get: g(e).fp), total(lambda e, g=get: g(e).fn)
        p, r, f = prf(tp, fp, fn)
        out.update({f"{prefix}.precision": p, f"{prefix}.recall": r, f"{prefix}.f1": f, f"{prefix}.tp": tp, f"{prefix}.fp": fp, f"{prefix}.fn": fn})

    matched = total(lambda e: e.requirements.matched)
    out["requirements.chapter_accuracy"] = ratio(total(lambda e: e.requirements.chapter_correct), matched)
    out["requirements.verbatim_exact_rate"] = ratio(total(lambda e: e.requirements.verbatim_exact), matched)
    out["requirements.location_accuracy"] = ratio(total(lambda e: e.requirements.location_correct), matched)

    out["references.complete_set_accuracy"] = ratio(total(lambda e: e.references.complete_set_correct), total(lambda e: e.references.gold_requirements))
    out["references.complete_set_accuracy_matched"] = ratio(
        total(lambda e: e.references.complete_set_correct_matched), total(lambda e: e.references.matched_requirements)
    )
    out["references.duplicate_predictions"] = total(lambda e: e.references.duplicate_predictions)
    out["references.relation_accuracy"] = ratio(total(lambda e: e.references.relation_correct), out["references.tp"] or 0)
    out["references.availability_accuracy"] = ratio(total(lambda e: e.references.availability_correct), out["references.tp"] or 0)

    out["graph.structurally_valid_rate"] = ratio(sum(1 for e in evaluations if e.graph.structurally_valid), len(evaluations))
    out["graph.correct_reference_edges"] = total(lambda e: e.graph.correct_reference_edges)
    out["graph.missing_reference_edges"] = total(lambda e: e.graph.missing_reference_edges)
    out["graph.unsupported_reference_edges"] = total(lambda e: e.graph.unsupported_reference_edges)
    out["graph.duplicate_edges"] = total(lambda e: e.graph.duplicate_edges)
    out["graph.invalid_endpoints"] = total(lambda e: e.graph.invalid_endpoints)
    out["graph.node_resolution_rate"] = ratio(total(lambda e: e.graph.resolved_document_nodes), total(lambda e: e.graph.document_nodes))
    out["graph.unresolved_identifiers"] = total(lambda e: e.graph.unresolved_identifiers)
    out["graph.qualified_edge_f1"] = prf(
        total(lambda e: e.graph.qualified_edge_tp), total(lambda e: e.graph.qualified_edge_fp), total(lambda e: e.graph.qualified_edge_fn)
    )[2]

    out["conflicts.candidate_pair_recall"] = ratio(total(lambda e: e.conflicts.candidate_hits), total(lambda e: e.conflicts.gold_pairs))
    out["conflicts.candidate_pairs_considered"] = total(lambda e: e.conflicts.candidate_pairs_considered)
    out["conflicts.evidence_validity_rate"] = ratio(total(lambda e: e.conflicts.evidence_valid), total(lambda e: e.conflicts.predicted))
    out["conflicts.hard_negative_false_positives"] = total(lambda e: e.conflicts.hard_negative_false_positives)
    out["conflicts.ambiguous_recall"] = ratio(total(lambda e: e.conflicts.ambiguous_found), total(lambda e: e.conflicts.ambiguous_gold))
    out["conflicts.type_accuracy"] = ratio(total(lambda e: e.conflicts.type_correct), out["conflicts.tp"] or 0)
    out["conflicts.cross_document_recall"] = ratio(
        total(lambda e: e.conflicts.cross_document_found), total(lambda e: e.conflicts.cross_document_gold)
    )

    for key in _SUMMED_OPS:
        out[f"ops.{key}"] = sum((e.operational.get(key) or 0) for e in evaluations)
    for key in _NULLABLE_SUMS:
        values = [e.operational.get(key) for e in evaluations]
        out[f"ops.{key}"] = round(sum(values), 6) if values and all(v is not None for v in values) else None
    return out


def describe(values: list[float | None]) -> dict[str, float | int | None]:
    """Mean, sample standard deviation, min, max over repetitions (None values ignored)."""
    xs = [v for v in values if v is not None]
    if not xs:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "std": statistics.stdev(xs) if len(xs) > 1 else None,
        "min": min(xs),
        "max": max(xs),
    }


def flatten_case(evaluation: CaseEvaluation) -> dict[str, Any]:
    """One flat row per case for CSV/SQLite (lists of FP/FN items are excluded)."""
    row: dict[str, Any] = {"case_id": evaluation.case_id, "status": evaluation.status}
    for section in ("requirements", "references", "graph", "conflicts"):
        for key, value in getattr(evaluation, section).model_dump().items():
            if isinstance(value, list):
                continue
            row[f"{section}.{key}"] = value
    row.update({f"ops.{k}": v for k, v in evaluation.operational.items()})
    return row
