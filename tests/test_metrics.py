"""tests/test_metrics.py

Scoring: undefined ratios, a perfect gold oracle, reference and conflict matching, aggregation.
"""

import pytest
from helpers import GoldOracleProvider, case_documents, run_case

from csr_lab.evaluation.metrics import aggregate, describe, evaluate_case, prf
from csr_lab.schemas import AnalysisResult


def test_prf_undefined_values():
    assert prf(0, 0, 0) == (None, None, None)
    assert prf(0, 0, 3) == (None, 0.0, None)
    assert prf(2, 2, 0) == (0.5, 1.0, pytest.approx(2 / 3))
    assert prf(0, 2, 2) == (0.0, 0.0, 0.0)


def _base(dataset, config):
    _, _, gold = case_documents(dataset)
    return run_case(dataset, "single_pass", config, provider=GoldOracleProvider(gold))


def test_gold_oracle_scores_perfectly(dataset, config):
    """Gold answers fed through the real pipeline, as an experiment runs the case (documents.yaml,
    the dataset's reader), must score 1.0 on every metric."""
    result, gold = _base(dataset, config)
    ev = evaluate_case(result, gold, customer="Global Motors")
    assert (ev.requirements.precision, ev.requirements.recall, ev.requirements.verbatim_exact_rate, ev.requirements.location_accuracy) == (1, 1, 1, 1)
    assert (ev.references.precision, ev.references.recall, ev.references.relation_accuracy, ev.references.location_accuracy) == (1, 1, 1, 1), ev.references.errors
    assert ev.references.errors["unresolved_alias"] == 0  # name-only documents resolve through documents.yaml
    assert ev.references.out_of_scope_gold > 0  # references for Brennic/Ostra exist and are not expected
    assert ev.graph.chapter_link_f1 == 1.0
    assert ev.conflicts.f1 == 1.0 and ev.conflicts.evidence_validity_rate == 1.0 and ev.conflicts.precedence_accuracy == 1.0
    assert ev.graph.structurally_valid


def test_reference_on_wrong_requirement_is_an_error(dataset, config):
    result, gold = _base(dataset, config)
    data = result.model_dump(mode="json")
    reqs = data["requirements"]
    src = next(r for r in reqs if r["reference_mentions"])
    gold_chapters = {g.chapter for g in gold.requirements}
    dst = next(r for r in reqs if not r["reference_mentions"] and r["chapter"] in gold_chapters)
    moved = src["reference_mentions"].pop(0)
    src["referenced_document_ids"] = list(dict.fromkeys(m["document_id"] for m in src["reference_mentions"]))
    dst["reference_mentions"].append(moved)
    dst["referenced_document_ids"] = [moved["document_id"]]
    ev = evaluate_case(AnalysisResult.model_validate(data), gold, customer="Global Motors")
    assert ev.references.fp == 1 and ev.references.fn == 1
    assert ev.references.complete_set_correct == len(gold.requirements) - 2


def test_conflict_pairs_are_order_independent_and_failures_score_empty(dataset, config):
    result, gold = _base(dataset, config)
    data = result.model_dump(mode="json")
    for c in data["potential_conflicts"]:
        c["related_requirement_ids"].reverse()
    ev = evaluate_case(AnalysisResult.model_validate(data), gold)
    assert ev.conflicts.f1 == 1.0

    failed = evaluate_case(None, gold, status="error")
    assert failed.references.tp == 0 and failed.references.fn == sum(len(g.references) for g in gold.requirements)
    assert failed.references.precision is None and failed.references.recall == 0.0
    assert failed.conflicts.recall == 0.0


def test_hard_negatives_and_aggregation(dataset, config):
    result, gold = _base(dataset, config)
    data = result.model_dump(mode="json")
    a, b = gold.non_conflicting_pairs[0].requirement_ids

    def predicted(gold_id: str) -> str:  # 'HVG-CVS-20#1.1' -> the requirement of that document and chapter
        doc, chapter = gold_id.split("#")
        return next(r["requirement_id"] for r in data["requirements"] if r["chapter"] == chapter and r["source_document_id"].startswith(f"DOC-{doc}"))

    a_id, b_id = predicted(a), predicted(b)
    extra = dict(data["potential_conflicts"][0])
    extra.update({"conflict_id": "CON-extra", "related_requirement_ids": [a_id, b_id], "related_relationship_ids": []})
    data["potential_conflicts"].append(extra)
    ev = evaluate_case(AnalysisResult.model_validate(data), gold)
    assert ev.conflicts.fp == 1 and ev.conflicts.hard_negative_false_positives == 1

    good = evaluate_case(result, gold)
    agg = aggregate([good, ev])
    assert agg["conflicts.tp"] == good.conflicts.tp + ev.conflicts.tp
    assert agg["conflicts.hard_negative_false_positives"] == 1
    assert describe([1.0, 0.0])["std"] == pytest.approx(0.7071, abs=1e-3)
    assert describe([None])["mean"] is None
