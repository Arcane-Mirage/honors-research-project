"""tests/test_metrics.py

Scoring: undefined ratios, a perfect gold oracle, reference and conflict matching, aggregation.
"""

import pytest

from csr_lab.evaluation.metrics import aggregate, describe, evaluate_case, prf
from csr_lab.pipelines.run import run_analysis
from csr_lab.schemas import AnalysisResult
from helpers import GoldOracleProvider, case_documents


def test_prf_undefined_values():
    assert prf(0, 0, 0) == (None, None, None)
    assert prf(0, 0, 3) == (None, 0.0, None)
    assert prf(2, 2, 0) == (0.5, 1.0, pytest.approx(2 / 3))
    assert prf(0, 2, 2) == (0.0, 0.0, 0.0)


@pytest.mark.parametrize("case_id", ["dev_corvane_production", "dev_corvane_packaging", "heldout_tessaly_battery"])
def test_gold_oracle_scores_perfectly(dataset, config, case_id):
    """Feeding gold answers through the real pipeline must score 1.0 everywhere."""
    _, docs, gold = case_documents(dataset, case_id)
    result = run_analysis(docs, strategy="single_pass", config=config, provider=GoldOracleProvider(gold))
    ev = evaluate_case(result, gold)
    assert (ev.requirements.precision, ev.requirements.recall) == (1.0, 1.0)
    assert ev.requirements.chapter_accuracy == 1.0 and ev.requirements.verbatim_exact_rate == 1.0
    assert ev.requirements.location_accuracy == 1.0
    named = sum(1 for g in gold.requirements for r in g.references if r.resolution == "name_alias")
    assert ev.references.fn == named, ev.references.false_negatives  # name aliases need graph resolution
    assert ev.references.fp == named
    assert ev.conflicts.f1 == 1.0 and ev.conflicts.evidence_validity_rate == 1.0
    assert ev.graph.structurally_valid


def _base(dataset, config):
    _, docs, gold = case_documents(dataset, "dev_corvane_production")
    result = run_analysis(docs, strategy="single_pass", config=config, provider=GoldOracleProvider(gold))
    return result, gold


def test_reference_on_wrong_requirement_is_an_error(dataset, config):
    result, gold = _base(dataset, config)
    data = result.model_dump(mode="json")
    reqs = data["requirements"]
    src = next(r for r in reqs if r["reference_mentions"])
    dst = next(r for r in reqs if not r["reference_mentions"])
    moved = src["reference_mentions"].pop(0)
    src["referenced_document_ids"] = list(dict.fromkeys(m["document_id"] for m in src["reference_mentions"]))
    dst["reference_mentions"].append(moved)
    dst["referenced_document_ids"] = [moved["document_id"]]
    ev = evaluate_case(AnalysisResult.model_validate(data), gold)
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
    ids = {r["chapter"]: r["requirement_id"] for r in data["requirements"]}
    a_id, b_id = ids[a.split("#")[1]], ids[b.split("#")[1]]
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
