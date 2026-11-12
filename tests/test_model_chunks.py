"""tests/test_model_chunks.py

Model-made chunks (llm_chunked strategies): fidelity, repair, fallback and scoring.
"""

import json

from helpers import ScriptedProvider, run_case

from csr_lab.evaluation.metrics import evaluate_case
from csr_lab.pipelines.run import run_analysis


def _rewording_chunker(config):
    """The mock model, except that semantic chunking rewords 2.3 ('shall' -> 'must')."""
    from csr_lab.llm.mock_provider import MockProvider

    mock = MockProvider(config.llm.mock)

    def fn(task, payload, request):
        if task == "semantic_chunking":
            chunks = json.loads(mock.generate(request).text)["chunks"]
            for c in chunks:
                c["text"] = c["text"].replace("Parts shall be validated", "Parts must be validated")
            return {"chunks": chunks}
        return mock.generate(request).text

    return ScriptedProvider(fn)


def test_reworded_model_chunk_is_flagged_or_repaired(simple_docs, config):
    as_given = run_analysis(simple_docs, strategy="llm_chunked", config=config, provider=_rewording_chunker(config))
    unit = next(r for r in as_given.requirements if "must be validated" in r.requirement_text)
    assert unit.source_location.match == "not_found"  # analysed as given: not verbatim
    repaired = run_analysis(simple_docs, strategy="llm_chunked_repaired", config=config, provider=_rewording_chunker(config))
    unit = next(r for r in repaired.requirements if "HVG-TM-320" in r.requirement_text and "validated" in r.requirement_text)
    assert unit.source_location.match in ("exact", "whitespace_normalized")  # the aligned source text
    assert "shall be validated" in unit.requirement_text


def test_llm_chunked_strategy_records_fidelity_and_costs(simple_docs, config):
    result = run_analysis(simple_docs, strategy="llm_chunked", config=config)
    assert result.status == "completed"
    assert result.metadata.input_fidelity.origin == "llm"
    ops = result.metadata.operational
    assert "llm_chunk" in ops.stage_costs and ops.stage_costs["llm_chunk"].calls >= 1
    assert ops.token_source == "estimated_from_chars" and ops.estimated_tokens > 0
    assert ops.requirements == len(result.requirements) and ops.calls_per_requirement > 0


def test_failed_chunking_falls_back_to_whole_batch(simple_docs, config):
    from csr_lab.llm.mock_provider import MockProvider

    mock = MockProvider(config.llm.mock)

    def fn(task, payload, request):
        if task == "semantic_chunking":
            return "not json"
        return mock.generate(request).text

    result = run_analysis(simple_docs, strategy="llm_chunked", config=config, provider=ScriptedProvider(fn))
    assert result.status == "completed_with_errors"
    assert any("FALLBACK" in f.fragment_id for f in result.metadata.input_fidelity.fragments)
    assert any(lim.kind == "llm_call_failed" for lim in result.coverage.limitations)
    assert result.requirements  # the batch is still analysed
    assert result.metadata.operational.token_source == "reported"  # the scripted provider reports tokens


def test_unit_metrics_credit_chunks_but_strict_metrics_do_not(dataset, config):
    chunked_result, gold = run_case(dataset, "llm_chunked", config)
    staged_result, _ = run_case(dataset, "staged", config)
    chunked = evaluate_case(chunked_result, gold, customer="Global Motors")
    staged = evaluate_case(staged_result, gold, customer="Global Motors")
    assert chunked.units.mean_gold_per_unit > 1 and staged.units.mean_gold_per_unit == 1
    assert chunked.units.recall > 0.5  # documents are found ...
    assert chunked.references.tp == 0  # ... but not attributed to individual requirements
    assert chunked.references.errors["wrong_requirement"] + chunked.references.errors["missed"] > 0
    assert staged.units.f1 == staged.references.f1
