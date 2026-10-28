"""tests/test_chunk_input.py

Pre-chunked input (chunk files) and model-made chunks (llm_chunked strategies).
"""

import json

from helpers import ScriptedProvider

from csr_lab.evaluation.metrics import evaluate_case
from csr_lab.ingestion.loader import load_bundle
from csr_lab.pipelines.registry import derive_strategy
from csr_lab.pipelines.run import run_analysis

from conftest import SIMPLE_DOC


def _chunk_file(tmp_path, chunks, source="simple.txt", **extra):
    (tmp_path / "simple.txt").write_text(SIMPLE_DOC, encoding="utf-8")
    body = {"chunks": chunks, **extra}
    if source:
        body["source"] = {"file": source}
    path = tmp_path / "simple.chunks.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


CHUNKS = [
    {"title": "Scope", "content": "1 Scope\n1.1 This document applies to test suppliers.", "source_pages": [1]},
    {
        "title": "Requirements",
        "content": "2.1 The supplier shall comply with Specification CVM-QS-110 and Drawing DRW-045.\n"
        "2.2 Packaging Standard CVM-PKG-050 does not apply and shall not be used.",
    },
    # Reworded: 'shall' -> 'must', line break joined.
    {"content": "2.3 Parts must be validated at a maximum test temperature of 85 °C according to CVM-TM-320."},
]


def test_chunk_file_loads_with_source_and_lenient_keys(tmp_path):
    path = _chunk_file(tmp_path, [{"id": "x", "text": "2.5 The supplier shall keep records for five years.", "pages": "1, 2"}])
    (doc,) = load_bundle([path])
    assert doc.document_id == "DOC-CVM-CSR-900-REV-A"  # header of the source file
    assert doc.provided_chunks[0].chunk_id == "x" and doc.provided_chunks[0].source_pages == [1, 2]
    assert any("pre-chunked input" in w for w in doc.warnings)
    assert "provided_chunks" not in doc.model_dump()  # never exported


def test_folder_with_chunk_file_and_source_keeps_only_chunked(tmp_path):
    _chunk_file(tmp_path, CHUNKS)
    docs = load_bundle([tmp_path])
    assert len(docs) == 1 and docs[0].provided_chunks


def test_chunk_file_without_source_uses_chunks_as_text(tmp_path):
    path = _chunk_file(tmp_path, CHUNKS, source=None, identifier="EXT-1")
    (doc,) = load_bundle([path])
    assert doc.canonical_identifier == "EXT-1"
    assert any("cannot be checked" in w for w in doc.warnings)


def test_requirement_units_inside_provided_chunks(tmp_path, config):
    docs = load_bundle([_chunk_file(tmp_path, CHUNKS)])
    result = run_analysis(docs, strategy="staged_verified", config=config)
    fid = result.metadata.input_fidelity
    assert fid is not None and fid.origin == "provided"
    assert [f.match for f in fid.fragments] == ["exact", "exact", "fuzzy"]
    chapters = {r.chapter for r in result.requirements}
    assert {"2.1", "2.2", "2.3"} <= chapters  # requirements identified inside the chunks
    reworded = next(r for r in result.requirements if r.chapter == "2.3")
    assert reworded.source_location.match == "not_found"  # analysed as given: not verbatim
    # Untouched source text (2.4, 2.5) shows up as a requirement-like gap.
    assert fid.summary.requirement_like_gaps >= 1


def test_repair_replaces_reworded_chunk_with_source(tmp_path, config):
    docs = load_bundle([_chunk_file(tmp_path, CHUNKS)])
    spec = derive_strategy("t_repaired", "staged_verified", {"chunking": {"repair": "source_slice"}})
    result = run_analysis(docs, strategy=spec, config=config)
    reworded = next(r for r in result.requirements if r.chapter == "2.3")
    assert reworded.source_location.match in ("exact", "whitespace_normalized")  # verbatim source text
    assert "shall be validated" in reworded.requirement_text and "CVM-TM-320" in reworded.requirement_text


def test_chunk_units_extract_references_per_chunk(tmp_path, config):
    docs = load_bundle([_chunk_file(tmp_path, CHUNKS)])
    spec = derive_strategy("t_units", "staged", {"chunking": {"unit": "chunk"}, "conflicts": {"enabled": False}})
    result = run_analysis(docs, strategy=spec, config=config)
    assert len(result.requirements) == 3  # one unit per chunk
    unit = next(r for r in result.requirements if "CVM-QS-110" in r.requirement_text)
    assert {"DOC-CVM-QS-110", "DOC-DRW-045", "DOC-CVM-PKG-050"} <= set(unit.referenced_document_ids)


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
    case = next(c for c in dataset.cases if c.case_id == "dev_corvane_production")
    docs = load_bundle(list(case.document_paths))
    gold = case.load_gold()
    chunked = evaluate_case(run_analysis(docs, strategy="llm_chunked", config=config), gold)
    staged = evaluate_case(run_analysis(docs, strategy="staged", config=config), gold)
    assert chunked.units.mean_gold_per_unit > 1 and staged.units.mean_gold_per_unit == 1
    assert chunked.units.recall > 0.5  # documents are found ...
    assert chunked.references.tp == 0  # ... but not attributed to individual requirements
    assert chunked.references.errors["wrong_requirement"] + chunked.references.errors["missed"] > 0
    assert staged.units.f1 == staged.references.f1
