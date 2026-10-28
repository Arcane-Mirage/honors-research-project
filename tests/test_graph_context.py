"""tests/test_graph_context.py

Graph context formats for the conflict prompt (graph.context_format).
"""

import json

import pytest

from csr_lab.conflicts.detector import CONTEXT_FORMATS, graph_context_for_pair
from csr_lab.pipelines.registry import derive_strategy
from csr_lab.pipelines.run import run_analysis
from csr_lab.relationships.builder import graph_from_result
from helpers import ScriptedProvider, case_documents

from csr_lab.llm.mock_provider import MockProvider


@pytest.fixture
def packaging(dataset, config):
    _, docs, _ = case_documents(dataset, "dev_corvane_packaging")
    result = run_analysis(docs, strategy="graph_assisted", config=config)
    graph = graph_from_result(result)
    by_chapter = {(r.source_document_id, r.chapter): r.requirement_id for r in result.requirements}
    pair = (by_chapter[("DOC-CVM-CSR-200-REV-A", "1.3")], by_chapter[("DOC-CVM-PKG-210-REV-D", "2.1")])
    return graph, pair


def test_every_format_uses_a_b_labels_except_ids_for_other_requirements(packaging):
    graph, pair = packaging
    out = {f: graph_context_for_pair(graph, pair, 6, f) for f in CONTEXT_FORMATS}
    for fmt in ("sentences", "triples", "json"):
        text = json.dumps(out[fmt])
        assert "REQ-" not in text, fmt  # no raw requirement IDs left for the model to decode
    assert any("REQ-" in item for item in out["ids"])  # the original rendering keeps them
    assert out["sentences"][0] == "Requirement A states that document CVM-PKG-050 does not apply (content not provided)."
    assert all(isinstance(item, dict) for item in out["json"])
    assert all(item.count(" | ") == 3 for item in out["triples"])


def test_redundant_reference_is_collapsed_but_not_in_ids(packaging):
    graph, pair = packaging
    sentences = graph_context_for_pair(graph, pair, 6, "sentences")
    ids = graph_context_for_pair(graph, pair, 6, "ids")
    assert not any(s.startswith("Requirement A refers to document CVM-PKG-050") for s in sentences)
    assert "A references CVM-PKG-050 (not provided)" in ids


def test_context_is_deterministic_and_capped(packaging):
    graph, pair = packaging
    first = graph_context_for_pair(graph, pair, 2, "sentences")
    assert first == graph_context_for_pair(graph, pair, 2, "sentences") and len(first) == 2
    with pytest.raises(ValueError):
        graph_context_for_pair(graph, pair, 6, "yaml")


@pytest.mark.parametrize("fmt", ["sentences", "json", "ids"])
def test_prompt_receives_the_chosen_format(dataset, config, fmt):
    _, docs, _ = case_documents(dataset, "dev_corvane_packaging")
    mock = MockProvider(config.llm.mock)
    seen = []

    def fn(task, payload, request):
        if task == "conflict_analysis":
            seen.extend(item for p in payload["pairs"] for item in p.get("related_context", []))
        return mock.generate(request).text

    spec = derive_strategy(f"t_ctx_{fmt}", "graph_assisted", {"graph": {"context_format": fmt}})
    result = run_analysis(docs, strategy=spec, config=config, provider=ScriptedProvider(fn))
    assert seen
    assert all(isinstance(x, dict) for x in seen) == (fmt == "json")
    decision = next(d for d in result.metadata.graph_decisions if d.decision_type == "context_retrieved")
    assert f"({fmt})" in decision.detail
