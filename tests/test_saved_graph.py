"""tests/test_saved_graph.py

Documents without chapters (passages, no requirements), conflicts with content already in
the saved graph, and pruning conflicts of superseded revisions.
"""

from csr_lab.config import ChunkingSettings
from csr_lab.ingestion.chapters import split_chapters
from csr_lab.ingestion.loader import load_bundle
from csr_lab.pipelines.run import run_analysis
from csr_lab.relationships.builder import graph_from_result
from csr_lab.relationships.render import to_llm_markdown
from csr_lab.relationships.store import GraphStore
from csr_lab.schemas import RelationshipType

CSR = """\
HALVARD MOTORS
Thermal Requirements
Document ID: HVG-CSR-910
Revision: {rev}
Effective Date: {date}

1 Scope
1.1 This document applies to test suppliers.

2 Requirements
2.1 Parts shall be validated at a maximum test temperature of 85 °C according to
HVG-TM-320.
2.2 The supplier shall keep records for five years.
"""

# A contract: no numbered chapters, only paragraphs.
CONTRACT = """\
SUPPLY AGREEMENT
Document ID: HVG-SA-001

This agreement is made between Halvard Motors and the supplier for the supply of
thermal sensors. Deliveries shall follow the Halvard Logistics Manual HVG-LG-140.

Thermal validation according to HVG-TM-320 shall be performed at a minimum test
temperature of 105 °C for every delivered lot.

The supplier keeps the quality records described in the Quality Manual HVG-QM-007
and makes them available on request.
"""

NEW_SPEC = """\
HALVARD MOTORS
Thermal Test Specification
Document ID: HVG-TS-930
Revision: A

1 Validation
1.1 Thermal validation according to HVG-TM-320 shall be performed at a minimum test
temperature of 105 °C.
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _conflict_edges(store_or_graph):
    edges = store_or_graph.edges.values() if isinstance(store_or_graph, GraphStore) else store_or_graph.edges
    return [e for e in edges if e.relationship_type == RelationshipType.POTENTIALLY_CONFLICTS_WITH]


def test_document_without_chapters_becomes_passages(tmp_path):
    (doc,) = load_bundle([_write(tmp_path, "contract.txt", CONTRACT)])
    segs, notes = split_chapters(doc, ChunkingSettings(passage_words=50))
    assert "no chapters" in notes[0]
    assert len(segs) >= 2 and all(s.kind == "passage" and s.chapter is None for s in segs)
    assert all((doc.text or "")[s.start_offset : s.end_offset] == s.text for s in segs)
    assert "HVG-TM-320" in " ".join(s.text for s in segs)


def test_passages_give_references_and_conflicts_but_no_requirements(tmp_path, config):
    docs = load_bundle([_write(tmp_path, "csr.txt", CSR.format(rev="A", date="2026-01-01")), _write(tmp_path, "contract.txt", CONTRACT)])
    spec_config = config.model_copy(deep=True)
    spec_config.processing.chunk_max_chars = 100000
    result = run_analysis(docs, strategy="graph_assisted", config=spec_config)
    contract_id = next(d.document_id for d in docs if d.canonical_identifier == "HVG-SA-001")
    assert not [r for r in result.requirements if r.source_document_id == contract_id]  # no requirement from the contract
    passages = [p for p in result.document_passages if p.source_document_id == contract_id]
    assert passages and all(p.passage_id.startswith("PSG-HVG-SA-001-P") for p in passages)

    # A document that was never provided (the logistics manual) is still a node.
    graph = graph_from_result(result)
    manual = graph.node("DOC-HVG-LG-140")
    assert manual is not None and manual.attributes["content_available"] is False
    stated = [e for e in graph.edges if e.source_node_id == contract_id and e.metadata.get("passage_ids")]
    assert {e.target_node_id for e in stated} >= {"DOC-HVG-LG-140", "DOC-HVG-TM-320", "DOC-HVG-QM-007"}
    assert graph.node(contract_id).attributes["has_chapters"] is False
    assert len(graph.node(contract_id).attributes["passages"]) == len(passages)

    # The 85 °C chapter and the 105 °C contract passage: a conflict drawn to the contract node.
    conflict = next(c for c in result.potential_conflicts if c.related_passage_ids)
    assert len(conflict.related_requirement_ids) == 1 and conflict.related_requirement_ids[0].endswith("-2.1")
    edge = next(e for e in _conflict_edges(graph) if contract_id in (e.source_node_id, e.target_node_id))
    assert edge.metadata["passage_ids"] == conflict.related_passage_ids
    text = to_llm_markdown(graph)
    assert "## Documents without chapters" in text and "no chapters" in text and "(text)" in text


def test_new_document_is_checked_against_the_saved_graph(tmp_path, config):
    store = GraphStore()
    first = run_analysis(load_bundle([_write(tmp_path, "csr.txt", CSR.format(rev="A", date="2026-01-01"))]), strategy="graph_assisted", config=config)
    store.merge_result(first)
    assert not _conflict_edges(store)

    later = load_bundle([_write(tmp_path, "spec.txt", NEW_SPEC)])
    units = store.saved_units()
    assert "REQ-HVG-CSR-910-REV-A-2.1" in units and "DOC-HVG-TM-320" in units["REQ-HVG-CSR-910-REV-A-2.1"].referenced
    result = run_analysis(later, strategy="graph_assisted", config=config, saved_graph=store)
    found = [c for c in result.potential_conflicts if c.related_saved_ids]
    assert found and found[0].related_saved_ids == ["REQ-HVG-CSR-910-REV-A-2.1"]
    assert "saved_graph_check" in found[0].detection_methods
    assert any(e.verified for e in found[0].evidence if e.requirement_id == "REQ-HVG-CSR-910-REV-A-2.1")
    assert result.metadata.saved_graph["saved_units"] == len(units)
    assert any(d.decision_type == "saved_graph_conflict_check" for d in result.metadata.graph_decisions)

    report = store.merge_result(result)
    assert report.saved_graph_conflicts_added == 1
    edge = next(e for e in _conflict_edges(store) if e.metadata.get("found_by") == "saved_graph_check")
    assert {edge.source_node_id, edge.target_node_id} == {"REQ-HVG-CSR-910-REV-A-2.1", "REQ-HVG-TS-930-REV-A-1.1"}

    # Without a saved graph (or with the check off) the new document is analysed alone.
    alone = run_analysis(later, strategy="graph_assisted", config=config)
    assert not any(c.related_saved_ids for c in alone.potential_conflicts)


def test_conflicts_of_superseded_revisions_are_pruned(tmp_path, config):
    store = GraphStore()
    for name, text in (("csr.txt", CSR.format(rev="A", date="2026-01-01")), ("spec.txt", NEW_SPEC)):
        store.merge_result(run_analysis(load_bundle([_write(tmp_path, name, text)]), strategy="graph_assisted", config=config, saved_graph=store))
    assert any(e.metadata.get("found_by") == "saved_graph_check" for e in _conflict_edges(store))

    # Revision B of the CSR no longer has the 85 °C limit: the old conflict goes with revision A.
    rev_b = CSR.format(rev="B", date="2026-06-01").replace("maximum test temperature of 85 °C", "test temperature of 105 °C")
    result = run_analysis(load_bundle([_write(tmp_path, "csr_b.txt", rev_b)]), strategy="graph_assisted", config=config, saved_graph=store)
    report = store.merge_result(result, prune_superseded_conflicts=True)
    assert report.superseded_conflicts_pruned == 1
    assert store.nodes["REQ-HVG-CSR-910-REV-A-2.1"].attributes["revision_status"] == "superseded"
    assert not _conflict_edges(store)


def test_reanalysing_a_document_without_chapters_replaces_its_passages(tmp_path, config):
    store = GraphStore()
    path = _write(tmp_path, "contract.txt", CONTRACT)
    store.merge_result(run_analysis(load_bundle([path]), strategy="staged", config=config))
    doc_id = "DOC-HVG-SA-001"
    first = len(store.nodes[doc_id].attributes["passages"])
    assert store.saved_units() and all(u.kind == "passage" for u in store.saved_units().values())
    path.write_text(CONTRACT.replace("HVG-QM-007", "HVG-QM-008"), encoding="utf-8")
    store.merge_result(run_analysis(load_bundle([path]), strategy="staged", config=config))
    targets = {e.target_node_id for e in store.edges.values() if e.source_node_id == doc_id and e.metadata.get("passage_ids")}
    assert "DOC-HVG-QM-008" in targets and "DOC-HVG-QM-007" not in targets
    assert len(store.nodes[doc_id].attributes["passages"]) == first


def test_saved_graph_experiment_grades_both_arms(tmp_path, config):
    from csr_lab.config import ExperimentConfig
    from csr_lab.evaluation.runner import run_experiment
    from csr_lab.pipelines.registry import register_variants

    register_variants({"graph_assisted_no_saved": {"base": "graph_assisted", "options": {"graph": {"use_saved": False}}}})
    experiment = ExperimentConfig(
        name="t_saved",
        dataset="data/synthetic/incremental/manifest.json",
        split="all",
        case_ids=["heldout_halvard_delta"],
        saved_graph={"build_from": "data/synthetic/incremental/graph_input"},
        strategies=["graph_assisted", "graph_assisted_no_saved"],
        llm={"provider": "mock"},
    )
    outcome = run_experiment(config, experiment, output_root=tmp_path, database_path=tmp_path / "e.sqlite")
    agg = outcome.summary["aggregates"]
    with_graph, alone = agg["graph_assisted"], agg["graph_assisted_no_saved"]
    assert with_graph["conflicts.saved_graph_hits"]["mean"] > 0 and alone["conflicts.saved_graph_predicted"]["mean"] == 0
    assert with_graph["conflicts.recall"]["mean"] > alone["conflicts.recall"]["mean"]
    # Only the new document is scored for requirements; the graph documents are not missed requirements.
    assert with_graph["requirements.recall"]["mean"] == 1.0
    info = outcome.summary["saved_graph"]
    assert info["build_llm_calls"] > 0 and (outcome.output_dir / "saved_graph.json").is_file()
    assert "Saved graph" in (outcome.output_dir / "report.md").read_text()


def test_sweep_builds_the_saved_graph_once(tmp_path, config):
    from csr_lab.config import ExperimentConfig
    from csr_lab.evaluation.sweep import SweepConfig, run_sweep

    base = ExperimentConfig(
        name="t_saved",
        dataset="data/synthetic/incremental/manifest.json",
        split="all",
        case_ids=["heldout_halvard_delta"],
        saved_graph={"build_from": "data/synthetic/incremental/graph_input"},
        strategies=["graph_assisted"],
        llm={"provider": "mock"},
    )
    sweep = SweepConfig(name="t", experiment="t_saved", grid={"options.graph.saved_max_pairs": [1, 50]}, metrics=["conflicts.saved_graph_predicted"])
    out_dir, rows = run_sweep(sweep, base, config, output_root=tmp_path, database_path=tmp_path / "s.sqlite")
    assert [p.parent for p in out_dir.rglob("saved_graph.json")] == [out_dir]  # one graph for every setting
    assert rows[0]["conflicts.saved_graph_predicted"] <= 1 < rows[1]["conflicts.saved_graph_predicted"]
