"""tests/test_store.py

Requirement identity, document revisions and the saved (persistent) graph store.
"""

import json

from csr_lab.cli import main
from csr_lab.ids import requirement_id
from csr_lab.ingestion.loader import load_bundle
from csr_lab.pipelines.run import run_analysis
from csr_lab.relationships.builder import graph_from_result
from csr_lab.relationships.revisions import RequirementVersion, compare_revisions
from csr_lab.relationships.store import GraphStore
from csr_lab.relationships.validation import validate_graph
from csr_lab.schemas import RelationshipType

HEADER = "CORVANE MOTORS\n{title}\nDocument ID: {ident}\nRevision: {rev}\nEffective Date: {date}\n\n"

REV_C = HEADER.format(title="Supplier Requirements", ident="CVM-CSR-700", rev="C", date="2026-01-10") + """\
1 Scope
1.1 This document applies to test suppliers.

2 Requirements
2.1 The supplier shall comply with Specification CVM-QS-110.
2.2 The supplier shall report quality incidents within 10 working days.
2.3 The supplier shall keep records for five years.
"""

# Rev D: 2.1 unchanged, 2.2 changed, 2.3 removed, 2.4 added.
REV_D = HEADER.format(title="Supplier Requirements", ident="CVM-CSR-700", rev="D", date="2026-08-01") + """\
1 Scope
1.1 This document applies to test suppliers.

2 Requirements
2.1 The supplier shall comply with Specification CVM-QS-110.
2.2 The supplier shall report quality incidents within 5 working days.
2.4 The supplier shall label every shipment according to Standard CVM-LBL-220.
"""

NAMES_WITH_ID = HEADER.format(title="Production Requirements", ident="CVM-CSR-710", rev="A", date="2026-01-10") + """\
1 Requirements
1.1 The supplier shall maintain a quality system that conforms to the Corvane Supplier Quality
Manual CVM-SQM-001.
"""

NAME_ONLY = HEADER.format(title="Service Requirements", ident="CVM-CSR-720", rev="B", date="2026-03-01") + """\
1 Requirements
1.1 Service parts shall be packed as described in the Corvane Supplier Quality Manual.
"""


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _run(tmp_path, name, text, config, **kwargs):
    return run_analysis(load_bundle([_write(tmp_path, name, text)]), strategy=kwargs.pop("strategy", "graph_assisted"), config=config, **kwargs)


# -- identity ------------------------------------------------------------------


def test_requirement_id_comes_from_revision_chapter_and_text():
    rid = requirement_id("DOC-CVM-CSR-700-REV-C", "2.1", "The supplier shall comply.")
    assert rid.startswith("REQ-CVM-CSR-700-REV-C-2.1-")
    assert rid == requirement_id("DOC-CVM-CSR-700-REV-C", "2.1", "The supplier\nshall  comply.")  # wrapping ignored
    assert rid != requirement_id("DOC-CVM-CSR-700-REV-C", "2.2", "The supplier shall comply.")
    assert rid != requirement_id("DOC-CVM-CSR-700-REV-D", "2.1", "The supplier shall comply.")
    assert rid != requirement_id("DOC-CVM-CSR-700-REV-C", "2.1", "The supplier shall not comply.")
    assert "-NOCHAPTER-" in requirement_id("DOC-X", None, "text")


def test_pipeline_ids_are_stable_across_runs(tmp_path, config):
    a = _run(tmp_path, "c.txt", REV_C, config, run_seed=1)
    b = _run(tmp_path, "c.txt", REV_C, config, run_seed=2)
    assert [r.requirement_id for r in a.requirements] == [r.requirement_id for r in b.requirements]
    assert all(r.source_document_id == "DOC-CVM-CSR-700-REV-C" for r in a.requirements)


def test_compare_revisions():
    old = [RequirementVersion("o1", "1", "Same text."), RequirementVersion("o2", "2", "Old wording."),
           RequirementVersion("o3", "3", "Moved text."), RequirementVersion("o4", "4", "Dropped.")]
    new = [RequirementVersion("n1", "1", "Same  text."), RequirementVersion("n2", "2", "New wording."),
           RequirementVersion("n3", "9", "Moved text."), RequirementVersion("n5", "5", "Brand new.")]
    changes = {(c.change, c.old.requirement_id if c.old else None, c.new.requirement_id if c.new else None) for c in compare_revisions(old, new)}
    assert changes == {("unchanged", "o1", "n1"), ("changed", "o2", "n2"), ("moved", "o3", "n3"), ("removed", "o4", None), ("added", None, "n5")}


# -- revisions inside one run ----------------------------------------------------


def test_reference_to_provided_revision_meets_input_node(tmp_path, config):
    provided = HEADER.format(title="Packaging Specification", ident="CVM-PKG-210", rev="D", date="2026-01-01") + (
        "1 Requirements\n1.1 Parts shall be packed in returnable containers.\n"
    )
    citing = HEADER.format(title="Supplier Requirements", ident="CVM-CSR-730", rev="A", date="2026-01-01") + (
        "1 Requirements\n"
        "1.1 Packaging shall conform to CVM-PKG-210 Rev D.\n"
        "1.2 Labels shall follow CVM-PKG-210.\n"
        "1.3 Legacy parts shall use CVM-PKG-210 Rev C.\n"
    )
    docs = load_bundle([_write(tmp_path, "pkg.txt", provided), _write(tmp_path, "csr.txt", citing)])
    result = run_analysis(docs, strategy="graph_assisted", config=config)
    available = {d.document_id: d.content_available for d in result.referenced_documents}
    assert available == {"DOC-CVM-PKG-210-REV-D": True, "DOC-CVM-PKG-210": True, "DOC-CVM-PKG-210-REV-C": False}
    graph = graph_from_result(result)
    assert validate_graph(graph).valid
    assert graph.current_revision("DOC-CVM-PKG-210") == "DOC-CVM-PKG-210-REV-D"
    assert set(graph.revisions_of("DOC-CVM-PKG-210")) == {"DOC-CVM-PKG-210-REV-C", "DOC-CVM-PKG-210-REV-D"}
    assert not graph.same_document("DOC-CVM-PKG-210-REV-C", "DOC-CVM-PKG-210-REV-D")


# -- the store -----------------------------------------------------------------------


def test_store_tracks_revisions(tmp_path, config):
    store = GraphStore()
    rev_c = _run(tmp_path, "c.txt", REV_C, config)
    rev_d = _run(tmp_path, "d.txt", REV_D, config)
    store.merge_result(rev_d)  # merge order does not matter: effective dates decide
    store.merge_result(rev_c)
    # Another document citing an older, never-analysed revision must not disturb the order.
    citing = HEADER.format(title="Other", ident="CVM-CSR-740", rev="A", date="2026-01-01") + "1 Requirements\n1.1 Old parts shall follow CVM-CSR-700 Rev B.\n"
    store.merge_result(_run(tmp_path, "e.txt", citing, config))
    assert "DOC-CVM-CSR-700-REV-B" in store.nodes

    graph = store.to_graph()
    assert validate_graph(graph).valid
    assert graph.revisions_of("DOC-CVM-CSR-700") == ["DOC-CVM-CSR-700-REV-C", "DOC-CVM-CSR-700-REV-D", "DOC-CVM-CSR-700-REV-B"]
    assert graph.current_revision("DOC-CVM-CSR-700") == "DOC-CVM-CSR-700-REV-D"
    status = {n.node_id: n.attributes["revision_status"] for n in graph.nodes_of_type("requirement")}
    assert {status[r.requirement_id] for r in rev_c.requirements} == {"superseded"}
    assert {status[r.requirement_id] for r in rev_d.requirements} == {"current"}

    changes = {(c["change"], c["old_chapter"], c["new_chapter"]) for c in store.revision_changes("DOC-CVM-CSR-700")}
    assert {("changed", "2.2", "2.2"), ("removed", "2.3", None), ("added", None, "2.4")} <= changes
    assert ("unchanged", "2.1", "2.1") in changes
    lineage = [e for e in graph.edges if e.relationship_type == RelationshipType.SUPERSEDED_BY]
    assert any(e.source_node_id == "DOC-CVM-CSR-700-REV-C" and e.target_node_id == "DOC-CVM-CSR-700-REV-D" for e in lineage)
    assert any(e.metadata.get("change") == "changed" for e in lineage)

    # A declared order overrides dates.
    store.set_revision_order("DOC-CVM-CSR-700", ["D", "C"])
    assert store.to_graph().current_revision("DOC-CVM-CSR-700") == "DOC-CVM-CSR-700-REV-C"


def test_store_save_load_remove_and_prune(tmp_path, config):
    path = tmp_path / "store.json"
    store = GraphStore()
    rev_c = _run(tmp_path, "c.txt", REV_C, config)
    rev_d = _run(tmp_path, "d.txt", REV_D, config)
    store.merge_result(rev_c)
    store.merge_result(rev_d)
    store.save(path)
    loaded = GraphStore.load(path)
    assert loaded.fingerprint() == store.fingerprint()

    # Re-merging the same result changes nothing; re-analysing a revision replaces its requirements.
    before = loaded.fingerprint()
    loaded.merge_result(rev_d)
    assert loaded.fingerprint() == before
    assert loaded.merge_result(_run(tmp_path, "d.txt", REV_D, config, run_seed=7)).requirements_replaced == len(rev_d.requirements)

    report = loaded.prune(drop_superseded=True)
    assert report.nodes_removed >= len(rev_c.requirements)
    assert not any(n.attributes.get("revision_status") == "superseded" for n in loaded.nodes.values())

    loaded.remove_source(rev_c.analysis_id)
    assert "DOC-CVM-CSR-700-REV-C" not in loaded.nodes and "DOC-CVM-CSR-700-REV-D" in loaded.nodes
    assert validate_graph(loaded.to_graph()).valid

    loaded.prune(drop_mock=True)
    assert loaded.nodes == {} and loaded.sources == {}


def test_saved_graph_resolves_a_name_seen_in_an_earlier_run(tmp_path, config):
    store = GraphStore()
    store.merge_result(_run(tmp_path, "a.txt", NAMES_WITH_ID, config))

    fresh = _run(tmp_path, "b.txt", NAME_ONLY, config)
    with_saved = _run(tmp_path, "b.txt", NAME_ONLY, config, saved_graph=store)

    def refs(result):
        return {d.canonical_identifier or d.document_name for d in result.referenced_documents}

    assert "CVM-SQM-001" not in refs(fresh)
    assert "CVM-SQM-001" in refs(with_saved)
    assert any("saved graph" in d.detail for d in with_saved.metadata.graph_decisions)
    assert with_saved.metadata.saved_graph["fingerprint"] == store.fingerprint()
    assert with_saved.analysis_id != fresh.analysis_id
    # The exported result holds only this run's own documents and references.
    assert {d.document_id for d in with_saved.input_documents} == {"DOC-CVM-CSR-720-REV-B"}
    assert "DOC-CVM-CSR-710-REV-A" not in {d.document_id for d in with_saved.referenced_documents}


def test_saved_graph_is_ignored_without_graph_use(tmp_path, config):
    store = GraphStore()
    store.merge_result(_run(tmp_path, "a.txt", NAMES_WITH_ID, config))
    result = _run(tmp_path, "b.txt", NAME_ONLY, config, strategy="staged", saved_graph=store)
    assert result.metadata.saved_graph is None
    assert any("saved graph ignored" in w for w in result.metadata.warnings)


def test_graph_store_cli(tmp_path, capsys):
    store = tmp_path / "store.json"
    a, b = _write(tmp_path, "c.txt", REV_C), _write(tmp_path, "d.txt", REV_D)
    for doc in (a, b):
        assert main(["analyze", str(doc), "--graph-store", str(store), "--out", str(tmp_path / f"{doc.stem}.json")]) == 0
    capsys.readouterr()
    assert main(["graph-store", "info", str(store)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["validation"]["valid"] and info["analysed_document_revisions"] == 2
    assert info["requirements_by_revision_status"]["superseded"] > 0
    assert main(["graph-store", "revisions", str(store), "DOC-CVM-CSR-700"]) == 0
    assert any(c["change"] == "added" for c in json.loads(capsys.readouterr().out)["changes"])
    assert main(["graph-store", "prune", str(store), "--superseded"]) == 0
    assert main(["analyze", str(a), "--use-saved-graph"]) == 1  # needs --graph-store
