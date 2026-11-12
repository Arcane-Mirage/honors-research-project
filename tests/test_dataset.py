"""tests/test_dataset.py

Synthetic dataset: deterministic generation, committed files up to date, gold consistent with
how the pipeline cuts the files, self-checks that catch gold mistakes, an error-free model
scoring perfectly on every structural metric, and gold kept out of reach of the pipelines.
"""

import dataclasses
from pathlib import Path

import pytest

from csr_lab.config import PROJECT_ROOT, ExperimentConfig, apply_experiment, dataset_reader, load_config
from csr_lab.datasets.loader import load_dataset
from csr_lab.datasets.synthetic import READER, Ref, all_cases, build_case, write_dataset
from csr_lab.datasets.variants import write_noisy_dataset
from csr_lab.evaluation.metrics import evaluate_case
from csr_lab.pipelines.run import load_inputs, run_analysis

DATA = PROJECT_ROOT / "data" / "synthetic"


def _same_tree(fresh: Path, committed: Path, hint: str) -> None:
    for path in fresh.rglob("*"):
        if path.is_file():
            rel = path.relative_to(fresh)
            assert (committed / rel).read_bytes() == path.read_bytes(), f"{rel} is out of date: run `{hint}`"


def test_generator_is_deterministic(tmp_path):
    a = write_dataset(tmp_path / "a").parent
    b = write_dataset(tmp_path / "b").parent
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    assert files_a == sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    for rel in files_a:
        assert (a / rel).read_bytes() == (b / rel).read_bytes()


def test_committed_datasets_match_generator(tmp_path):
    fresh = write_dataset(tmp_path / "text").parent
    _same_tree(fresh, DATA / "text", "csr-lab generate-data")
    _same_tree(write_noisy_dataset(fresh / "manifest.json", tmp_path / "noisy").parent, DATA / "noisy", "csr-lab generate-data --noisy")


@pytest.mark.parametrize("name", ["text", "noisy"])
def test_gold_texts_are_exact_slices_of_the_files(name):
    dataset = load_dataset(DATA / name / "manifest.json")
    for case in dataset.cases:
        gold = case.load_gold()
        docs, _registry = load_inputs(list(case.document_paths), load_config())
        by_id = {d.document_id: d for d in docs}
        assert set(by_id) == {d.document_id for d in gold.documents}
        for g in gold.requirements:
            assert by_id[g.source_document_id].text[g.start_offset : g.end_offset] == g.requirement_text, g.gold_id
            for ref in g.references:
                assert ref.mention_text in g.requirement_text, (g.gold_id, ref.mention_text)


def test_gold_covers_the_real_csr_situations(dataset):
    (case,) = dataset.cases
    gold = case.load_gold()
    refs = [r for g in gold.requirements for r in g.references]
    assert len(gold.requirements) >= 15 and len(refs) >= 15 and len(gold.conflicts) >= 3
    assert any(r.applies_to for r in refs) and any(r.location for r in refs) and any(r.resolution == "name_alias" for r in refs)
    assert {link.kind for link in gold.chapter_links} == {"inherits_from", "amends", "references_chapter"}
    assert {c.precedence for c in gold.conflicts} == {"resolved_by_precedence", "unresolved"}
    assert dataset_reader(DATA / "text" / "manifest.json") == READER


def test_noise_changes_text_but_not_gold_keys():
    clean, noisy = load_dataset(DATA / "text" / "manifest.json"), load_dataset(DATA / "noisy" / "manifest.json")
    for a, b in zip(clean.cases, noisy.cases):
        ga, gb = a.load_gold(), b.load_gold()
        assert [g.gold_id for g in ga.requirements] == [g.gold_id for g in gb.requirements]
        assert [[r.canonical_identifier for r in g.references] for g in ga.requirements] == [[r.canonical_identifier for r in g.references] for g in gb.requirements]
        assert all(p.read_text() != q.read_text() for p, q in zip(a.document_paths, b.document_paths))
    assert sum(q.read_text().count("-\n") for b in noisy.cases for q in b.document_paths) >= 2


def test_dataset_hash_changes_with_content(tmp_path):
    root = write_dataset(tmp_path / "d").parent
    before = load_dataset(root).manifest_sha256
    doc = next((root / "documents").rglob("*.txt"))
    doc.write_text(doc.read_text() + "\n", encoding="utf-8")
    assert load_dataset(root).manifest_sha256 != before


# -- the generator's self-checks reject gold mistakes --------------------------------


def _spec():
    return all_cases()[0]


def _block(case, chapter: str):
    return next(b for d in case.documents for b in d.blocks if getattr(b, "chapter", None) == chapter and d.identifier == "HVG-CVS-20")


def test_self_check_rejects_wrong_applies_to():
    case = _spec()
    block = _block(case, "0")
    block.refs = [dataclasses.replace(r, applies_to=["Brennic"]) if r.canonical == "GLB-SWR-001" else r for r in block.refs]
    with pytest.raises(AssertionError, match="applies_to"):
        build_case(case)


def test_self_check_rejects_unlabelled_reference_marked_for_a_customer():
    case = _spec()
    block = _block(case, "4.1")
    block.text = "Container labels shall follow HVG-LB-001 and be printed with a minimum barcode height of 25 mm."
    block.refs = [Ref("HVG-LB-001", "HVG-LB-001", applies_to=["Ostra"])]
    with pytest.raises(AssertionError, match="applies_to"):
        build_case(case)


def test_self_check_rejects_wrong_chapter_link_kind():
    case = _spec()
    case.chapter_links = [(a, b, "amends" if k == "inherits_from" else k) for a, b, k in case.chapter_links]
    with pytest.raises(AssertionError, match="pointer chapter"):
        build_case(case)


def test_self_check_rejects_missing_chapter_link():
    case = _spec()
    case.chapter_links = case.chapter_links[1:]
    with pytest.raises(AssertionError, match="shared chapter numbers"):
        build_case(case)


def test_self_check_rejects_wrong_prevailing_document():
    case = _spec()
    case.conflicts[0].prevailing = "DOC-ISO-9001"
    with pytest.raises(AssertionError, match="prevailing"):
        build_case(case)


# -- an error-free model scores perfectly ---------------------------------------------


def test_error_free_model_scores_perfectly_on_structure():
    perfect = {"reference_miss_rate": 0, "hallucination_rate": 0, "quote_paraphrase_rate": 0, "conflict_miss_rate": 0, "conflict_false_positive_rate": 0}
    experiment = ExperimentConfig(name="t", dataset="data/synthetic/text/manifest.json", split="all", llm={"provider": "mock", "mock": perfect})
    cfg = apply_experiment(load_config(), experiment)
    assert (cfg.system_prompt.customer, cfg.evaluation.customer) == ("Global Motors", "Global Motors")  # the dataset's reader
    (case,) = load_dataset(DATA / "text" / "manifest.json").cases
    docs, registry = load_inputs(list(case.document_paths), cfg)
    result = run_analysis(docs, strategy="graph_assisted", config=cfg, registry=registry)
    ev = evaluate_case(result, case.load_gold(), customer=cfg.evaluation.customer)
    assert (ev.requirements.precision, ev.requirements.recall, ev.requirements.verbatim_exact_rate, ev.requirements.location_accuracy) == (1, 1, 1, 1)
    assert ev.graph.chapter_link_f1 == 1.0
    assert (ev.conflicts.precision, ev.conflicts.recall, ev.conflicts.precedence_accuracy) == (1.0, 1.0, 1.0)
    assert ev.references.location_accuracy == 1.0 and ev.references.errors["unresolved_alias"] == 0


def test_gold_is_not_reachable_from_pipeline_modules():
    """Pipelines and the mock model never import the gold schema loader."""
    src = Path(__file__).parents[1] / "src" / "csr_lab"
    for folder in ("pipelines", "llm", "extraction", "conflicts", "relationships", "prompts", "ingestion"):
        for py in (src / folder).glob("*.py"):
            text = py.read_text(encoding="utf-8")
            assert "load_gold" not in text and "CaseGold" not in text, py
