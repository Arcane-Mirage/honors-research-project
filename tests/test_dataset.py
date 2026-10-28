"""tests/test_dataset.py

Synthetic dataset: deterministic generation, committed files up to date, gold spans,
and gold annotations kept out of reach of the pipelines.
"""

from pathlib import Path

from csr_lab.config import PROJECT_ROOT
from csr_lab.datasets.loader import load_dataset
from csr_lab.datasets.synthetic import build_case, generate_scaled_case, write_dataset
from csr_lab.ingestion.loader import load_bundle


def test_generator_is_deterministic(tmp_path):
    a = write_dataset(tmp_path / "a").parent
    b = write_dataset(tmp_path / "b").parent
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert files_a == files_b
    for rel in files_a:
        assert (a / rel).read_bytes() == (b / rel).read_bytes()


def test_committed_dataset_matches_generator(tmp_path):
    committed = PROJECT_ROOT / "data" / "synthetic"
    fresh = write_dataset(tmp_path / "fresh").parent
    for path in fresh.rglob("*"):
        if path.is_file():
            rel = path.relative_to(fresh)
            assert (committed / rel).read_bytes() == path.read_bytes(), f"{rel} is out of date: run `csr-lab generate-data`"


def test_gold_spans_and_minimum_counts(dataset):
    totals = {"req": 0, "ref": 0, "conflicts": 0, "no_ref": 0, "available_cross": 0}
    splits = set()
    for case in dataset.cases:
        splits.add(case.split)
        gold = case.load_gold()
        docs = {d.document_id: d for d in load_bundle(list(case.document_paths))}
        assert set(docs) == {d.document_id for d in gold.documents}
        for req in gold.requirements:
            assert docs[req.source_document_id].text[req.start_offset : req.end_offset] == req.requirement_text
            for ref in req.references:
                assert " ".join(ref.mention_text.split()) in " ".join(req.requirement_text.split())
        totals["req"] += len(gold.requirements)
        totals["ref"] += sum(len(r.references) for r in gold.requirements)
        totals["conflicts"] += len(gold.conflicts)
        totals["no_ref"] += sum(1 for r in gold.requirements if not r.references)
        provided = {d.canonical_identifier for d in gold.documents}
        totals["available_cross"] += sum(
            1 for r in gold.requirements for ref in r.references if ref.canonical_identifier in provided
        )
    assert len(dataset.cases) >= 5 and splits == {"dev", "heldout"}
    assert totals["req"] >= 25 and totals["ref"] >= 40 and totals["conflicts"] >= 10
    assert totals["no_ref"] >= 3 and totals["available_cross"] >= 2


def test_dataset_hash_changes_with_content(tmp_path):
    root = write_dataset(tmp_path / "d").parent
    before = load_dataset(root).manifest_sha256
    doc = next((root / "documents").rglob("*.txt"))
    doc.write_text(doc.read_text() + "\n", encoding="utf-8")
    assert load_dataset(root).manifest_sha256 != before


def test_scaled_case_is_consistent():
    case = generate_scaled_case(n_requirements=25, refs_per_requirement=3, filler_sentences=4, position="end", seed=2)
    files, gold = build_case(case)
    assert len(gold.requirements) == 25
    text = next(iter(files.values()))
    for req in gold.requirements:
        assert text[req.start_offset : req.end_offset] == req.requirement_text
        assert 1 <= len(req.references) <= 3


def test_gold_is_not_reachable_from_pipeline_modules():
    """Pipelines and the mock model never import the gold schema loader."""
    src = Path(__file__).parents[1] / "src" / "csr_lab"
    for folder in ("pipelines", "llm", "extraction", "conflicts", "relationships", "prompts", "ingestion"):
        for py in (src / folder).glob("*.py"):
            text = py.read_text(encoding="utf-8")
            assert "load_gold" not in text and "CaseGold" not in text, py
