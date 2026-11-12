"""tests/test_pdf_and_sweep.py

PDF and incremental variants, header/footer removal, readers, gold re-anchoring, sweeps, CLI.
"""

import json

from helpers import run_case

from csr_lab.cli import main
from csr_lab.config import PROJECT_ROOT, ExperimentConfig, load_config
from csr_lab.datasets.loader import load_dataset
from csr_lab.datasets.variants import write_incremental_dataset, write_pdf_dataset
from csr_lab.evaluation.matching import reanchor_gold
from csr_lab.evaluation.metrics import evaluate_case
from csr_lab.evaluation.sweep import SweepConfig, run_sweep
from csr_lab.ingestion.loader import load_bundle, strip_running_lines


def test_strip_repeated_lines_only_touches_page_edges():
    pages = [
        "ACME | DOC-1 Rev A\nBody line one\nshared body line\nPage 1 of 3",
        "ACME | DOC-1 Rev A\nshared body line\nBody line two\nmore\nmore\nmore\nPage 2 of 3",
        "ACME | DOC-1 Rev A\nBody line three\nmore\nmore\nmore\nshared body line\nmore\nmore\nmore\nPage 3 of 3",
    ]
    cleaned, removed, _originals = strip_running_lines(pages)
    assert "page of" in removed and "acme | doc- rev a" in removed  # digits are ignored when lines are compared
    assert all("Page" not in p and "ACME" not in p for p in cleaned)
    assert "shared body line" in cleaned[2]  # repeated, but in the body: kept


def test_pdf_dataset_loads_and_scores_like_the_text(tmp_path, config, dataset):
    manifest = write_pdf_dataset(tmp_path / "pdf", lines_per_page=30)
    pdf = load_dataset(manifest)
    (case,) = pdf.cases
    gold = case.load_gold()
    for strip in (False, True):
        docs = load_bundle(list(case.document_paths), pdf_strip_repeated_lines=strip)
        assert {d.document_id for d in docs} == {d.document_id for d in gold.documents}
        assert all(d.format == "pdf" and len(d.pages) >= 2 for d in docs)
        anchored = reanchor_gold(gold, docs)
        texts = {d.document_id: d.text for d in docs}
        # Every gold requirement is found in the extracted text after re-anchoring.
        assert all(" ".join(texts[g.source_document_id][g.start_offset : g.end_offset].split()).startswith(" ".join(g.requirement_text.split()[:5])) for g in anchored.requirements)
    config = config.model_copy(update={"processing": config.processing.model_copy(update={"pdf_strip_repeated_lines": True})})
    as_pdf, _ = run_case(pdf, "staged_verified", config)
    as_text, text_gold = run_case(dataset, "staged_verified", config)
    ev_pdf = evaluate_case(as_pdf, gold, customer="Global Motors")
    ev_text = evaluate_case(as_text, text_gold, customer="Global Motors")
    assert ev_pdf.requirements.recall == 1.0
    # Same case, same gold: with running headers and page breaks removed, the PDF scores like the text.
    assert (ev_pdf.references.f1, ev_pdf.conflicts.f1, ev_pdf.graph.chapter_link_f1) == (ev_text.references.f1, ev_text.conflicts.f1, ev_text.graph.chapter_link_f1)


def test_pdf_layout_backend(tmp_path):
    manifest = write_pdf_dataset(tmp_path / "pdf")
    (case,) = load_dataset(manifest).cases
    docs = {d.filename: d for d in load_bundle(list(case.document_paths), pdf_backend="pypdf_layout")}
    base = docs["halvard_q_agreement.pdf"]
    assert "Part labels shall show the part number" in base.text and any("pypdf_layout" in w for w in base.warnings)


def test_committed_variant_datasets_match_generator(tmp_path):
    for name, writer in (("pdf", write_pdf_dataset), ("incremental", write_incremental_dataset)):
        fresh = writer(tmp_path / name).parent
        for path in fresh.rglob("*"):
            if path.is_file():
                rel = path.relative_to(fresh)
                committed = PROJECT_ROOT / "data" / "synthetic" / name / rel
                assert committed.read_bytes() == path.read_bytes(), f"data/synthetic/{name}/{rel} is out of date: regenerate it"


def test_sweep_over_strategy_option_and_mock_rate(tmp_path):
    sweep = SweepConfig(
        name="t",
        experiment="demo_mock",
        strategies=["llm_chunked"],
        repetitions=1,
        grid={"options.chunking.target_chunk_words": [60, 500], "llm.mock.reference_miss_rate": [0.0]},
        metrics=["units.f1", "units.mean_gold_per_unit"],
    )
    base = ExperimentConfig(name="demo_mock", split="all", strategies=["llm_chunked"], llm={"provider": "mock"})
    out_dir, rows = run_sweep(sweep, base, load_config(), output_root=tmp_path, database_path=tmp_path / "s.sqlite")
    assert len(rows) == 2 and {r["strategy"] for r in rows} == {"llm_chunked"}
    small, large = rows
    assert small["units.mean_gold_per_unit"] < large["units.mean_gold_per_unit"]  # smaller chunks hold fewer requirements
    report = (out_dir / "sweep_report.md").read_text()
    assert "target_chunk_words=60" in report and (out_dir / "sweep.csv").is_file()


def test_check_text_and_chunk_commands(tmp_path, capsys, simple_doc_path):
    frags = tmp_path / "frags.json"
    frags.write_text(json.dumps(["2.5 The supplier shall keep records for five years.", "2.5 The supplier must keep records for 5 years."]))
    out = tmp_path / "report.json"
    assert main(["check-text", str(simple_doc_path), str(frags), "--out", str(out)]) == 0
    assert "fuzzy (rewritten): 1" in capsys.readouterr().out
    assert json.loads(out.read_text())["summary"]["exact"] == 1
    assert main(["chunk", str(simple_doc_path), "--out-dir", str(tmp_path / "chunks"), "--target-words", "30"]) == 0
    chunk_file = tmp_path / "chunks" / "simple.chunks.json"
    body = json.loads(chunk_file.read_text())
    assert body and {"id", "title", "text"} <= set(body[0])
    assert main(["check-text", str(simple_doc_path), str(chunk_file)]) == 0  # the written file can be checked
    assert main(["llm-check"]) == 0
