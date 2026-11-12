"""tests/test_ingestion.py

Document loading: header detection, filename fallback, line endings, text and scanned PDFs, JSON input.
"""

import json
from pathlib import Path

import pytest
from helpers import make_pdf

from csr_lab.extraction.normalization import NameIndex, name_match_key
from csr_lab.ingestion.loader import RegistryEntry, load_bundle, load_document, load_registry, running_title, strip_running_lines
from csr_lab.pipelines.run import analyze_files, run_analysis
from csr_lab.schemas import ResolutionMethod


def test_text_header_detection(simple_doc_path):
    doc = load_document(simple_doc_path)
    assert (doc.document_id, doc.canonical_identifier, doc.revision) == ("DOC-HVG-CSR-900-REV-A", "HVG-CSR-900", "A")
    assert doc.title == "Test Requirements" and doc.identifier_source == "header"


def test_filename_fallback(tmp_path):
    path = tmp_path / "abc_spec_7.txt"
    path.write_text("1 Scope\n1.1 The supplier shall comply.\n", encoding="utf-8")
    doc = load_document(path)
    assert doc.canonical_identifier == "ABC-SPEC-7" and doc.identifier_source == "filename"
    assert any("filename" in w for w in doc.warnings)


def test_crlf_is_normalised(tmp_path):
    path = tmp_path / "x.txt"
    path.write_bytes(b"Document ID: X-1\r\n\r\n1 Scope\r\n1.1 The supplier shall comply.\r\n")
    assert "\r" not in load_document(path).text


def test_pdf_with_text(tmp_path, config):
    lines = ["HALVARD MOTORS", "PDF Requirements", "Document ID: HVG-CSR-910", "", "1 Requirements", "1.1 The supplier shall comply with Specification HVG-QS-110."]
    path = tmp_path / "req.pdf"
    path.write_bytes(make_pdf([lines, ["1.2 The supplier shall keep records for five years."]]))
    doc = load_document(path)
    assert doc.format == "pdf" and doc.parse_status == "ok" and len(doc.pages) == 2
    assert doc.canonical_identifier == "HVG-CSR-910"
    assert any("pypdf" in w for w in doc.warnings)
    result = run_analysis([doc], strategy="staged_verified", config=config)
    assert {r.chapter for r in result.requirements} == {"1.1", "1.2"}
    first = next(r for r in result.requirements if r.chapter == "1.1")
    assert first.referenced_document_ids == ["DOC-HVG-QS-110"]
    assert first.source_location.page_start == 1


def test_scanned_pdf_is_unsupported(tmp_path, config):
    path = tmp_path / "scan.pdf"
    path.write_bytes(make_pdf([[], []]))
    doc = load_document(path)
    assert doc.parse_status == "unsupported"
    assert any("OCR" in w for w in doc.warnings)
    result = run_analysis([doc], strategy="single_pass", config=config)
    assert result.status == "failed" and result.requirements == []


def test_json_input_and_duplicate_ids(tmp_path):
    payload = {"identifier": "J-1", "title": "Json doc", "text": "1 Scope\n1.1 The supplier shall comply.\n"}
    (tmp_path / "a.json").write_text(json.dumps(payload), encoding="utf-8")
    (tmp_path / "b.json").write_text(json.dumps(payload), encoding="utf-8")
    docs = load_bundle([tmp_path])
    assert [d.document_id for d in docs] == ["DOC-J-1", "DOC-J-1-DUP2"]
    assert docs[0].identifier_source == "manifest" and docs[0].title == "Json doc"


# ---------------------------------------------------------------------------
# documents.yaml: identifiers, titles and aliases for documents known only by name; running
# header/footer lines kept as metadata; the share of references that resolve to a loaded document.
# ---------------------------------------------------------------------------

BASE = """\
Halvard Q Agreement
Quality management agreement between Halvard Group companies and their suppliers

1 General
1.1 Scope
The supplier shall keep a quality management system.
1.2 Records
The supplier shall keep records for 10 years.
"""

CSR = """\
HALVARD VEHICLE GROUP
Commercial Vehicle Standard
Document ID: HVG-CVS-20
Revision: 2025-04

1 General
1.1 Scope
The supplier shall follow the Halvard Q Agreement.
1.2 Records
New parts shall follow the Halvard New Parts Program; see also the Halvard Neuteil Program.
"""

REGISTRY = """\
documents:
  - file: halvard_q_agreement.txt
    title: Halvard Q Agreement
    aliases: [the Q Agreement]
  - identifier: HALVARD-NEW-PARTS-PROGRAM
    title: Halvard New Parts Program
    aliases: [Halvard Neuteil Program]
"""


def _name_index(entries: list[RegistryEntry]) -> NameIndex:
    index = NameIndex()
    for e in entries:
        for name in e.names:
            index.add(name, e.identifier)
    return index


def test_name_keys_ignore_case_punctuation_and_spacing():
    index = _name_index([RegistryEntry(identifier="FQK", title="Formel Q – konkret", aliases=["Formula Q concretely"])])
    assert name_match_key("Formel Q – konkret") == "FORMEL Q KONKRET"
    for name in ("Formel Q Konkret", "formel q konkret", "Form el Q konkret", "VW Formel Q Konkret", "Formula Q concretely"):
        assert index.resolve(name) == "FQK", name
    assert index.resolve("Formel Q Capability") is None
    assert index.resolve("Q konkret") is None  # a two-word suffix is too short to be safe


def test_ambiguous_names_are_not_resolved():
    index = _name_index([RegistryEntry(identifier="A", title="Supplier Quality Manual"), RegistryEntry(identifier="B", title="Supplier Quality Manual")])
    assert index.resolve("Supplier Quality Manual") is None


def test_entry_needs_a_file_or_an_identifier():
    with pytest.raises(ValueError):
        RegistryEntry(title="no identifier")


def _bundle(tmp_path: Path, registry: bool) -> Path:
    folder = tmp_path / "bundle"
    folder.mkdir()
    (folder / "halvard_q_agreement.txt").write_text(BASE, encoding="utf-8")
    (folder / "hvg_cvs_20.txt").write_text(CSR, encoding="utf-8")
    if registry:
        (folder / "documents.yaml").write_text(REGISTRY, encoding="utf-8")
    return folder


def test_registry_sets_identifier_title_and_aliases(tmp_path):
    folder = _bundle(tmp_path, registry=True)
    assert [p.name for p in load_registry([folder / "hvg_cvs_20.txt"]).paths] == ["documents.yaml"]
    docs = {d.filename: d for d in load_bundle([folder], registry=load_registry([folder / "hvg_cvs_20.txt"]))}
    base = docs["halvard_q_agreement.txt"]
    assert (base.canonical_identifier, base.title, base.title_source, base.aliases) == ("HALVARD-Q-AGREEMENT", "Halvard Q Agreement", "registry", ["the Q Agreement"])
    assert docs["hvg_cvs_20.txt"].identifier_source == "header"  # no file entry: the header is used
    assert {d.filename for d in load_bundle([folder])} == set(docs)


@pytest.mark.parametrize("registry", [False, True])
def test_registry_names_resolve_references_in_every_strategy(tmp_path, config, registry):
    folder = _bundle(tmp_path, registry=registry)
    config = config.model_copy(update={"processing": config.processing.model_copy(update={"document_registry": registry})})
    config.llm.mock.reference_miss_rate = 0.0
    config.llm.mock.hallucination_rate = 0.0
    result = analyze_files([folder], strategy="staged", config=config)
    mentions = {m.raw_identifier: m for r in result.requirements for m in r.reference_mentions}
    agreement = mentions["Halvard Q Agreement"]
    refs = result.coverage.references
    if registry:
        assert agreement.document_id == "DOC-HALVARD-Q-AGREEMENT" and agreement.resolution == ResolutionMethod.REGISTRY
        assert mentions["Halvard Neuteil Program"].document_id == mentions["Halvard New Parts Program"].document_id == "DOC-HALVARD-NEW-PARTS-PROGRAM"
        assert refs.to_input_documents == 1 and refs.by_resolution.get("registry") == 3
    else:
        assert agreement.document_id.startswith("DOC-NAME-") and agreement.resolution == ResolutionMethod.UNRESOLVED_NAME
        assert refs.to_input_documents == 0 and refs.name_only == 3
    assert refs.mentions == 3 and refs.input_resolution_rate == round(refs.to_input_documents / 3, 4)


def test_running_lines_are_kept_and_give_a_title():
    pages = [
        "Formel Q – konkret 2\nBody one\nmore\nmore\nmore\n© Volkswagen AG 6th Edition, October 2021\nK-GQF | CSD 0.2 | Internal",
        "Formel Q – konkret\n3\nBody two\nmore\nmore\nmore\n© Volkswagen AG 6th Edition, October 2021\nK-GQF | CSD 0.2 | Internal",
        "Formel Q – konkret 4\nBody three\nmore\nmore\nmore\n© Volkswagen AG 6th Edition, October 2021\nK-GQF | CSD 0.2 | Internal",
    ]
    cleaned, _patterns, originals = strip_running_lines(pages)
    assert all("Formel Q" not in p and "Volkswagen" not in p for p in cleaned)
    assert originals[0] == "Formel Q – konkret 2"
    assert running_title(originals) == "Formel Q – konkret"  # page number dropped, © and | lines skipped
    assert running_title(["Document Number: ISQ-011-PR", "Page 1/6"]) is None
