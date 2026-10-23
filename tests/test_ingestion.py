import json

from csr_lab.ingestion.loader import load_bundle, load_document
from csr_lab.pipelines.run import run_analysis
from helpers import make_pdf


def test_text_header_detection(simple_doc_path):
    doc = load_document(simple_doc_path)
    assert (doc.document_id, doc.canonical_identifier, doc.revision) == ("DOC-CVM-CSR-900", "CVM-CSR-900", "A")
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
    lines = ["CORVANE MOTORS", "PDF Requirements", "Document ID: CVM-CSR-910", "", "1 Requirements", "1.1 The supplier shall comply with Specification CVM-QS-110."]
    path = tmp_path / "req.pdf"
    path.write_bytes(make_pdf([lines, ["1.2 The supplier shall keep records for five years."]]))
    doc = load_document(path)
    assert doc.format == "pdf" and doc.parse_status == "ok" and len(doc.pages) == 2
    assert doc.canonical_identifier == "CVM-CSR-910"
    assert any("pypdf" in w for w in doc.warnings)
    result = run_analysis([doc], strategy="staged_verified", config=config)
    assert {r.chapter for r in result.requirements} == {"1.1", "1.2"}
    first = next(r for r in result.requirements if r.chapter == "1.1")
    assert first.referenced_document_ids == ["DOC-CVM-QS-110"]
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
