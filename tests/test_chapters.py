"""tests/test_chapters.py

Whole chapters as requirements: cutting by the table of contents or by numbered
headings, chapter IDs, and the single-pass baseline folded onto chapters.
"""

from pathlib import Path

from csr_lab.config import ChunkingSettings
from csr_lab.ids import requirement_id
from csr_lab.ingestion.chapters import split_chapters
from csr_lab.ingestion.loader import load_document
from csr_lab.pipelines.run import run_analysis

# Laid out like a real CSR: cover page, contents with dot leaders, unnumbered chapters,
# a numbered list inside a chapter, a sub-heading missing from the contents.
WITH_TOC = """\
SUPPLIER STANDARD
Quality requirements for purchased parts
Contents
Foreword .............................................. 1
Scope ................................................ 2
1 General ............................................ 2
1.1 Documentation .................................... 2
1.2 Part approval .................................... 3
2 Warranty ........................................... 3
Annex A (informative) Change history ................ 4
Foreword
This standard was written jointly by the customers.
Scope
The order of precedence is:
1. Customer Requirements XS-10
2. Quality Manual QM-1
1 General
1.1 Documentation
The supplier shall keep records according to XS-20.
1.2 Part approval
Samples shall be delivered free of charge.
1.2.1 Sampling elements
The supplier shall follow VDA Volume 2.
2 Warranty
The warranty period is 24 months.
Annex A (informative) Change history
2024-08 First release.
"""


def _doc(tmp_path: Path, text: str, name: str = "XS10_2024-08.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return load_document(path)


def test_chapters_from_table_of_contents(tmp_path):
    doc = _doc(tmp_path, WITH_TOC)
    segs, notes = split_chapters(doc, ChunkingSettings())
    assert "7 of 7 entries found" in notes[0]
    units = {s.chapter: s for s in segs if s.is_candidate}
    assert list(units) == ["Foreword", "Scope", "1.1", "1.2", "2", "Annex A (informative) Change history"]
    # The numbered list stays inside Scope; the sub-heading missing from the contents stays in 1.2.
    assert "1. Customer Requirements XS-10" in units["Scope"].text
    assert "1.2.1 Sampling elements" in units["1.2"].text and "VDA Volume 2" in units["1.2"].text
    # '1 General' has no text of its own; the contents lines are never analysed.
    assert next(s for s in segs if s.chapter == "1").kind == "heading"
    assert not any("......" in s.text for s in segs if s.is_candidate)
    assert units["1.1"].section_path == ["1 General"] and units["1.1"].title == "Documentation"
    # Chapter text is an exact slice of the source, without the heading line.
    text = doc.text or ""
    assert all(text[s.start_offset : s.end_offset] == s.text for s in units.values())
    assert units["1.1"].text == "The supplier shall keep records according to XS-20."
    cover = next(s for s in segs if s.kind == "front_matter")
    assert "SUPPLIER STANDARD" in cover.text


def test_chapter_settings(tmp_path):
    doc = _doc(tmp_path, WITH_TOC)
    no_toc, notes = split_chapters(doc, ChunkingSettings(chapter_use_toc=False))
    assert "no usable table of contents" in notes[0]
    assert "1.2.1" in {s.chapter for s in no_toc}  # numbered headings alone: the sub-heading is a chapter
    shallow, _ = split_chapters(doc, ChunkingSettings(chapter_max_depth=1))
    one = next(s for s in shallow if s.chapter == "1")
    assert one.is_candidate and "Documentation" in one.text and "free of charge" in one.text
    front, _ = split_chapters(doc, ChunkingSettings(chapter_include_front_matter=True))
    assert any(s.chapter == "Front matter" and s.is_candidate for s in front)


def test_chapter_ids_and_filename_revision(tmp_path):
    doc = _doc(tmp_path, WITH_TOC)
    assert (doc.canonical_identifier, doc.revision, doc.effective_date) == ("XS10", "2024-08", "2024-08")
    assert requirement_id(doc.document_id, "1.1") == "REQ-XS10-REV-2024-08-1.1"
    assert requirement_id("DOC-CVS10", "Scope") == "REQ-CVS10-SCOPE"


def test_every_strategy_uses_one_requirement_per_chapter(tmp_path, config):
    docs = [_doc(tmp_path, WITH_TOC)]
    for strategy in ("single_pass", "staged", "graph_assisted"):
        result = run_analysis(docs, strategy=strategy, config=config)
        by_chapter = {r.chapter: r for r in result.requirements}
        assert len(by_chapter) == len(result.requirements) == 6, strategy
        assert by_chapter["1.1"].requirement_id == "REQ-XS10-REV-2024-08-1.1"
        assert by_chapter["1.1"].source_location.match == "exact"
        refs = {d.canonical_identifier for d in result.referenced_documents if d.document_id in by_chapter["1.1"].referenced_document_ids}
        assert refs == {"XS-20"}, strategy


def test_simple_doc_chapters(simple_docs):
    segs, _ = split_chapters(simple_docs[0], ChunkingSettings())
    assert [s.chapter for s in segs if s.is_candidate] == ["1.1", "2.1", "2.2", "2.3", "2.4", "2.5"]
    two_three = next(s for s in segs if s.chapter == "2.3")
    assert two_three.text.startswith("Parts shall be validated") and two_three.text.endswith("CVM-TM-320.")
