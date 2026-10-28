"""src/csr_lab/datasets/variants.py

Input variants of the synthetic dataset: the same cases and gold, different input form.

* ``write_pdf_dataset``: every document rendered as a PDF with running headers, page
  footers, page breaks inside requirements and drawn tables.
* ``write_chunked_dataset``: every document handed over as a ``chunks.json`` file made
  by a *simulated external chunker*, next to its source text. Two chunkers:
  - ``paragraph``: packs blank-line-separated paragraphs up to ``words`` words, ignoring
    sections (like a word-budget batcher), copying the text verbatim;
  - ``window``: blind windows of ``words`` words that cut through sentences.

Gold annotations are the same as for the text dataset (PDF: table rows as cell text).
Because the cases are unchanged, results on a variant are directly comparable with
results on the text dataset.

**PDF rendering.** No PDF library is needed: the writer emits Helvetica text with
absolute positions. What makes the PDFs realistic for parser testing:

* a running header and a "Page n of N" footer on every page (they end up *inside*
  requirements that cross a page break when the text is extracted);
* fixed lines per page, so page breaks fall wherever the text happens to be;
* tables drawn as positioned cells without the ``|`` characters of the text version.

``pdf_gold`` adapts a case's gold annotations to the PDF: table rows become their cell
text (what a reader sees), and offsets point into the "ideal" extraction (the drawn
lines in order). The evaluator then re-anchors them to the text the parser actually extracted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from csr_lab.datasets.synthetic import DATASET_VERSION, all_cases, build_case
from csr_lab.ids import document_node_id
from csr_lab.ingestion.provided_chunks import write_chunk_file
from csr_lab.schemas import CaseGold

# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

PAGE_WIDTH, PAGE_HEIGHT = 612, 792
LEFT, TOP, LINE_HEIGHT, FONT_SIZE = 50, 742, 14, 10
LINES_PER_PAGE = 44
_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")


@dataclass
class Line:
    cells: list[tuple[float, str]]  # (x offset, text)

    @property
    def text(self) -> str:
        return " ".join(t for _, t in self.cells)


def _escape(text: str) -> str:
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return text.encode("cp1252", errors="replace").decode("latin-1")


def text_to_lines(text: str) -> list[Line]:
    """Document text -> drawable lines; table rows become positioned cells."""
    lines: list[Line] = []
    for raw in text.rstrip("\n").split("\n"):
        m = _TABLE_ROW.match(raw)
        if m:
            cells = [c.strip() for c in m.group(1).split("|")]
            width = 500 / max(len(cells), 1)
            lines.append(Line([(i * width, c) for i, c in enumerate(cells) if c]))
        else:
            lines.append(Line([(0, raw)] if raw else []))
    return lines


def paginate(lines: list[Line], header: str | None, lines_per_page: int = LINES_PER_PAGE) -> list[list[Line]]:
    body = lines_per_page - (2 if header else 0)
    pages = [lines[i : i + body] for i in range(0, len(lines), body)] or [[]]
    out = []
    for n, page in enumerate(pages, start=1):
        framed = list(page)
        if header:
            framed = [Line([(0, header)]), Line([]), *framed, Line([]), Line([(380, f"Page {n} of {len(pages)}")])]
        out.append(framed)
    return out


def render_pdf(pages: list[list[Line]]) -> bytes:
    objects: list[bytes] = []
    page_ids = [4 + 2 * i for i in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for i, lines in enumerate(pages):
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>".encode()
        )
        ops = []
        for row, line in enumerate(lines):
            y = TOP - row * LINE_HEIGHT
            for x, text in line.cells:
                ops.append(f"BT /F1 {FONT_SIZE} Tf {LEFT + x:.1f} {y} Td ({_escape(text)}) Tj ET")
        stream = "\n".join(ops).encode("latin-1")
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def ideal_text(pages: list[list[Line]]) -> str:
    """The drawn lines in reading order: what a perfect parser would extract."""
    return "\n".join("\n".join(line.text for line in page) for page in pages)


def flatten_tables(text: str) -> str:
    return "\n".join(
        " ".join(c.strip() for c in m.group(1).split("|") if c.strip()) if (m := _TABLE_ROW.match(line)) else line
        for line in text.split("\n")
    )


def pdf_gold(gold: CaseGold, ideal: dict[str, str], filenames: dict[str, str]) -> CaseGold:
    """Gold for the PDF version of a case (``ideal``/``filenames`` keyed by document ID)."""
    requirements = []
    for g in gold.requirements:
        text = flatten_tables(g.requirement_text)
        src = ideal.get(g.source_document_id, "")
        start = src.find(text)
        if start == -1:  # interrupted by a page break: anchor at the first line
            start = max(src.find(text.split("\n", 1)[0]), 0)
        requirements.append(g.model_copy(update={"requirement_text": text, "start_offset": start, "end_offset": start + len(text)}))
    documents = [d.model_copy(update={"filename": filenames.get(d.document_id, d.filename)}) for d in gold.documents]
    return gold.model_copy(
        update={
            "requirements": requirements,
            "documents": documents,
            "notes": [*gold.notes, "PDF rendering: table rows are cell text; offsets are re-anchored to the extracted text when scoring"],
        }
    )


# ---------------------------------------------------------------------------
# Dataset writers: PDF variant
# ---------------------------------------------------------------------------

def _manifest(out: Path, name: str, description: str, cases: list[dict], extra: dict | None = None) -> Path:
    manifest = {"name": name, "version": DATASET_VERSION, "description": description, "generator": "csr_lab.datasets.variants", **(extra or {}), "cases": cases}
    path = out / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def write_pdf_dataset(out_dir: str | Path, *, lines_per_page: int = 44) -> Path:
    out = Path(out_dir)
    manifest_cases = []
    for case in all_cases():
        files, gold = build_case(case)
        ideal: dict[str, str] = {}
        filenames: dict[str, str] = {}
        doc_paths = []
        for spec in case.documents:
            header = f"{spec.company} | {spec.identifier} Rev {spec.revision} | Confidential"
            pages = paginate(text_to_lines(files[spec.filename]), header, lines_per_page)
            pdf_name = Path(spec.filename).with_suffix(".pdf").name
            path = out / "documents" / case.case_id / pdf_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(render_pdf(pages))
            doc_id = document_node_id(spec.identifier, spec.revision)
            ideal[doc_id] = ideal_text(pages)
            filenames[doc_id] = pdf_name
            doc_paths.append(path.relative_to(out).as_posix())
        gold_path = out / "ground_truth" / f"{case.case_id}.json"
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        gold_path.write_text(pdf_gold(gold, ideal, filenames).model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        manifest_cases.append(
            {"case_id": case.case_id, "split": case.split, "description": case.description + " (PDF rendering)", "documents": doc_paths, "ground_truth": gold_path.relative_to(out).as_posix()}
        )
    return _manifest(
        out,
        "synthetic_csr_pdf",
        "The synthetic CSR cases rendered as PDFs with running headers, page footers and drawn tables.",
        manifest_cases,
        {"lines_per_page": lines_per_page},
    )


# ---------------------------------------------------------------------------
# Pre-chunked variant: simulated external chunkers
# ---------------------------------------------------------------------------

_PARA_BREAK = re.compile(r"\n[ \t]*\n")


def paragraph_chunks(text: str, words: int) -> list[dict]:
    paragraphs = [p for p in _PARA_BREAK.split(text) if p.strip()]
    groups: list[list[str]] = [[]]
    count = 0
    for para in paragraphs:
        n = len(para.split())
        if groups[-1] and count + n > words:
            groups.append([])
            count = 0
        groups[-1].append(para.strip("\n"))
        count += n
    chunks = []
    for i, group in enumerate(g for g in groups if g):
        body = "\n\n".join(group)
        chunks.append({"chunk_id": f"C{i + 1:03d}", "title": " ".join(group[0].split()[:6]), "content": body})
    return chunks


def window_chunks(text: str, words: int) -> list[dict]:
    spans = [m.span() for m in re.finditer(r"\S+", text)]
    chunks = []
    for i in range(0, len(spans), words):
        part = spans[i : i + words]
        body = text[part[0][0] : part[-1][1]]
        chunks.append({"chunk_id": f"W{i // words + 1:03d}", "title": None, "content": body})
    return chunks


def write_chunked_dataset(
    out_dir: str | Path, *, chunker: str = "paragraph", words: int = 120
) -> Path:
    if chunker not in ("paragraph", "window"):
        raise ValueError("chunker must be 'paragraph' or 'window'")
    out = Path(out_dir)
    manifest_cases = []
    for case in all_cases():
        files, gold = build_case(case)
        doc_paths = []
        for filename, text in files.items():
            folder = out / "documents" / case.case_id
            folder.mkdir(parents=True, exist_ok=True)
            (folder / filename).write_text(text, encoding="utf-8", newline="\n")
            chunks = paragraph_chunks(text, words) if chunker == "paragraph" else window_chunks(text, words)
            chunk_path = write_chunk_file(
                folder / (Path(filename).stem + ".chunks.json"),
                chunks,
                source_file=filename,
                chunker=f"simulated-{chunker}(words={words})",
            )
            doc_paths.append(chunk_path.relative_to(out).as_posix())
        gold_path = out / "ground_truth" / f"{case.case_id}.json"
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        gold_path.write_text(gold.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        manifest_cases.append(
            {"case_id": case.case_id, "split": case.split, "description": case.description + f" (pre-chunked: {chunker})", "documents": doc_paths, "ground_truth": gold_path.relative_to(out).as_posix()}
        )
    return _manifest(
        out,
        f"synthetic_csr_chunked_{chunker}",
        "The synthetic CSR cases as chunk files from a simulated external chunker, with their source text.",
        manifest_cases,
        {"chunker": chunker, "words": words},
    )
