"""src/csr_lab/datasets/variants.py

Input variants of the synthetic dataset: the same cases and gold, different input form.

* ``write_pdf_dataset``: every document rendered as a PDF with running headers, page
  footers and page breaks inside requirements.
* ``write_incremental_dataset``: the cases with several documents, split for the
  saved-graph experiment. The supporting documents go to ``graph_input/`` (built into a
  graph first, with ``build-graph``); the CSR is the new document of the case. The gold
  lists the supporting documents as ``graph_documents``.
* ``write_noisy_dataset``: a copy of any text dataset with PDF extraction artifacts: letters
  split off words ('s upplier') and words hyphenated at a line end ('require-\\nments'). Only
  words of six letters or more inside requirements change; never identifiers, gold mention
  texts or customer labels. Gold texts and offsets are mapped through the same edits, and every
  mapped span is checked to hold the same letters as the clean one.

Gold annotations are the same as for the text dataset, so results on a variant are directly
comparable with results on the text dataset.

**PDF rendering.** No PDF library is needed: the writer emits Helvetica text with
absolute positions. What makes the PDFs realistic for parser testing:

* a running header and a "Page n of N" footer on every page (they end up *inside*
  requirements that cross a page break when the text is extracted);
* fixed lines per page, so page breaks fall wherever the text happens to be.

``pdf_gold`` points a case's gold offsets into the "ideal" extraction (the drawn lines in
order); the evaluator then re-anchors them to the text the parser actually extracted.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

from csr_lab.datasets.synthetic import all_cases, build_case, write_manifest
from csr_lab.extraction.grounding import phrase_pattern
from csr_lab.ids import document_node_id
from csr_lab.schemas import CaseGold

# ---------------------------------------------------------------------------
# PDF rendering
# ---------------------------------------------------------------------------

PAGE_WIDTH, PAGE_HEIGHT = 612, 792
LEFT, TOP, LINE_HEIGHT, FONT_SIZE = 50, 742, 14, 10
LINES_PER_PAGE = 44


@dataclass
class Line:
    text: str = ""
    x: float = 0  # offset from the left margin (the page footer is right-aligned)


def _escape(text: str) -> str:
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return text.encode("cp1252", errors="replace").decode("latin-1")


def text_to_lines(text: str) -> list[Line]:
    return [Line(raw) for raw in text.rstrip("\n").split("\n")]


def paginate(lines: list[Line], header: str | None, lines_per_page: int = LINES_PER_PAGE) -> list[list[Line]]:
    body = lines_per_page - (2 if header else 0)
    pages = [lines[i : i + body] for i in range(0, len(lines), body)] or [[]]
    out = []
    for n, page in enumerate(pages, start=1):
        framed = list(page)
        if header:
            framed = [Line(header), Line(), *framed, Line(), Line(f"Page {n} of {len(pages)}", x=380)]
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
            if line.text:
                ops.append(f"BT /F1 {FONT_SIZE} Tf {LEFT + line.x:.1f} {y} Td ({_escape(line.text)}) Tj ET")
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


def pdf_gold(gold: CaseGold, ideal: dict[str, str], filenames: dict[str, str]) -> CaseGold:
    """Gold for the PDF version of a case (``ideal``/``filenames`` keyed by document ID)."""
    requirements = []
    for g in gold.requirements:
        text = g.requirement_text
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
            "notes": [*gold.notes, "PDF rendering: offsets are re-anchored to the extracted text when scoring"],
        }
    )


# ---------------------------------------------------------------------------
# Dataset writers: PDF variant
# ---------------------------------------------------------------------------

def _manifest(out: Path, name: str, description: str, cases: list[dict], extra: dict | None = None) -> Path:
    return write_manifest(out, name=name, description=description, generator="csr_lab.datasets.variants", cases=cases, extra=extra)


def write_pdf_dataset(out_dir: str | Path, *, lines_per_page: int = 44) -> Path:
    out = Path(out_dir)
    manifest_cases = []
    for case in all_cases():
        files, gold = build_case(case)
        ideal: dict[str, str] = {}
        filenames: dict[str, str] = {}
        doc_paths = []
        registry = case.registry
        for spec in case.documents:
            # A running header like a real one ('Formel Q – konkret'): the title, plus the code and
            # revision only for documents that state them (a name-only document has none).
            header = f"{spec.title} | {spec.identifier} {spec.revision}" if spec.revision else spec.title
            pages = paginate(text_to_lines(files[spec.filename]), header, lines_per_page)
            pdf_name = Path(spec.filename).with_suffix(".pdf").name
            registry = registry.replace(spec.filename, pdf_name) if registry else registry
            path = out / "documents" / case.case_id / pdf_name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(render_pdf(pages))
            doc_id = document_node_id(spec.identifier, spec.revision)
            ideal[doc_id] = ideal_text(pages)
            filenames[doc_id] = pdf_name
            doc_paths.append(path.relative_to(out).as_posix())
        if registry:
            (out / "documents" / case.case_id / "documents.yaml").write_text(registry, encoding="utf-8", newline="\n")
        gold_path = out / "ground_truth" / f"{case.case_id}.json"
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        gold_path.write_text(pdf_gold(gold, ideal, filenames).model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        manifest_cases.append(
            {"case_id": case.case_id, "split": case.split, "description": case.description + " (PDF rendering)", "documents": doc_paths, "ground_truth": gold_path.relative_to(out).as_posix()}
        )
    return _manifest(
        out,
        "synthetic_csr_pdf",
        "The synthetic CSR cases rendered as PDFs with running headers, page footers and page breaks inside chapters.",
        manifest_cases,
        {"lines_per_page": lines_per_page},
    )


# ---------------------------------------------------------------------------
# Incremental variant: a saved graph first, then a new document
# ---------------------------------------------------------------------------


def write_incremental_dataset(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    manifest_cases = []
    for case in all_cases():
        if len(case.documents) < 2:
            continue
        files, gold = build_case(case)
        new, *supporting = case.documents  # the CSR comes first in every case
        new_path = out / "documents" / case.case_id / new.filename
        new_path.parent.mkdir(parents=True, exist_ok=True)
        new_path.write_text(files[new.filename], encoding="utf-8", newline="\n")
        for spec in supporting:
            path = out / "graph_input" / case.case_id / spec.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(files[spec.filename], encoding="utf-8", newline="\n")
        if case.registry:  # both runs need the names: the graph build and the new document's analysis
            for folder in (out / "graph_input" / case.case_id, new_path.parent):
                (folder / "documents.yaml").write_text(case.registry, encoding="utf-8", newline="\n")
        new_id = document_node_id(new.identifier, new.revision)
        split_gold = gold.model_copy(
            update={
                "documents": [d for d in gold.documents if d.document_id == new_id],
                "graph_documents": [d for d in gold.documents if d.document_id != new_id],
            }
        )
        gold_path = out / "ground_truth" / f"{case.case_id}.json"
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        gold_path.write_text(split_gold.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        manifest_cases.append(
            {
                "case_id": case.case_id,
                "split": case.split,
                "description": case.description + f" (new document {new.identifier}; the others are in the saved graph)",
                "documents": [new_path.relative_to(out).as_posix()],
                "ground_truth": gold_path.relative_to(out).as_posix(),
            }
        )
    return _manifest(
        out,
        "synthetic_csr_incremental",
        "Multi-document synthetic cases split into a saved graph (graph_input/) and a new CSR to analyse.",
        manifest_cases,
        {"graph_input": "graph_input"},
    )


# ---------------------------------------------------------------------------
# Dataset writers: PDF-noise variant
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"(?<![\w\-./])[A-Za-z]{6,}(?![\w\-./])")
_BARE = re.compile(r"[\s\-]+")


def _protected_spans(gold: CaseGold, doc_id: str, text: str) -> list[tuple[int, int]]:
    """Gold mention texts and customer labels ('Global:') of one document: never changed."""
    spans = []
    for g in gold.requirements:
        if g.source_document_id != doc_id:
            continue
        for ref in g.references:
            pattern = phrase_pattern(ref.mention_text)
            spans += [(g.start_offset + m.start(), g.start_offset + m.end()) for m in re.finditer(pattern, g.requirement_text)]
        for m in re.finditer(r"(?m)^[A-Z][\w/ ]{0,40}:", g.requirement_text):
            spans.append((g.start_offset + m.start(), g.start_offset + m.end()))
    return spans


def add_noise(
    text: str, requirement_spans: list[tuple[int, int]], protected: list[tuple[int, int]], rate: float, seed: int, hyphen_rate: float = 0.25
) -> tuple[str, list[int]]:
    """(noisy text, map old offset -> new offset for 0..len(text)). ``rate``: share of long words
    with a letter split off; ``hyphen_rate``: share of long words at a line end hyphenated."""
    rng = random.Random(seed)
    inserts: dict[int, str] = {}  # old position -> text inserted before it
    replace: dict[int, str] = {}  # old position -> replacement character (same length)
    for s_lo, s_hi in requirement_spans:
        for m in _WORD_RE.finditer(text, s_lo, s_hi):
            s, e = m.start(), m.end()
            if any(s < pe and ps < e for ps, pe in protected):
                continue
            r = rng.random()
            line_end = text.find("\n", e)
            at_line_end = line_end == e or (line_end != -1 and not text[e:line_end].strip())
            next_line = text[line_end + 1 : text.find("\n", line_end + 1) if text.find("\n", line_end + 1) != -1 else len(text)] if line_end != -1 else ""
            joins_safely = not re.match(r"\s*(?:[A-Z][\w/ ]{0,40}:|\|)", next_line)  # never pull a label or a table row up
            if at_line_end and joins_safely and line_end < s_hi - 1 and line_end not in replace and rng.random() < hyphen_rate:
                k = rng.randint(3, len(m.group()) - 3)
                inserts[s + k] = "-\n"  # 'require-\nments'
                replace[line_end] = " "
            elif r < rate:
                inserts[s + rng.randint(1, len(m.group()) - 2)] = " "  # 's upplier'
    out: list[str] = []
    mapping: list[int] = []
    pos = 0
    for i, ch in enumerate(text):
        if i in inserts:
            out.append(inserts[i])
            pos += len(inserts[i])
        mapping.append(pos)
        out.append(replace.get(i, ch))
        pos += 1
    mapping.append(pos)
    return "".join(out), mapping


def _mapped(old: str, new: str, mapping: list[int], start: int, end: int) -> tuple[int, int]:
    ns, ne = mapping[start], mapping[end - 1] + 1
    if _BARE.sub("", old[start:end]) != _BARE.sub("", new[ns:ne]):
        raise AssertionError(f"noise changed the letters of a gold span: {old[start:end]!r} -> {new[ns:ne]!r}")
    return ns, ne


def noisy_gold(gold: CaseGold, texts: dict[str, tuple[str, str, list[int]]]) -> CaseGold:
    """Gold for the noisy texts (document ID -> (old text, new text, offset map))."""
    reqs = []
    for g in gold.requirements:
        old, new, mapping = texts[g.source_document_id]
        ns, ne = _mapped(old, new, mapping, g.start_offset, g.end_offset)
        refs = []
        for ref in g.references:
            m = re.search(phrase_pattern(ref.mention_text), g.requirement_text)
            if m is None:
                raise AssertionError(f"{g.gold_id}: mention {ref.mention_text!r} not in requirement")
            ms, me = _mapped(old, new, mapping, g.start_offset + m.start(), g.start_offset + m.end())
            refs.append(ref.model_copy(update={"mention_text": new[ms:me]}))
        reqs.append(g.model_copy(update={"requirement_text": new[ns:ne], "start_offset": ns, "end_offset": ne, "references": refs}))
    by_id = {g.gold_id: g for g in gold.requirements}
    conflicts = []
    for c in gold.conflicts:
        evidence = []
        for ev in c.evidence:
            g = by_id[ev.requirement_id]
            old, new, mapping = texts[g.source_document_id]
            m = re.search(phrase_pattern(ev.quoted_text), g.requirement_text)
            if m is None:
                raise AssertionError(f"{c.gold_conflict_id}: quote {ev.quoted_text!r} not in {g.gold_id}")
            qs, qe = _mapped(old, new, mapping, g.start_offset + m.start(), g.start_offset + m.end())
            evidence.append(ev.model_copy(update={"quoted_text": new[qs:qe]}))
        conflicts.append(c.model_copy(update={"evidence": evidence}))
    out = gold.model_copy(update={"requirements": reqs, "conflicts": conflicts, "notes": [*gold.notes, "noise variant: PDF extraction artifacts added; texts and offsets mapped"]})
    return CaseGold.model_validate(out.model_dump())


def write_noisy_dataset(source_manifest: str | Path, out_dir: str | Path, *, rate: float = 0.08, seed: int = 0) -> Path:
    """A copy of a text dataset with PDF extraction artifacts in its requirement texts."""
    from csr_lab.datasets.loader import load_dataset

    source = load_dataset(source_manifest)
    out = Path(out_dir)
    (out / "ground_truth").mkdir(parents=True, exist_ok=True)
    entries = []
    for case in source.cases:
        gold = case.load_gold()
        assert gold is not None, f"{case.case_id}: noise needs gold"
        by_file = {d.filename: d.document_id for d in gold.documents}
        texts: dict[str, tuple[str, str, list[int]]] = {}
        paths = []
        folder = out / "documents" / case.case_id
        folder.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(case.document_paths):
            old = path.read_text(encoding="utf-8")
            doc_id = by_file[path.name]
            spans = [(g.start_offset, g.end_offset) for g in gold.requirements if g.source_document_id == doc_id]
            new, mapping = add_noise(old, spans, _protected_spans(gold, doc_id, old), rate, seed * 1000 + i + sum(map(ord, case.case_id)))
            texts[doc_id] = (old, new, mapping)
            (folder / path.name).write_text(new, encoding="utf-8", newline="\n")
            paths.append((folder / path.name).relative_to(out).as_posix())
        for extra in ("documents.yaml", "documents.yml"):
            if (case.document_paths[0].parent / extra).is_file():
                (folder / extra).write_text((case.document_paths[0].parent / extra).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
        gold_path = out / "ground_truth" / f"{case.case_id}.json"
        gold_path.write_text(noisy_gold(gold, texts).model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
        entries.append({"case_id": case.case_id, "split": case.split, "description": case.description + " (PDF noise)", "documents": paths,
                        "ground_truth": gold_path.relative_to(out).as_posix()})
    return _manifest(out, f"{source.name}_noisy", f"{source.name} with PDF extraction artifacts (rate {rate}, seed {seed}).", entries,
                           {"source": source.name, "source_version": source.version, "noise": {"rate": rate, "seed": seed}})
