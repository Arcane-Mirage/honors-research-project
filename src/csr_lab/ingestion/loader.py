"""src/csr_lab/ingestion/loader.py

Load input documents (.txt, .pdf, .json) into ``InputDocument`` records.

The document header (identifier, title, revision, effective date) is read with a
few regular expressions.

Each input document is one *revision* of a document: its ID includes the
revision when the header states one ('DOC-CVM-PKG-210-REV-D'), using the same
function as revision-qualified references, so the two meet in one graph node.
Without a stated revision the ID is the plain document ID and a warning is added.

Text fidelity rules:
* Line endings are normalised to ``\\n``; nothing else is changed. All offsets
  refer to this normalised text.
* PDF text is whatever ``pypdf`` extracts. Requirement text is therefore
  verbatim *relative to the parser's output*, not to the visual PDF.
* A PDF with no extractable text is marked ``unsupported``; OCR is not attempted.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from csr_lab.extraction.normalization import canonical_input_identifier, normalize_revision
from csr_lab.ids import document_node_id, slug
from csr_lab.schemas import InputDocument, PageSpan

SUPPORTED_SUFFIXES = {".txt", ".pdf", ".json"}
_HEADER_SCAN_LINES = 25

_DOC_ID_RE = re.compile(
    r"^\s*(?:document\s*(?:id|number|no\.?)|doc(?:ument)?[\s_-]*id)\s*[:#]\s*(?P<value>\S.*?)\s*$",
    re.IGNORECASE,
)
_REVISION_RE = re.compile(r"^\s*(?:revision|rev\.?)\s*[:#]\s*(?P<value>\S+)\s*$", re.IGNORECASE)
_TITLE_RE = re.compile(r"^\s*title\s*:\s*(?P<value>\S.*?)\s*$", re.IGNORECASE)
_DATE_RE = re.compile(r"^\s*(?:effective|issue|release)\s*date\s*[:#]\s*(?P<value>\S.*?)\s*$", re.IGNORECASE)


class UnsupportedInputError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def parse_effective_date(text: str) -> str | None:
    """'Effective Date: 2026-02-15' in the first lines of a document, as written."""
    for line in text.split("\n")[:_HEADER_SCAN_LINES]:
        if m := _DATE_RE.match(line):
            return m.group("value")
    return None


def parse_header(text: str) -> tuple[str | None, str | None, str | None]:
    """Return (identifier, title, revision) found in the first lines of a document."""
    identifier = title = revision = None
    previous_lines: list[str] = []
    for line in text.split("\n")[:_HEADER_SCAN_LINES]:
        if identifier is None and (m := _DOC_ID_RE.match(line)):
            identifier = m.group("value")
            # Without an explicit "Title:" line, use the last non-empty line above the ID.
            if title is None and previous_lines:
                title = previous_lines[-1]
            continue
        if revision is None and (m := _REVISION_RE.match(line)):
            revision = m.group("value")
            continue
        if title is None and (m := _TITLE_RE.match(line)):
            title = m.group("value")
            continue
        if line.strip() and identifier is None:
            previous_lines.append(line.strip())
    return identifier, title, revision


def _identifier_from_filename(path: Path) -> str:
    return slug(path.stem)


def _build_document(
    *,
    path: Path,
    text: str,
    raw: bytes,
    fmt: str,
    pages: list[PageSpan] | None = None,
    parse_status: str = "ok",
    warnings: list[str] | None = None,
    identifier: str | None = None,
    title: str | None = None,
    revision: str | None = None,
    effective_date: str | None = None,
    identifier_source: str | None = None,
) -> InputDocument:
    warnings = list(warnings or [])
    header_id, header_title, header_rev = parse_header(text)
    if identifier:
        source = identifier_source or "manifest"
    elif header_id:
        identifier, source = header_id, "header"
    else:
        identifier, source = _identifier_from_filename(path), "filename"
        warnings.append("no 'Document ID:' header found; identifier taken from the filename")
    canonical = canonical_input_identifier(identifier)
    rev = normalize_revision(revision or header_rev)
    if rev is None:
        warnings.append("no revision found; the document is treated as revision-unknown")
    return InputDocument(
        document_id=document_node_id(canonical, rev),
        canonical_identifier=canonical,
        identifier_source=source,  # type: ignore[arg-type]
        filename=path.name,
        title=title or header_title,
        revision=rev,
        effective_date=effective_date or parse_effective_date(text),
        format=fmt,  # type: ignore[arg-type]
        sha256=_sha256(raw),
        char_count=len(text),
        pages=pages or [],
        parse_status=parse_status,  # type: ignore[arg-type]
        warnings=warnings,
        text=text,
    )


def load_text_file(path: Path) -> InputDocument:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        return InputDocument(
            document_id=document_node_id(_identifier_from_filename(path)),
            canonical_identifier=_identifier_from_filename(path),
            identifier_source="filename",
            filename=path.name,
            format="txt",
            sha256=_sha256(raw),
            char_count=0,
            parse_status="failed",
            warnings=[f"file is not valid UTF-8: {exc}"],
            text="",
        )
    text = _normalise_newlines(text)
    status, warnings = ("ok", []) if text.strip() else ("unsupported", ["file contains no text"])
    return _build_document(path=path, text=text, raw=raw, fmt="txt", parse_status=status, warnings=warnings)


PDF_BACKENDS = ("pypdf", "pypdf_layout", "pymupdf")
_DIGITS_RE = re.compile(r"\d+")
_EDGE_LINES = 3  # header/footer zone: this many non-empty lines at the top and bottom of a page


def _page_texts(path: Path, backend: str) -> list[str]:
    if backend == "pymupdf":
        try:
            import fitz  # type: ignore[import-not-found]  # PyMuPDF, optional
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise UnsupportedInputError("pdf_backend 'pymupdf' needs `pip install pymupdf`") from exc
        with fitz.open(str(path)) as doc:
            return [page.get_text() or "" for page in doc]
    from pypdf import PdfReader  # imported lazily: only needed for PDF input

    reader = PdfReader(str(path))
    if backend == "pypdf_layout":
        return [page.extract_text(extraction_mode="layout") or "" for page in reader.pages]
    return [page.extract_text() or "" for page in reader.pages]


def strip_repeated_lines(page_texts: list[str], min_share: float = 0.5) -> tuple[list[str], list[str]]:
    """Remove running headers/footers: lines in the top/bottom zone of many pages.

    Lines are compared with digits replaced ('Page 3 of 9' == 'Page 4 of 9'). Only the
    first and last few non-empty lines of each page are candidates, so repeated body
    text is never removed. Returns (cleaned pages, removed line patterns).
    """
    if len(page_texts) < 2:
        return page_texts, []

    def key(line: str) -> str:
        return _DIGITS_RE.sub("#", " ".join(line.split())).lower()

    zones: list[set[str]] = []
    for text in page_texts:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        zones.append({key(ln) for ln in lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]})
    counts: dict[str, int] = {}
    for zone in zones:
        for k in zone:
            counts[k] = counts.get(k, 0) + 1
    needed = max(2, int(min_share * len(page_texts) + 0.999))
    repeated = {k for k, n in counts.items() if n >= needed}
    if not repeated:
        return page_texts, []
    cleaned = []
    for text in page_texts:
        lines = text.split("\n")
        non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
        edge = set(non_empty[:_EDGE_LINES] + non_empty[-_EDGE_LINES:])
        kept = [ln for i, ln in enumerate(lines) if not (i in edge and key(ln) in repeated)]
        cleaned.append("\n".join(kept).strip("\n"))
    return cleaned, sorted(repeated)


def load_pdf_file(
    path: Path,
    min_chars_per_page: int = 20,
    *,
    backend: str = "pypdf",
    strip_repeated: bool = False,
    repeated_min_share: float = 0.5,
) -> InputDocument:
    if backend not in PDF_BACKENDS:
        raise UnsupportedInputError(f"unknown pdf_backend {backend!r}; choose from {PDF_BACKENDS}")
    raw = path.read_bytes()
    try:
        page_texts = [_normalise_newlines(t) for t in _page_texts(path, backend)]
    except UnsupportedInputError:
        raise
    except Exception as exc:  # PDF libraries raise many exception types for damaged files
        return InputDocument(
            document_id=document_node_id(_identifier_from_filename(path)),
            canonical_identifier=_identifier_from_filename(path),
            identifier_source="filename",
            filename=path.name,
            format="pdf",
            sha256=_sha256(raw),
            char_count=0,
            parse_status="failed",
            warnings=[f"PDF could not be parsed: {exc}"],
            text="",
        )

    warnings: list[str] = []
    if strip_repeated:
        page_texts, removed = strip_repeated_lines(page_texts, repeated_min_share)
        if removed:
            warnings.append(f"removed {len(removed)} repeated header/footer line patterns: {removed[:6]}")

    pages: list[PageSpan] = []
    parts: list[str] = []
    offset = 0
    empty_pages: list[int] = []
    for number, page_text in enumerate(page_texts, start=1):
        page_text = page_text.rstrip("\n")
        if len(page_text.strip()) < min_chars_per_page:
            empty_pages.append(number)
        pages.append(PageSpan(page_number=number, start_offset=offset, end_offset=offset + len(page_text)))
        parts.append(page_text)
        offset += len(page_text) + 1  # +1 for the joining newline
    text = "\n".join(parts)

    if not page_texts or len(empty_pages) == len(page_texts):
        status = "unsupported"
        warnings.append(
            "PDF has no extractable text (possibly a scanned document). OCR is not supported; "
            "the document was not analysed."
        )
    elif empty_pages:
        status = "partial"
        warnings.append(f"pages without extractable text were skipped: {empty_pages}")
    else:
        status = "ok"
    warnings.append(f"requirement text is verbatim relative to the {backend} extracted text")
    return _build_document(path=path, text=text, raw=raw, fmt="pdf", pages=pages, parse_status=status, warnings=warnings)


def load_json_file(path: Path, *, load_source=None) -> InputDocument:
    """JSON input: {"identifier": ..., "title": ..., "revision": ..., "effective_date": ..., "text": ...},
    or a pre-chunked file with a ``chunks`` list (see ``ingestion.provided_chunks``)."""
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    from csr_lab.ingestion.provided_chunks import is_chunk_payload, load_chunk_file

    if is_chunk_payload(data):
        return load_chunk_file(path, data, raw, load_source=load_source or load_document)
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise UnsupportedInputError(f"{path}: JSON input needs an object with a 'text' string")
    text = _normalise_newlines(data["text"])
    return _build_document(
        path=path,
        text=text,
        raw=raw,
        fmt="json",
        identifier=data.get("identifier"),
        title=data.get("title"),
        revision=data.get("revision"),
        effective_date=data.get("effective_date"),
        identifier_source="manifest" if data.get("identifier") else None,
        parse_status="ok" if text.strip() else "unsupported",
    )


def load_document(
    path: str | Path,
    *,
    min_chars_per_page: int = 20,
    pdf_backend: str = "pypdf",
    pdf_strip_repeated_lines: bool = False,
    pdf_repeated_line_min_share: float = 0.5,
) -> InputDocument:
    path = Path(path)
    suffix = path.suffix.lower()
    options = dict(
        min_chars_per_page=min_chars_per_page,
        pdf_backend=pdf_backend,
        pdf_strip_repeated_lines=pdf_strip_repeated_lines,
        pdf_repeated_line_min_share=pdf_repeated_line_min_share,
    )
    if suffix == ".txt":
        return load_text_file(path)
    if suffix == ".pdf":
        return load_pdf_file(
            path,
            min_chars_per_page=min_chars_per_page,
            backend=pdf_backend,
            strip_repeated=pdf_strip_repeated_lines,
            repeated_min_share=pdf_repeated_line_min_share,
        )
    if suffix == ".json":
        return load_json_file(path, load_source=lambda p: load_document(p, **options))
    raise UnsupportedInputError(f"unsupported input type: {path.name} (supported: {sorted(SUPPORTED_SUFFIXES)})")


def processing_load_options(processing) -> dict:
    """Loader keyword arguments from ``ProcessingSettings``."""
    return {
        "min_chars_per_page": processing.pdf_min_chars_per_page,
        "pdf_backend": processing.pdf_backend,
        "pdf_strip_repeated_lines": processing.pdf_strip_repeated_lines,
        "pdf_repeated_line_min_share": processing.pdf_repeated_line_min_share,
    }


def expand_inputs(paths: list[str | Path]) -> list[Path]:
    """Expand directories into their supported files (sorted, non-recursive)."""
    files: list[Path] = []
    for item in paths:
        p = Path(item)
        if p.is_dir():
            files.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_SUFFIXES))
        else:
            files.append(p)
    return files


def load_bundle(paths: list[str | Path], **options) -> list[InputDocument]:
    """Load several files as one document bundle. Duplicate identifiers are made unique.

    ``options`` are passed to ``load_document`` (see ``processing_load_options``).
    """
    loaded = [load_document(path, **options) for path in expand_inputs(paths)]
    # A folder holding a chunk file and the source it points to: keep only the chunked version.
    chunked_ids = {d.document_id for d in loaded if d.provided_chunks}
    loaded = [d for d in loaded if d.provided_chunks or d.document_id not in chunked_ids]
    documents: list[InputDocument] = []
    seen: dict[str, int] = {}
    for doc in loaded:
        if doc.document_id in seen:
            seen[doc.document_id] += 1
            new_id = f"{doc.document_id}-DUP{seen[doc.document_id]}"
            doc = doc.model_copy(
                update={
                    "document_id": new_id,
                    "warnings": [*doc.warnings, f"duplicate document identifier; renamed to {new_id}"],
                }
            )
        else:
            seen[doc.document_id] = 1
        documents.append(doc)
    return documents
