"""Load input documents (.txt, .pdf, .json) into ``InputDocument`` records.

File reading and header parsing are combined in this one module because the
header parser (document identifier, title, revision) is only a few regular
expressions.

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

from csr_lab.extraction.normalization import canonical_input_identifier
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


class UnsupportedInputError(ValueError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


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
    return InputDocument(
        document_id=document_node_id(canonical),
        canonical_identifier=canonical,
        identifier_source=source,  # type: ignore[arg-type]
        filename=path.name,
        title=title or header_title,
        revision=revision or header_rev,
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


def load_pdf_file(path: Path, min_chars_per_page: int = 20) -> InputDocument:
    from pypdf import PdfReader  # imported lazily: only needed for PDF input

    raw = path.read_bytes()
    try:
        reader = PdfReader(str(path))
        page_texts = [_normalise_newlines(page.extract_text() or "") for page in reader.pages]
    except Exception as exc:  # pypdf raises many exception types for damaged files
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

    warnings: list[str] = []
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
    warnings.append("requirement text is verbatim relative to pypdf's extracted text")
    return _build_document(path=path, text=text, raw=raw, fmt="pdf", pages=pages, parse_status=status, warnings=warnings)


def load_json_file(path: Path) -> InputDocument:
    """JSON input: {"identifier": ..., "title": ..., "revision": ..., "text": ...}."""
    raw = path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
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
        identifier_source="manifest" if data.get("identifier") else None,
        parse_status="ok" if text.strip() else "unsupported",
    )


def load_document(path: str | Path, *, min_chars_per_page: int = 20) -> InputDocument:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return load_text_file(path)
    if suffix == ".pdf":
        return load_pdf_file(path, min_chars_per_page=min_chars_per_page)
    if suffix == ".json":
        return load_json_file(path)
    raise UnsupportedInputError(f"unsupported input type: {path.name} (supported: {sorted(SUPPORTED_SUFFIXES)})")


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


def load_bundle(paths: list[str | Path], *, min_chars_per_page: int = 20) -> list[InputDocument]:
    """Load several files as one document bundle. Duplicate identifiers are made unique."""
    documents: list[InputDocument] = []
    seen: dict[str, int] = {}
    for path in expand_inputs(paths):
        doc = load_document(path, min_chars_per_page=min_chars_per_page)
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
