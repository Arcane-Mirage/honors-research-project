"""src/csr_lab/ingestion/loader.py

Load input documents (.txt, .pdf, .json) into ``InputDocument`` records.

The document header (identifier, title, revision, effective date) is read with a
few regular expressions. For PDFs, the running header/footer lines that
``strip_running_lines`` removes are kept as ``InputDocument.running_lines``: field lines
among them ('Document Number: ISQ-011-PR') count as header lines, and the first
title-like line ('Formel Q – konkret') becomes the title when nothing better is known.

Real CSRs often carry no ``Document ID:`` header, and many documents are known only by
name ("Formel Q Konkret", "Formel Q Capability"), sometimes under several names ("Formel Q
Neuteile Integral" is "Formel Q New Parts Integral"). A *document registry*,
``documents.yaml`` next to the files, states what the text does not, and wins over the header::

    # ~/Documents/CSRs/traton/documents.yaml
    documents:
      - file: Formel Q Konkret.pdf          # a file in this folder ...
        identifier: FORMEL-Q-KONKRET
        title: Formel Q konkret
        revision: 6th edition               # optional; becomes 'EDITION 6'
        aliases: [Formel Q Konkret, Formula Q concretely]
      - identifier: FORMEL-Q-NEW-PARTS-INTEGRAL   # ... or a document you do not have
        title: Formel Q New Parts Integral
        aliases: [Formel Q Neuteile Integral]
      - file: CVS10_2024-08.pdf
        follows_structure_of: FORMEL-Q-KONKRET    # same chapter numbering (a delta document)
        takes_precedence_over: [FORMEL-Q-KONKRET, IATF-16949, ISO-9001]

What it is used for:

* a **file entry** sets the loaded document's identifier, title, revision and effective date
  (it wins over the header and the file name) and adds its aliases;
* **every entry's title and aliases** resolve name-only references deterministically, in
  every strategy (``ResolutionMethod.REGISTRY``): a mention "Formel Q Neuteile Integral"
  becomes FORMEL-Q-NEW-PARTS-INTEGRAL. A name matches when its words are equal (case,
  punctuation and spaces ignored, so "Form el Q" matches too) or when the shorter of the
  two (at least three words) is a word-suffix of the longer ("VW Formel Q Konkret").
  A name that matches several identifiers is left unresolved;
* ``follows_structure_of`` and ``takes_precedence_over`` declare a delta document and a
  precedence order (``relationships/builder.py``, ``conflicts/detector.py``).

The registry is found next to the files (``documents.yaml`` in a file's folder or the
folder above it) and used when ``processing.document_registry`` is on. It is user-provided
knowledge, so experiments can switch it off to measure what it adds.

Each input document is one *revision* of a document: its ID includes the
revision when the header states one ('DOC-HVG-CVS-20-REV-D'), using the same
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
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from csr_lab.config import ChunkingSettings
from csr_lab.extraction.normalization import canonical_input_identifier, normalize_revision
from csr_lab.ids import document_node_id, slug
from csr_lab.ingestion.chapters import split_chapters
from csr_lab.schemas import InputDocument, PageSpan

SUPPORTED_SUFFIXES = {".txt", ".pdf", ".json"}
_HEADER_SCAN_LINES = 25

_DOC_ID_RE = re.compile(
    r"^\s*(?:document\s*(?:id|number|no\.?)|doc(?:ument)?[\s_-]*id)\s*[:#]\s*(?P<value>\S.*?)\s*$",
    re.IGNORECASE,
)
_REVISION_RE = re.compile(r"^\s*(?:revision|rev\.?)\s*[:#]\s*(?P<value>\S+)\s*$", re.IGNORECASE)
_TITLE_RE = re.compile(r"^\s*title\s*:\s*(?P<value>\S.*?)\s*$", re.IGNORECASE)
_DATE_RE = re.compile(r"^\s*(?:effective|issue|released?)\s*date\s*[:#]\s*(?P<value>\S.*?)\s*$", re.IGNORECASE)
# Running lines that are not titles: copyright, page numbers, classification, URLs, field lines.
_NOT_TITLE_RE = re.compile(r"©|\(c\)|\||https?:|www\.|@|:|\bpage\b|\binternal\b|\bconfidential\b|\bcopyright\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# documents.yaml (the document registry)
# ---------------------------------------------------------------------------

REGISTRY_FILENAMES = ("documents.yaml", "documents.yml")


class RegistryEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file: str | None = Field(default=None, description="file name in this folder; omit for a document you do not have")
    identifier: str | None = None
    title: str | None = None
    revision: str | None = None
    effective_date: str | None = None
    aliases: list[str] = Field(default_factory=list)
    follows_structure_of: str | None = Field(default=None, description="identifier of the document whose chapter numbering this one follows")
    takes_precedence_over: list[str] = Field(default_factory=list, description="identifiers this document takes precedence over")

    @model_validator(mode="after")
    def _identified(self) -> RegistryEntry:
        if not self.file and not self.identifier:
            raise ValueError("a registry entry needs a file or an identifier")
        return self

    @property
    def names(self) -> list[str]:
        return [n for n in dict.fromkeys([self.title, *self.aliases]) if n]


class RegistryFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    documents: list[RegistryEntry] = Field(default_factory=list)


@dataclass
class DocumentRegistry:
    entries: list[RegistryEntry] = field(default_factory=list)
    paths: list[Path] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.entries)

    def for_file(self, filename: str) -> RegistryEntry | None:
        return next((e for e in self.entries if e.file and e.file == filename), None)


def registry_paths_for(files: list[str | Path]) -> list[Path]:
    """``documents.yaml`` files next to the given files (their folder, then the folder above)."""
    found: list[Path] = []
    for f in files:
        p = Path(f)
        folder = p if p.is_dir() else p.parent
        for d in (folder, folder.parent):
            for name in REGISTRY_FILENAMES:
                candidate = d / name
                if candidate.is_file() and candidate.resolve() not in [x.resolve() for x in found]:
                    found.append(candidate)
    return found


def load_registry_file(path: str | Path) -> list[RegistryEntry]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping with a `documents:` list")  # noqa: TRY004
    try:
        return RegistryFile.model_validate(data).documents
    except Exception as exc:  # pydantic gives a precise message; add the file name
        raise ValueError(f"{path}: {exc}") from exc


def load_registry(files: list[str | Path]) -> DocumentRegistry:
    """The registry for a bundle of files. Entries of the file's own folder come first."""
    paths = registry_paths_for(files)
    entries: list[RegistryEntry] = []
    for path in paths:
        for e in load_registry_file(path):
            key = (e.file, e.identifier)
            if not any((x.file, x.identifier) == key for x in entries):
                entries.append(e)
    return DocumentRegistry(entries=entries, paths=paths)


# ---------------------------------------------------------------------------
# Loading files
# ---------------------------------------------------------------------------


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


# 'CVS10_2024-08.pdf' -> CVS10, revision 2024-08; 'XYZ-100_rev-C.pdf' -> XYZ-100, revision C
_FILENAME_REV_RE = re.compile(
    r"^(?P<id>.+?)[_ ]+(?:(?P<date>\d{4}-\d{2}(?:-\d{2})?)|rev(?:ision)?[-_ ]?(?P<rev>[A-Za-z0-9]{1,4}))$",
    re.IGNORECASE,
)


def _identifier_from_filename(path: Path) -> str:
    return slug(path.stem)


def _filename_parts(path: Path) -> tuple[str, str | None, str | None]:
    """(identifier, revision, ISO date) from a file name such as 'CVS10_2024-08.pdf'."""
    if m := _FILENAME_REV_RE.match(path.stem):
        date = m.group("date")
        return slug(m.group("id")), date or m.group("rev"), date
    return _identifier_from_filename(path), None, None


def running_title(lines: list[str]) -> str | None:
    """The first running header/footer line that reads like a title ('Formel Q – konkret 7' ->
    'Formel Q – konkret'): two or more words of letters, no copyright, page or field text."""
    for line in lines:
        candidate = re.sub(r"^\s*\d+\s+|\s+\d+\s*$", "", " ".join(line.split()))
        words = re.findall(r"[^\W\d_]{2,}", candidate)
        if len(words) >= 2 and len(candidate.split()) <= 12 and not _NOT_TITLE_RE.search(candidate) and "/" not in candidate:
            return candidate
    return None


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
    running_lines: list[str] | None = None,
    registry_entry: RegistryEntry | None = None,
) -> InputDocument:
    warnings = list(warnings or [])
    running_lines = list(running_lines or [])
    header_id, header_title, header_rev = parse_header(text)
    if running_lines:  # header fields printed on every page ('Document Number: ISQ-011-PR')
        run_id, _run_title, run_rev = parse_header("\n".join(running_lines))
        header_id, header_rev = header_id or run_id, header_rev or run_rev
    title_source = "explicit" if title else ("header" if header_title else None)
    entry = registry_entry
    if entry is not None and entry.identifier:
        identifier, source = entry.identifier, "registry"
    elif identifier:
        source = identifier_source or "manifest"
    elif header_id:
        identifier, source = header_id, "header"
    else:
        identifier, file_rev, file_date = _filename_parts(path)
        source = "filename"
        revision = revision or (file_rev if not header_rev else None)
        effective_date = effective_date or file_date
        note = " (and revision)" if file_rev else ""
        warnings.append(f"no 'Document ID:' header found; identifier{note} taken from the filename")
    if entry is not None:
        revision = entry.revision or revision
        effective_date = entry.effective_date or effective_date
        if entry.title:
            title, title_source = entry.title, "registry"
    canonical = canonical_input_identifier(identifier)
    rev = normalize_revision(revision or header_rev)
    if rev is None:
        warnings.append("no revision found; the document is treated as revision-unknown")
    title = title or header_title
    if title is None and (found := running_title(running_lines)):
        title, title_source = found, "running_line"
    aliases = [a for a in dict.fromkeys(entry.aliases if entry else []) if a and a != title]
    follows = canonical_input_identifier(entry.follows_structure_of) if entry and entry.follows_structure_of else None
    return InputDocument(
        document_id=document_node_id(canonical, rev),
        canonical_identifier=canonical,
        identifier_source=source,  # type: ignore[arg-type]
        filename=path.name,
        title=title,
        title_source=title_source,  # type: ignore[arg-type]
        aliases=aliases,
        running_lines=running_lines,
        takes_precedence_over=[canonical_input_identifier(x) for x in (entry.takes_precedence_over if entry else [])],
        follows_structure_of=follows,
        structure_source="registry" if follows else None,
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


def load_text_file(path: Path, registry_entry: RegistryEntry | None = None) -> InputDocument:
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
    return _build_document(path=path, text=text, raw=raw, fmt="txt", parse_status=status, warnings=warnings, registry_entry=registry_entry)


PDF_BACKENDS = ("pypdf", "pypdf_layout", "pymupdf")
_DIGITS_RE = re.compile(r"\d+")
_EDGE_LINES = 4  # header/footer zone: this many non-empty lines at the top and bottom of a page


def _page_texts(path: Path, backend: str) -> list[str]:
    if backend == "pymupdf":
        try:
            import pymupdf as fitz  # type: ignore[import-not-found]  # PyMuPDF >= 1.24, optional (AGPL)
        except ImportError:
            try:
                import fitz  # type: ignore[import-not-found,no-redef]  # older PyMuPDF
            except ImportError as exc:  # pragma: no cover - depends on the environment
                raise UnsupportedInputError("pdf_backend 'pymupdf' needs `pip install pymupdf`") from exc
        with fitz.open(str(path)) as doc:
            return [page.get_text() or "" for page in doc]
    from pypdf import PdfReader  # imported lazily: only needed for PDF input

    reader = PdfReader(str(path))
    if backend == "pypdf_layout":
        return [page.extract_text(extraction_mode="layout") or "" for page in reader.pages]
    return [page.extract_text() or "" for page in reader.pages]


def _line_key(line: str) -> str:
    """Comparison form of a header/footer line: digits ignored, so 'Page 3 of 9' == 'Page 4 of 9'
    and 'Formel Q – konkret 7' == 'Formel Q – konkret'. A line of digits only is '#'."""
    collapsed = " ".join(line.split()).lower()
    key = " ".join(_DIGITS_RE.sub(" ", collapsed).split())
    return key if key.strip(" .,:;-–/") else ("#" if collapsed else "")


def strip_running_lines(page_texts: list[str], min_share: float = 0.5) -> tuple[list[str], list[str], list[str]]:
    """Remove running headers/footers: lines in the top/bottom zone of many pages.

    Lines are compared with digits ignored (``_line_key``). Only the first and last few
    non-empty lines of each page are candidates, so repeated body text is never removed.
    Returns (cleaned pages, removed line patterns, the removed lines as first printed).
    """
    if len(page_texts) < 2:
        return page_texts, [], []

    key = _line_key

    zones: list[set[str]] = []
    for text in page_texts:
        lines = [ln for ln in text.split("\n") if ln.strip()]
        zones.append({key(ln) for ln in lines[:_EDGE_LINES] + lines[-_EDGE_LINES:]})
    counts: dict[str, int] = {}
    for zone in zones:
        for k in zone:
            counts[k] = counts.get(k, 0) + 1
    needed = max(2, int(min_share * len(page_texts) + 0.999))
    repeated = {k for k, n in counts.items() if n >= needed and k}
    if not repeated:
        return page_texts, [], []
    cleaned = []
    originals: dict[str, str] = {}  # pattern -> the line as printed the first time
    for text in page_texts:
        lines = text.split("\n")
        non_empty = [i for i, ln in enumerate(lines) if ln.strip()]
        edge = set(non_empty[:_EDGE_LINES] + non_empty[-_EDGE_LINES:])
        kept = []
        for i, ln in enumerate(lines):
            if i in edge and key(ln) in repeated:
                originals.setdefault(key(ln), " ".join(ln.split()))
            else:
                kept.append(ln)
        cleaned.append("\n".join(kept).strip("\n"))
    return cleaned, sorted(repeated), [v for v in originals.values() if v and not v.isdigit()]


def load_pdf_file(
    path: Path,
    min_chars_per_page: int = 20,
    *,
    backend: str = "pypdf",
    strip_repeated: bool = False,
    repeated_min_share: float = 0.5,
    registry_entry: RegistryEntry | None = None,
) -> InputDocument:
    if backend not in PDF_BACKENDS:
        raise UnsupportedInputError(f"unknown pdf_backend {backend!r}; choose from {PDF_BACKENDS}")
    raw = path.read_bytes()
    try:
        page_texts = [_normalise_newlines(t) for t in _page_texts(path, backend)]
    except UnsupportedInputError:
        raise
    except Exception as exc:  # noqa: BLE001 - PDF libraries raise many exception types for damaged files
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
    running: list[str] = []
    if strip_repeated:
        page_texts, removed, running = strip_running_lines(page_texts, repeated_min_share)
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
    return _build_document(
        path=path, text=text, raw=raw, fmt="pdf", pages=pages, parse_status=status, warnings=warnings,
        running_lines=running, registry_entry=registry_entry,
    )


def load_json_file(path: Path, registry_entry: RegistryEntry | None = None) -> InputDocument:
    """JSON input: {"identifier": ..., "title": ..., "revision": ..., "effective_date": ..., "text": ...}."""
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
        effective_date=data.get("effective_date"),
        identifier_source="manifest" if data.get("identifier") else None,
        parse_status="ok" if text.strip() else "unsupported",
        registry_entry=registry_entry,
    )


def load_document(
    path: str | Path,
    *,
    min_chars_per_page: int = 20,
    pdf_backend: str = "pypdf",
    pdf_strip_repeated_lines: bool = False,
    pdf_repeated_line_min_share: float = 0.5,
    registry: DocumentRegistry | None = None,
) -> InputDocument:
    """Load one file; ``registry`` (``documents.yaml``) entries for it win over its header."""
    path = Path(path)
    suffix = path.suffix.lower()
    entry = registry.for_file(path.name) if registry else None
    if suffix == ".txt":
        return load_text_file(path, entry)
    if suffix == ".pdf":
        return load_pdf_file(
            path,
            min_chars_per_page=min_chars_per_page,
            backend=pdf_backend,
            strip_repeated=pdf_strip_repeated_lines,
            repeated_min_share=pdf_repeated_line_min_share,
            registry_entry=entry,
        )
    if suffix == ".json":
        return load_json_file(path, entry)
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


def load_bundle(paths: list[str | Path], *, registry: DocumentRegistry | None = None, **options) -> list[InputDocument]:
    """Load several files as one document bundle. Duplicate identifiers are made unique.

    ``options`` are passed to ``load_document`` (see ``processing_load_options``); ``registry``:
    the bundle's ``documents.yaml`` (``pipelines.run.load_inputs`` finds it).
    """
    loaded = [load_document(path, registry=registry, **options) for path in expand_inputs(paths)]
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


# ---------------------------------------------------------------------------
# Comparing PDF backends (csr-lab pdf-backends)
# ---------------------------------------------------------------------------
# Compare PDF text backends on real documents (``csr-lab pdf-backends``). No gold is needed.
#
# For every file and backend (pypdf, pypdf_layout, pymupdf) the same loader and chapter splitter
# as an analysis run are used, and what can be counted without an answer key is reported:
#
# * **artifacts** in the extracted text: words hyphenated at a line end ('require-\\nments'),
#   hyphens with a space before them ('B -50'), letters split off a word ('s upplier'), and lone
#   capital letters on a line ('V\\nolkswagen');
# * **structure**: contents entries found in the body (and how many needed the number/title
#   fallback), chapters, words outside any chapter (front matter);
# * **page furniture**: header/footer line patterns removed, and the title read from them;
# * extraction time.
#
# Fewer artifacts with the same chapters is better. The artifact counts are estimates (a real
# 'B -50' and an artifact look alike), but they compare backends fairly on the same file.

ARTIFACTS = {
    "hyphen_breaks": re.compile(r"[a-z]-[ \t]*\n[ \t]*[a-z]"),
    "space_before_hyphen": re.compile(r"[a-z] -[a-z]"),
    "split_letters": re.compile(r"(?<![\w-])[b-hj-zB-HJ-Z] (?=[a-z]{3,})"),
    "lone_capitals": re.compile(r"(?m)^[ \t]*[A-Z][ \t]*$"),
}


_REMOVED_RE = re.compile(r"removed (\d+) repeated")


def backend_row(path: Path, backend: str, settings: ChunkingSettings | None = None, strip: bool = True) -> dict:
    settings = settings or ChunkingSettings()
    started = time.perf_counter()
    try:
        doc = load_document(path, pdf_backend=backend, pdf_strip_repeated_lines=strip)
    except UnsupportedInputError as exc:
        return {"file": path.name, "backend": backend, "error": str(exc)}
    seconds = time.perf_counter() - started
    text = doc.text or ""
    segments, notes = split_chapters(doc, settings)
    toc = next((n for n in notes if n.startswith("chapters from the table of contents")), "")
    m = re.search(r"(\d+) of (\d+) entries", toc)
    words = len(text.split())
    row = {
        "file": path.name,
        "backend": backend,
        "status": doc.parse_status,
        "pages": len(doc.pages),
        "words": words,
        **{k: len(rx.findall(text)) for k, rx in ARTIFACTS.items()},
        "toc_found": f"{m.group(1)}/{m.group(2)}" if m else "no contents",
        "toc_fallbacks": sum(1 for n in notes if "matched by its" in n),
        "chapters": sum(1 for s in segments if s.is_candidate and s.kind != "passage"),
        "passages": sum(1 for s in segments if s.kind == "passage"),
        "front_matter_words": sum(len(s.text.split()) for s in segments if s.kind == "front_matter"),
        "header_patterns": next((int(m.group(1)) for w in doc.warnings if (m := _REMOVED_RE.search(w))), 0),
        "title": doc.title,
        "seconds": round(seconds, 2),
    }
    row["artifacts_per_1000_words"] = round(1000 * sum(row[k] for k in ARTIFACTS) / words, 2) if words else None
    return row


def compare_backends(paths: list[Path], backends: list[str] | None = None, settings: ChunkingSettings | None = None, strip: bool = True) -> list[dict]:
    backends = backends or list(PDF_BACKENDS)
    return [backend_row(p, b, settings, strip) for p in paths for b in backends]


_COLUMNS = [
    ("backend", "Backend"), ("words", "Words"), ("hyphen_breaks", "Line-end hyphens"), ("space_before_hyphen", "Space before hyphen"),
    ("split_letters", "Split letters"), ("lone_capitals", "Lone capitals"), ("artifacts_per_1000_words", "Artifacts / 1000 words"),
    ("toc_found", "Contents found"), ("toc_fallbacks", "Fallbacks"), ("chapters", "Chapters"), ("front_matter_words", "Front-matter words"),
    ("header_patterns", "Header patterns"), ("title", "Title"), ("seconds", "Seconds"),
]


def backends_markdown(rows: list[dict]) -> str:
    out = ["# PDF backends compared", "", "Same loader and chapter splitter as an analysis; counts need no gold (see `ingestion/loader.py`).", ""]
    for name in dict.fromkeys(r["file"] for r in rows):
        out += [f"## {name}", "", "| " + " | ".join(h for _, h in _COLUMNS) + " |", "|" + "---|" * len(_COLUMNS)]
        for r in (r for r in rows if r["file"] == name):
            if "error" in r:
                out.append(f"| {r['backend']} | error: {r['error']} |" + " |" * (len(_COLUMNS) - 2))
                continue
            out.append("| " + " | ".join("–" if r.get(k) is None else str(r.get(k)) for k, _ in _COLUMNS) + " |")
        out.append("")
    totals: dict[str, list[int]] = {}
    for r in rows:
        if "error" not in r:
            t = totals.setdefault(r["backend"], [0, 0])
            t[0] += sum(r[k] for k in ARTIFACTS)
            t[1] += r["words"]
    out += ["## All files", "", "| Backend | Artifacts | Words | Artifacts / 1000 words |", "|---|---|---|---|"]
    out += [f"| {b} | {a} | {w} | {1000 * a / w:.2f} |" for b, (a, w) in totals.items() if w]
    return "\n".join(out) + "\n"
