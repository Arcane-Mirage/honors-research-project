"""src/csr_lab/ingestion/provided_chunks.py

Load pre-chunked input: chunks produced by another tool, analysed by csr-lab's later stages.

File format (``*.json`` with a ``chunks`` list; unknown keys are ignored)::

    {
      "format": "csr-lab.chunks/1",
      "chunker": "name of the tool that made the chunks (optional)",
      "source": {                      # optional, but needed to check the chunks
        "file": "manual.pdf",          #   .txt/.pdf/.json, relative to this file, or
        "text": "...full text...",     #   the text itself
        "identifier": "CVM-CSR-100", "title": "...", "revision": "C"   # optional header values
      },
      "chunks": [
        {"chunk_id": "c1", "title": "Scope", "content": "...", "source_pages": [1, 2]}
      ]
    }

``content`` may also be called ``text``; ``source_pages`` may be called ``pages``;
``chunk_id`` may be ``id`` or left out. A plain list of chunk objects under
``chunks`` is the only required part.

When the source text is given, every chunk is located in it (``extraction.fidelity``)
and requirement offsets refer to the source. Without it, the chunks are joined into
the document text and the fidelity check is meaningless (recorded as a warning).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from csr_lab.schemas import InputDocument, ProvidedChunk

CHUNKS_FORMAT = "csr-lab.chunks/1"


def is_chunk_payload(data: Any) -> bool:
    return isinstance(data, dict) and isinstance(data.get("chunks"), list)


def parse_chunks(items: list[Any]) -> list[ProvidedChunk]:
    chunks: list[ProvidedChunk] = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, str):
            item = {"content": item}
        if not isinstance(item, dict):
            raise ValueError(f"chunk {i} must be an object or a string")
        content = item.get("content", item.get("text"))
        if not isinstance(content, str) or not content.strip():
            continue
        pages = item.get("source_pages", item.get("pages")) or []
        if isinstance(pages, str):  # "3, 4"
            pages = [p for p in pages.replace(";", ",").split(",") if p.strip()]
        chunks.append(
            ProvidedChunk(
                chunk_id=str(item.get("chunk_id") or item.get("id") or f"C{i:03d}"),
                title=item.get("title"),
                content=content,
                source_pages=sorted({int(p) for p in pages if str(p).strip().isdigit()}),
            )
        )
    ids = [c.chunk_id for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("chunk_id values must be unique")
    return chunks


def load_chunk_file(path: Path, data: dict, raw: bytes, *, load_source) -> InputDocument:
    """Build an InputDocument carrying ``provided_chunks``. ``load_source(path)`` loads the source file."""
    from csr_lab.ingestion.loader import _build_document  # local import: loader imports this module lazily

    chunks = parse_chunks(data["chunks"])
    if not chunks:
        raise ValueError(f"{path.name}: no non-empty chunks")
    source = data.get("source") or {}
    source_file = source.get("file") or data.get("source_file")
    warnings = [f"pre-chunked input: {len(chunks)} chunks" + (f" from {data['chunker']}" if data.get("chunker") else "")]
    digest = hashlib.sha256(raw)
    if source_file:
        src_path = (path.parent / source_file).resolve()
        if not src_path.is_file():
            raise FileNotFoundError(f"{path.name}: source file not found: {src_path}")
        doc = load_source(src_path)
        digest.update(src_path.read_bytes())
        update: dict[str, Any] = {}
        if source.get("identifier") and doc.identifier_source != "header":
            update["warnings"] = [*doc.warnings, "chunk file's source.identifier ignored: the source header is used when present"]
        doc = doc.model_copy(update=update)
        warnings.append(f"chunks are checked against the source file {source_file}")
    else:
        text = source.get("text", data.get("source_text"))
        if text is None:
            text = "\n\n".join(c.content for c in chunks)
            warnings.append("no source text given: chunks are analysed as the document text and cannot be checked against a source")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        doc = _build_document(
            path=path,
            text=text,
            raw=raw,
            fmt="json",
            identifier=source.get("identifier") or data.get("identifier"),
            title=source.get("title") or data.get("title"),
            revision=source.get("revision") or data.get("revision"),
            effective_date=source.get("effective_date"),
            identifier_source="manifest" if (source.get("identifier") or data.get("identifier")) else None,
            parse_status="ok" if text.strip() else "unsupported",
        )
    return doc.model_copy(
        update={
            "filename": path.name,
            "sha256": digest.hexdigest(),
            "warnings": [*doc.warnings, *warnings],
            "provided_chunks": chunks,
        }
    )


def write_chunk_file(
    path: str | Path,
    chunks: list[dict],
    *,
    source_file: str | None = None,
    source_text: str | None = None,
    chunker: str | None = None,
    identifier: str | None = None,
) -> Path:
    """Write a chunk file in the format above (used by `csr-lab chunk` and the dataset generator)."""
    path = Path(path)
    source: dict[str, Any] = {}
    if source_file:
        source["file"] = source_file
    elif source_text is not None:
        source["text"] = source_text
    if identifier:
        source["identifier"] = identifier
    body = {"format": CHUNKS_FORMAT, "chunker": chunker, "source": source or None, "chunks": chunks}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    return path
