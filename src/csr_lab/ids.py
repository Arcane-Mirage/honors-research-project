"""src/csr_lab/ids.py

Stable identifier helpers.

IDs are derived from content (not random) so that the same input and
configuration produce the same IDs on every run.
"""

from __future__ import annotations

import hashlib
import re

_NON_ALNUM = re.compile(r"[^A-Z0-9]+")


def slug(value: str) -> str:
    """Upper-case slug: 'cvm qs/110' -> 'CVM-QS-110'."""
    return _NON_ALNUM.sub("-", value.upper()).strip("-")


def stable_hash(*parts: object, length: int = 10) -> str:
    joined = "\x1f".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()[:length]


def document_node_id(canonical: str | None, revision: str | None = None, name: str | None = None) -> str:
    """Node ID for a document or one revision of it.

    'CVM-PKG-210' -> 'DOC-CVM-PKG-210' (the document, any revision);
    'CVM-PKG-210', 'D' -> 'DOC-CVM-PKG-210-REV-D' (that revision only).
    """
    if canonical:
        base = f"DOC-{slug(canonical)}"
        rev = slug(revision) if revision else ""
        return f"{base}-REV-{rev}" if rev else base
    if name:
        return f"DOC-NAME-{slug(name)}"
    raise ValueError("a document node needs a canonical identifier or a name")


def document_family_id(canonical: str) -> str:
    """ID of the revision-independent document node: 'CVM-PKG-210' -> 'DOC-CVM-PKG-210'."""
    return document_node_id(canonical)


_CHAPTER_CHARS = re.compile(r"[^A-Z0-9.]+")


def chapter_slug(chapter: str | None) -> str:
    """'TSA 2.1' -> 'TSA-2.1'; a missing chapter -> 'NOCHAPTER'."""
    value = _CHAPTER_CHARS.sub("-", (chapter or "").upper()).strip("-.")
    return value or "NOCHAPTER"


def requirement_id(document_id: str, chapter: str | None) -> str:
    """A requirement is a whole chapter, identified by its document and chapter label.

    'DOC-CVS10-REV-2024-08', '1.1.4' -> 'REQ-CVS10-REV-2024-08-1.1.4'; 'Scope' -> '...-SCOPE'
    """
    base = document_id[4:] if document_id.startswith("DOC-") else slug(document_id)
    return f"REQ-{base}-{chapter_slug(chapter)}"


def relationship_id(source: str, relationship_type: str, target: str) -> str:
    return f"REL-{stable_hash(source, relationship_type, target, length=12)}"


def conflict_id(requirement_ids: list[str] | tuple[str, ...]) -> str:
    return f"CON-{stable_hash(*sorted(requirement_ids), length=10)}"
