"""Stable identifier helpers.

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
    """Node ID for a document. Revision-qualified references are distinct nodes."""
    if canonical:
        base = f"DOC-{slug(canonical)}"
        return f"{base}-REV-{slug(revision)}" if revision else base
    if name:
        return f"DOC-NAME-{slug(name)}"
    raise ValueError("a document node needs a canonical identifier or a name")


def requirement_id(document_id: str, index: int) -> str:
    """'DOC-CVM-CSR-100', 3 -> 'REQ-CVM-CSR-100-003'."""
    base = document_id[4:] if document_id.startswith("DOC-") else slug(document_id)
    return f"REQ-{base}-{index:03d}"


def relationship_id(source: str, relationship_type: str, target: str) -> str:
    return f"REL-{stable_hash(source, relationship_type, target, length=12)}"


def conflict_id(requirement_ids: list[str] | tuple[str, ...]) -> str:
    return f"CON-{stable_hash(*sorted(requirement_ids), length=10)}"
