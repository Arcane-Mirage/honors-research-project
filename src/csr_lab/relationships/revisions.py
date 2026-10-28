"""src/csr_lab/relationships/revisions.py

Compare the requirements of two revisions of one document.

A requirement is identified by its document, chapter and verbatim text, so
comparing two revisions needs no model:

* same chapter, same text           -> ``unchanged``
* same text, different chapter      -> ``moved``
* same chapter, different text      -> ``changed``
* only in the older revision        -> ``removed``
* only in the newer revision        -> ``added``

Text is compared after collapsing whitespace, so different line wrapping does
not count as a change. Passes run in the order above; inside a chapter with
several requirements, leftovers are paired in source order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Change = Literal["unchanged", "moved", "changed", "removed", "added"]


@dataclass(frozen=True)
class RequirementVersion:
    requirement_id: str
    chapter: str | None
    text: str

    @property
    def norm(self) -> str:
        return " ".join(self.text.split())

    @property
    def chapter_key(self) -> str | None:
        return " ".join(self.chapter.split()).upper() if self.chapter else None


@dataclass(frozen=True)
class RevisionChange:
    change: Change
    old: RequirementVersion | None
    new: RequirementVersion | None

    def as_dict(self) -> dict:
        return {
            "change": self.change,
            "old_requirement_id": self.old.requirement_id if self.old else None,
            "new_requirement_id": self.new.requirement_id if self.new else None,
            "old_chapter": self.old.chapter if self.old else None,
            "new_chapter": self.new.chapter if self.new else None,
        }


def compare_revisions(old: list[RequirementVersion], new: list[RequirementVersion]) -> list[RevisionChange]:
    """Pair the requirements of an older and a newer revision (both in source order)."""
    old_left, new_left = list(old), list(new)
    changes: list[RevisionChange] = []

    def take(change: Change, match) -> None:
        for o in list(old_left):
            n = next((n for n in new_left if match(o, n)), None)
            if n is not None:
                changes.append(RevisionChange(change, o, n))
                old_left.remove(o)
                new_left.remove(n)

    take("unchanged", lambda o, n: o.chapter_key == n.chapter_key and o.norm == n.norm)
    take("moved", lambda o, n: o.norm == n.norm)
    take("changed", lambda o, n: o.chapter_key is not None and o.chapter_key == n.chapter_key)
    changes += [RevisionChange("removed", o, None) for o in old_left]
    changes += [RevisionChange("added", None, n) for n in new_left]
    return changes
