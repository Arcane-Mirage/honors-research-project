"""Load evaluation datasets described by a ``manifest.json``.

Manifest format (paths are relative to the manifest)::

    {"name": "...", "version": "...",
     "cases": [{"case_id": "...", "split": "dev|heldout",
                "documents": ["documents/x/a.txt", ...],
                "ground_truth": "ground_truth/x.json"}]}

Public datasets (for example PURE) can be added later with the same format;
cases without ``ground_truth`` can be analysed but not scored.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from csr_lab.config import PROJECT_ROOT
from csr_lab.schemas import CaseGold


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    description: str
    document_paths: tuple[Path, ...]
    gold_path: Path | None

    def load_gold(self) -> CaseGold | None:
        """Gold annotations. Called only by the evaluator, never by a pipeline."""
        if self.gold_path is None:
            return None
        return CaseGold.model_validate_json(self.gold_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Dataset:
    name: str
    version: str
    manifest_path: Path
    manifest_sha256: str
    cases: tuple[Case, ...]

    def select(self, split: str = "all", case_ids: list[str] | None = None) -> list[Case]:
        cases = [c for c in self.cases if split == "all" or c.split == split]
        if case_ids:
            unknown = set(case_ids) - {c.case_id for c in self.cases}
            if unknown:
                raise KeyError(f"unknown case ids: {sorted(unknown)}")
            cases = [c for c in cases if c.case_id in case_ids]
        return cases


def load_dataset(path: str | Path) -> Dataset:
    manifest_path = Path(path)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_ROOT / manifest_path
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    raw = manifest_path.read_bytes()
    data = json.loads(raw)
    root = manifest_path.parent
    cases = []
    for item in data["cases"]:
        cases.append(
            Case(
                case_id=item["case_id"],
                split=item.get("split", "heldout"),
                description=item.get("description", ""),
                document_paths=tuple(root / p for p in item["documents"]),
                gold_path=(root / item["ground_truth"]) if item.get("ground_truth") else None,
            )
        )
    digest = hashlib.sha256(raw)
    for case in cases:
        for p in (*case.document_paths, *( [case.gold_path] if case.gold_path else [])):
            digest.update(p.read_bytes())
    return Dataset(
        name=data.get("name", manifest_path.parent.name),
        version=data.get("version", "unversioned"),
        manifest_path=manifest_path,
        manifest_sha256=digest.hexdigest(),
        cases=tuple(cases),
    )
