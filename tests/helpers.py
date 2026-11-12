"""tests/helpers.py

Test helpers: scripted providers, a tiny PDF writer, and a gold-oracle provider.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from csr_lab.ingestion.loader import load_bundle, load_registry
from csr_lab.llm.interface import GenerationRequest, GenerationResponse, LLMProvider
from csr_lab.prompts import parse_input_payload
from csr_lab.schemas import CaseGold


class ScriptedProvider(LLMProvider):
    """Returns responses from a function of (task, payload, attempt_index)."""

    is_mock = True

    def __init__(self, fn: Callable[[str, dict, GenerationRequest], str | dict]):
        self.fn = fn
        self.requests: list[GenerationRequest] = []

    def describe(self) -> dict[str, Any]:
        return {"provider": "scripted"}

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        self.requests.append(request)
        user = next(m["content"] for m in reversed(request.messages) if "<input_json>" in m["content"])
        out = self.fn(request.task, parse_input_payload(user), request)
        return GenerationResponse(text=out if isinstance(out, str) else json.dumps(out), input_tokens=10, output_tokens=5)


class GoldOracleProvider(LLMProvider):
    """TEST ONLY: answers single-pass prompts (one call, all chapters) from gold annotations to check the evaluator."""

    is_mock = True

    def __init__(self, gold: CaseGold, customer: str | None = "Global Motors"):
        self.gold = gold
        self.customer = customer  # like a model told to skip text for other customers

    def describe(self) -> dict[str, Any]:
        return {"provider": "gold_oracle"}

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        assert request.task == "single_pass_extraction"
        user = next(m["content"] for m in reversed(request.messages) if "<input_json>" in m["content"])
        units = parse_input_payload(user)["units"]

        def flat(text: str) -> str:
            return " ".join(text.split())

        # Each gold requirement is one unit (chapter) of its document. Chapters can share their
        # text ('See Halvard Q Agreement. No additional requirements.'), so the chapter must match too.
        holder: dict[str, str] = {}
        for g in self.gold.requirements:
            for u in units:
                if u["document_id"] == g.source_document_id and u.get("chapter") == g.chapter and flat(g.requirement_text) in flat(u["text"]):
                    holder[g.gold_id] = u["key"]
                    break
        results = []
        for u in units:
            if u["context_only"]:
                continue
            refs = []
            for g in self.gold.requirements:
                if holder.get(g.gold_id) != u["key"]:
                    continue
                for r in g.references:
                    if self.customer and r.applies_to and self.customer not in r.applies_to:
                        continue
                    ident = r.canonical_identifier if r.resolution == "identifier" else r.mention_text
                    refs.append({"identifier": ident, "revision": r.revision, "relation": r.relation.value, "location": r.location, "quoted_text": r.mention_text})
            results.append({"key": u["key"], "references": refs})
        conflicts = [
            {
                "requirement_keys": [holder[c.requirement_ids[0]], holder[c.requirement_ids[1]]],
                "conflict_type": c.conflict_type.value,
                "description": c.note or "gold",
                "ambiguous": c.category == "ambiguous",
                "evidence": [{"requirement_key": holder[e.requirement_id], "quoted_text": e.quoted_text} for e in c.evidence],
            }
            for c in self.gold.conflicts
            if c.requirement_ids[0] in holder and c.requirement_ids[1] in holder
        ]
        return GenerationResponse(text=json.dumps({"results": results, "potential_conflicts": conflicts}))


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: list[list[str]]) -> bytes:
    """Minimal valid PDF with Helvetica text lines (an empty list makes a blank page)."""
    objects: list[bytes] = []
    n_pages = len(pages)
    page_ids = [4 + 2 * i for i in range(n_pages)]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{p} 0 R" for p in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, lines in enumerate(pages):
        content_id = page_ids[i] + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>".encode()
        )
        if lines:
            ops = ["BT", "/F1 10 Tf", "14 TL", "50 750 Td"]
            for j, line in enumerate(lines):
                ops.append(f"({_escape(line)}) Tj" if j == 0 else f"T* ({_escape(line)}) Tj")
            ops.append("ET")
            stream = "\n".join(ops).encode("latin-1")
        else:
            stream = b""
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


CASE = "heldout_halvard_delta"  # the synthetic dataset's case


def case_documents(dataset, case_id: str = CASE):
    """(case, documents with the case's documents.yaml applied, gold)."""
    case = next(c for c in dataset.cases if c.case_id == case_id)
    registry = load_registry(list(case.document_paths)) or None
    return case, load_bundle(list(case.document_paths), registry=registry), case.load_gold()


def with_reader(config, dataset):
    """``config`` read for the dataset's reader (manifest ``reader``), as experiments do."""
    from csr_lab.config import ExperimentConfig, apply_experiment

    return apply_experiment(config, ExperimentConfig(name="test", dataset=str(dataset.manifest_path)))


def run_case(dataset, strategy, config, case_id: str = CASE, **kwargs):
    """Run a case as an experiment does (documents.yaml, the dataset's reader); returns (result, gold)."""
    from csr_lab.pipelines.run import load_inputs, run_analysis

    case = next(c for c in dataset.cases if c.case_id == case_id)
    config = with_reader(config, dataset)
    docs, registry = load_inputs(list(case.document_paths), config)
    return run_analysis(docs, strategy=strategy, config=config, case_id=case_id, registry=registry, **kwargs), case.load_gold()


# A two-document bundle for the graph rules the synthetic case does not contain: an
# exclusion contradicted by the referenced specification, mutual precedence claims, a shared
# reference, an unavailable document, and a provided revision under its document node.
RULES_CSR = """\
HALVARD VEHICLE GROUP
Logistics and Packaging Requirements for Production Suppliers
Document ID: HVG-CSR-200
Revision: A
Effective Date: 2026-01-10

1 General
1.1 This document applies to all shipments of production parts to Halvard assembly plants.

1.2 The supplier shall package all production parts in accordance with the Packaging and
Labeling Specification HVG-PKG-210.

1.3 Packaging Standard HVG-PKG-050 does not apply to parts shipped under this document and
shall not be used for packaging design.

2 Labels
2.1 Each container shall carry a shipping label that conforms to HVG-LBL-220.

2.2 Container labels shall be printed with a minimum barcode height of 25 mm.

3 Containers
3.1 Production parts shall be shipped only in returnable containers approved by Halvard
Logistics.

4 Deliveries
4.1 The supplier shall transmit an advance shipping notice using the message format defined in
HVG-LG-140.

5 Precedence
5.1 Where this document conflicts with HVG-PKG-210, the requirements of this document shall
take precedence.
"""

RULES_SPEC = """\
HALVARD VEHICLE GROUP
Packaging and Labeling Specification
Document ID: HVG-PKG-210
Revision: D
Effective Date: 2025-11-03

1 Scope
1.1 This specification defines packaging and labeling requirements for production parts.

2 Packaging Design
2.1 Packaging design shall follow the dimensional guidelines of Packaging Standard
HVG-PKG-050.

2.2 Expendable (one-way) cardboard containers shall be used for all parts weighing less than 2
kg.

3 Labeling
3.1 Labels shall conform to HVG-LBL-220 and shall have a barcode height of no more than 20
mm.

4 Precedence
4.1 In the event of a conflict between this specification and HVG-CSR-200, the requirements
of this specification shall take precedence.
"""


def write_rules_bundle(folder: Path) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = [folder / "hvg_csr_200.txt", folder / "hvg_pkg_210.txt"]
    for path, text in zip(paths, (RULES_CSR, RULES_SPEC)):
        path.write_text(text, encoding="utf-8")
    return paths
