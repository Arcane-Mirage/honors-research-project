"""Test helpers: scripted providers, a tiny PDF writer, and a gold-oracle provider."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from csr_lab.llm.interface import GenerationRequest, GenerationResponse, LLMProvider
from csr_lab.prompts import parse_input_payload
from csr_lab.ingestion.loader import load_bundle
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
    """TEST ONLY: answers single-pass prompts from gold annotations to check the evaluator end to end."""

    is_mock = True

    def __init__(self, gold: CaseGold):
        self.gold = gold

    def describe(self) -> dict[str, Any]:
        return {"provider": "gold_oracle"}

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        assert request.task == "single_pass_extraction"
        reqs, keys = [], {}
        for i, g in enumerate(self.gold.requirements, start=1):
            key = f"R{i}"
            keys[g.gold_id] = key
            refs = []
            for r in g.references:
                ident = r.canonical_identifier if r.resolution == "identifier" else r.mention_text
                refs.append(
                    {
                        "identifier": ident,
                        "revision": r.revision,
                        "relation": r.relation.value,
                        "quoted_text": r.mention_text,
                    }
                )
            reqs.append(
                {"key": key, "source_document_id": g.source_document_id, "chapter": g.chapter, "requirement_text": g.requirement_text, "references": refs}
            )
        conflicts = [
            {
                "requirement_keys": [keys[c.requirement_ids[0]], keys[c.requirement_ids[1]]],
                "conflict_type": c.conflict_type.value,
                "description": c.note or "gold",
                "ambiguous": c.category == "ambiguous",
                "evidence": [{"requirement_key": keys[e.requirement_id], "quoted_text": e.quoted_text} for e in c.evidence],
            }
            for c in self.gold.conflicts
        ]
        return GenerationResponse(text=json.dumps({"requirements": reqs, "potential_conflicts": conflicts}))


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


def case_documents(dataset, case_id: str):
    case = next(c for c in dataset.cases if c.case_id == case_id)
    return case, load_bundle(list(case.document_paths)), case.load_gold()
