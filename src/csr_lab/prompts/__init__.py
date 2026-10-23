"""Versioned prompt templates and the response models they request.

Each prompt module defines a ``SPEC`` (name, version, system text, user
instructions) and the Pydantic response model(s). The input data is always sent
as a JSON block between ``<input_json>`` tags so that the text is unambiguous
for real models and machine-readable for the mock model.

``prompt_versions()`` returns ``name -> version@hash``. The hash covers the
prompt text, so an edited prompt is visible in stored results even if someone
forgets to bump the version.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

INPUT_OPEN = "<input_json>"
INPUT_CLOSE = "</input_json>"
_INPUT_RE = re.compile(re.escape(INPUT_OPEN) + r"\s*(?P<body>.*?)\s*" + re.escape(INPUT_CLOSE), re.DOTALL)


@dataclass(frozen=True)
class PromptSpec:
    name: str
    version: str
    system: str
    instructions: str

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(f"{self.system}\n{self.instructions}".encode()).hexdigest()[:8]
        return f"{self.version}@{digest}"

    def render(self, payload: dict[str, Any]) -> tuple[str, str]:
        body = json.dumps(payload, ensure_ascii=False, indent=1)
        return self.system, f"{self.instructions}\n\n{INPUT_OPEN}\n{body}\n{INPUT_CLOSE}"


def parse_input_payload(text: str) -> dict[str, Any]:
    match = _INPUT_RE.search(text)
    if not match:
        raise ValueError("message has no <input_json> block")
    return json.loads(match.group("body"))


COMMON_RULES = """\
Definitions used in every task:
- A requirement is a statement that obliges, permits under conditions, or prohibits something,
  usually with "shall", "must" or "is required". Headings, titles, document identifiers, scope
  descriptions and purely informational sentences are not requirements.
- Requirement text is copied verbatim from the source. Never summarise, correct or paraphrase it.
- A document reference is an explicit mention of another document (specification, standard,
  drawing, procedure, manual, form, customer-specific requirements document) by its identifier
  or by its specific name. Section or clause numbers, part numbers, dates, quantities,
  measurement values and lighting or material designations are not document references.
  Generic phrases such as "the applicable drawing" are not references.
- Reference relation: "excludes" when the text says the document does not apply or must not be
  used; "supersedes" when the text says it replaces or supersedes the document;
  "takes_precedence_over" when the text says the current document takes precedence over or
  governs the referenced one; "depends_on" when it must be used in conjunction with it;
  otherwise "references".
- Every quoted_text value must be copied character for character from the input text.
- Never invent identifiers, names, revisions or content of documents you were not given."""


def prompt_versions() -> dict[str, str]:
    from csr_lab.prompts import baseline, conflicts, references, requirements, verification

    specs = [baseline.SPEC, requirements.SPEC, references.SPEC, verification.SPEC, conflicts.SPEC]
    return {spec.name: spec.fingerprint for spec in specs}
