"""src/csr_lab/prompts/baseline.py

Single-pass baseline prompt: everything in one structured call per chunk.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from csr_lab.prompts import COMMON_RULES, PromptSpec
from csr_lab.prompts.references import ExtractedReference
from csr_lab.schemas import ConflictType


class SinglePassRequirement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str = Field(description="Local key: R1, R2, ...")
    source_document_id: str
    chapter: str | None = None
    requirement_text: str
    references: list[ExtractedReference] = Field(default_factory=list)


class SinglePassEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requirement_key: str
    quoted_text: str


class SinglePassConflict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requirement_keys: list[str] = Field(min_length=2, max_length=2)
    conflict_type: ConflictType
    description: str
    ambiguous: bool = False
    evidence: list[SinglePassEvidence] = Field(default_factory=list)


class SinglePassResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requirements: list[SinglePassRequirement]
    potential_conflicts: list[SinglePassConflict] = Field(default_factory=list)


SPEC = PromptSpec(
    name="single_pass_extraction",
    version="baseline.v1",
    system=(
        "You analyse customer-specific requirements (CSR) documents for automotive supplier quality "
        "engineering and return structured JSON.\n\n" + COMMON_RULES + "\n"
        "- chapter: the section label exactly as printed (e.g. '4.2.1' or 'TSA 2.1'). For a paragraph "
        "under a heading without its own number, use the heading's label. Do not include the label in "
        "requirement_text.\n"
        "- A potential conflict is a pair of requirements whose obligations cannot both be met as "
        "written: contradictory obligations, incompatible numeric limits, incompatible timing, "
        "incompatible processes, contradictory precedence statements, or a document excluded in one "
        "place and required in another. Sharing a referenced document is not a conflict by itself. "
        "Set ambiguous=true when review is warranted but incompatibility is uncertain. Never decide "
        "which requirement prevails."
    ),
    instructions=(
        "Extract every requirement from the documents below, the documents each requirement "
        "references, and potential conflicts between requirements (also across documents). Give each "
        "requirement a key R1, R2, ... and use these keys in potential_conflicts. Use the document_id "
        "values from the input as source_document_id."
    ),
)
