"""src/csr_lab/prompts/baseline.py

Single-pass baseline prompt: references and conflicts for a group of chapters in one call.

Each unit in the input is one requirement (a whole chapter, or a model-made chunk). Units
marked ``context_only`` repeat the end of the previous group: they may appear in conflicts
but their references are reported where they are not context.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from csr_lab.prompts import COMMON_RULES, PromptSpec
from csr_lab.prompts.references import ExtractedReference
from csr_lab.schemas import ConflictType


class SinglePassUnitReferences(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str = Field(description="The unit's key from the input: R1, R2, ...")
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

    results: list[SinglePassUnitReferences]
    potential_conflicts: list[SinglePassConflict] = Field(default_factory=list)


SPEC = PromptSpec(
    name="single_pass_extraction",
    version="baseline.v2",
    system=(
        "You analyse customer-specific requirements (CSR) documents for automotive supplier quality "
        "engineering and return structured JSON.\n\n" + COMMON_RULES + "\n"
        "- Each input unit is one requirement: a whole chapter, keyed R1, R2, ...\n"
        "- A potential conflict is a pair of requirements whose obligations cannot both be met as "
        "written: contradictory obligations, incompatible numeric limits, incompatible timing, "
        "incompatible processes, contradictory precedence statements, or a document excluded in one "
        "place and required in another. Sharing a referenced document is not a conflict by itself. "
        "Set ambiguous=true when review is warranted but incompatibility is uncertain. Never decide "
        "which requirement prevails."
    ),
    instructions=(
        "For each unit below that is not context_only, list every document its text references "
        "(an empty list when it references none), with one entry in `results` per unit key. Then list "
        "potential conflicts between units, also across documents, using the unit keys; context_only "
        "units may take part in conflicts."
    ),
)
