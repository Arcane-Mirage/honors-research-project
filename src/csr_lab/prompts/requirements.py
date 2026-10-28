"""src/csr_lab/prompts/requirements.py

Stage prompt: identify requirements and chapters in pre-segmented text.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from csr_lab.prompts import COMMON_RULES, PromptSpec


class IdentifiedRequirement(BaseModel):
    model_config = ConfigDict(extra="ignore")

    segment_id: str
    chapter: str | None = None
    requirement_text: str


class RequirementIdentificationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requirements: list[IdentifiedRequirement] = Field(default_factory=list)


SPEC = PromptSpec(
    name="requirement_identification",
    version="requirements.v1",
    system=(
        "You identify requirements in customer-specific requirements (CSR) documents for automotive "
        "supplier quality engineering.\n\n" + COMMON_RULES
    ),
    instructions=(
        "Each segment below is a block of text from a CSR document, already split at section "
        "boundaries. Return every requirement found in the segments. Use the segment's chapter unless "
        "the text clearly states a different one. If one segment holds several separate requirements, "
        "return each separately with its own verbatim text. Do not return requirements from segments "
        "marked context_only: true; they are shown only for context. Segments that contain no "
        "requirement are simply left out."
    ),
)
