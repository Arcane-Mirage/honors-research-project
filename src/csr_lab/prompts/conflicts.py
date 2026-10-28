"""src/csr_lab/prompts/conflicts.py

Stage prompt: assess candidate requirement pairs for potential conflicts.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from csr_lab.prompts import COMMON_RULES, PromptSpec
from csr_lab.schemas import ConflictType


class PairEvidence(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requirement_id: str
    quoted_text: str


class PairAssessment(BaseModel):
    model_config = ConfigDict(extra="ignore")

    pair_id: str
    potential_conflict: bool
    conflict_type: ConflictType | None = None
    ambiguous: bool = False
    description: str = ""
    evidence: list[PairEvidence] = Field(default_factory=list)


class ConflictAnalysisResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    assessments: list[PairAssessment] = Field(default_factory=list)


SPEC = PromptSpec(
    name="conflict_analysis",
    version="conflicts.v2",
    system=(
        "You flag potentially conflicting requirements in customer-specific requirements (CSR) "
        "documents for human review. You do not resolve conflicts, decide precedence, or give legal "
        "conclusions.\n\n" + COMMON_RULES + "\n"
        "- A potential conflict exists when both requirements cannot be met as written: contradictory "
        "obligations, incompatible numeric limits, incompatible timing, incompatible processes, "
        "contradictory precedence statements, or a document excluded in one place and required in "
        "another. Consider whether different products, conditions, scopes or time frames explain the "
        "difference; if they plausibly do, it is not a conflict. Sharing a referenced document is not "
        "a conflict by itself. Use ambiguous=true when review is warranted but incompatibility is "
        "uncertain."
    ),
    instructions=(
        "Assess each requirement pair below. Return one assessment per pair_id. For a potential "
        "conflict, give the conflict_type, a one-sentence description of why the statements may be "
        "incompatible, and a verbatim quote from each requirement (requirement_id 'A' or 'B' says "
        "which requirement the quote is from). If a pair has related_context, it "
        "lists relationships taken from the document relationship graph, where A and B mean the two "
        "requirements of that pair; use it as background only and base every conclusion on the "
        "quoted requirement text."
    ),
)
