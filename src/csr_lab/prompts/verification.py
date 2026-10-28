"""src/csr_lab/prompts/verification.py

Verification prompt: targeted re-examination of candidate references.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from csr_lab.prompts import COMMON_RULES, PromptSpec
from csr_lab.schemas import ReferenceRelation


class CandidateDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    is_document_reference: bool
    identifier: str | None = None
    revision: str | None = None
    document_name: str | None = None
    relation: ReferenceRelation = ReferenceRelation.REFERENCES
    quoted_text: str | None = None
    reason: str = ""


class TargetedVerificationResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decisions: list[CandidateDecision] = Field(default_factory=list)


SPEC = PromptSpec(
    name="reference_verification",
    version="verification.v1",
    system=(
        "You verify document references in customer-specific requirements (CSR) for automotive "
        "supplier quality engineering.\n\n" + COMMON_RULES
    ),
    instructions=(
        "A deterministic scan found the candidate mentions below in one requirement. For each "
        "candidate decide whether it is an explicit reference to a document. Candidates of kind "
        "'possible_missed' were not in the first extraction; 'doubtful_existing' were extracted but "
        "may be a number, section label or other non-document; 'known_document' match a document "
        "that is referenced elsewhere in the analysed documents (see related_documents). If a "
        "candidate is a reference, give its identifier exactly as written (or its name if no "
        "identifier is written), any stated revision, the relation, and a verbatim quote. Return one "
        "decision per candidate_id."
    ),
)
