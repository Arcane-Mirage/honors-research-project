"""src/csr_lab/prompts/references.py

Stage prompt: extract referenced documents for a batch of requirements.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from csr_lab.prompts import COMMON_RULES, PromptSpec
from csr_lab.schemas import DocumentType, ReferenceRelation


class ExtractedReference(BaseModel):
    """One document reference as reported by a model (shared by several prompts)."""

    model_config = ConfigDict(extra="ignore")

    identifier: str = Field(
        description="Document identifier exactly as written (e.g. 'CVM-QS-110', 'ES 4.12'). "
        "If the document is named without an identifier, its name as written."
    )
    revision: str | None = Field(default=None, description="Revision stated next to the reference, else null.")
    document_name: str | None = Field(
        default=None, description="Document title only if the text states it next to the reference, else null."
    )
    document_type: DocumentType | None = None
    relation: ReferenceRelation = ReferenceRelation.REFERENCES
    quoted_text: str = Field(description="Short verbatim excerpt of the requirement containing the reference.")


class RequirementReferences(BaseModel):
    model_config = ConfigDict(extra="ignore")

    requirement_id: str
    references: list[ExtractedReference] = Field(default_factory=list)


class ReferenceExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    results: list[RequirementReferences]


SPEC = PromptSpec(
    name="reference_extraction",
    version="references.v2",
    system=(
        "You extract document references from customer-specific requirements (CSR) for automotive "
        "supplier quality engineering.\n\n" + COMMON_RULES
    ),
    instructions=(
        "For each requirement in the input, list every document it explicitly references. Check the "
        "whole text, including tables and the end of long requirements. A requirement may reference "
        "several documents or none. Return one entry in `results` for every requirement_id in the "
        "input, with an empty `references` list when it references no document. Report the same "
        "document only once per requirement."
    ),
)
