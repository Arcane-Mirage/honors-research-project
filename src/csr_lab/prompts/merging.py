"""src/csr_lab/prompts/merging.py

Stage prompts for reference merging.

* ``reference_merging`` (``llm`` method): group every reference label of a document bundle
  by the document it denotes and name each group.
* ``reference_alias_check`` (``hybrid`` method): for one candidate group chosen by
  deterministic rules, decide which labels denote the same document.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from csr_lab.prompts import PromptSpec


class ReferenceGroup(BaseModel):
    model_config = ConfigDict(extra="ignore")

    canonical_name: str = Field(description="Name to use for the document, taken from one of the members.")
    members: list[str] = Field(description="Input labels that refer to this document.")


class ReferenceMergingResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    groups: list[ReferenceGroup] = Field(default_factory=list)


class AliasDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    candidate_id: str
    same_document_as: str | None = Field(
        default=None, description="The option_id of the document this label refers to, or null if none or unsure."
    )
    reason: str = ""


class AliasCheckResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decisions: list[AliasDecision] = Field(default_factory=list)


MERGING_SPEC = PromptSpec(
    name="reference_merging",
    version="merging.v1",
    system=(
        "You consolidate the list of documents referenced in customer-specific requirements (CSR) "
        "documents for automotive supplier quality engineering. The same document is often written in "
        "different ways: with or without its type word, with different spacing or punctuation, by its "
        "title instead of its identifier."
    ),
    instructions=(
        "Group the labels in `labels` so that each group holds the labels that refer to the same "
        "document, and give each group a canonical_name copied from its clearest member. Every label "
        "must appear in exactly one group; a label that matches no other label is a group of its own. "
        "Labels for different documents must stay in different groups."
    ),
)

ALIAS_SPEC = PromptSpec(
    name="reference_alias_check",
    version="alias_check.v1",
    system=(
        "You check document references in customer-specific requirements (CSR) documents for automotive "
        "supplier quality engineering. Answer only from the information given; when unsure, answer null."
    ),
    instructions=(
        "Each candidate is a document label found in a requirement, with the requirement sentence it "
        "came from. For each candidate decide whether it refers to one of its `options` (documents known "
        "in this bundle, with their identifiers and names). Return the option_id it refers to, or null "
        "when it refers to none of them or the text does not make it clear. Different revisions of a "
        "document are different documents."
    ),
)
