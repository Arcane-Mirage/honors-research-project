"""src/csr_lab/drafts.py

Intermediate prediction records passed between pipeline stages.

These are working representations. They are converted to the export schema
(``csr_lab.schemas``) by ``csr_lab.pipelines.assembly`` at the end of every
strategy, using the same code for all strategies.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from csr_lab.extraction.normalization import reference_key
from csr_lab.ids import document_node_id
from csr_lab.schemas import (
    ConflictType,
    DocumentType,
    Evidence,
    MentionStatus,
    ReferenceRelation,
    ResolutionMethod,
    TextMatch,
)


class RequirementDraft(BaseModel):
    requirement_id: str
    document_id: str
    chapter: str | None
    text: str
    start_offset: int | None = None
    end_offset: int | None = None
    match: TextMatch = "not_checked"
    section_path: list[str] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    segmentation_confidence: Literal["high", "low", "not_segmented"] = "not_segmented"
    page_start: int | None = None
    page_end: int | None = None
    reference_status: Literal["not_run", "unverified", "verified", "failed"] = "not_run"
    notes: list[str] = Field(default_factory=list)

    @property
    def span(self) -> tuple[int, int] | None:
        if self.start_offset is None or self.end_offset is None:
            return None
        return self.start_offset, self.end_offset


class MentionDraft(BaseModel):
    mention_id: str
    requirement_id: str
    raw_identifier: str
    canonical: str | None
    revision: str | None = None
    name: str | None = None
    document_type: DocumentType = DocumentType.UNKNOWN
    relation: ReferenceRelation = ReferenceRelation.REFERENCES
    evidence: Evidence
    status: MentionStatus = MentionStatus.FOUND_INITIALLY
    resolution: ResolutionMethod = ResolutionMethod.IDENTIFIER
    verified: bool = False
    active: bool = True
    inactive_reason: str | None = None

    @property
    def key(self) -> str:
        return reference_key(self.canonical, self.revision, self.name)

    @property
    def document_id(self) -> str:
        return document_node_id(self.canonical, self.revision, self.name)


class ConflictDraft(BaseModel):
    requirement_ids: list[str] = Field(min_length=2, max_length=2)
    conflict_type: ConflictType
    description: str
    evidence: list[Evidence] = Field(default_factory=list)
    assessment: Literal["apparent_incompatibility", "ambiguous"] = "apparent_incompatibility"
    detection_methods: list[str] = Field(default_factory=list)
    evidence_valid: bool | None = None
    reason_for_review: str = ""
    related_document_ids: list[str] = Field(default_factory=list)
    related_relationship_ids: list[str] = Field(default_factory=list)
    graph_informed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def pair(self) -> tuple[str, str]:
        a, b = sorted(self.requirement_ids)
        return a, b


class CandidatePair(BaseModel):
    requirement_ids: tuple[str, str]
    selectors: list[str] = Field(default_factory=list)
    graph_derived: bool = False
    analyzed: bool = False


class PipelineError(BaseModel):
    stage: str
    message: str
    recoverable: bool = True
    chunk_id: str | None = None
    requirement_id: str | None = None


def ordered_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)
