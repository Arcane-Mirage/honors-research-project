"""src/csr_lab/schemas.py

Pydantic models for the exported analysis JSON and the gold-standard annotations.

Two groups of models live here:

* The **export schema** (``AnalysisResult`` and everything it contains). This is
  the integration contract with downstream consumers such as an industrial CSR
  evaluator. Bump ``SCHEMA_VERSION`` whenever it changes in a way consumers can see.
* The **ground-truth schema** (``CaseGold``). Gold files are written by the
  synthetic data generator and read only by the evaluator. Pipelines never see them.

Every strategy, including the single-pass baseline, must produce an
``AnalysisResult`` so all strategies are scored by the same code.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from csr_lab.extraction.fidelity import FidelityReport
from csr_lab.ids import document_family_id

# 0.2.0: revision-specific document IDs, content-based requirement IDs
# 0.3.0: input fidelity report, reference-merge decisions, per-stage cost (all additive)
SCHEMA_VERSION = "0.3.0"
GOLD_SCHEMA_VERSION = "0.2.0"


class _Model(BaseModel):
    """Base model: unknown fields are errors, so schema drift is caught early."""

    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class DocumentType(StrEnum):
    STANDARD = "standard"
    TECHNICAL_SPECIFICATION = "technical_specification"
    ENGINEERING_DRAWING = "engineering_drawing"
    PROCEDURE = "procedure"
    QUALITY_MANUAL = "quality_manual"
    CUSTOMER_SPECIFICATION = "customer_specification"
    FORM = "form"
    UNKNOWN = "unknown"


class ReferenceRelation(StrEnum):
    """How a requirement relates to a document it mentions.

    Every mention is a reference. The other values record an *explicit*
    qualifier stated in the source text; they are never inferred from
    revision numbers or dates.
    """

    REFERENCES = "references"
    EXCLUDES = "excludes"  # "X does not apply", "X shall not be used"
    SUPERSEDES = "supersedes"  # "this document supersedes X"
    DEPENDS_ON = "depends_on"  # "to be used in conjunction with X"
    TAKES_PRECEDENCE_OVER = "takes_precedence_over"  # "this document takes precedence over X"


class RelationshipType(StrEnum):
    CONTAINS = "contains"
    REFERENCES = "references"
    EXPLICITLY_EXCLUDES = "explicitly_excludes"
    EXPLICITLY_SUPERSEDES = "explicitly_supersedes"
    EXPLICITLY_DEPENDS_ON = "explicitly_depends_on"
    EXPLICITLY_TAKES_PRECEDENCE_OVER = "explicitly_takes_precedence_over"
    POTENTIALLY_CONFLICTS_WITH = "potentially_conflicts_with"
    # Revision tracking (derived from document headers, never from requirement text):
    HAS_REVISION = "has_revision"  # document -> one revision of it
    SUPERSEDED_BY = "superseded_by"  # older revision -> newer revision; older requirement -> its newer version


RELATION_EDGE_TYPES: dict[ReferenceRelation, RelationshipType] = {
    ReferenceRelation.EXCLUDES: RelationshipType.EXPLICITLY_EXCLUDES,
    ReferenceRelation.SUPERSEDES: RelationshipType.EXPLICITLY_SUPERSEDES,
    ReferenceRelation.DEPENDS_ON: RelationshipType.EXPLICITLY_DEPENDS_ON,
    ReferenceRelation.TAKES_PRECEDENCE_OVER: RelationshipType.EXPLICITLY_TAKES_PRECEDENCE_OVER,
}


class ConflictType(StrEnum):
    CONTRADICTORY_OBLIGATION = "contradictory_obligation"
    NUMERIC_INCOMPATIBILITY = "numeric_incompatibility"
    TIMING_INCOMPATIBILITY = "timing_incompatibility"
    PROCESS_INCOMPATIBILITY = "process_incompatibility"
    POTENTIAL_PRECEDENCE_CONFLICT = "potential_precedence_conflict"
    APPLICABILITY_CONFLICT = "applicability_conflict"
    OTHER_OR_UNCERTAIN = "other_or_uncertain"


class MentionStatus(StrEnum):
    """Where a reference came from. Lets experiments separate stages."""

    FOUND_INITIALLY = "found_initially"
    ADDED_DURING_VERIFICATION = "added_during_verification"
    ADDED_BY_GRAPH_AUDIT = "added_by_graph_audit"


class ResolutionMethod(StrEnum):
    IDENTIFIER = "identifier"  # resolved from an explicit identifier in the text
    NAME_ALIAS = "name_alias"  # a document name resolved to a known identifier
    UNRESOLVED_NAME = "unresolved_name"  # a document name with no known identifier
    MERGED = "merged"  # renamed by the reference-merging stage (see metadata.reference_merges)


# How a piece of quoted text was located in the extracted source text.
TextMatch = Literal["exact", "whitespace_normalized", "not_found", "not_checked"]


# ---------------------------------------------------------------------------
# Source traceability
# ---------------------------------------------------------------------------


class PageSpan(_Model):
    page_number: int = Field(ge=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)


class SourceLocation(_Model):
    """Where something sits in the *extracted* text of an input document.

    Offsets are character offsets into ``InputDocument.text``. They are ``None``
    when the location could not be established; the pipeline never guesses.
    """

    source_document_id: str
    chapter: str | None = None
    section_path: list[str] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    match: TextMatch = "not_checked"

    @model_validator(mode="after")
    def _offsets_consistent(self) -> SourceLocation:
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("start_offset and end_offset must both be set or both be null")
        if self.start_offset is not None and self.end_offset < self.start_offset:  # type: ignore[operator]
            raise ValueError("end_offset must not be smaller than start_offset")
        return self


class Evidence(_Model):
    """A quotation from the source text that supports a claim.

    ``verified`` is True only when ``quoted_text`` was found in the extracted
    source text. Model explanations are never stored here.
    """

    source_document_id: str
    requirement_id: str | None = None
    quoted_text: str = Field(min_length=1)
    source_location: SourceLocation | None = None
    start_offset: int | None = None
    end_offset: int | None = None
    verified: bool | None = None
    verification: TextMatch = "not_checked"
    repaired: bool = Field(
        default=False,
        description="True when a verification step replaced an unlocatable model quote with source text.",
    )

    @model_validator(mode="after")
    def _verified_has_offsets(self) -> Evidence:
        if (self.start_offset is None) != (self.end_offset is None):
            raise ValueError("start_offset and end_offset must both be set or both be null")
        if self.verified and self.start_offset is None:
            raise ValueError("verified evidence must carry source offsets")
        return self


# ---------------------------------------------------------------------------
# Input documents
# ---------------------------------------------------------------------------


class ProvidedChunk(_Model):
    """A chunk produced outside csr-lab (another chunker), analysed instead of csr-lab's own segmentation."""

    chunk_id: str
    title: str | None = None
    content: str
    source_pages: list[int] = Field(default_factory=list, description="Pages the chunker said the chunk came from.")


class InputDocument(_Model):
    document_id: str
    canonical_identifier: str | None
    identifier_source: Literal["header", "filename", "manifest"]
    filename: str
    title: str | None = None
    revision: str | None = None
    effective_date: str | None = Field(default=None, description="As stated in the header; ISO dates are used to order revisions.")
    format: Literal["txt", "pdf", "json"]
    sha256: str
    char_count: int
    pages: list[PageSpan] = Field(default_factory=list)
    parse_status: Literal["ok", "partial", "unsupported", "failed"]
    warnings: list[str] = Field(default_factory=list)
    text: str | None = Field(
        default=None,
        description="Extracted text. Verbatim relative to the parser's output; offsets refer to it.",
    )
    provided_chunks: list[ProvidedChunk] | None = Field(
        default=None,
        exclude=True,
        description="Pre-chunked input (not exported; see metadata.input_fidelity for how the chunks matched the source).",
    )


# ---------------------------------------------------------------------------
# Requirements and references
# ---------------------------------------------------------------------------


class ReferenceMention(_Model):
    """One occurrence of a document reference inside one requirement."""

    mention_id: str
    document_id: str
    raw_identifier: str
    relation: ReferenceRelation = ReferenceRelation.REFERENCES
    evidence: Evidence
    status: MentionStatus
    resolution: ResolutionMethod
    verified: bool = Field(description="Passed the verification stage. False when no verification ran.")


class RequirementExtractionMetadata(_Model):
    strategy: str
    chunk_ids: list[str] = Field(default_factory=list)
    segment_ids: list[str] = Field(default_factory=list)
    verbatim_match: TextMatch = "not_checked"
    segmentation_confidence: Literal["high", "low", "not_segmented"] = "not_segmented"
    reference_analysis_status: Literal["verified", "unverified", "failed", "not_run"] = Field(
        description=(
            "'verified' + empty reference list means no explicit reference was found and the "
            "result was checked. 'unverified' means the extractor ran without verification. "
            "'failed' means reference analysis did not complete for this requirement."
        )
    )
    notes: list[str] = Field(default_factory=list)


class Requirement(_Model):
    requirement_id: str
    source_document_id: str
    chapter: str | None
    requirement_text: str = Field(min_length=1)
    source_location: SourceLocation
    referenced_document_ids: list[str] = Field(default_factory=list)
    reference_mentions: list[ReferenceMention] = Field(default_factory=list)
    department: str | None = None
    department_status: Literal["deferred", "assigned"] = "deferred"
    extraction_metadata: RequirementExtractionMetadata

    @model_validator(mode="after")
    def _consistency(self) -> Requirement:
        if self.department_status == "deferred" and self.department is not None:
            raise ValueError("department must be null while department assignment is deferred")
        if len(set(self.referenced_document_ids)) != len(self.referenced_document_ids):
            raise ValueError("referenced_document_ids must not contain duplicates")
        mention_docs = {m.document_id for m in self.reference_mentions}
        if mention_docs != set(self.referenced_document_ids):
            raise ValueError("referenced_document_ids must match the documents in reference_mentions")
        return self


class ReferencedDocument(_Model):
    document_id: str
    raw_identifiers: list[str]
    canonical_identifier: str | None
    document_name: str | None = None
    revision: str | None = None
    document_type: DocumentType = DocumentType.UNKNOWN
    content_available: bool
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Relationship graph edges and conflicts
# ---------------------------------------------------------------------------


class Relationship(_Model):
    relationship_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: RelationshipType
    evidence: list[Evidence] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="origin: document_structure | extraction | aggregation | analysis. "
        "analytical_hypothesis: true for generated (not source-stated) relationships.",
    )


class PotentialConflict(_Model):
    conflict_id: str
    related_requirement_ids: list[str] = Field(min_length=2)
    related_document_ids: list[str] = Field(default_factory=list)
    related_relationship_ids: list[str] = Field(default_factory=list)
    conflict_type: ConflictType
    description: str
    evidence: list[Evidence] = Field(default_factory=list)
    # Deliberately a single-value Literal: the system never marks a conflict resolved.
    review_status: Literal["requires_human_review"] = "requires_human_review"
    reason_for_review: str
    assessment: Literal["apparent_incompatibility", "ambiguous"] = "apparent_incompatibility"
    detection_methods: list[str] = Field(default_factory=list)
    evidence_valid: bool | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_requirements(self) -> PotentialConflict:
        if len(set(self.related_requirement_ids)) != len(self.related_requirement_ids):
            raise ValueError("related_requirement_ids must be distinct")
        return self


# ---------------------------------------------------------------------------
# Coverage: what was and was not examined
# ---------------------------------------------------------------------------


class DocumentCoverage(_Model):
    document_id: str
    status: Literal["analyzed", "partially_analyzed", "unsupported", "failed"]
    chars_total: int
    chars_processed: int
    segments_total: int
    segments_processed: int
    notes: list[str] = Field(default_factory=list)


class ReferencedDocumentCoverage(_Model):
    document_id: str
    content_available: bool
    conflict_analysis_status: Literal["examined", "not_examined_source_unavailable"]


class CoverageLimitation(_Model):
    kind: Literal[
        "unavailable_referenced_document",
        "unparsed_section",
        "unresolved_identifier",
        "truncated_input",
        "unsupported_input",
        "llm_call_failed",
        "conflict_analysis_incomplete",
        "segmentation_uncertain",
        "oversized_segment",
    ]
    message: str
    document_id: str | None = None
    requirement_id: str | None = None
    chunk_id: str | None = None


class ConflictCoverage(_Model):
    status: Literal["completed", "partial", "disabled", "failed"]
    outcome: Literal[
        "potential_conflicts_found",
        "no_conflicts_detected_in_available_text",
        "not_examined",
    ]
    candidate_pairs: int = 0
    pairs_analyzed: int = 0
    pairs_failed: int = 0
    total_possible_pairs: int = 0
    notes: list[str] = Field(default_factory=list)


class Coverage(_Model):
    documents: list[DocumentCoverage] = Field(default_factory=list)
    referenced_documents: list[ReferencedDocumentCoverage] = Field(default_factory=list)
    unresolved_identifiers: list[str] = Field(default_factory=list)
    conflict_analysis: ConflictCoverage
    limitations: list[CoverageLimitation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


class StageCost(_Model):
    """Model usage of one pipeline stage."""

    calls: int = 0
    failed_calls: int = 0
    input_chars: int = 0
    output_chars: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_tokens: int = Field(0, description="input+output tokens; reported counts when available, else characters / 4")


class OperationalStats(_Model):
    """Cost and effort measures. ``None`` means not measurable, never zero-filled."""

    total_seconds: float | None = None
    stage_seconds: dict[str, float] = Field(default_factory=dict)
    llm_calls: int = 0
    llm_failed_calls: int = 0
    json_validation_failures: int = 0
    retries: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_chars: int = 0
    output_chars: int = 0
    estimated_cost_usd: float | None = None
    graph_informed_calls: int = 0
    # Per-stage and per-unit cost (for comparing pipelines of different shapes).
    stage_costs: dict[str, StageCost] = Field(default_factory=dict)
    token_source: Literal["reported", "estimated_from_chars", "none"] = "none"
    estimated_tokens: int = Field(0, description="Total input+output tokens (reported, or characters / 4 when the server reports none).")
    pages: float | None = Field(default=None, description="PDF pages; for text input, characters / 3000 (see pages_source).")
    pages_source: Literal["pdf", "estimated", "mixed", "none"] = "none"
    requirements: int = Field(0, description="Requirement units in the result (the denominator of the per-requirement figures).")
    calls_per_page: float | None = None
    tokens_per_page: float | None = None
    calls_per_requirement: float | None = None
    tokens_per_requirement: float | None = None


class LLMCallSummary(_Model):
    call_id: str
    stage: str
    task: str
    prompt_version: str
    success: bool
    attempts: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    input_chars: int
    output_chars: int
    latency_seconds: float
    graph_informed: bool = False
    errors: list[str] = Field(default_factory=list)


class VerificationEvent(_Model):
    requirement_id: str
    raw_identifier: str
    action: Literal[
        "kept",
        "rejected",
        "added",
        "merged_duplicate",
        "evidence_repaired",
        "flagged_near_duplicate",
        "candidate_rejected",
        "candidate_unresolved",
        "resolved_by_alias",
    ]
    reason: str
    attempt: int = 0
    mention_id: str | None = None


class ReferenceVerificationReport(_Model):
    """Initial predictions are kept next to the verified ones for analysis."""

    enabled: bool
    attempts_used: int = 0
    initial_mentions: list[ReferenceMention] = Field(default_factory=list)
    events: list[VerificationEvent] = Field(default_factory=list)
    counts: dict[str, int] = Field(default_factory=dict)


class CandidatePairRecord(_Model):
    requirement_ids: list[str] = Field(min_length=2, max_length=2)
    selectors: list[str]
    graph_derived: bool = False
    analyzed: bool = False


class GraphDecision(_Model):
    """A processing decision that used the relationship graph (graph_assisted only)."""

    decision_type: Literal[
        "alias_resolved",
        "audit_candidate",
        "conflict_candidate_added",
        "context_retrieved",
        "rule_flagged_conflict",
    ]
    detail: str
    node_ids: list[str] = Field(default_factory=list)


class ReferenceMergeDecision(_Model):
    """One grouping decision of the reference-merging stage."""

    method: Literal["llm", "hybrid"]
    members: list[str] = Field(description="Reference labels (as extracted) the decision covers.")
    canonical: str = Field(description="Label the members were renamed to.")
    applied: bool
    reason: str = ""
    mention_ids: list[str] = Field(default_factory=list)


class AnalysisMetadata(_Model):
    strategy: str
    strategy_description: str = ""
    strategy_options: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    package_version: str
    provider: dict[str, Any] = Field(default_factory=dict)
    provider_is_mock: bool
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    run_seed: int | None = None
    case_id: str | None = None
    operational: OperationalStats = Field(default_factory=OperationalStats)
    reference_verification: ReferenceVerificationReport | None = None
    conflict_candidates: list[CandidatePairRecord] = Field(default_factory=list)
    graph_decisions: list[GraphDecision] = Field(default_factory=list)
    reference_merges: list[ReferenceMergeDecision] = Field(default_factory=list)
    input_fidelity: FidelityReport | None = Field(
        default=None,
        description="How the analysed text units (provided or model-made chunks) matched the source text.",
    )
    saved_graph: dict[str, Any] | None = Field(
        default=None,
        description="Set when a graph-assisted run started from a saved (persistent) graph: its path, fingerprint and size.",
    )
    llm_calls: list[LLMCallSummary] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


class AnalysisResult(_Model):
    schema_version: str = SCHEMA_VERSION
    analysis_id: str
    status: Literal["completed", "completed_with_errors", "failed"]
    input_documents: list[InputDocument]
    requirements: list[Requirement] = Field(default_factory=list)
    referenced_documents: list[ReferencedDocument] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    potential_conflicts: list[PotentialConflict] = Field(default_factory=list)
    coverage: Coverage
    metadata: AnalysisMetadata

    @model_validator(mode="after")
    def _referential_integrity(self) -> AnalysisResult:
        req_ids = [r.requirement_id for r in self.requirements]
        if len(set(req_ids)) != len(req_ids):
            raise ValueError("requirement_id values must be unique")
        doc_ids = [d.document_id for d in self.referenced_documents]
        if len(set(doc_ids)) != len(doc_ids):
            raise ValueError("referenced document_id values must be unique")
        known_docs = set(doc_ids)
        for req in self.requirements:
            missing = set(req.referenced_document_ids) - known_docs
            if missing:
                raise ValueError(f"{req.requirement_id} references unknown documents: {sorted(missing)}")
        known_reqs = set(req_ids)
        for conflict in self.potential_conflicts:
            missing = set(conflict.related_requirement_ids) - known_reqs
            if missing:
                raise ValueError(f"{conflict.conflict_id} names unknown requirements: {sorted(missing)}")
        node_ids = known_reqs | known_docs | {d.document_id for d in self.input_documents}
        # Document-level nodes of provided revisions ('DOC-X' for 'DOC-X-REV-D').
        node_ids |= {document_family_id(d.canonical_identifier) for d in self.input_documents if d.canonical_identifier}
        for rel in self.relationships:
            if rel.source_node_id not in node_ids or rel.target_node_id not in node_ids:
                raise ValueError(f"{rel.relationship_id} has an endpoint that is not a known node")
        return self


# ---------------------------------------------------------------------------
# Ground truth (evaluator-only)
# ---------------------------------------------------------------------------


class GoldReference(_Model):
    canonical_identifier: str
    revision: str | None = None
    mention_text: str = Field(description="Exact substring of the requirement text that names the document.")
    relation: ReferenceRelation = ReferenceRelation.REFERENCES
    resolution: Literal["identifier", "name_alias"] = "identifier"


class GoldRequirement(_Model):
    gold_id: str
    source_document_id: str
    chapter: str
    requirement_text: str
    start_offset: int
    end_offset: int
    references: list[GoldReference] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class GoldEvidence(_Model):
    requirement_id: str
    quoted_text: str


class GoldConflict(_Model):
    gold_conflict_id: str
    requirement_ids: list[str] = Field(min_length=2, max_length=2)
    conflict_type: ConflictType
    category: Literal["conflict", "ambiguous"] = "conflict"
    cross_document: bool = False
    evidence: list[GoldEvidence] = Field(default_factory=list)
    note: str = ""


class GoldPair(_Model):
    requirement_ids: list[str] = Field(min_length=2, max_length=2)
    note: str = ""


class GoldInputDocument(_Model):
    document_id: str
    canonical_identifier: str
    filename: str
    title: str
    revision: str | None = None


class GoldReferencedDocument(_Model):
    canonical_identifier: str
    revision: str | None = None
    content_available: bool
    document_type: DocumentType = DocumentType.UNKNOWN


class CaseGold(_Model):
    schema_version: str = GOLD_SCHEMA_VERSION
    dataset_name: str
    dataset_version: str
    case_id: str
    split: Literal["dev", "heldout"]
    description: str
    documents: list[GoldInputDocument]
    requirements: list[GoldRequirement]
    referenced_documents: list[GoldReferencedDocument]
    conflicts: list[GoldConflict] = Field(default_factory=list)
    non_conflicting_pairs: list[GoldPair] = Field(
        default_factory=list,
        description="Hard negatives: related requirements that should NOT be flagged.",
    )
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _ids_known(self) -> CaseGold:
        ids = {r.gold_id for r in self.requirements}
        for c in self.conflicts:
            if not set(c.requirement_ids) <= ids:
                raise ValueError(f"gold conflict {c.gold_conflict_id} names unknown requirements")
        for p in self.non_conflicting_pairs:
            if not set(p.requirement_ids) <= ids:
                raise ValueError("non_conflicting_pairs names unknown requirements")
        return self
