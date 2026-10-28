"""src/csr_lab/pipelines/state.py

LangGraph workflow state.

Documents are stored once (``documents``); every other field refers to them
by ID and offset. Fields annotated with a reducer accumulate across nodes.
Non-serialisable objects (the LLM client, recorder) are in the runtime
context, not here.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from csr_lab.drafts import CandidatePair, ConflictDraft, MentionDraft, PipelineError, RequirementDraft
from csr_lab.extraction.verification import AuditCandidate
from csr_lab.ingestion.chunking import ChunkPlan
from csr_lab.ingestion.segmentation import Segment
from csr_lab.relationships.builder import RelationshipGraph
from csr_lab.extraction.fidelity import FragmentCheck
from csr_lab.schemas import AnalysisResult, CoverageLimitation, GraphDecision, InputDocument, ReferenceMergeDecision, VerificationEvent


def sum_seconds(a: dict[str, float] | None, b: dict[str, float] | None) -> dict[str, float]:
    out = dict(a or {})
    for key, value in (b or {}).items():
        out[key] = round(out.get(key, 0.0) + value, 6)
    return out


class PipelineState(TypedDict, total=False):
    case_id: str | None
    documents: list[InputDocument]
    segments: list[Segment]
    chunk_plan: ChunkPlan
    chunk_checks: Annotated[list[FragmentCheck], operator.add]  # how chunk units matched the source
    requirements: list[RequirementDraft]
    initial_mentions: list[MentionDraft]  # snapshot taken right after first extraction
    mentions: list[MentionDraft]  # working set (includes inactive = rejected/merged)
    mention_seq: int
    pending_candidates: list[AuditCandidate]
    examined_candidates: list[str]
    verification_attempts: int
    verification_events: Annotated[list[VerificationEvent], operator.add]
    graph: RelationshipGraph | None
    conflict_candidates: list[CandidatePair]
    total_pairs: int
    conflicts: list[ConflictDraft]
    conflict_stats: dict[str, Any]
    graph_decisions: Annotated[list[GraphDecision], operator.add]
    merge_decisions: Annotated[list[ReferenceMergeDecision], operator.add]
    limitations: Annotated[list[CoverageLimitation], operator.add]
    errors: Annotated[list[PipelineError], operator.add]
    warnings: Annotated[list[str], operator.add]
    stage_seconds: Annotated[dict[str, float], sum_seconds]
    fatal: str | None
    result: AnalysisResult | None


def initial_state(documents: list[InputDocument], case_id: str | None) -> PipelineState:
    return PipelineState(
        case_id=case_id,
        documents=documents,
        requirements=[],
        initial_mentions=[],
        mentions=[],
        mention_seq=0,
        pending_candidates=[],
        examined_candidates=[],
        verification_attempts=0,
        verification_events=[],
        graph=None,
        conflict_candidates=[],
        total_pairs=0,
        conflicts=[],
        conflict_stats={},
        graph_decisions=[],
        merge_decisions=[],
        chunk_checks=[],
        limitations=[],
        errors=[],
        warnings=[],
        stage_seconds={},
        fatal=None,
        result=None,
    )
