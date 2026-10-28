"""src/csr_lab/pipelines/assembly.py

Convert working drafts into the exported ``AnalysisResult`` (same code for every strategy).

This step also builds the exported relationship graph from the *final*
predictions. That graph is output only: nothing here feeds back into
predictions, so building it for every strategy does not give non-graph
strategies access to graph information.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version

from csr_lab.drafts import ConflictDraft, MentionDraft, RequirementDraft
from csr_lab.extraction.fidelity import FidelityReport, source_gaps, summarize
from csr_lab.extraction.normalization import reference_key
from csr_lab.ids import conflict_id, document_family_id, stable_hash
from csr_lab.ingestion.chunking import ChunkPlan
from csr_lab.ingestion.segmentation import Segment
from csr_lab.prompts import prompt_versions
from csr_lab.relationships.builder import RelationshipGraph, build_graph
from csr_lab.relationships.validation import validate_graph
from csr_lab.schemas import (
    AnalysisMetadata,
    AnalysisResult,
    CandidatePairRecord,
    ConflictCoverage,
    Coverage,
    CoverageLimitation,
    DocumentCoverage,
    DocumentType,
    InputDocument,
    PotentialConflict,
    ReferencedDocument,
    ReferencedDocumentCoverage,
    ReferenceMention,
    ReferenceVerificationReport,
    Requirement,
    RequirementExtractionMetadata,
    SourceLocation,
)


def package_version() -> str:
    try:
        return version("csr-relationship-lab")
    except PackageNotFoundError:
        return "0.0.0+local"


def to_reference_mention(m: MentionDraft) -> ReferenceMention:
    return ReferenceMention(
        mention_id=m.mention_id,
        document_id=m.document_id,
        raw_identifier=m.raw_identifier,
        relation=m.relation,
        evidence=m.evidence,
        status=m.status,
        resolution=m.resolution,
        verified=m.verified,
    )


def requirement_records(
    requirements: list[RequirementDraft], mentions: list[MentionDraft], strategy: str
) -> list[Requirement]:
    by_req: dict[str, list[MentionDraft]] = {}
    for m in mentions:
        if m.active:
            by_req.setdefault(m.requirement_id, []).append(m)
    out = []
    for r in requirements:
        req_mentions = by_req.get(r.requirement_id, [])
        out.append(
            Requirement(
                requirement_id=r.requirement_id,
                source_document_id=r.document_id,
                chapter=r.chapter,
                requirement_text=r.text,
                source_location=SourceLocation(
                    source_document_id=r.document_id,
                    chapter=r.chapter,
                    section_path=r.section_path,
                    page_start=r.page_start,
                    page_end=r.page_end,
                    start_offset=r.start_offset,
                    end_offset=r.end_offset,
                    match=r.match,
                ),
                referenced_document_ids=list(dict.fromkeys(m.document_id for m in req_mentions)),
                reference_mentions=[to_reference_mention(m) for m in req_mentions],
                extraction_metadata=RequirementExtractionMetadata(
                    strategy=strategy,
                    chunk_ids=r.chunk_ids,
                    segment_ids=r.segment_ids,
                    verbatim_match=r.match,
                    segmentation_confidence=r.segmentation_confidence,
                    reference_analysis_status=r.reference_status,
                    notes=r.notes,
                ),
            )
        )
    return out


def referenced_document_records(mentions: list[MentionDraft], documents: list[InputDocument]) -> list[ReferencedDocument]:
    """One record per referenced document node.

    Content is available when the exact node was provided, or when an unrevisioned
    reference ("CVM-PKG-210") names a document of which one revision was provided.
    A reference to a *different* revision than the one provided is not available.
    """
    inputs = {d.document_id: d for d in documents}
    by_family: dict[str, list[InputDocument]] = {}
    for d in documents:
        if d.canonical_identifier:
            by_family.setdefault(document_family_id(d.canonical_identifier), []).append(d)
    for family, docs in by_family.items():
        if family not in inputs and len(docs) == 1:
            inputs[family] = docs[0]
    available = {doc_id for doc_id, d in inputs.items() if d.parse_status in ("ok", "partial")}
    grouped: dict[str, list[MentionDraft]] = {}
    for m in mentions:
        if m.active:
            grouped.setdefault(m.document_id, []).append(m)
    out = []
    for doc_id, ms in grouped.items():
        first = ms[0]
        types = Counter(m.document_type for m in ms if m.document_type != DocumentType.UNKNOWN)
        names = list(dict.fromkeys(m.name for m in ms if m.name))
        input_doc = inputs.get(doc_id)
        out.append(
            ReferencedDocument(
                document_id=doc_id,
                raw_identifiers=list(dict.fromkeys(m.raw_identifier for m in ms)),
                canonical_identifier=first.canonical,
                document_name=names[0] if names else (input_doc.title if input_doc else None),
                revision=first.revision,
                document_type=types.most_common(1)[0][0] if types else DocumentType.UNKNOWN,
                content_available=doc_id in available,
                metadata={
                    "key": reference_key(first.canonical, first.revision, first.name),
                    "names": names,
                    "is_input_document": input_doc is not None,
                    "resolution": sorted({m.resolution.value for m in ms}),
                    "referencing_requirement_ids": sorted({m.requirement_id for m in ms}),
                },
            )
        )
    return sorted(out, key=lambda d: d.document_id)


def conflict_records(conflicts: list[ConflictDraft], relationship_ids: set[str] | None = None) -> list[PotentialConflict]:
    out = []
    for c in conflicts:
        rel_ids = [r for r in c.related_relationship_ids if relationship_ids is None or r in relationship_ids]
        out.append(
            PotentialConflict(
                conflict_id=conflict_id(c.requirement_ids),
                related_requirement_ids=sorted(c.requirement_ids),
                related_document_ids=c.related_document_ids,
                related_relationship_ids=rel_ids,
                conflict_type=c.conflict_type,
                description=c.description,
                evidence=c.evidence,
                reason_for_review=c.reason_for_review or "Requires human review.",
                assessment=c.assessment,
                detection_methods=c.detection_methods,
                evidence_valid=c.evidence_valid,
                metadata={**c.metadata, "graph_informed": c.graph_informed},
            )
        )
    return out


def provisional_graph(
    documents: list[InputDocument], requirements: list[RequirementDraft], mentions: list[MentionDraft]
) -> RelationshipGraph:
    """Graph of the current predictions, used mid-pipeline by graph-assisted strategies."""
    reqs = requirement_records(requirements, mentions, "provisional")
    return build_graph(documents, reqs, referenced_document_records(mentions, documents))


def _coverage(
    documents: list[InputDocument],
    segments: list[Segment],
    plan: ChunkPlan | None,
    referenced: list[ReferencedDocument],
    conflict_stats: dict,
    conflicts_enabled: bool,
    n_conflicts: int,
    limitations: list[CoverageLimitation],
    fatal: str | None,
) -> Coverage:
    processed = set(plan.processed_segment_ids) if plan else set()
    doc_cov = []
    for d in documents:
        segs = [s for s in segments if s.document_id == d.document_id]
        done = [s for s in segs if s.segment_id in processed]
        chars_done = sum(s.end_offset - s.line_start_offset for s in done)
        if d.parse_status in ("unsupported", "failed"):
            status = d.parse_status
        elif len(done) < len(segs) or d.parse_status == "partial" or (fatal and not done):
            status = "partially_analyzed" if done else "failed"
        else:
            status = "analyzed"
        doc_cov.append(
            DocumentCoverage(
                document_id=d.document_id,
                status=status,  # type: ignore[arg-type]
                chars_total=d.char_count,
                chars_processed=chars_done,
                segments_total=len(segs),
                segments_processed=len(done),
                notes=d.warnings,
            )
        )

    ref_cov = [
        ReferencedDocumentCoverage(
            document_id=r.document_id,
            content_available=r.content_available,
            conflict_analysis_status="examined" if r.content_available else "not_examined_source_unavailable",
        )
        for r in referenced
    ]
    limits = list(limitations)
    for r in referenced:
        if not r.content_available:
            limits.append(
                CoverageLimitation(
                    kind="unavailable_referenced_document",
                    document_id=r.document_id,
                    message=(
                        f"{r.canonical_identifier or r.document_name} is referenced but its content was not "
                        "provided; it was not examined for conflicts"
                    ),
                )
            )
        if r.canonical_identifier is None:
            limits.append(
                CoverageLimitation(
                    kind="unresolved_identifier",
                    document_id=r.document_id,
                    message=f"document named '{r.document_name}' has no identifier that could be resolved",
                )
            )

    if not conflicts_enabled:
        cc = ConflictCoverage(status="disabled", outcome="not_examined")
    else:
        total = conflict_stats.get("total_pairs", 0)
        analysed = conflict_stats.get("pairs_analyzed", 0)
        failed = conflict_stats.get("pairs_failed", 0)
        if fatal or (failed and not analysed):
            status = "failed"
        elif failed or analysed < total:
            status = "partial"
        else:
            status = "completed"
        if n_conflicts:
            outcome = "potential_conflicts_found"
        elif status in ("completed", "partial"):
            outcome = "no_conflicts_detected_in_available_text"
        else:
            outcome = "not_examined"
        cc = ConflictCoverage(
            status=status,  # type: ignore[arg-type]
            outcome=outcome,  # type: ignore[arg-type]
            candidate_pairs=conflict_stats.get("candidate_pairs", 0),
            pairs_analyzed=analysed,
            pairs_failed=failed,
            total_possible_pairs=total,
            notes=conflict_stats.get("notes", []),
        )
    unresolved = sorted({i for r in referenced if r.canonical_identifier is None for i in r.raw_identifiers})
    return Coverage(
        documents=doc_cov,
        referenced_documents=ref_cov,
        unresolved_identifiers=unresolved,
        conflict_analysis=cc,
        limitations=limits,
    )


CHARS_PER_PAGE = 3000  # a typical page of a text-heavy specification; used only for text input


def add_unit_costs(stats, documents: list[InputDocument], n_requirements: int) -> None:
    """Per-page and per-requirement cost, so pipelines of different shapes compare fairly."""
    usable = [d for d in documents if d.parse_status in ("ok", "partial")]
    pdf = [d for d in usable if d.pages]
    text = [d for d in usable if not d.pages]
    pages = sum(len(d.pages) for d in pdf) + sum(d.char_count for d in text) / CHARS_PER_PAGE
    stats.pages = round(pages, 2) if usable else None
    stats.pages_source = "none" if not usable else ("pdf" if not text else ("estimated" if not pdf else "mixed"))
    stats.requirements = n_requirements
    if stats.pages:
        stats.calls_per_page = round(stats.llm_calls / stats.pages, 3)
        stats.tokens_per_page = round(stats.estimated_tokens / stats.pages, 1)
    if n_requirements:
        stats.calls_per_requirement = round(stats.llm_calls / n_requirements, 3)
        stats.tokens_per_requirement = round(stats.estimated_tokens / n_requirements, 1)


def fidelity_report(state: dict) -> FidelityReport | None:
    """How the chunk units (provided or model-made) matched the source text."""
    checks = state.get("chunk_checks", [])
    if not checks:
        return None
    docs = {d.document_id: d for d in state["documents"]}
    origins = sorted({s.origin for s in state.get("segments", []) if s.origin != "deterministic"})
    gaps = []
    covered_words = total_words = 0
    for doc_id in dict.fromkeys(c.document_id for c in checks):
        text = (docs[doc_id].text or "") if doc_id in docs else ""
        located = [(c.start_offset, c.end_offset) for c in checks if c.document_id == doc_id and c.start_offset is not None]
        doc_gaps, coverage = source_gaps(text, located, document_id=doc_id)  # type: ignore[arg-type]
        gaps += doc_gaps
        n = len(text.split())
        total_words += n
        covered_words += round((coverage or 0) * n)
    coverage = round(covered_words / total_words, 4) if total_words else None
    return FidelityReport(origin="+".join(origins) or "chunks", fragments=checks, gaps=gaps, summary=summarize(checks, gaps, coverage))


def assemble(state: dict, ctx) -> AnalysisResult:
    """Build and validate the final result from workflow state."""
    from csr_lab.extraction.verification import verification_counts

    options = ctx.options
    documents: list[InputDocument] = state["documents"]
    requirements: list[RequirementDraft] = state.get("requirements", [])
    mentions: list[MentionDraft] = state.get("mentions", [])
    fatal = state.get("fatal")

    reqs = requirement_records(requirements, mentions, ctx.strategy.name)
    referenced = referenced_document_records(mentions, documents)
    conflicts_preview = conflict_records(state.get("conflicts", []))
    graph = build_graph(documents, reqs, referenced, conflicts_preview)
    edge_ids = {e.relationship_id for e in graph.edges}
    conflicts = conflict_records(state.get("conflicts", []), edge_ids)
    graph_report = validate_graph(graph)

    errors = [f"[{e.stage}] {e.message}" for e in state.get("errors", [])]
    if fatal:
        errors.insert(0, f"[fatal] {fatal}")
    if not graph_report.valid:
        errors.append(f"[graph] structural validation failed: {graph_report.model_dump(exclude_defaults=True)}")

    initial = state.get("initial_mentions", [])
    events = state.get("verification_events", [])
    verification = None
    if options.workflow == "staged":
        verification = ReferenceVerificationReport(
            enabled=options.verification.enabled,
            attempts_used=state.get("verification_attempts", 0),
            initial_mentions=[to_reference_mention(m) for m in initial],
            events=events,
            counts=verification_counts(initial, mentions, events) if options.verification.enabled else {},
        )

    stats = ctx.recorder.stats(
        ctx.config.llm.pricing.input_usd_per_million_tokens, ctx.config.llm.pricing.output_usd_per_million_tokens
    )
    stats.stage_seconds = state.get("stage_seconds", {})
    add_unit_costs(stats, documents, len(reqs))

    include_text = ctx.config.processing.include_source_text
    exported_docs = [d if include_text else d.model_copy(update={"text": None}) for d in documents]
    options_dump = options.model_dump(mode="json")
    analysis_id = "ANL-" + stable_hash(
        ctx.strategy.name,
        ctx.case_id,
        ctx.run_seed,
        json.dumps(options_dump, sort_keys=True),
        *[d.sha256 for d in documents],
        *([ctx.saved_graph_info["fingerprint"]] if ctx.saved_graph_info else []),
        length=16,
    )
    metadata = AnalysisMetadata(
        strategy=ctx.strategy.name,
        strategy_description=ctx.strategy.description,
        strategy_options=options_dump,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        package_version=package_version(),
        provider=ctx.provider.describe(),
        provider_is_mock=ctx.provider.is_mock,
        prompt_versions=prompt_versions(),
        run_seed=ctx.run_seed,
        case_id=ctx.case_id,
        operational=stats,
        reference_verification=verification,
        conflict_candidates=[
            CandidatePairRecord(
                requirement_ids=list(c.requirement_ids),
                selectors=c.selectors,
                graph_derived=c.graph_derived,
                analyzed=c.analyzed,
            )
            for c in state.get("conflict_candidates", [])
        ],
        graph_decisions=state.get("graph_decisions", []),
        reference_merges=state.get("merge_decisions", []),
        input_fidelity=fidelity_report(state),
        saved_graph=ctx.saved_graph_info,
        llm_calls=list(ctx.recorder.calls),
        warnings=[*ctx.recorder.warnings, *state.get("warnings", [])],
        errors=errors,
    )
    status = "failed" if fatal else ("completed_with_errors" if errors else "completed")
    return AnalysisResult(
        analysis_id=analysis_id,
        status=status,  # type: ignore[arg-type]
        input_documents=exported_docs,
        requirements=reqs,
        referenced_documents=referenced,
        relationships=graph.edges,
        potential_conflicts=conflicts,
        coverage=_coverage(
            documents,
            state.get("segments", []),
            state.get("chunk_plan"),
            referenced,
            state.get("conflict_stats", {}),
            options.conflicts.enabled,
            len(conflicts),
            state.get("limitations", []),
            fatal,
        ),
        metadata=metadata,
    )
