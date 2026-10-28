"""src/csr_lab/pipelines/nodes.py

LangGraph node functions shared by the built-in workflows.

Each node receives the workflow state and the runtime context
(``PipelineContext``) and returns a partial state update. Deterministic
stages are plain Python; only the nodes marked "LLM" call the model.
Recoverable failures are recorded in ``errors``/``limitations`` and the
workflow continues; only unusable input sets ``fatal``.
"""

from __future__ import annotations

import functools
import time
from typing import Any, Callable

from langgraph.runtime import Runtime

from csr_lab.conflicts.candidate_selection import select_candidates
from csr_lab.conflicts.detector import (
    REVIEW_REASON,
    analyze_pairs,
    build_conflict_evidence,
    graph_context_for_pair,
    graph_rule_conflicts,
    merge_conflicts,
    validate_conflict_evidence,
)
from csr_lab.drafts import CandidatePair, ConflictDraft, MentionDraft, PipelineError, ordered_pair
from csr_lab.extraction.fidelity import WordIndex
from csr_lab.extraction.graph_audit import known_document_candidates, resolve_aliases
from csr_lab.extraction.grounding import TextIndex
from csr_lab.extraction.merging import hybrid_merge, llm_merge
from csr_lab.extraction.references import MentionCounter, extract_references, mentions_from_model
from csr_lab.extraction.requirements import ReportedRequirement, ground_requirements
from csr_lab.extraction.verification import targeted_reextraction, verify_pass
from csr_lab.ingestion.chunking import chunk_raw_text, plan_chunks
from csr_lab.ingestion.segmentation import segment_document
from csr_lab.ingestion.units import ChunkInput, model_chunks, segments_from_chunks
from csr_lab.pipelines.assembly import assemble, provisional_graph
from csr_lab.pipelines.base import PipelineContext
from csr_lab.pipelines.state import PipelineState
from csr_lab.prompts import baseline as baseline_prompt
from csr_lab.prompts import requirements as requirement_prompt
from csr_lab.relationships.store import with_saved_documents
from csr_lab.schemas import ConflictType, CoverageLimitation, GraphDecision, MentionStatus, VerificationEvent

Node = Callable[[PipelineState, Runtime[PipelineContext]], dict[str, Any]]


def timed(name: str) -> Callable[[Node], Node]:
    """Record wall-clock time per stage in ``stage_seconds``."""

    def decorator(fn: Node) -> Node:
        @functools.wraps(fn)
        def wrapper(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
            start = time.perf_counter()
            update = fn(state, runtime) or {}
            update["stage_seconds"] = {name: time.perf_counter() - start}
            return update

        return wrapper

    return decorator


def _copy_mentions(state: PipelineState) -> list[MentionDraft]:
    return [m.model_copy(deep=True) for m in state.get("mentions", [])]


# ---------------------------------------------------------------------------
# Shared preparation and finalisation (deterministic)
# ---------------------------------------------------------------------------


def _uncertain_segment_limitations(segments) -> list[CoverageLimitation]:
    return [
        CoverageLimitation(kind="segmentation_uncertain", document_id=seg.document_id, message=f"{seg.segment_id}: {'; '.join(seg.notes)}")
        for seg in segments
        if seg.is_candidate and any(n.startswith("numbering jumps") or n.startswith("long block") for n in seg.notes)
    ]


@timed("prepare_input")
def prepare_input(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    """Make the analysable units: csr-lab segments, or the chunks of a chunk file.

    Documents without provided chunks are left to ``llm_chunk`` when the strategy's
    chunking method is ``llm``.
    """
    ctx = runtime.context
    documents = state["documents"]
    limitations: list[CoverageLimitation] = []
    for d in documents:
        if d.parse_status in ("unsupported", "failed"):
            limitations.append(
                CoverageLimitation(kind="unsupported_input", document_id=d.document_id, message="; ".join(d.warnings) or d.parse_status)
            )
    usable = [d for d in documents if d.parse_status in ("ok", "partial")]
    if not usable:
        return {"fatal": "no input document contains analysable text", "limitations": limitations}

    chunking = ctx.options.chunking
    segments = []
    checks = []
    pending_llm = False
    for d in usable:
        if d.provided_chunks:
            inputs = [ChunkInput(c.chunk_id, c.content, c.title, c.source_pages) for c in d.provided_chunks]
            segs, chk = segments_from_chunks(d, inputs, origin="provided", settings=chunking)
            segments += segs
            checks += chk
        elif chunking.method == "llm" and ctx.options.workflow == "staged":
            pending_llm = True
        else:
            segments += segment_document(d)
    plan = plan_chunks(usable, segments, ctx.config.processing)
    limitations += plan.limitations + _uncertain_segment_limitations(segments)
    update: dict[str, Any] = {"segments": segments, "chunk_plan": plan, "limitations": limitations, "chunk_checks": checks}
    if not pending_llm and not any(s.is_candidate for s in segments):
        update["fatal"] = "no requirement candidates were found in the input"
    return update


@timed("llm_chunk")
def llm_chunk(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    """LLM: split each document into word-count batches and ask the model for semantic chunks."""
    ctx = runtime.context
    settings = ctx.options.chunking
    limit = ctx.config.processing.max_document_chars
    usable = [d for d in state["documents"] if d.parse_status in ("ok", "partial")]
    targets = [d for d in usable if not d.provided_chunks]
    made = model_chunks(ctx.llm, targets, settings, limit)
    by_doc = made.chunks
    limitations = [
        CoverageLimitation(kind="truncated_input", document_id=doc_id, message=f"document exceeds max_document_chars={limit}; text after offset {limit} was not chunked")
        for doc_id in made.truncated
    ]
    errors: list[PipelineError] = []
    for doc_id, b, err in made.failed_batches:
        errors.append(PipelineError(stage="llm_chunk", message=f"{doc_id} batch {b}: chunking failed; batch kept as one chunk: {err}"))
        limitations.append(CoverageLimitation(kind="llm_call_failed", document_id=doc_id, message=f"semantic chunking failed for batch {b}; the whole batch is one unit"))

    segments = list(state.get("segments", []))
    checks = []
    for d in targets:
        text = d.text or ""
        segs, chk = segments_from_chunks(d, by_doc.get(d.document_id, []), origin="llm", settings=settings, indexes=(TextIndex(text), WordIndex(text)))
        segments += segs
        checks += chk
    plan = plan_chunks(usable, segments, ctx.config.processing)
    known = {lim.message for lim in state.get("limitations", [])}
    limitations += [lim for lim in plan.limitations if lim.message not in known]
    update: dict[str, Any] = {"segments": segments, "chunk_plan": plan, "chunk_checks": checks, "limitations": limitations, "errors": errors}
    if not any(s.is_candidate for s in segments):
        update["fatal"] = "semantic chunking produced no analysable text"
    return update


@timed("finalize")
def finalize(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    return {"result": assemble(dict(state), runtime.context)}


def route_after_prepare(state: PipelineState) -> str:
    return "fatal" if state.get("fatal") else "ok"


# ---------------------------------------------------------------------------
# Single-pass baseline (LLM)
# ---------------------------------------------------------------------------


@timed("single_pass_extract")
def single_pass_extract(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    ctx = runtime.context
    plan = state["chunk_plan"]
    seg_by_id = {s.segment_id: s for s in state["segments"]}
    docs = {d.document_id: d for d in state["documents"]}

    def run(chunk):
        payload = {"documents": chunk_raw_text(chunk, seg_by_id, docs)}
        system, user = baseline_prompt.SPEC.render(payload)
        return chunk, ctx.llm.call(
            task="single_pass_extraction",
            stage="single_pass_extract",
            system=system,
            user=user,
            response_model=baseline_prompt.SinglePassResponse,
            prompt_version=baseline_prompt.SPEC.fingerprint,
        )

    reported: list[ReportedRequirement] = []
    refs_by_key: dict[str, list] = {}
    raw_conflicts: list[tuple[str, Any]] = []
    errors: list[PipelineError] = []
    limitations: list[CoverageLimitation] = []
    failed_chunks: set[str] = set()
    for chunk, result in ctx.llm.map(run, plan.chunks):
        if not result.ok or result.value is None:
            failed_chunks.add(chunk.chunk_id)
            errors.append(PipelineError(stage="single_pass_extract", chunk_id=chunk.chunk_id, message="; ".join(result.errors)[:300]))
            limitations.append(CoverageLimitation(kind="llm_call_failed", chunk_id=chunk.chunk_id, message="single-pass extraction failed for this chunk"))
            continue
        for req in result.value.requirements:
            reported.append(
                ReportedRequirement(
                    document_id=req.source_document_id,
                    chapter=req.chapter,
                    text=req.requirement_text,
                    chunk_id=chunk.chunk_id,
                    local_key=req.key,
                )
            )
            refs_by_key.setdefault(f"{chunk.chunk_id}:{req.key}", req.references)
        raw_conflicts += [(chunk.chunk_id, c) for c in result.value.potential_conflicts]

    chunk_docs = {c.chunk_id: c.document_ids for c in plan.chunks}
    drafts, key_map, warnings = ground_requirements(reported, sources=ctx.sources, segments=state["segments"], chunk_documents=chunk_docs)
    by_id = {d.requirement_id: d for d in drafts}

    counter = MentionCounter(state.get("mention_seq", 0))
    mentions: list[MentionDraft] = []
    done: set[str] = set()
    for local, req_id in key_map.items():
        if req_id in done:
            continue  # the same requirement seen in an overlapping chunk: keep the first report
        done.add(req_id)
        mentions += mentions_from_model(refs_by_key.get(local, []), by_id[req_id], sources=ctx.sources, counter=counter)
    for d in drafts:
        d.reference_status = "failed" if set(d.chunk_ids) <= failed_chunks else "unverified"

    conflicts: list[ConflictDraft] = []
    for chunk_id, c in raw_conflicts:
        ids = [key_map.get(f"{chunk_id}:{k}") for k in c.requirement_keys]
        if None in ids or ids[0] == ids[1]:
            errors.append(PipelineError(stage="single_pass_extract", chunk_id=chunk_id, message=f"conflict names unknown requirement keys {c.requirement_keys}"))
            continue
        quotes = [(key_map.get(f"{chunk_id}:{e.requirement_key}"), e.quoted_text) for e in c.evidence]
        evidence = build_conflict_evidence([(r, q) for r, q in quotes if r in ids], by_id, ctx.sources)
        valid = bool(evidence) and all(e.verified for e in evidence) and set(ids) <= {e.requirement_id for e in evidence}
        conflicts.append(
            ConflictDraft(
                requirement_ids=list(ids),  # type: ignore[arg-type]
                conflict_type=c.conflict_type,
                description=c.description,
                evidence=evidence,
                assessment="ambiguous" if c.ambiguous else "apparent_incompatibility",
                detection_methods=["llm_single_pass"],
                evidence_valid=valid,
                reason_for_review=REVIEW_REASON,
                related_document_ids=sorted({by_id[i].document_id for i in ids}),  # type: ignore[index]
            )
        )

    # Implicit candidates: every pair of requirements that shared a prompt.
    candidates: dict[tuple[str, str], CandidatePair] = {}
    for chunk in plan.chunks:
        ids = sorted({d.requirement_id for d in drafts if chunk.chunk_id in d.chunk_ids})
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                pair = ordered_pair(a, b)
                candidates.setdefault(pair, CandidatePair(requirement_ids=pair, selectors=[f"single_pass_context:{chunk.chunk_id}"], analyzed=True))
    n = len(drafts)
    return {
        "requirements": drafts,
        "mentions": mentions,
        "initial_mentions": [m.model_copy(deep=True) for m in mentions],
        "mention_seq": counter.value,
        "conflicts": merge_conflicts(conflicts),
        "conflict_candidates": list(candidates.values()),
        "conflict_stats": {
            "candidate_pairs": len(candidates),
            "pairs_analyzed": len(candidates),
            "pairs_failed": 0,
            "total_pairs": n * (n - 1) // 2,
            "notes": [] if len(candidates) == n * (n - 1) // 2 else ["single-pass prompts only compare requirements within the same chunk"],
        },
        "errors": errors,
        "limitations": limitations,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Staged workflow nodes
# ---------------------------------------------------------------------------


@timed("identify_requirements")
def identify_requirements(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    ctx = runtime.context
    plan = state["chunk_plan"]
    seg_by_id = {s.segment_id: s for s in state["segments"]}
    docs = {d.document_id: d for d in state["documents"]}

    def run(chunk):
        segs = []
        for sid, context_only in [(s, True) for s in chunk.context_segment_ids] + [(s, False) for s in chunk.segment_ids]:
            seg = seg_by_id[sid]
            if not seg.is_candidate:
                continue
            segs.append(
                {
                    "segment_id": seg.segment_id,
                    "document_id": seg.document_id,
                    "chapter": seg.chapter,
                    "section_path": seg.section_path,
                    "text": seg.text,
                    "context_only": context_only,
                }
            )
        payload = {
            "documents": [
                {"document_id": d, "identifier": docs[d].canonical_identifier, "title": docs[d].title} for d in chunk.document_ids
            ],
            "segments": segs,
        }
        system, user = requirement_prompt.SPEC.render(payload)
        return chunk, ctx.llm.call(
            task="requirement_identification",
            stage="identify_requirements",
            system=system,
            user=user,
            response_model=requirement_prompt.RequirementIdentificationResponse,
            prompt_version=requirement_prompt.SPEC.fingerprint,
        )

    reported: list[ReportedRequirement] = []
    errors: list[PipelineError] = []
    limitations: list[CoverageLimitation] = []
    for chunk, result in ctx.llm.map(run, plan.chunks):
        if not result.ok or result.value is None:
            errors.append(PipelineError(stage="identify_requirements", chunk_id=chunk.chunk_id, message="; ".join(result.errors)[:300]))
            limitations.append(CoverageLimitation(kind="llm_call_failed", chunk_id=chunk.chunk_id, message="requirement identification failed for this chunk"))
            continue
        for item in result.value.requirements:
            seg = seg_by_id.get(item.segment_id)
            reported.append(
                ReportedRequirement(
                    document_id=seg.document_id if seg else None,
                    chapter=item.chapter,
                    text=item.requirement_text,
                    chunk_id=chunk.chunk_id,
                    segment_id=item.segment_id if seg else None,
                )
            )
    chunk_docs = {c.chunk_id: c.document_ids for c in plan.chunks}
    drafts, _, warnings = ground_requirements(reported, sources=ctx.sources, segments=state["segments"], chunk_documents=chunk_docs)
    return {"requirements": drafts, "errors": errors, "limitations": limitations, "warnings": warnings}


@timed("chunks_as_requirements")
def chunks_as_requirements(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    """``unit: chunk``: every unit is analysed as one requirement-like unit (no requirement identification).

    The unit text goes through the same grounding as model-reported requirements, so a
    reworded chunk is marked not verbatim and gets no offsets.
    """
    ctx = runtime.context
    plan = state["chunk_plan"]
    owner = {sid: c.chunk_id for c in plan.chunks for sid in c.segment_ids}
    reported = [
        ReportedRequirement(document_id=seg.document_id, chapter=seg.chapter, text=seg.text, chunk_id=owner[seg.segment_id], segment_id=seg.segment_id)
        for seg in state["segments"]
        if seg.is_candidate and seg.segment_id in owner
    ]
    chunk_docs = {c.chunk_id: c.document_ids for c in plan.chunks}
    drafts, _, warnings = ground_requirements(reported, sources=ctx.sources, segments=state["segments"], chunk_documents=chunk_docs)
    return {"requirements": drafts, "warnings": warnings}


@timed("merge_references")
def merge_references(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    """LLM: merge reference labels that denote the same document (method llm or hybrid)."""
    ctx = runtime.context
    method = ctx.options.reference_merging.method
    mentions = _copy_mentions(state)
    if method == "llm":
        decisions, errs = llm_merge(ctx, mentions)
    else:
        texts = {r.requirement_id: r.text for r in state["requirements"]}
        decisions, errs = hybrid_merge(ctx, mentions, state["documents"], texts)
    return {
        "mentions": mentions,
        "merge_decisions": decisions,
        "errors": [PipelineError(stage="merge_references", message=e) for e in errs],
    }


@timed("extract_references")
def extract_references_node(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    ctx = runtime.context
    requirements = [r.model_copy(deep=True) for r in state["requirements"]]
    counter = MentionCounter(state.get("mention_seq", 0))
    mentions, statuses, errs = extract_references(ctx, requirements, counter=counter, batch_size=ctx.options.reference_batch_size)
    for r in requirements:
        r.reference_status = statuses.get(r.requirement_id, "failed")  # type: ignore[assignment]
    return {
        "requirements": requirements,
        "mentions": mentions,
        "initial_mentions": [m.model_copy(deep=True) for m in mentions],
        "mention_seq": counter.value,
        "errors": [PipelineError(stage="extract_references", message=e) for e in errs],
    }


@timed("verify_references")
def verify_references(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    ctx = runtime.context
    settings = ctx.options.verification
    mentions = _copy_mentions(state)
    attempt = state.get("verification_attempts", 0)
    examined = set(state.get("examined_candidates", []))
    vp = verify_pass(state["requirements"], mentions, settings, sources=ctx.sources, attempt=attempt, examined=examined)
    events = list(vp.events)
    pending = vp.candidates
    update: dict[str, Any] = {"mentions": mentions}
    if pending and attempt >= settings.max_attempts:
        for c in pending:
            events.append(
                VerificationEvent(requirement_id=c.requirement_id, raw_identifier=c.text, action="candidate_unresolved", reason="verification attempt budget exhausted", attempt=attempt)
            )
        update["limitations"] = [
            CoverageLimitation(kind="unresolved_identifier", requirement_id=c.requirement_id, message=f"candidate '{c.text}' left unexamined (attempt budget exhausted)")
            for c in pending
            if c.kind == "possible_missed"
        ]
        for m in mentions:  # doubtful mentions that were never checked stay, unconfirmed
            if m.active and not m.verified and any(c.mention_id == m.mention_id for c in pending):
                m.verified = False
        pending = []
    if not pending:
        requirements = [r.model_copy() for r in state["requirements"]]
        for r in requirements:
            if r.reference_status == "unverified":
                r.reference_status = "verified"
        update["requirements"] = requirements
    update["pending_candidates"] = pending
    update["verification_events"] = events
    return update


def route_after_verification(state: PipelineState) -> str:
    return "reextract" if state.get("pending_candidates") else "done"


@timed("targeted_reextraction")
def targeted_reextraction_node(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    ctx = runtime.context
    mentions = _copy_mentions(state)
    attempt = state.get("verification_attempts", 0) + 1
    counter = MentionCounter(state.get("mention_seq", 0))
    req_map = {r.requirement_id: r for r in state["requirements"]}
    new, events, errs, examined = targeted_reextraction(
        ctx, req_map, mentions, state["pending_candidates"], counter=counter, attempt=attempt
    )
    return {
        "mentions": mentions + new,
        "mention_seq": counter.value,
        "verification_events": events,
        "examined_candidates": [*state.get("examined_candidates", []), *sorted(examined)],
        "verification_attempts": attempt,
        "pending_candidates": [],
        "errors": [PipelineError(stage="targeted_reextraction", message=e) for e in errs],
    }


# ---------------------------------------------------------------------------
# Graph-assisted nodes
# ---------------------------------------------------------------------------


def _working_graph(state: PipelineState, ctx: PipelineContext, mentions: list) -> Any:
    """Graph of the current predictions, plus saved-graph documents when the run has them."""
    graph = provisional_graph(state["documents"], state["requirements"], mentions)
    return with_saved_documents(graph, ctx.saved_graph)


@timed("build_graph")
def build_graph_node(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    return {"graph": _working_graph(state, runtime.context, state.get("mentions", []))}


@timed("graph_reference_audit")
def graph_reference_audit(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    ctx = runtime.context
    gs = ctx.options.graph
    mentions = _copy_mentions(state)
    attempt = state.get("verification_attempts", 0)
    events: list[VerificationEvent] = []
    decisions: list[GraphDecision] = []
    errors: list[PipelineError] = []
    examined = list(state.get("examined_candidates", []))
    counter = MentionCounter(state.get("mention_seq", 0))
    graph = state["graph"]

    if gs.alias_resolution:
        ev, dec = resolve_aliases(mentions, graph, attempt)
        events += ev
        decisions += dec
        graph = _working_graph(state, ctx, mentions)

    if gs.reference_audit:
        candidates, dec = known_document_candidates(
            state["requirements"],
            mentions,
            graph,
            examined=set(examined),
            max_per_requirement=ctx.options.verification.max_candidates_per_requirement,
            attempt=attempt,
        )
        decisions += dec
        if candidates:
            req_map = {r.requirement_id: r for r in state["requirements"]}
            new, ev, errs, done = targeted_reextraction(
                ctx,
                req_map,
                mentions,
                candidates,
                counter=counter,
                attempt=attempt,
                added_status=MentionStatus.ADDED_BY_GRAPH_AUDIT,
                graph_informed=True,
                stage="graph_reference_audit",
            )
            mentions += new
            events += ev
            errors += [PipelineError(stage="graph_reference_audit", message=e) for e in errs]
            examined += sorted(done)
            if new and ctx.options.verification.enabled:
                check = ctx.options.verification.model_copy(update={"audit_enabled": False})
                vp = verify_pass(state["requirements"], mentions, check, sources=ctx.sources, attempt=attempt, examined=set(examined))
                events += vp.events
                for m in mentions:  # no further targeted round inside the graph audit
                    if m.active and not m.verified and any(c.mention_id == m.mention_id for c in vp.candidates):
                        m.verified = True
            if new and gs.alias_resolution:
                ev, dec = resolve_aliases(mentions, _working_graph(state, ctx, mentions), attempt)
                events += ev
                decisions += dec
        graph = _working_graph(state, ctx, mentions)

    return {
        "mentions": mentions,
        "mention_seq": counter.value,
        "graph": graph,
        "verification_events": events,
        "graph_decisions": decisions,
        "examined_candidates": examined,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Conflict nodes
# ---------------------------------------------------------------------------


@timed("select_conflict_candidates")
def select_conflict_candidates(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    ctx = runtime.context
    use_graph = ctx.options.graph.enabled
    candidates, total, notes, decisions = select_candidates(
        state["requirements"],
        ctx.options.conflicts,
        graph=state.get("graph") if use_graph else None,
        graph_settings=ctx.options.graph if use_graph else None,
    )
    return {
        "conflict_candidates": candidates,
        "total_pairs": total,
        "conflict_stats": {"candidate_pairs": len(candidates), "total_pairs": total, "notes": notes},
        "graph_decisions": decisions,
    }


@timed("analyze_conflicts")
def analyze_conflicts(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    ctx = runtime.context
    gs = ctx.options.graph
    graph = state.get("graph") if gs.enabled else None
    req_map = {r.requirement_id: r for r in state["requirements"]}
    candidates = [c.model_copy(deep=True) for c in state.get("conflict_candidates", [])]

    context_for = None
    decisions: list[GraphDecision] = []
    if graph is not None and gs.context_retrieval:
        context_for = functools.partial(graph_context_for_pair, graph, max_items=gs.max_context_items, context_format=gs.context_format)

    conflicts, analysed, failed, errs = analyze_pairs(
        ctx, candidates, req_map, pairs_per_call=ctx.options.conflicts.pairs_per_call, context_for=context_for, graph_informed=graph is not None
    )
    if context_for is not None:
        sizes = [len(context_for(c.requirement_ids)) for c in candidates]
        with_context = sum(1 for n in sizes if n)
        mean_items = sum(sizes) / with_context if with_context else 0.0
        decisions.append(
            GraphDecision(
                decision_type="context_retrieved",
                detail=f"graph context ({gs.context_format}) added to {with_context} of {len(candidates)} pairs, {mean_items:.1f} items each",
            )
        )

    if graph is not None:
        rule = graph_rule_conflicts(graph, gs, set(req_map))
        for c in rule:
            decisions.append(GraphDecision(decision_type="rule_flagged_conflict", detail=c.description, node_ids=c.requirement_ids))
        conflicts += rule

    merged = merge_conflicts(conflicts)
    repairs = 0
    if ctx.options.conflicts.validate_evidence:
        repairs = validate_conflict_evidence(merged, req_map, ctx.sources)
    for c in merged:
        if c.conflict_type is None:
            c.conflict_type = ConflictType.OTHER_OR_UNCERTAIN
    stats = dict(state.get("conflict_stats", {}))
    stats.update({"pairs_analyzed": len(analysed), "pairs_failed": failed, "evidence_repairs": repairs})
    return {
        "conflicts": merged,
        "conflict_candidates": candidates,
        "conflict_stats": stats,
        "graph_decisions": decisions,
        "errors": [PipelineError(stage="analyze_conflicts", message=e) for e in errs],
    }
