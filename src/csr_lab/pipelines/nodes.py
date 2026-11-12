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
from collections.abc import Callable
from typing import Any

from langgraph.runtime import Runtime

from csr_lab.conflicts.candidate_selection import select_candidates
from csr_lab.conflicts.detector import (
    REVIEW_REASON,
    analyze_pairs,
    build_conflict_evidence,
    graph_context_for_pair,
    graph_rule_conflicts,
    merge_conflicts,
    saved_graph_candidates,
    saved_unit_evidence,
    validate_conflict_evidence,
)
from csr_lab.drafts import CandidatePair, ConflictDraft, MentionDraft, PipelineError, RequirementDraft, comparable, ordered_pair
from csr_lab.extraction.fidelity import WordIndex
from csr_lab.extraction.graph_audit import known_document_candidates, resolve_aliases
from csr_lab.extraction.grounding import TextIndex
from csr_lab.extraction.merging import hybrid_merge, llm_merge, merge_spacing_variants
from csr_lab.extraction.references import MentionCounter, extract_references, mentions_from_model
from csr_lab.extraction.requirements import UnitText, ground_requirements
from csr_lab.extraction.verification import targeted_reextraction, verify_pass
from csr_lab.ids import document_family_id
from csr_lab.ingestion.chapters import split_chapters
from csr_lab.ingestion.chunking import plan_chunks
from csr_lab.ingestion.segmentation import CustomerMatcher, find_blocks, hide_other_customers
from csr_lab.ingestion.units import model_chunks, segments_from_chunks
from csr_lab.pipelines.assembly import assemble, provisional_graph
from csr_lab.pipelines.base import PipelineContext
from csr_lab.pipelines.state import PipelineState
from csr_lab.prompts import baseline as baseline_prompt
from csr_lab.relationships.builder import annotate_structure, saved_documents
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


@timed("prepare_input")
def prepare_input(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    """Cut every document into whole chapters (``ingestion/chapters.py``).

    With ``chunking.method: llm`` (staged workflow) the documents are left to ``llm_chunk``.
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
    if chunking.method == "llm" and ctx.options.workflow == "staged":
        return {"segments": [], "limitations": limitations}
    segments = []
    warnings: list[str] = []
    for d in usable:
        segs, notes = split_chapters(d, chunking)
        segments += segs
        warnings += [f"{d.document_id}: {n}" for n in notes]
    plan = plan_chunks(usable, segments, ctx.config.processing)
    update: dict[str, Any] = {"segments": segments, "chunk_plan": plan, "limitations": limitations + plan.limitations, "warnings": warnings}
    if not any(s.is_candidate for s in segments):
        update["fatal"] = "no chapter with text was found in the input"
    return update


@timed("llm_chunk")
def llm_chunk(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    """LLM: split each document into word-count batches and ask the model for semantic chunks."""
    ctx = runtime.context
    settings = ctx.options.chunking
    limit = ctx.config.processing.max_document_chars
    targets = [d for d in state["documents"] if d.parse_status in ("ok", "partial")]
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
    plan = plan_chunks(targets, segments, ctx.config.processing)
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
    """One call per chunk of chapters: references of every chapter and conflicts between them."""
    ctx = runtime.context
    plan = state["chunk_plan"]
    drafts, warnings = _units_as_requirements(state, ctx)
    by_id = {d.requirement_id: d for d in drafts}
    by_segment = {sid: d for d in drafts for sid in d.segment_ids}
    docs = {d.document_id: d for d in state["documents"]}

    def run(chunk):
        keys: dict[str, str] = {}  # R1 -> requirement ID
        units = []
        shown = [(sid, True) for sid in chunk.context_segment_ids] + [(sid, False) for sid in chunk.segment_ids]
        for sid, context_only in shown:
            req = by_segment.get(sid)
            if req is None:
                continue
            key = f"R{len(units) + 1}"
            keys[key] = req.requirement_id
            units.append(
                {
                    "key": key,
                    "document_id": req.document_id,
                    "identifier": docs[req.document_id].canonical_identifier,
                    "chapter": req.chapter,
                    "text": req.prompt_text,
                    "context_only": context_only,
                }
            )
        system, user = baseline_prompt.SPEC.render({"units": units})
        result = ctx.llm.call(
            task="single_pass_extraction",
            stage="single_pass_extract",
            system=system,
            user=user,
            response_model=baseline_prompt.SinglePassResponse,
            prompt_version=baseline_prompt.SPEC.fingerprint,
        )
        return chunk, keys, result

    counter = MentionCounter(state.get("mention_seq", 0))
    mentions: list[MentionDraft] = []
    conflicts: list[ConflictDraft] = []
    errors: list[PipelineError] = []
    limitations: list[CoverageLimitation] = []
    answered: set[str] = set()
    for chunk, keys, result in ctx.llm.map(run, plan.chunks):
        own = {by_segment[sid].requirement_id for sid in chunk.segment_ids if sid in by_segment}
        if not result.ok or result.value is None:
            errors.append(PipelineError(stage="single_pass_extract", chunk_id=chunk.chunk_id, message="; ".join(result.errors)[:300]))
            limitations.append(CoverageLimitation(kind="llm_call_failed", chunk_id=chunk.chunk_id, message="single-pass extraction failed for this chunk"))
            continue
        for item in result.value.results:
            rid = keys.get(item.key)
            if rid is None or rid not in own or rid in answered:
                continue  # unknown key, a context unit, or already answered
            answered.add(rid)
            mentions += mentions_from_model(item.references, by_id[rid], sources=ctx.sources, counter=counter)
        for c in result.value.potential_conflicts:
            ids = [keys.get(k) for k in c.requirement_keys]
            if None in ids or ids[0] == ids[1]:
                errors.append(PipelineError(stage="single_pass_extract", chunk_id=chunk.chunk_id, message=f"conflict names unknown unit keys {c.requirement_keys}"))
                continue
            if not comparable(by_id[ids[0]], by_id[ids[1]]):
                continue  # two passages of one document are not compared
            quotes = [(keys.get(e.requirement_key), e.quoted_text) for e in c.evidence]
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
    for d in drafts:
        d.reference_status = "unverified" if d.requirement_id in answered else "failed"
    merges = merge_spacing_variants(mentions, state["documents"]) if ctx.options.reference_merging.merge_spacing_variants else []

    # Implicit candidates: every pair of requirements that shared a prompt.
    candidates: dict[tuple[str, str], CandidatePair] = {}
    for chunk in plan.chunks:
        ids = sorted({d.requirement_id for d in drafts if chunk.chunk_id in d.chunk_ids})
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                if not comparable(by_id[a], by_id[b]):
                    continue
                pair = ordered_pair(a, b)
                candidates.setdefault(pair, CandidatePair(requirement_ids=pair, selectors=[f"single_pass_context:{chunk.chunk_id}"], analyzed=True))
    n = sum(1 for i, a in enumerate(drafts) for b in drafts[i + 1 :] if comparable(a, b))
    return {
        "requirements": drafts,
        "mentions": mentions,
        "initial_mentions": [m.model_copy(deep=True) for m in mentions],
        "mention_seq": counter.value,
        "merge_decisions": merges,
        "conflicts": merge_conflicts(conflicts),
        "conflict_candidates": list(candidates.values()),
        "conflict_stats": {
            "candidate_pairs": len(candidates),
            "pairs_analyzed": len(candidates),
            "pairs_failed": 0,
            "total_pairs": n,
            "notes": [] if len(candidates) == n else ["single-pass prompts only compare requirements within the same chunk"],
        },
        "errors": errors,
        "limitations": limitations,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Staged workflow nodes
# ---------------------------------------------------------------------------


def _units_as_requirements(state: PipelineState, ctx: PipelineContext):
    """Every unit (a chapter, or a model-made chunk) becomes one requirement, grounded in the source."""
    plan = state["chunk_plan"]
    owner = {sid: c.chunk_id for c in plan.chunks for sid in c.segment_ids}
    units = [
        UnitText(document_id=seg.document_id, chapter=seg.chapter, text=seg.text, chunk_id=owner[seg.segment_id], segment_id=seg.segment_id)
        for seg in state["segments"]
        if seg.is_candidate and seg.segment_id in owner
    ]
    chunk_docs = {c.chunk_id: c.document_ids for c in plan.chunks}
    drafts, warnings = ground_requirements(units, sources=ctx.sources, segments=state["segments"], chunk_documents=chunk_docs)
    apply_customer_scoping(drafts, state["segments"], ctx)
    return drafts, warnings


def apply_customer_scoping(drafts: list[RequirementDraft], segments, ctx: PipelineContext) -> None:
    """Find customer-labelled blocks (tag), and hide other customers' blocks from the model (filter).

    Only units whose text is an exact source slice are scoped (block offsets must be real)."""
    mode = ctx.options.customer_scoping
    matcher = CustomerMatcher.from_settings(ctx.config.system_prompt) if mode != "off" else None
    if matcher is None:
        return
    titles = {s.segment_id: s.title for s in segments}
    for d in drafts:
        if d.start_offset is None or d.match != "exact":
            continue
        d.title = titles.get(d.segment_ids[0]) if d.segment_ids else None
        d.scope_blocks = find_blocks(d.text, d.start_offset, matcher, d.title)
        if mode == "filter" and any(not b.applies_to_target for b in d.scope_blocks):
            d.model_text = hide_other_customers(d.text, d.start_offset, d.scope_blocks)


@timed("units_as_requirements")
def units_as_requirements(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    """Each unit is one requirement: no requirement identification.

    The unit text goes through grounding, so a reworded model chunk is marked not verbatim
    and gets no offsets.
    """
    drafts, warnings = _units_as_requirements(state, runtime.context)
    return {"requirements": drafts, "warnings": warnings}


@timed("merge_references")
def merge_references(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    """Merge reference labels that denote the same document: spacing variants (deterministic),
    then the model-based method (LLM: method llm or hybrid)."""
    ctx = runtime.context
    settings = ctx.options.reference_merging
    mentions = _copy_mentions(state)
    decisions, errs = [], []
    if settings.merge_spacing_variants:
        decisions += merge_spacing_variants(mentions, state["documents"])
    if settings.method == "llm":
        more, errs = llm_merge(ctx, mentions)
        decisions += more
    elif settings.method == "hybrid":
        texts = {r.requirement_id: r.text for r in state["requirements"]}
        more, errs = hybrid_merge(ctx, mentions, state["documents"], texts)
        decisions += more
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
    limitations: list[CoverageLimitation] = []
    for r in requirements:
        r.reference_status = statuses.get(r.requirement_id, "failed")  # type: ignore[assignment]
        if r.reference_status == "failed":
            limitations.append(
                CoverageLimitation(
                    kind="llm_call_failed",
                    document_id=r.document_id,
                    requirement_id=r.requirement_id,
                    message=f"reference extraction failed for {r.chapter or r.requirement_id}; its references were not checked",
                )
            )
    return {
        "requirements": requirements,
        "limitations": limitations,
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
    graph = provisional_graph(
        state["documents"], state["requirements"], mentions, structure_mode=ctx.options.structure_links, names=ctx.sources.names
    )
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


@timed("check_saved_conflicts")
def check_saved_conflicts(state: PipelineState, runtime: Runtime[PipelineContext]) -> dict[str, Any]:
    """LLM: check the new units against requirements and passages already in the saved graph.

    Only pairs whose documents are linked by references are checked (see
    ``saved_graph_candidates``). The saved units are read from the graph; their quotes are
    located in the stored text.
    """
    ctx = runtime.context
    gs = ctx.options.graph
    saved = ctx.saved_units
    if not saved or not gs.saved_conflicts:
        return {}
    documents = state["documents"]
    families = {d.document_id: document_family_id(d.canonical_identifier) for d in documents if d.canonical_identifier}
    unit_refs: dict[str, set[str]] = {}
    for m in state.get("mentions", []):
        if m.active:
            refs = unit_refs.setdefault(m.requirement_id, set())
            refs.add(m.document_id)
            if m.canonical:
                refs.add(document_family_id(m.canonical))
    units = state["requirements"]
    follows = {}
    if ctx.options.structure_links != "off":  # a delta document added after its base agreement
        annotated = annotate_structure(documents, units, state.get("mentions", []), mode=ctx.options.structure_links, names=ctx.sources.names, known=saved_documents(ctx.saved_graph))
        follows = {d.document_id: document_family_id(d.follows_structure_of) for d in annotated if d.follows_structure_of}
    candidates, notes = saved_graph_candidates(
        units,
        unit_refs,
        saved,
        max_pairs=gs.saved_max_pairs,
        min_score=gs.saved_min_score,
        max_fanout=gs.max_shared_reference_fanout,
        families=families,
        follows=follows,
    )
    stats = dict(state.get("conflict_stats", {}))
    stats["saved_graph_units"] = len(saved)
    stats["saved_graph_pairs"] = len(candidates)
    decision = GraphDecision(
        decision_type="saved_graph_conflict_check",
        detail=f"{len(candidates)} pairs of new and saved units (linked by references or aligned chapters) out of {len(saved)} saved units",
    )
    if not candidates:
        return {"conflict_stats": stats | {"notes": [*stats.get("notes", []), *notes]}, "graph_decisions": [decision]}

    stored = {
        u.unit_id: RequirementDraft(requirement_id=u.unit_id, document_id=u.document_id, chapter=u.chapter, text=u.text, unit_kind="saved")
        for u in saved.values()
    }
    req_map = {r.requirement_id: r for r in units} | stored
    found, analysed, failed, errs = analyze_pairs(
        ctx,
        candidates,
        req_map,
        pairs_per_call=ctx.options.conflicts.pairs_per_call,
        graph_informed=True,
        evidence_for=lambda quotes: saved_unit_evidence(quotes, req_map, ctx.sources, saved),
        document_info={u.document_id: (u.identifier, u.revision) for u in saved.values()},
        stage="check_saved_conflicts",
        method="saved_graph_check",
    )
    for c in found:
        c.graph_informed = True
    stats.update(saved_graph_pairs_analyzed=len(analysed), saved_graph_pairs_failed=failed, saved_graph_conflicts=len(found))
    stats["notes"] = [*stats.get("notes", []), *notes]
    return {
        "conflicts": merge_conflicts([*state.get("conflicts", []), *found]),
        "conflict_candidates": [*state.get("conflict_candidates", []), *candidates],
        "conflict_stats": stats,
        "graph_decisions": [decision],
        "errors": [PipelineError(stage="check_saved_conflicts", message=e) for e in errs],
    }
