"""src/csr_lab/extraction/verification.py

Reference verification: deterministic checks, coverage audit, targeted re-extraction.

Used by strategies with ``verification.enabled``. For each predicted mention:

1. the identifier (or name) must occur in *its own* requirement text;
2. it must not look like a section number, date, quantity or part number;
3. duplicates within a requirement are merged;
4. evidence that cannot be located inside the requirement is replaced by the
   sentence that contains the identifier (recorded as ``evidence_repaired``);
5. identifiers without any document cue are sent to a targeted model check.

The coverage audit scans each requirement for identifier-like tokens and
document names that no mention covers and sends them, a bounded number of
times, to a targeted re-extraction call. Every action is logged as a
``VerificationEvent``; the caller keeps the initial predictions separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from csr_lab.config import VerificationSettings
from csr_lab.drafts import MentionDraft, RequirementDraft
from csr_lab.extraction.grounding import build_evidence, evidence_inside
from csr_lab.extraction.normalization import near_duplicates, non_document_reason, parse_identifier
from csr_lab.extraction.references import MentionCounter, find_mention_span, mention_from_reference
from csr_lab.prompts import verification as verification_prompt
from csr_lab.prompts.references import ExtractedReference
from csr_lab.schemas import MentionStatus, VerificationEvent

if TYPE_CHECKING:
    from csr_lab.pipelines.base import PipelineContext

CandidateKind = Literal["possible_missed", "doubtful_existing", "known_document"]

_ID_CANDIDATE_RE = re.compile(
    r"(?<![A-Za-z0-9-])(?:[A-Z][A-Z0-9]{0,7}[-\s]){0,3}[A-Z]{0,8}\d[\dA-Z]*(?:[.\-][\dA-Z]+)*"
    r"(?:,?\s*(?:Rev(?:ision)?\.?|rev\.?)\s*[A-Z0-9]{1,3}\b)?(?![A-Za-z0-9-])"
)
# Names directly followed by an identifier are skipped: the identifier is the reference.
_NAME_CANDIDATE_RE = re.compile(
    r"\bthe\s+(?P<name>[A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|and|of|for))*\s+(?:Manual|Specification|Standard|Procedure|Policy|Drawing))\b"
    r"(?!\s*\(?\s*[A-Z]{2,6}[-\s][A-Z0-9.-]*\d)"
)
_TYPE_CUE_RE = re.compile(r"(specification|standard|procedure|drawing|manual|method|form|policy|according to|per)\s*$", re.I)


@dataclass
class AuditCandidate:
    candidate_id: str
    requirement_id: str
    text: str
    kind: CandidateKind
    start: int
    end: int
    key: str
    mention_id: str | None = None
    related: list[dict] = field(default_factory=list)


def _sentence_around(text: str, start: int, end: int, limit: int = 200) -> str:
    left = max(text.rfind(". ", 0, start), text.rfind("; ", 0, start), text.rfind("\n\n", 0, start))
    left = 0 if left == -1 else left + 2
    right_candidates = [p for p in (text.find(". ", end), text.find("; ", end)) if p != -1]
    right = min(right_candidates) + 1 if right_candidates else len(text)
    if right - left > limit:
        left = max(left, start - limit // 2)
        right = min(right, end + limit // 2)
    return text[left:right].strip()


def candidate_key(text: str) -> str:
    parsed = parse_identifier(text)
    return parsed.key


def scan_candidates(text: str) -> list[tuple[int, int, str]]:
    """Liberal scan for identifier-like tokens and specific document names."""
    found: list[tuple[int, int, str]] = []
    for m in _ID_CANDIDATE_RE.finditer(text):
        token = m.group(0).strip()
        if any(c.isalpha() for c in token) and any(c.isdigit() for c in token):
            found.append((m.start(), m.start() + len(token), token))
    for m in _NAME_CANDIDATE_RE.finditer(text):
        words = [w for w in m.group("name").split() if w[:1].isupper()]
        if len(words) >= 3:
            found.append((m.start("name"), m.end("name"), " ".join(m.group("name").split())))
    return found


@dataclass
class VerificationPass:
    events: list[VerificationEvent]
    candidates: list[AuditCandidate]


def verify_pass(
    requirements: list[RequirementDraft],
    mentions: list[MentionDraft],
    settings: VerificationSettings,
    *,
    sources,
    attempt: int,
    examined: set[str],
) -> VerificationPass:
    """One deterministic verification + audit pass. Mutates the given (working) mentions."""
    events: list[VerificationEvent] = []
    candidates: list[AuditCandidate] = []
    by_req: dict[str, list[MentionDraft]] = {}
    for m in mentions:
        if m.active:
            by_req.setdefault(m.requirement_id, []).append(m)

    def event(req_id: str, m: MentionDraft | None, raw: str, action: str, reason: str) -> None:
        events.append(
            VerificationEvent(
                requirement_id=req_id,
                raw_identifier=raw,
                action=action,  # type: ignore[arg-type]
                reason=reason,
                attempt=attempt,
                mention_id=m.mention_id if m else None,
            )
        )

    for req in requirements:
        text = req.text
        seen: dict[str, MentionDraft] = {}
        req_mentions = sorted(by_req.get(req.requirement_id, []), key=lambda x: (not x.verified, x.mention_id))
        for m in req_mentions:
            if m.verified:
                seen.setdefault(m.key, m)
                continue
            span = find_mention_span(m, text)
            if span is None:
                m.active, m.inactive_reason = False, "identifier not found in its requirement text"
                event(req.requirement_id, m, m.raw_identifier, "rejected", m.inactive_reason)
                continue
            preceding = text[max(0, span[0] - 30) : span[0]]
            reason = non_document_reason(m.canonical, preceding) if settings.reject_distractors else None
            if reason:
                m.active, m.inactive_reason = False, reason
                event(req.requirement_id, m, m.raw_identifier, "rejected", reason)
                continue
            if m.key in seen:
                m.active, m.inactive_reason = False, f"duplicate of {seen[m.key].mention_id}"
                event(req.requirement_id, m, m.raw_identifier, "merged_duplicate", m.inactive_reason)
                continue
            seen[m.key] = m
            if settings.repair_evidence and req.span and not evidence_inside(m.evidence, req.span):
                quote = _sentence_around(text, *span)
                repaired = build_evidence(
                    sources,
                    document_id=req.document_id,
                    quote=quote,
                    requirement_id=req.requirement_id,
                    span=req.span,
                    chapter=req.chapter,
                )
                m.evidence = repaired.model_copy(update={"repaired": True})
                event(req.requirement_id, m, m.raw_identifier, "evidence_repaired", "model quote not found in requirement")
            doubtful = bool(m.canonical) and "-" not in (m.canonical or "") and not _TYPE_CUE_RE.search(preceding)
            key = f"doubt:{req.requirement_id}:{m.key}"
            if doubtful and key not in examined:
                candidates.append(
                    AuditCandidate(
                        candidate_id="",
                        requirement_id=req.requirement_id,
                        text=text[span[0] : span[1]],
                        kind="doubtful_existing",
                        start=span[0],
                        end=span[1],
                        key=key,
                        mention_id=m.mention_id,
                    )
                )
            else:
                m.verified = True
                event(req.requirement_id, m, m.raw_identifier, "kept", "identifier present in requirement text")

        if settings.audit_enabled:
            active = [m for m in by_req.get(req.requirement_id, []) if m.active]
            spans = [s for s in (find_mention_span(m, text) for m in active) if s]
            keys = {m.key for m in active} | {m.canonical for m in active if m.canonical}
            req_candidates = []
            for start, end, token in scan_candidates(text):
                if any(start < e and s < end for s, e in spans):
                    continue
                parsed = parse_identifier(token)
                if parsed.key in keys or (parsed.canonical and parsed.canonical in keys):
                    continue
                if non_document_reason(parsed.canonical, text[max(0, start - 30) : start]):
                    continue
                key = f"audit:{req.requirement_id}:{parsed.key}"
                if key in examined or any(c.key == key for c in req_candidates):
                    continue
                req_candidates.append(
                    AuditCandidate(
                        candidate_id="",
                        requirement_id=req.requirement_id,
                        text=token,
                        kind="possible_missed",
                        start=start,
                        end=end,
                        key=key,
                    )
                )
            candidates.extend(req_candidates[: settings.max_candidates_per_requirement])

    flagged = set()
    for a, b in near_duplicates([m.canonical for m in mentions if m.active and m.canonical]):
        if (a, b) not in flagged:
            flagged.add((a, b))
            events.append(
                VerificationEvent(
                    requirement_id="*",
                    raw_identifier=f"{a} / {b}",
                    action="flagged_near_duplicate",
                    reason="identifiers differ only in separators or leading zeros; kept distinct for review",
                    attempt=attempt,
                )
            )
    for i, cand in enumerate(candidates, start=1):
        cand.candidate_id = f"C{attempt}-{i:03d}"
    return VerificationPass(events=events, candidates=candidates)


def targeted_reextraction(
    ctx: PipelineContext,
    requirements: dict[str, RequirementDraft],
    mentions: list[MentionDraft],
    candidates: list[AuditCandidate],
    *,
    counter: MentionCounter,
    attempt: int,
    added_status: MentionStatus = MentionStatus.ADDED_DURING_VERIFICATION,
    graph_informed: bool = False,
    stage: str = "targeted_reextraction",
) -> tuple[list[MentionDraft], list[VerificationEvent], list[str], set[str]]:
    """Ask the model about each requirement's candidates. Returns (new mentions, events, errors, examined keys)."""
    by_req: dict[str, list[AuditCandidate]] = {}
    for cand in candidates:
        by_req.setdefault(cand.requirement_id, []).append(cand)
    mention_by_id = {m.mention_id: m for m in mentions}

    def run(item: tuple[str, list[AuditCandidate]]):
        req_id, cands = item
        req = requirements[req_id]
        current = [
            m.raw_identifier for m in mentions if m.requirement_id == req_id and m.active and m.mention_id not in {c.mention_id for c in cands}
        ]
        payload = {
            "requirement": {"requirement_id": req_id, "chapter": req.chapter, "text": req.text},
            "current_references": current,
            "candidates": [
                {
                    "candidate_id": c.candidate_id,
                    "text": c.text,
                    "kind": c.kind,
                    "context": _sentence_around(req.text, c.start, c.end),
                }
                for c in cands
            ],
        }
        related = [r for c in cands for r in c.related]
        if related:
            payload["related_documents"] = related
        system, user = verification_prompt.SPEC.render(payload)
        result = ctx.llm.call(
            task="reference_verification",
            stage=stage,
            system=system,
            user=user,
            response_model=verification_prompt.TargetedVerificationResponse,
            prompt_version=verification_prompt.SPEC.fingerprint,
            graph_informed=graph_informed,
        )
        return req_id, cands, result

    new_mentions: list[MentionDraft] = []
    events: list[VerificationEvent] = []
    errors: list[str] = []
    examined: set[str] = set()
    for req_id, cands, result in ctx.llm.map(run, list(by_req.items())):
        req = requirements[req_id]
        decisions = {d.candidate_id: d for d in result.value.decisions} if result.ok and result.value else {}
        if not result.ok:
            errors.append(f"targeted re-extraction failed for {req_id}: {'; '.join(result.errors)[:200]}")
        for cand in cands:
            examined.add(cand.key)
            decision = decisions.get(cand.candidate_id)
            target = mention_by_id.get(cand.mention_id) if cand.mention_id else None
            if decision is None:
                events.append(VerificationEvent(requirement_id=req_id, raw_identifier=cand.text, action="candidate_unresolved", reason="no decision returned", attempt=attempt))
                if target is not None:
                    target.verified = True  # keep, but note it was not confirmed
                continue
            if cand.kind == "doubtful_existing" and target is not None:
                if decision.is_document_reference:
                    target.verified = True
                    events.append(VerificationEvent(requirement_id=req_id, raw_identifier=cand.text, action="kept", reason="confirmed by targeted check", attempt=attempt, mention_id=target.mention_id))
                else:
                    target.active, target.inactive_reason = False, "targeted check: not a document reference"
                    events.append(VerificationEvent(requirement_id=req_id, raw_identifier=cand.text, action="rejected", reason=target.inactive_reason, attempt=attempt, mention_id=target.mention_id))
                continue
            if decision.is_document_reference and (decision.identifier or cand.text):
                ref = ExtractedReference(
                    identifier=decision.identifier or cand.text,
                    revision=decision.revision,
                    document_name=decision.document_name,
                    relation=decision.relation,
                    quoted_text=decision.quoted_text or _sentence_around(req.text, cand.start, cand.end),
                )
                mention = mention_from_reference(ref, req, sources=ctx.sources, counter=counter, status=added_status)
                if mention is not None:
                    new_mentions.append(mention)
                    events.append(VerificationEvent(requirement_id=req_id, raw_identifier=mention.raw_identifier, action="added", reason=f"{cand.kind} confirmed by targeted check", attempt=attempt, mention_id=mention.mention_id))
            else:
                events.append(VerificationEvent(requirement_id=req_id, raw_identifier=cand.text, action="candidate_rejected", reason=decision.reason or "not a document reference", attempt=attempt))
    return new_mentions, events, errors, examined


def verification_counts(initial: list[MentionDraft], final: list[MentionDraft], events: list[VerificationEvent]) -> dict[str, int]:
    return {
        "found_initially": len(initial),
        "kept_from_initial": sum(1 for m in final if m.active and m.status == MentionStatus.FOUND_INITIALLY),
        "added_during_verification": sum(1 for m in final if m.active and m.status == MentionStatus.ADDED_DURING_VERIFICATION),
        "added_by_graph_audit": sum(1 for m in final if m.active and m.status == MentionStatus.ADDED_BY_GRAPH_AUDIT),
        "rejected_unsupported": sum(1 for e in events if e.action == "rejected"),
        "merged_duplicates": sum(1 for e in events if e.action == "merged_duplicate"),
        "evidence_repaired": sum(1 for e in events if e.action == "evidence_repaired"),
        "unresolved": sum(1 for e in events if e.action == "candidate_unresolved")
        + sum(1 for m in final if m.active and m.canonical is None),
        "near_duplicate_flags": sum(1 for e in events if e.action == "flagged_near_duplicate"),
    }
