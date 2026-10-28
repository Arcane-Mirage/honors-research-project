"""src/csr_lab/extraction/references.py

Build reference mentions from model output, and the staged reference-extraction call.

``mentions_from_model`` is used by every strategy. It canonicalises identifiers
(``normalization.parse_identifier``) and records evidence, but it does not
reject anything: rejection is the job of the verification stage, which only
some strategies run.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from csr_lab.drafts import MentionDraft, RequirementDraft
from csr_lab.extraction.grounding import SourceIndex, build_evidence, collapse_ws
from csr_lab.extraction.normalization import parse_identifier
from csr_lab.prompts import references as reference_prompt
from csr_lab.prompts.references import ExtractedReference
from csr_lab.schemas import DocumentType, MentionStatus, ReferenceRelation, ResolutionMethod

if TYPE_CHECKING:
    from csr_lab.pipelines.base import PipelineContext


class MentionCounter:
    """Deterministic mention IDs within one analysis run."""

    def __init__(self, start: int = 0):
        self.value = start

    def next(self) -> str:
        self.value += 1
        return f"MEN-{self.value:04d}"


def _name_in_text(name: str | None, text: str) -> bool:
    return bool(name) and collapse_ws(name).lower() in collapse_ws(text).lower()  # type: ignore[union-attr]


def mention_from_reference(
    ref: ExtractedReference | dict,
    requirement: RequirementDraft,
    *,
    sources: SourceIndex,
    counter: MentionCounter,
    status: MentionStatus,
) -> MentionDraft | None:
    if isinstance(ref, dict):
        ref = ExtractedReference.model_validate(ref)
    raw = ref.identifier.strip()
    if not raw:
        return None
    parsed = parse_identifier(raw, revision_hint=ref.revision)
    name = None
    if parsed.canonical is None:
        name = parsed.name  # name-only mention: the mention itself is the name
        if not name:
            return None
    else:
        # A document name is kept only when it appears in the requirement text (never invented).
        for candidate in (ref.document_name, parsed.name):
            if candidate and _name_in_text(candidate, requirement.text):
                name = candidate.strip()
                break
    doc_type = ref.document_type if ref.document_type not in (None, DocumentType.UNKNOWN) else parsed.type_hint
    quote = ref.quoted_text or raw
    evidence = build_evidence(
        sources,
        document_id=requirement.document_id,
        quote=quote,
        requirement_id=requirement.requirement_id,
        span=requirement.span,
        chapter=requirement.chapter,
    )
    return MentionDraft(
        mention_id=counter.next(),
        requirement_id=requirement.requirement_id,
        raw_identifier=raw,
        canonical=parsed.canonical,
        revision=parsed.revision,
        name=name,
        document_type=doc_type or DocumentType.UNKNOWN,
        relation=ref.relation or ReferenceRelation.REFERENCES,
        evidence=evidence,
        status=status,
        resolution=ResolutionMethod.IDENTIFIER if parsed.canonical else ResolutionMethod.UNRESOLVED_NAME,
    )


def mentions_from_model(
    refs: list[ExtractedReference] | list[dict],
    requirement: RequirementDraft,
    *,
    sources: SourceIndex,
    counter: MentionCounter,
    status: MentionStatus = MentionStatus.FOUND_INITIALLY,
) -> list[MentionDraft]:
    out = []
    for ref in refs:
        mention = mention_from_reference(ref, requirement, sources=sources, counter=counter, status=status)
        if mention is not None:
            out.append(mention)
    return out


def extract_references(
    ctx: PipelineContext,
    requirements: list[RequirementDraft],
    *,
    counter: MentionCounter,
    batch_size: int,
) -> tuple[list[MentionDraft], dict[str, str], list[str]]:
    """Staged reference extraction in batches of requirements.

    Returns (mentions, requirement_id -> status, errors). A requirement that the
    model skipped or whose batch failed gets status 'failed', which is distinct
    from 'no references found'.

    The model sees short batch-local keys (R1, R2, ...) instead of the long
    requirement IDs, so it never has to copy a hash back exactly.
    """
    batches = [requirements[i : i + batch_size] for i in range(0, len(requirements), batch_size)]

    def run(batch: list[RequirementDraft]):
        payload = {
            "requirements": [
                {"requirement_id": f"R{i}", "chapter": r.chapter, "text": r.text} for i, r in enumerate(batch, start=1)
            ]
        }
        system, user = reference_prompt.SPEC.render(payload)
        return batch, ctx.llm.call(
            task="reference_extraction",
            stage="extract_references",
            system=system,
            user=user,
            response_model=reference_prompt.ReferenceExtractionResponse,
            prompt_version=reference_prompt.SPEC.fingerprint,
        )

    mentions: list[MentionDraft] = []
    statuses: dict[str, str] = {}
    errors: list[str] = []
    for batch, result in ctx.llm.map(run, batches):
        if not result.ok or result.value is None:
            errors.append(f"reference extraction failed for {len(batch)} requirements: {'; '.join(result.errors)[:300]}")
            for r in batch:
                statuses[r.requirement_id] = "failed"
            continue
        by_id = {item.requirement_id.strip(): item for item in result.value.results}
        for i, r in enumerate(batch, start=1):
            item = by_id.get(f"R{i}")
            if item is None:
                statuses[r.requirement_id] = "failed"
                errors.append(f"model returned no reference result for {r.requirement_id}")
                continue
            statuses[r.requirement_id] = "unverified"
            mentions.extend(mentions_from_model(item.references, r, sources=ctx.sources, counter=counter))
    return mentions, statuses, errors


def identifier_pattern(canonical: str) -> re.Pattern[str]:
    """Regex matching an identifier with flexible separators ('TSA-MP-215' ~ 'TSA MP 215')."""
    tokens = [t for t in canonical.split("-") if t]
    sep = r"[\s\-‐-―_]*"
    return re.compile(r"(?<![A-Za-z0-9])" + sep.join(re.escape(t) for t in tokens) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def find_mention_span(mention: MentionDraft, text: str) -> tuple[int, int] | None:
    """Where the mention's identifier (or name) occurs in ``text``, if anywhere."""
    if mention.canonical:
        if m := identifier_pattern(mention.canonical).search(text):
            return m.start(), m.end()
        return None
    if mention.name:
        pattern = r"\s+".join(re.escape(w) for w in mention.name.split())
        if m := re.search(pattern, text, re.IGNORECASE):
            return m.start(), m.end()
    return None
