"""src/csr_lab/extraction/merging.py

Reference merging: decide which extracted reference labels denote the same document.

Three methods (``PipelineOptions.reference_merging.method``):

* ``deterministic``: nothing here runs; identifiers are only normalised
  (``normalization.parse_identifier``), so "Specification CVM-QS-110" and "CVM-QS-110"
  already share a key, but a document *name* is never linked to an identifier.
* ``llm``: one model call per bundle sees only the list of reference labels and returns
  groups with a canonical name; the grouping is applied as given. This mirrors the
  common "ask the model to deduplicate the reference list" step, including its risks
  (similar identifiers merged, revisions dropped).
* ``hybrid``: deterministic keys stay. Candidate groups are built by rules:
  (a) a name-only label and the known documents whose names share words with it;
  (b) near-duplicate identifiers ("QP-012" / "QP-12"), only when ``protect_identifiers`` is off.
  The model answers one targeted question per candidate; guard rules veto merges
  across different explicit revisions (``protect_revisions``).

Every decision is returned as a ``ReferenceMergeDecision``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from csr_lab.drafts import MentionDraft
from csr_lab.extraction.normalization import normalize_name, normalize_revision, parse_identifier, squash
from csr_lab.prompts import merging as merging_prompt
from csr_lab.schemas import InputDocument, ReferenceMergeDecision, ResolutionMethod

if TYPE_CHECKING:
    from csr_lab.pipelines.base import PipelineContext

_STOP = {"the", "of", "and", "for", "a", "an", "to", "in", "on", "corvane", "tessaly"}


def mention_label(m: MentionDraft) -> str:
    """The reference as a human would list it: name (if stated), identifier, revision."""
    raw = " ".join(m.raw_identifier.split())
    label = raw
    if m.canonical and m.name and normalize_name(m.name) not in normalize_name(raw):
        label = f"{m.name} {raw}"
    if m.revision and parse_identifier(raw).revision is None:
        label = f"{label} Rev {m.revision}"
    return label


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2}


def _dedupe_per_requirement(mentions: list[MentionDraft]) -> None:
    seen: set[tuple[str, str]] = set()
    for m in mentions:
        if not m.active:
            continue
        key = (m.requirement_id, m.key)
        if key in seen:
            m.active = False
            m.inactive_reason = "duplicate after reference merging"
        seen.add(key)


def _apply(m: MentionDraft, canonical: str | None, revision: str | None, name: str | None) -> bool:
    before = m.key
    m.canonical, m.revision = canonical, normalize_revision(revision)
    m.name = name if canonical is None else (m.name or name)
    if m.key != before:
        m.resolution = ResolutionMethod.MERGED
        return True
    return False


# ---------------------------------------------------------------------------
# llm: model-only grouping of the label list
# ---------------------------------------------------------------------------


def llm_merge(ctx: PipelineContext, mentions: list[MentionDraft]) -> tuple[list[ReferenceMergeDecision], list[str]]:
    active = [m for m in mentions if m.active]
    by_label: dict[str, list[MentionDraft]] = {}
    for m in active:
        by_label.setdefault(mention_label(m), []).append(m)
    labels = list(by_label)
    if len(labels) < 2:
        return [], []
    system, user = merging_prompt.MERGING_SPEC.render({"labels": labels})
    result = ctx.llm.call(
        task="reference_merging",
        stage="merge_references",
        system=system,
        user=user,
        response_model=merging_prompt.ReferenceMergingResponse,
        prompt_version=merging_prompt.MERGING_SPEC.fingerprint,
    )
    if not result.ok or result.value is None:
        return [], [f"reference merging failed; labels kept unmerged: {'; '.join(result.errors)[:300]}"]
    decisions: list[ReferenceMergeDecision] = []
    covered: set[str] = set()
    for group in result.value.groups:
        members = [x for x in dict.fromkeys(group.members) if x in by_label and x not in covered]
        if not members:
            continue
        covered |= set(members)
        canonical_name = " ".join(group.canonical_name.split()) or members[0]
        if len(members) == 1 and canonical_name == members[0]:
            continue
        parsed = parse_identifier(canonical_name)
        changed = []
        for label in members:
            for m in by_label[label]:
                if _apply(m, parsed.canonical, parsed.revision, parsed.name):
                    changed.append(m.mention_id)
        if not changed:
            continue  # the group only restated keys that were already equal
        decisions.append(
            ReferenceMergeDecision(method="llm", members=members, canonical=canonical_name, applied=True, reason="model grouping", mention_ids=changed)
        )
    _dedupe_per_requirement(mentions)
    return decisions, []


# ---------------------------------------------------------------------------
# hybrid: rule-selected candidates, one targeted question each, guard rules
# ---------------------------------------------------------------------------


def _known_documents(mentions: list[MentionDraft], documents: list[InputDocument]) -> dict[str, dict]:
    """Documents with an identifier, keyed 'IDENT' (family) or 'IDENT@REV'."""
    known: dict[str, dict] = {}

    def add(canonical: str, revision: str | None, name: str | None) -> None:
        key = canonical if revision is None else f"{canonical}@{revision}"
        entry = known.setdefault(key, {"identifier": canonical, "revision": revision, "names": []})
        if name and name not in entry["names"]:
            entry["names"].append(name)

    for d in documents:
        if d.canonical_identifier:
            add(d.canonical_identifier, None, d.title)  # the document (any revision) ...
            if d.revision:
                add(d.canonical_identifier, d.revision, d.title)  # ... and the provided revision
    for m in mentions:
        if m.active and m.canonical:
            add(m.canonical, None, m.name)
    return known


def hybrid_merge(
    ctx: PipelineContext, mentions: list[MentionDraft], documents: list[InputDocument], requirement_text: dict[str, str]
) -> tuple[list[ReferenceMergeDecision], list[str]]:
    settings = ctx.options.reference_merging
    known = _known_documents(mentions, documents)
    decisions: list[ReferenceMergeDecision] = []
    candidates: list[dict] = []
    targets: dict[str, tuple[list[MentionDraft], dict[str, dict]]] = {}

    # (a) name-only labels -> documents whose names share content words
    names: dict[str, list[MentionDraft]] = {}
    for m in mentions:
        if m.active and m.canonical is None and m.name:
            names.setdefault(normalize_name(m.name), []).append(m)
    for i, (norm, ms) in enumerate(sorted(names.items()), start=1):
        words = _content_words(ms[0].name or "")
        scored = []
        for key, doc in known.items():
            overlap = max((len(words & _content_words(n)) / max(len(words), 1) for n in doc["names"]), default=0.0)
            if overlap > 0:
                scored.append((overlap, key))
        scored.sort(key=lambda x: (-x[0], x[1]))
        options = {f"O{j}": known[key] for j, (_, key) in enumerate(scored[: settings.max_candidates_per_name], start=1)}
        if not options:
            decisions.append(ReferenceMergeDecision(method="hybrid", members=[mention_label(ms[0])], canonical=mention_label(ms[0]), applied=False, reason="no known document shares a name word; left unresolved"))
            continue
        cid = f"N{i}"
        targets[cid] = (ms, options)
        candidates.append(
            {
                "candidate_id": cid,
                "label": ms[0].name,
                "requirement_sentence": ms[0].evidence.quoted_text or requirement_text.get(ms[0].requirement_id, "")[:300],
                "options": [{"option_id": oid, "identifier": d["identifier"], "revision": d["revision"], "names": d["names"]} for oid, d in options.items()],
            }
        )

    # (b) near-duplicate identifiers
    by_canon: dict[str, list[MentionDraft]] = {}
    for m in mentions:
        if m.active and m.canonical:
            by_canon.setdefault(m.canonical, []).append(m)
    canons = sorted(by_canon)
    for i, a in enumerate(canons):
        for b in canons[i + 1 :]:
            if squash(a) != squash(b):
                continue
            if settings.protect_identifiers:
                decisions.append(ReferenceMergeDecision(method="hybrid", members=[a, b], canonical=a, applied=False, reason="different identifiers kept apart (protect_identifiers)"))
                continue
            cid = f"I{len(targets) + 1}"
            targets[cid] = (by_canon[b], {"O1": {"identifier": a, "revision": None, "names": []}})
            candidates.append(
                {
                    "candidate_id": cid,
                    "label": b,
                    "requirement_sentence": by_canon[b][0].evidence.quoted_text,
                    "options": [{"option_id": "O1", "identifier": a, "names": known.get(a, {}).get("names", [])}],
                }
            )

    if not candidates:
        return decisions, []
    system, user = merging_prompt.ALIAS_SPEC.render({"candidates": candidates})
    result = ctx.llm.call(
        task="reference_alias_check",
        stage="merge_references",
        system=system,
        user=user,
        response_model=merging_prompt.AliasCheckResponse,
        prompt_version=merging_prompt.ALIAS_SPEC.fingerprint,
    )
    if not result.ok or result.value is None:
        return decisions, [f"reference alias check failed; nothing merged: {'; '.join(result.errors)[:300]}"]
    answers = {d.candidate_id: d for d in result.value.decisions}
    for cid, (ms, options) in targets.items():
        answer = answers.get(cid)
        label = mention_label(ms[0])
        if answer is None or not answer.same_document_as or answer.same_document_as not in options:
            decisions.append(ReferenceMergeDecision(method="hybrid", members=[label], canonical=label, applied=False, reason="model: no matching document"))
            continue
        target = options[answer.same_document_as]
        target_rev = normalize_revision(target["revision"])
        revs = {normalize_revision(m.revision) for m in ms if m.revision}
        if settings.protect_revisions and target_rev and revs and revs != {target_rev}:
            decisions.append(ReferenceMergeDecision(method="hybrid", members=[label], canonical=f"{target['identifier']} Rev {target_rev}", applied=False, reason="vetoed: different explicit revisions (protect_revisions)"))
            continue
        # With the guard on, a mention keeps the revision its own text states (none = the document);
        # with it off, the model's choice of revision wins.
        changed = [
            m.mention_id
            for m in ms
            if _apply(m, target["identifier"], m.revision if settings.protect_revisions else (target_rev or m.revision), None)
        ]
        decisions.append(
            ReferenceMergeDecision(
                method="hybrid",
                members=[label],
                canonical=target["identifier"],
                applied=bool(changed),
                reason=f"model: {answer.reason}"[:200] if answer.reason else "model: same document",
                mention_ids=changed,
            )
        )
    _dedupe_per_requirement(mentions)
    return decisions, []
