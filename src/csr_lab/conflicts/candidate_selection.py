"""src/csr_lab/conflicts/candidate_selection.py

Choose which requirement pairs are sent to conflict analysis.

Non-graph selectors (available to every staged strategy):
* ``all_pairs`` when the number of pairs is at most ``all_pairs_max``;
* otherwise ``attribute_overlap``: shared technical words and units (a simple
  topic match), keeping the best-scoring ``max_candidate_pairs`` pairs.

Graph selectors (graph-assisted strategies only) are *added* to the non-graph
set, never used alone, because requirements can conflict without sharing a
referenced document:
* ``shared_reference:<doc>``     both requirements reference the same document;
* ``relation_statement:<rule>``  an exclusion or precedence statement links them;
* ``linked_documents:<a>-><b>``  their documents reference each other and the
  texts share technical vocabulary.

Every selected pair records which selectors chose it, so candidate recall can
be analysed per selector.
"""

from __future__ import annotations

import re
from itertools import combinations

from csr_lab.config import ConflictSettings, GraphSettings
from csr_lab.drafts import CandidatePair, RequirementDraft, ordered_pair
from csr_lab.relationships.builder import RelationshipGraph
from csr_lab.relationships.queries import (
    find_applicability_conflicts,
    find_precedence_conflicts,
    requirements_in_linked_documents,
    requirements_sharing_references,
)
from csr_lab.schemas import GraphDecision

_STOP = set(
    """the a an and or of to in on for with by be is are shall must not no any all each this that these those as at
    from per its it which when where than then into under over such other only also may can will been being has have
    had their there was were within before after least more most less according accordance document supplier
    suppliers requirement requirements""".split()
)
_UNIT_RE = re.compile(r"°C|°F|\bmm\b|\bkg\b|%|µg|\bhours?\b|\bdays?\b|\byears?\b|\bminutes?\b|\blux\b")


def _terms(text: str) -> set[str]:
    terms = set()
    for w in re.findall(r"[a-z][a-z-]+", text.lower()):
        if w in _STOP or len(w) < 3:
            continue
        for suffix in ("ing", "ed", "es", "s"):
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                w = w[: -len(suffix)]
                break
        terms.add(w)
    return terms


def attribute_score(a: str, b: str) -> float:
    ta, tb = _terms(a), _terms(b)
    if not ta or not tb:
        return 0.0
    score = len(ta & tb) / len(ta | tb)
    units = set(_UNIT_RE.findall(a)) & set(_UNIT_RE.findall(b))
    return score + 0.05 * len(units)


def select_candidates(
    requirements: list[RequirementDraft],
    settings: ConflictSettings,
    *,
    graph: RelationshipGraph | None = None,
    graph_settings: GraphSettings | None = None,
) -> tuple[list[CandidatePair], int, list[str], list[GraphDecision]]:
    """Return (candidates, total possible pairs, notes, graph decisions)."""
    ids = [r.requirement_id for r in requirements]
    texts = {r.requirement_id: r.text for r in requirements}
    total = len(ids) * (len(ids) - 1) // 2
    notes: list[str] = []
    selected: dict[tuple[str, str], CandidatePair] = {}

    if total <= settings.all_pairs_max:
        for a, b in combinations(ids, 2):
            pair = ordered_pair(a, b)
            selected[pair] = CandidatePair(requirement_ids=pair, selectors=["all_pairs"])
    else:
        scored = []
        for a, b in combinations(ids, 2):
            score = attribute_score(texts[a], texts[b])
            if score >= settings.attribute_min_score:
                scored.append((score, ordered_pair(a, b)))
        scored.sort(key=lambda x: (-x[0], x[1]))
        for score, pair in scored[: settings.max_candidate_pairs]:
            selected[pair] = CandidatePair(requirement_ids=pair, selectors=[f"attribute_overlap:{score:.2f}"])
        notes.append(
            f"{total} possible pairs exceed all_pairs_max={settings.all_pairs_max}; "
            f"{len(selected)} pairs selected by attribute overlap, the rest were not analysed"
        )

    decisions: list[GraphDecision] = []
    if graph is not None and graph_settings is not None and graph_settings.enabled and graph_settings.candidate_selection:
        known = set(ids)
        graph_pairs: dict[tuple[str, str], list[str]] = {}
        for a, b, docs in requirements_sharing_references(graph, max_fanout=graph_settings.max_shared_reference_fanout):
            if a in known and b in known:
                graph_pairs.setdefault(ordered_pair(a, b), []).extend(f"shared_reference:{d}" for d in docs)
        rule_flags = []
        if graph_settings.applicability_rules:
            rule_flags += find_applicability_conflicts(graph, graph_settings.related_document_hops)
        if graph_settings.precedence_rules:
            rule_flags += find_precedence_conflicts(graph)
        for flag in rule_flags:
            a, b = flag.requirement_ids
            if a in known and b in known:
                graph_pairs.setdefault(ordered_pair(a, b), []).append(f"relation_statement:{flag.rule}")
        for a, b, doc_a, doc_b in requirements_in_linked_documents(graph):
            if a in known and b in known and attribute_score(texts[a], texts[b]) >= settings.attribute_min_score / 2:
                graph_pairs.setdefault(ordered_pair(a, b), []).append(f"linked_documents:{doc_a}->{doc_b}")

        added = 0
        for pair, selectors in sorted(graph_pairs.items()):
            selectors = list(dict.fromkeys(selectors))
            if pair in selected:
                selected[pair].selectors.extend(selectors)
                continue
            if added >= settings.max_candidate_pairs:
                continue
            added += 1
            selected[pair] = CandidatePair(requirement_ids=pair, selectors=selectors, graph_derived=True)
            decisions.append(
                GraphDecision(
                    decision_type="conflict_candidate_added",
                    detail=f"pair added by {', '.join(selectors)}",
                    node_ids=list(pair),
                )
            )
        if added:
            notes.append(f"{added} additional pairs selected from the relationship graph")

    return [selected[p] for p in sorted(selected)], total, notes, decisions
