"""src/csr_lab/extraction/issuers.py

Issuer classification: who published each document of a bundle, so documents can be grouped by
issuer (``AnalysisResult.issuers``, the saved graph and its views). It runs once per bundle after
the references are final, because the cited documents are only known then.

Each document is classified once for all its revisions (its family node, ``DOC-CVS10``). The first
source that answers wins:

1. ``registry``: ``issuer:`` in documents.yaml (for a file, or for a document known only by identifier);
2. ``saved_graph``: the issuer the saved graph already has for it (graph-assisted runs, ``reuse_saved``);
3. ``rules``: the known issuers' ``patterns`` on its identifier, raw spellings and names
   (methods ``rules`` and ``hybrid``);
4. ``llm``: batched model calls (methods ``llm`` and ``hybrid``). The prompt lists the known issuers
   (config ``issuers.known``) plus those added earlier in the run or found in the saved graph; the
   model answers with one of them, a new name (``allow_new``), or 'unknown'. Batches run one after
   another so that a new issuer named for one document is offered for the next ones, and the same
   publisher is not named two ways.

Answers are matched to the known names leniently (case, punctuation, aliases, the name without its
parenthesis: 'European Standards' -> 'European Standards (EN/DIN)'). An answer that matches none
becomes a new issuer when ``allow_new`` is on, else the document stays unclassified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from csr_lab.config import IssuerSpec
from csr_lab.drafts import MentionDraft, PipelineError
from csr_lab.ids import document_family_id
from csr_lab.prompts import issuers as issuer_prompt
from csr_lab.schemas import CoverageLimitation, InputDocument, IssuerAssignment

if TYPE_CHECKING:
    from csr_lab.pipelines.base import PipelineContext
    from csr_lab.relationships.builder import RelationshipGraph

_NO_ANSWER = {"", "unknown", "none", "n/a", "na", "other", "unclear", "not known", "null"}
_MAX_NEW_NAME = 60


def match_key(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "", name.casefold())


def family_key(document_id: str, canonical: str | None) -> str:
    return document_family_id(canonical) if canonical else document_id


@dataclass
class IssuerCatalog:
    """The issuers a run can assign: the configured ones, then those added during the run."""

    specs: list[IssuerSpec]
    added: list[str] = field(default_factory=list)
    _lookup: dict[str, str] = field(default_factory=dict, init=False)
    _patterns: list[tuple[str, list[re.Pattern]]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        for spec in self.specs:
            for n in [spec.name, *spec.aliases]:
                self._register(n, spec.name)
        for spec in self.specs:  # a customer's names mean its issuer ('Audi' -> Volkswagen Group), after the issuers' own names
            for c in spec.customers:
                for n in [c.name, *(a for a in c.aliases if a not in c.ambiguous_aliases)]:
                    self._register(n, spec.name)
        self._patterns = [(s.name, [re.compile(p, re.IGNORECASE) for p in s.patterns]) for s in self.specs]
        for name in list(self.added):
            self._register(name, name)

    def _register(self, spelling: str, name: str) -> None:
        for key in {match_key(spelling), match_key(spelling.split(" (", 1)[0])}:
            if key:
                self._lookup.setdefault(key, name)

    @property
    def known_names(self) -> list[str]:
        return [s.name for s in self.specs]

    @property
    def names(self) -> list[str]:
        return [*self.known_names, *self.added]

    def is_known(self, name: str) -> bool:
        return name in self.known_names

    def resolve(self, answer: str | None) -> str | None:
        """The catalog name an answer means, or None when it matches none."""
        if not answer:
            return None
        return self._lookup.get(match_key(answer)) or self._lookup.get(match_key(answer.split(" (", 1)[0]))

    def add(self, name: str) -> str:
        """Add a new issuer (or return the existing name it matches)."""
        if existing := self.resolve(name):
            return existing
        self.added.append(name)
        self._register(name, name)
        return name

    def by_rules(self, texts: list[str]) -> str | None:
        """The first issuer whose pattern matches one of ``texts`` (tried in order: identifier first)."""
        for text in texts:
            for name, patterns in self._patterns:
                if any(p.search(text) for p in patterns):
                    return name
        return None

    def prompt_entries(self) -> list[dict]:
        entries = [
            {"name": s.name, "description": s.description, "examples": s.examples, **({"customers": [c.name for c in s.customers]} if s.customers else {})}
            for s in self.specs
        ]
        return entries + [{"name": n, "description": "added for an earlier document", "examples": []} for n in self.added]


def clean_new_name(answer: str) -> str | None:
    name = " ".join(answer.split()).strip(" .,;:\"'")
    if match_key(name) in {match_key(x) for x in _NO_ANSWER} or not name:
        return None
    return name[:_MAX_NEW_NAME]


@dataclass
class DocumentForIssuer:
    """One document (family) as the classifier sees it."""

    key: str
    document_ids: list[str] = field(default_factory=list)
    identifier: str | None = None
    names: list[str] = field(default_factory=list)  # titles, aliases, raw spellings
    header: str | None = None  # first text of an input document
    context: str | None = None  # a sentence citing a referenced document
    is_input: bool = False
    registry_issuer: str | None = None

    def rule_texts(self) -> list[str]:
        return [t for t in [self.identifier, *self.names] if t]


def collect_documents(
    documents: list[InputDocument], mentions: list[MentionDraft], *, header_chars: int, context_chars: int
) -> list[DocumentForIssuer]:
    """Input documents and every document the active mentions point at, one entry per family."""
    out: dict[str, DocumentForIssuer] = {}

    def entry(document_id: str, canonical: str | None) -> DocumentForIssuer:
        key = family_key(document_id, canonical)
        e = out.setdefault(key, DocumentForIssuer(key=key, identifier=canonical))
        e.identifier = e.identifier or canonical
        if document_id not in e.document_ids:
            e.document_ids.append(document_id)
        if key not in e.document_ids:
            e.document_ids.append(key)
        return e

    for d in documents:
        e = entry(d.document_id, d.canonical_identifier)
        e.is_input = True
        e.names = list(dict.fromkeys([*e.names, *(n for n in [d.title, *d.aliases] if n)]))
        if d.issuer and d.issuer_source == "registry":
            e.registry_issuer = d.issuer
        if d.text and header_chars and not e.header:
            e.header = " ".join(d.text[: header_chars * 2].split())[:header_chars]
    for m in mentions:
        if not m.active:
            continue
        e = entry(m.document_id, m.canonical)
        spellings = [" ".join(m.raw_identifier.split()), m.name]
        e.names = list(dict.fromkeys([*e.names, *(s for s in spellings if s)]))[:6]
        if context_chars and not e.context and m.evidence.quoted_text:
            e.context = " ".join(m.evidence.quoted_text.split())[:context_chars]
    return list(out.values())


def saved_issuers(graph: RelationshipGraph | None) -> dict[str, str]:
    """Family node -> issuer, from a saved graph's document nodes."""
    if graph is None:
        return {}
    out: dict[str, str] = {}
    for n in graph.nodes:
        issuer = n.attributes.get("issuer")
        if n.node_type == "document" and issuer:
            out.setdefault(graph.family_of(n.node_id), issuer)
    return out


@dataclass
class IssuerOutcome:
    assignments: list[IssuerAssignment]
    catalog: IssuerCatalog
    errors: list[PipelineError] = field(default_factory=list)
    limitations: list[CoverageLimitation] = field(default_factory=list)


def classify_issuers(ctx: PipelineContext, documents: list[InputDocument], mentions: list[MentionDraft]) -> IssuerOutcome:
    settings = ctx.options.issuer_classification
    entries = collect_documents(documents, mentions, header_chars=settings.header_chars, context_chars=settings.context_chars)
    return classify_entries(ctx, entries)


def classify_entries(ctx: PipelineContext, entries: list[DocumentForIssuer]) -> IssuerOutcome:
    """Classify prepared entries (``collect_documents``; ``csr-lab issuer-eval`` builds its own).

    With ``use_registry: false`` the documents.yaml issuers are neither applied nor offered to the
    model, so they can serve as gold."""
    settings = ctx.options.issuer_classification
    catalog = IssuerCatalog(specs=list(ctx.config.issuers.known))
    registry = dict(ctx.registry_issuers) if settings.use_registry else {}
    if not settings.use_registry:
        entries = [DocumentForIssuer(**{**e.__dict__, "registry_issuer": None}) for e in entries]
    saved = saved_issuers(ctx.saved_graph) if settings.reuse_saved else {}
    in_use = {n for n in [*registry.values(), *saved.values(), *(e.registry_issuer for e in entries)] if n}
    for name in sorted(in_use):
        catalog.add(name)  # names already in use are offered to the model, so groups stay consistent
    preexisting = {catalog.resolve(n) for n in in_use}

    done: dict[str, IssuerAssignment] = {}

    def settle(e: DocumentForIssuer, issuer: str | None, source, *, new: bool = False, answer: str | None = None) -> None:
        done[e.key] = IssuerAssignment(
            document_id=e.key,
            document_ids=sorted(e.document_ids),
            issuer=issuer,
            source=source if issuer else None,
            new_issuer=new,
            model_answer=answer if answer is not None and answer != issuer else None,
        )

    ask: list[DocumentForIssuer] = []
    for e in entries:
        registered = e.registry_issuer or registry.get(e.key)
        if registered:
            settle(e, catalog.resolve(registered) or catalog.add(registered), "registry")
        elif e.key in saved:
            settle(e, catalog.resolve(saved[e.key]) or saved[e.key], "saved_graph")
        elif settings.method in ("rules", "hybrid") and (hit := catalog.by_rules(e.rule_texts())):
            settle(e, hit, "rules")
        elif settings.method == "rules":
            settle(e, None, None)
        else:
            ask.append(e)

    errors: list[PipelineError] = []
    limitations: list[CoverageLimitation] = []
    for start in range(0, len(ask), settings.batch_size):
        batch = ask[start : start + settings.batch_size]
        keys = {f"D{i}": e for i, e in enumerate(batch, start=1)}
        payload = {
            "new_issuers_allowed": settings.allow_new,
            "issuers": catalog.prompt_entries(),
            "documents": [
                {
                    "key": k,
                    "identifier": e.identifier,
                    "names": e.names,
                    **({"header": e.header} if e.is_input and e.header else {}),
                    **({"context": e.context} if e.context and not (e.is_input and e.header) else {}),
                }
                for k, e in keys.items()
            ],
        }
        system, user = issuer_prompt.SPEC.render(payload)
        result = ctx.llm.call(
            task="issuer_classification",
            stage="classify_issuers",
            system=system,
            user=user,
            response_model=issuer_prompt.IssuerClassificationResponse,
            prompt_version=issuer_prompt.SPEC.fingerprint,
        )
        if not result.ok or result.value is None:
            for e in batch:
                settle(e, None, None)
            errors.append(PipelineError(stage="classify_issuers", message=f"issuer classification failed for {len(batch)} documents: {'; '.join(result.errors)[:300]}"))
            limitations.append(CoverageLimitation(kind="llm_call_failed", message=f"issuer classification failed; {len(batch)} documents have no issuer"))
            continue
        answers = {a.key: a for a in result.value.results if a.key in keys}
        for k, e in keys.items():
            a = answers.get(k)
            raw = (a.issuer if a else "") or ""
            if (name := catalog.resolve(raw)) is not None:
                settle(e, name, "llm", new=not catalog.is_known(name) and name not in preexisting, answer=raw)
            elif settings.allow_new and (new := clean_new_name(raw)):
                settle(e, catalog.add(new), "llm", new=True, answer=raw)
            else:
                settle(e, None, None, answer=raw or None)
    return IssuerOutcome(assignments=[done[e.key] for e in entries], catalog=catalog, errors=errors, limitations=limitations)


def annotate_graph(graph: RelationshipGraph | None, assignments: list[IssuerAssignment]) -> RelationshipGraph | None:
    """A copy of the working graph with ``issuer`` / ``issuer_source`` on its document nodes."""
    if graph is None:
        return None
    by_node = {i: a for a in assignments if a.issuer for i in a.document_ids}
    graph = graph.model_copy(deep=True)
    for n in graph.nodes:
        a = by_node.get(n.node_id) or by_node.get(graph.family_of(n.node_id))
        if n.node_type == "document" and a is not None:
            n.attributes["issuer"] = a.issuer
            n.attributes["issuer_source"] = a.source
    graph.reindex()
    return graph
