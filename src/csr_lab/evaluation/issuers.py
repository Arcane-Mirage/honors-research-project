"""src/csr_lab/evaluation/issuers.py

Issuer classification against your own labels (``csr-lab issuer-eval``).

The ``issuer:`` entries of a bundle's documents.yaml are the gold: for loaded files and for
documents known only by identifier (cited documents you do not have). Each method (llm, hybrid,
rules) classifies the same documents with ``use_registry: false``, so the labels are neither
applied nor shown to the model; identifiers, titles and aliases from documents.yaml are still used.

Only the issuer stage runs (no reference extraction), so a run costs a few model calls. A loaded
document is shown with its first text (``header_chars``); a document known only by identifier
with the first sentence of the loaded documents that names it (``context_chars``), which is what
the pipeline shows for a cited document.

Scores: ``accuracy`` (correct / labelled), ``coverage`` (classified / labelled), ``precision``
(correct / classified). A label that is not a configured issuer (e.g. one only documents.yaml
names) cannot be chosen from the list; such documents are reported separately (``gold_known``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from csr_lab.config import AppConfig
from csr_lab.extraction.grounding import SourceIndex
from csr_lab.extraction.issuers import DocumentForIssuer, IssuerCatalog, classify_entries, collect_documents, family_key
from csr_lab.extraction.normalization import canonical_input_identifier
from csr_lab.ids import document_family_id
from csr_lab.ingestion.loader import DocumentRegistry
from csr_lab.llm.interface import CallRecorder, StructuredLLM
from csr_lab.llm.provider import create_provider
from csr_lab.pipelines.base import PipelineContext
from csr_lab.pipelines.registry import derive_strategy
from csr_lab.pipelines.run import load_inputs, registry_issuers
from csr_lab.schemas import InputDocument

METHODS = ("llm", "hybrid", "rules")


@dataclass
class IssuerEvalRow:
    document_id: str
    identifier: str | None
    loaded: bool
    gold: str
    gold_known: bool
    predicted: str | None
    source: str | None
    model_answer: str | None

    @property
    def correct(self) -> bool:
        return self.predicted == self.gold


@dataclass
class IssuerEvalResult:
    method: str
    rows: list[IssuerEvalRow]
    calls: int = 0
    failed_calls: int = 0
    tokens: int = 0
    token_source: str = "none"
    summary: dict[str, Any] = field(default_factory=dict)


def _phrase_re(text: str) -> re.Pattern | None:
    words = [w for w in re.split(r"[\s\-_/.]+", text) if w]
    if not words:
        return None
    return re.compile(r"(?<![A-Za-z0-9])" + r"[\s\-_/.]*".join(map(re.escape, words)) + r"(?![A-Za-z0-9])", re.IGNORECASE)


def citing_context(documents: list[InputDocument], spellings: list[str], chars: int) -> str | None:
    """The text around the first place a loaded document names one of ``spellings``."""
    if not chars:
        return None
    for spelling in spellings:
        pattern = _phrase_re(spelling)
        if pattern is None:
            continue
        for d in documents:
            if d.text and (m := pattern.search(d.text)):
                start = max(0, m.start() - chars // 2)
                return " ".join(d.text[start : start + chars].split())
    return None


def evaluation_entries(documents: list[InputDocument], registry: DocumentRegistry, chars: tuple[int, int]) -> list[DocumentForIssuer]:
    """Loaded documents, plus documents.yaml entries known only by identifier (with an issuer label)."""
    header_chars, context_chars = chars
    entries = collect_documents(documents, [], header_chars=header_chars, context_chars=context_chars)
    loaded = {e.key for e in entries}
    for e in registry.entries:
        if e.file or not e.identifier or not e.issuer:
            continue
        canonical = canonical_input_identifier(e.identifier)
        key = document_family_id(canonical)
        if key in loaded:
            continue
        spellings = list(dict.fromkeys([e.identifier, *e.names]))
        entries.append(
            DocumentForIssuer(
                key=key,
                document_ids=[key],
                identifier=canonical,
                names=spellings,
                context=citing_context(documents, spellings, context_chars),
            )
        )
        loaded.add(key)
    return entries


def evaluate_issuers(
    paths: list[str | Path], *, config: AppConfig, methods: list[str] | None = None, run_seed: int | None = None
) -> tuple[list[IssuerEvalResult], dict[str, Any]]:
    documents, registry = load_inputs(paths, config)
    if not registry:
        raise ValueError("no documents.yaml with issuer labels found next to the files")
    gold_raw = registry_issuers(registry, documents)
    if not gold_raw:
        raise ValueError("documents.yaml has no `issuer:` entries to evaluate against")
    catalog = IssuerCatalog(specs=list(config.issuers.known))
    gold = {k: catalog.resolve(v) or v for k, v in gold_raw.items()}  # 'TRATON Group' -> TRATON
    base = derive_strategy("issuer_eval", "staged")
    chars = (base.options.issuer_classification.header_chars, base.options.issuer_classification.context_chars)
    entries = [e for e in evaluation_entries(documents, registry, chars) if e.key in gold]
    loaded = {family_key(d.document_id, d.canonical_identifier) for d in documents}
    seed = run_seed if run_seed is not None else config.llm.seed
    results = []
    for method in methods or list(METHODS):
        spec = derive_strategy(f"issuer_eval_{method}", "staged", {"issuer_classification": {"method": method, "use_registry": False}})
        provider = create_provider(config.llm, run_seed=seed)
        recorder = CallRecorder()
        ctx = PipelineContext(
            config=config,
            strategy=spec,
            llm=StructuredLLM(provider, config.llm, recorder, seed=seed),
            provider=provider,
            recorder=recorder,
            sources=SourceIndex(documents),
            run_seed=seed,
            registry_issuers=gold_raw,  # ignored with use_registry: false; kept to show it is not leaked
        )
        try:
            outcome = classify_entries(ctx, entries)
        finally:
            provider.close()
        by_key = {a.document_id: a for a in outcome.assignments}
        rows = [
            IssuerEvalRow(
                document_id=e.key,
                identifier=e.identifier,
                loaded=e.key in loaded,
                gold=gold[e.key],
                gold_known=catalog.resolve(gold[e.key]) is not None and catalog.is_known(gold[e.key]),
                predicted=by_key[e.key].issuer,
                source=by_key[e.key].source,
                model_answer=by_key[e.key].model_answer,
            )
            for e in entries
        ]
        stats = recorder.stats(None, None)
        result = IssuerEvalResult(method=method, rows=rows, calls=stats.llm_calls, failed_calls=stats.llm_failed_calls, tokens=stats.estimated_tokens, token_source=stats.token_source)
        result.summary = summarize(rows)
        results.append(result)
    info = {"documents": len(documents), "labelled": len(entries), "model": getattr(config.llm, "model", None), "provider": config.llm.provider}
    return results, info


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def summarize(rows: list[IssuerEvalRow]) -> dict[str, Any]:
    def block(rs: list[IssuerEvalRow]) -> dict[str, Any]:
        classified = [r for r in rs if r.predicted]
        correct = [r for r in rs if r.correct]
        return {
            "labelled": len(rs),
            "classified": len(classified),
            "correct": len(correct),
            "accuracy": _rate(len(correct), len(rs)),
            "coverage": _rate(len(classified), len(rs)),
            "precision": _rate(len(correct), len(classified)),
        }

    return {
        "all": block(rows),
        "gold_in_list": block([r for r in rows if r.gold_known]),
        "loaded": block([r for r in rows if r.loaded]),
        "cited_only": block([r for r in rows if not r.loaded]),
    }


def _fmt(x: float | None) -> str:
    return "–" if x is None else f"{x:.0%}"


def report_markdown(results: list[IssuerEvalResult], info: dict[str, Any]) -> str:
    lines = [
        "# Issuer classification against documents.yaml labels",
        "",
        f"- Model: {info['provider']} / {info['model']}; {info['documents']} loaded files; {info['labelled']} labelled documents",
        "- The labels are used only for scoring: every method runs with `use_registry: false`, so the model never sees them.",
    ]
    if info["provider"] == "mock":
        lines.append("- **Offline mock model: these numbers test the machinery and are not evidence.**")
    lines += [
        "",
        "| Method | Accuracy | Coverage | Precision | Accuracy, label in list | Loaded files | Cited only | Calls (failed) | Tokens |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        s = r.summary
        lines.append(
            f"| {r.method} | {_fmt(s['all']['accuracy'])} ({s['all']['correct']}/{s['all']['labelled']}) | {_fmt(s['all']['coverage'])} | "
            f"{_fmt(s['all']['precision'])} | {_fmt(s['gold_in_list']['accuracy'])} | {_fmt(s['loaded']['accuracy'])} | {_fmt(s['cited_only']['accuracy'])} | "
            f"{r.calls} ({r.failed_calls}) | {r.tokens}{'' if r.token_source == 'reported' else ' est.'} |"
        )
    lines += ["", "Accuracy = correct / labelled; coverage = classified / labelled; precision = correct / classified.", ""]
    for r in results:
        wrong = [x for x in r.rows if not x.correct]
        lines += [f"## {r.method}: {len(wrong)} not matching the label", ""]
        if not wrong:
            lines.append("None.")
        else:
            lines += ["| Document | Label | Predicted | Source | Model answer |", "|---|---|---|---|---|"]
            for x in wrong:
                lines.append(f"| {x.identifier or x.document_id} | {x.gold}{'' if x.gold_known else ' (not in list)'} | {x.predicted or 'unclassified'} | {x.source or '–'} | {x.model_answer or '–'} |")
        lines.append("")
    return "\n".join(lines)


def result_rows(results: list[IssuerEvalResult]) -> list[dict[str, Any]]:
    return [
        {
            "method": r.method,
            "document_id": x.document_id,
            "identifier": x.identifier,
            "loaded": x.loaded,
            "gold": x.gold,
            "gold_in_list": x.gold_known,
            "predicted": x.predicted,
            "source": x.source,
            "model_answer": x.model_answer,
            "correct": x.correct,
        }
        for r in results
        for x in r.rows
    ]
