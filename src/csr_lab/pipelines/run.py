"""src/csr_lab/pipelines/run.py

Run one strategy on one document bundle.
"""

from __future__ import annotations

import time
from pathlib import Path

from csr_lab.config import AppConfig
from csr_lab.extraction.grounding import SourceIndex
from csr_lab.ingestion.loader import load_bundle, processing_load_options
from csr_lab.llm.interface import CallRecorder, LLMProvider, StructuredLLM
from csr_lab.llm.provider import create_provider
from csr_lab.pipelines.base import PipelineContext, StrategySpec
from csr_lab.pipelines.registry import build_workflow, get_strategy
from csr_lab.pipelines.state import initial_state
from csr_lab.relationships.store import GraphStore
from csr_lab.schemas import AnalysisResult, InputDocument

RECURSION_LIMIT = 100


def run_analysis(
    documents: list[InputDocument],
    *,
    strategy: str | StrategySpec,
    config: AppConfig,
    case_id: str | None = None,
    run_seed: int | None = None,
    provider: LLMProvider | None = None,
    saved_graph: GraphStore | None = None,
    saved_graph_path: str | Path | None = None,
) -> AnalysisResult:
    """Execute a strategy. A fresh provider, recorder and context are used for every call.

    ``saved_graph``: a persistent graph whose *documents* (identifiers, names,
    revisions) a graph-assisted strategy may use for alias resolution and the
    known-document audit. Ignored, with a warning, by strategies without graph use.
    The run never writes to it; merge the result afterwards if wanted.
    """
    spec = get_strategy(strategy) if isinstance(strategy, str) else strategy
    spec = StrategySpec(  # private copy of the options: runs cannot affect each other
        name=spec.name,
        description=spec.description,
        options=spec.options.model_copy(deep=True),
        builder=spec.builder,
        base=spec.base,
    )
    seed = run_seed if run_seed is not None else config.llm.seed
    own_provider = provider is None
    provider = provider or create_provider(config.llm, run_seed=seed)
    recorder = CallRecorder()
    ctx = PipelineContext(
        config=config,
        strategy=spec,
        llm=StructuredLLM(provider, config.llm, recorder, seed=seed),
        provider=provider,
        recorder=recorder,
        sources=SourceIndex(documents),
        run_seed=seed,
        case_id=case_id,
    )
    warnings: list[str] = []
    if saved_graph is not None:
        if spec.options.graph.enabled:
            ctx.saved_graph = saved_graph.document_graph()
            ctx.saved_graph_info = {
                "path": str(saved_graph_path) if saved_graph_path else None,
                "fingerprint": saved_graph.fingerprint(),
                "documents": len(ctx.saved_graph.nodes),
                "sources": len(saved_graph.sources),
            }
        else:
            warnings.append(f"saved graph ignored: strategy {spec.name} does not use the graph")
    workflow = build_workflow(spec)
    started = time.perf_counter()
    try:
        final = workflow.invoke(initial_state(documents, case_id), context=ctx, config={"recursion_limit": RECURSION_LIMIT})
    finally:
        if own_provider:
            provider.close()
    result: AnalysisResult = final["result"]
    result.metadata.warnings += warnings
    result.metadata.operational.total_seconds = round(time.perf_counter() - started, 4)
    return result


def analyze_files(
    paths: list[str | Path],
    *,
    strategy: str,
    config: AppConfig,
    run_seed: int | None = None,
    saved_graph: GraphStore | None = None,
    saved_graph_path: str | Path | None = None,
) -> AnalysisResult:
    documents = load_bundle(paths, **processing_load_options(config.processing))
    if not documents:
        raise ValueError("no supported input files found")
    return run_analysis(
        documents, strategy=strategy, config=config, run_seed=run_seed, saved_graph=saved_graph, saved_graph_path=saved_graph_path
    )
