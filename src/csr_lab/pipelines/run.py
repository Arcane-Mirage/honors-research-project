"""Run one strategy on one document bundle."""

from __future__ import annotations

import time
from pathlib import Path

from csr_lab.config import AppConfig
from csr_lab.extraction.grounding import SourceIndex
from csr_lab.ingestion.loader import load_bundle
from csr_lab.llm.interface import CallRecorder, LLMProvider, StructuredLLM
from csr_lab.llm.provider import create_provider
from csr_lab.pipelines.base import PipelineContext, StrategySpec
from csr_lab.pipelines.registry import build_workflow, get_strategy
from csr_lab.pipelines.state import initial_state
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
) -> AnalysisResult:
    """Execute a strategy. A fresh provider, recorder and context are used for every call."""
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
    workflow = build_workflow(spec)
    started = time.perf_counter()
    try:
        final = workflow.invoke(initial_state(documents, case_id), context=ctx, config={"recursion_limit": RECURSION_LIMIT})
    finally:
        if own_provider:
            provider.close()
    result: AnalysisResult = final["result"]
    result.metadata.operational.total_seconds = round(time.perf_counter() - started, 4)
    return result


def analyze_files(
    paths: list[str | Path], *, strategy: str, config: AppConfig, run_seed: int | None = None
) -> AnalysisResult:
    documents = load_bundle(paths, min_chars_per_page=config.processing.pdf_min_chars_per_page)
    if not documents:
        raise ValueError("no supported input files found")
    return run_analysis(documents, strategy=strategy, config=config, run_seed=run_seed)
