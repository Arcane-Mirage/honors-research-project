"""src/csr_lab/pipelines/base.py

Shared types for strategies: the run context and the strategy specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from csr_lab.config import AppConfig, PipelineOptions
from csr_lab.extraction.grounding import SourceIndex
from csr_lab.llm.interface import CallRecorder, LLMProvider, StructuredLLM
from csr_lab.relationships.builder import RelationshipGraph


@dataclass(frozen=True)
class StrategySpec:
    """A named, selectable experimental condition.

    ``options`` configure the built-in workflows. A strategy that needs a
    different workflow shape can supply ``builder``: a function that takes the
    options and returns a compiled LangGraph graph producing ``result``.
    """

    name: str
    description: str
    options: PipelineOptions
    builder: Callable[[PipelineOptions], Any] | None = None
    base: str | None = None


@dataclass
class PipelineContext:
    """Per-run objects that must not live in (serialisable) workflow state.

    Passed to LangGraph nodes as the runtime context. A new context, provider
    and recorder are created for every run so runs cannot share state.
    """

    config: AppConfig
    strategy: StrategySpec
    llm: StructuredLLM
    provider: LLMProvider
    recorder: CallRecorder
    sources: SourceIndex
    run_seed: int | None = None
    case_id: str | None = None
    # Documents from a saved graph (``GraphStore.document_graph()``); graph-assisted runs only.
    saved_graph: RelationshipGraph | None = None
    saved_graph_info: dict | None = None

    @property
    def options(self) -> PipelineOptions:
        return self.strategy.options
