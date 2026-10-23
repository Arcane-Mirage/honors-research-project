"""LangGraph workflow for the single-pass baseline.

    prepare_input -> single_pass_extract -> finalize

Input preparation and finalisation are the same nodes the staged workflows
use, so the baseline receives the same chunks and is scored from the same
output format. Malformed model output is retried inside ``StructuredLLM``.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from csr_lab.config import PipelineOptions
from csr_lab.pipelines import nodes
from csr_lab.pipelines.base import PipelineContext
from csr_lab.pipelines.state import PipelineState


def build_single_pass_workflow(options: PipelineOptions):
    g = StateGraph(PipelineState, context_schema=PipelineContext)
    g.add_node("prepare_input", nodes.prepare_input)
    g.add_node("single_pass_extract", nodes.single_pass_extract)
    g.add_node("finalize", nodes.finalize)
    g.add_edge(START, "prepare_input")
    g.add_conditional_edges("prepare_input", nodes.route_after_prepare, {"ok": "single_pass_extract", "fatal": "finalize"})
    g.add_edge("single_pass_extract", "finalize")
    g.add_edge("finalize", END)
    return g.compile(name="single_pass_workflow")
