"""src/csr_lab/pipelines/baseline.py

LangGraph workflow for the single-pass baseline.

    prepare_input -> single_pass_extract [-> merge_references] [-> classify_issuers] -> finalize

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
    after_refs = "finalize"
    if options.issuer_classification.enabled:
        g.add_node("classify_issuers", nodes.classify_issuers_node)
        g.add_edge("classify_issuers", "finalize")
        after_refs = "classify_issuers"
    if options.reference_merging.method != "deterministic":
        g.add_node("merge_references", nodes.merge_references)
        g.add_edge("single_pass_extract", "merge_references")
        g.add_edge("merge_references", after_refs)
    else:
        g.add_edge("single_pass_extract", after_refs)
    g.add_edge("finalize", END)
    return g.compile(name="single_pass_workflow")
