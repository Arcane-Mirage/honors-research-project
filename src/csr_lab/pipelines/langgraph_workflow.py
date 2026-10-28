"""src/csr_lab/pipelines/langgraph_workflow.py

LangGraph workflow for the staged strategies (staged, staged_verified, graph_assisted, variants).

The graph *shape* is derived from ``PipelineOptions`` when the workflow is
built, so a strategy's compiled graph contains only the stages it uses.
Runtime decisions use conditional edges:

* after input preparation: unusable input -> finalize (failed result);
* after verification: candidates left and attempt budget remaining ->
  targeted re-extraction -> verification again (bounded, no infinite loop).

    prepare_input [-> llm_chunk]                            (chunking.method == llm)
      -> identify_requirements | chunks_as_requirements      (chunking.unit)
      -> extract_references
      [-> verify_references <-> targeted_reextraction]      (verification.enabled)
      [-> merge_references]                                 (reference_merging.method != deterministic)
      [-> build_graph -> graph_reference_audit]             (graph.enabled)
      [-> (build_graph) -> select_conflict_candidates -> analyze_conflicts]  (conflicts.enabled)
      -> finalize
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from csr_lab.config import PipelineOptions
from csr_lab.pipelines import nodes
from csr_lab.pipelines.base import PipelineContext
from csr_lab.pipelines.state import PipelineState


def build_staged_workflow(options: PipelineOptions):
    g = StateGraph(PipelineState, context_schema=PipelineContext)
    g.add_node("prepare_input", nodes.prepare_input)
    unit_node = "identify_requirements" if options.chunking.unit == "requirement" else "chunks_as_requirements"
    g.add_node(unit_node, nodes.identify_requirements if unit_node == "identify_requirements" else nodes.chunks_as_requirements)
    g.add_node("extract_references", nodes.extract_references_node)
    g.add_node("finalize", nodes.finalize)

    g.add_edge(START, "prepare_input")
    if options.chunking.method == "llm":
        g.add_node("llm_chunk", nodes.llm_chunk)
        g.add_conditional_edges("prepare_input", nodes.route_after_prepare, {"ok": "llm_chunk", "fatal": "finalize"})
        g.add_conditional_edges("llm_chunk", nodes.route_after_prepare, {"ok": unit_node, "fatal": "finalize"})
    else:
        g.add_conditional_edges("prepare_input", nodes.route_after_prepare, {"ok": unit_node, "fatal": "finalize"})
    g.add_edge(unit_node, "extract_references")

    use_graph = options.graph.enabled
    use_conflicts = options.conflicts.enabled
    needs_graph_node = use_graph and (options.graph.reference_audit or options.graph.alias_resolution or use_conflicts)

    # The stage that follows reference extraction (and verification, if enabled).
    if needs_graph_node:
        after_refs = "build_graph"
    elif use_conflicts:
        after_refs = "select_conflict_candidates"
    else:
        after_refs = "finalize"

    if options.reference_merging.method != "deterministic":
        g.add_node("merge_references", nodes.merge_references)
        g.add_edge("merge_references", after_refs)
        after_refs = "merge_references"

    if options.verification.enabled:
        g.add_node("verify_references", nodes.verify_references)
        g.add_node("targeted_reextraction", nodes.targeted_reextraction_node)
        g.add_edge("extract_references", "verify_references")
        g.add_conditional_edges(
            "verify_references", nodes.route_after_verification, {"reextract": "targeted_reextraction", "done": after_refs}
        )
        g.add_edge("targeted_reextraction", "verify_references")
    else:
        g.add_edge("extract_references", after_refs)

    if needs_graph_node:
        g.add_node("build_graph", nodes.build_graph_node)
        after_graph = "select_conflict_candidates" if use_conflicts else "finalize"
        if options.graph.reference_audit or options.graph.alias_resolution:
            g.add_node("graph_reference_audit", nodes.graph_reference_audit)
            g.add_edge("build_graph", "graph_reference_audit")
            g.add_edge("graph_reference_audit", after_graph)
        else:
            g.add_edge("build_graph", after_graph)

    if use_conflicts:
        g.add_node("select_conflict_candidates", nodes.select_conflict_candidates)
        g.add_node("analyze_conflicts", nodes.analyze_conflicts)
        g.add_edge("select_conflict_candidates", "analyze_conflicts")
        g.add_edge("analyze_conflicts", "finalize")

    g.add_edge("finalize", END)
    return g.compile(name="staged_workflow")
