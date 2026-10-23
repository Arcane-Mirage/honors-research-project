"""The document relationship graph (plain JSON) and deterministic queries over it.

This graph describes *content* (requirements, documents, and their relations).
It is independent of the LangGraph *workflow* graph that runs the pipeline and
can be built from any ``AnalysisResult``, including one produced elsewhere.
"""
