# Architecture

## Processing flow

```
input files ──► loader ──► segmentation ──► chunk plan  (deterministic, shared by all strategies)
                                               │
                     ┌─────────────────────────┴────────────────────────┐
             single_pass                                          staged family
     one LLM call per chunk returns                 identify_requirements (LLM, per chunk)
     requirements + references + conflicts          extract_references   (LLM, batches)
                     │                              [verify_references ⇄ targeted_reextraction]   C, D
                     │                              [build_graph → graph_reference_audit]          D
                     │                              select_conflict_candidates → analyze_conflicts (LLM, pair batches)
                     └─────────────────────────┬────────────────────────┘
                                         finalize / assembly  (same code for all)
                          grounding annotations · referenced documents · export graph · coverage
                                               │
                                        AnalysisResult JSON
```

Print the exact compiled workflow of any strategy with `csr-lab workflow --strategy NAME`
(examples in `data/examples/workflow_*.mmd`).

## Two graphs

| | LangGraph workflow graph | Document relationship graph |
|---|---|---|
| Purpose | order of processing steps, retries, conditional routing | requirements, documents and their relations |
| Code | `pipelines/langgraph_workflow.py`, `pipelines/baseline.py` | `relationships/` |
| Lifetime | one run | exported JSON; rebuilt from any `AnalysisResult` |
| Depends on LangGraph | yes | no |

## Modules and responsibilities

| Module | Responsibility |
|---|---|
| `ingestion/loader.py` | Read `.txt`/`.pdf`/`.json`; detect `Document ID`, title, revision; record page spans and warnings. File reading and header parsing are combined because the parser is a few regexes. |
| `ingestion/segmentation.py` | Propose requirement blocks with chapter labels (`4.2.1`, `TSA 2.1`, heading + paragraph), section path and exact offsets. Never emits headings or front matter as candidates. |
| `ingestion/chunking.py` | Section-aware chunks, overlap context, truncation limits (recorded, never silent). |
| `llm/interface.py` | Provider interface; `StructuredLLM` (JSON-schema / JSON / prompt modes, validation, bounded retries, call records). |
| `llm/provider.py` | OpenAI-compatible HTTP client with transport retries and structured-output fallback. |
| `llm/mock_provider.py`, `llm/mock_rules.py` | Offline mock model: heuristic answers + seeded error injection. |
| `prompts/*` | Versioned prompt text with the response model it requests. Version and content hash are recorded. |
| `extraction/grounding.py` | Locate quotes (exact, then whitespace-insensitive) and build `Evidence`. |
| `extraction/normalization.py` | Conservative identifier canonicalisation used by every strategy and by the evaluator. |
| `extraction/requirements.py` | Ground model-reported requirements to source slices, merge overlap duplicates, assign stable IDs. |
| `extraction/references.py` | Build mentions from model output; staged reference-extraction calls. |
| `extraction/verification.py` | Deterministic checks, coverage audit, targeted re-extraction (Condition C+). |
| `extraction/graph_audit.py` | Alias resolution and known-document sweep using the graph (Condition D). |
| `relationships/builder.py` | Build the graph: `contains`, `references`, explicit-relation edges, document-level aggregates, conflict hypotheses. |
| `relationships/queries.py` | The query functions, applicability and precedence flagging rules, Mermaid export. |
| `relationships/validation.py` | Structural checks: endpoints, duplicates, ID consistency, evidence presence. |
| `conflicts/candidate_selection.py` | All pairs, topic overlap, and (D only) graph-derived pairs, with provenance. |
| `conflicts/detector.py` | Pairwise LLM analysis, evidence validation/repair, graph-rule conflicts, merging. |
| `pipelines/strategies.py`, `registry.py` | Built-in conditions (one file because B–D share one workflow and differ in options) and the registry. |
| `pipelines/nodes.py`, `state.py` | LangGraph nodes and typed state. |
| `pipelines/assembly.py` | Drafts → `AnalysisResult`; builds the export graph from final predictions only. |
| `evaluation/*` | Matching rules, metrics, experiment runner, reports. |
| `storage/database.py` | SQLite experiment store. |

## Workflow state and context

`PipelineState` (a `TypedDict`) stores the input documents once; everything else refers to
them by ID and offset. List fields such as `errors`, `limitations`, `verification_events` and
`graph_decisions` accumulate through reducers. The LLM client, provider and call recorder are
passed as LangGraph runtime context (`PipelineContext`), not state, so state stays
serialisable and each run gets fresh objects. LangGraph checkpointing is not used in this
version; experiment records live in SQLite.

Conditional edges:

- `prepare_input` → `finalize` when no input has analysable text (result `status: failed`);
- `verify_references` → `targeted_reextraction` while candidates remain and the attempt budget
  (`verification.max_attempts`) is not used up; then onward. The loop cannot run forever.

## How Condition D uses the graph

1. **Alias resolution**: a name-only mention is resolved to a document node when exactly one
   node carries that name (from another requirement's "Name ID" evidence or a provided
   document's title).
2. **Known-document sweep**: requirements are searched for identifiers/names of graph documents
   they are not linked to; hits get a targeted model check with graph context.
3. **Candidate selection**: shared references, linked documents and exclusion/precedence
   statements add pairs to the non-graph candidate set (never replace it).
4. **Context retrieval**: conflict prompts include short, evidence-backed graph statements.
5. **Rules**: deterministic flags for exclusions contradicted by a related document's reference
   and for mutual precedence claims.

Each use can be disabled under `graph:` in the strategy options.

## Failure handling

- Invalid model output: retried with the validation error, then recorded as a failed call.
  A requirement whose reference extraction failed is marked `failed`, distinct from "no
  references".
- Recoverable errors (one chunk or batch) are recorded in `metadata.errors` and
  `coverage.limitations`; the result status becomes `completed_with_errors`.
- In experiments, an exception in one case is stored with its traceback and the runner
  continues. Failed cases count as empty predictions in the strict aggregate.
