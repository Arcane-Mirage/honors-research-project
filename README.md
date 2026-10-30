# CSR Lab

A research framework for measuring how structured, multi-stage processing and a JSON
document-relationship graph affect two tasks on customer-specific requirement (CSR)
documents:

1. **Referenced-document extraction**: finding every document each requirement references.
2. **Potential conflict detection**: flagging requirements that may be incompatible, with
   source evidence, for human review.

> **Research question.** To what extent do structured, multi-stage processing and
> document relationship tracking improve the completeness and accuracy of referenced-document
> extraction and potential conflict identification in CSR documents, compared with a
> single-pass LLM approach?

This is an experimentation environment, not a single optimised evaluator. Strategies are
selectable, every strategy emits the same JSON schema, and all are scored by the same code
against hand-specified ground truth. Everything runs offline with a mock model; real models
are used through any OpenAI-compatible endpoint (a commercial API or a local LM Studio server).

---

## Quick start

Requires Python 3.11 or later.

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.lock      # exact, pinned versions
pip install -e . --no-deps            # this package (editable)
pytest                                # ~156 tests, ~5 s
python scripts/run_demo.py            # offline end-to-end demo
```

**Windows (PowerShell)**

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.lock
pip install -e . --no-deps
pytest
python scripts\run_demo.py
```

Never install the dependencies globally. Every command below assumes the virtual
environment is active (or call `.venv/bin/python -m csr_lab ...` directly).

The lockfile was produced with `pip-compile` on Python 3.14/Linux. On another platform pip may
add a platform-only dependency (for example `colorama` on Windows); the pinned versions are
otherwise identical. To change dependencies, edit `pyproject.toml` and run:

```bash
pip-compile --extra dev --strip-extras --no-emit-index-url --output-file requirements.lock pyproject.toml
```

## Commands

| Command | What it does |
|---|---|
| `csr-lab llms` / `llms models lmstudio` / `llms use lmstudio` | List LLM profiles, show what a server has loaded, change the default |
| `csr-lab build-graph [FOLDER] [--graph PATH] [--llm lmstudio]` | Analyse the PDFs in `input/` (or FOLDER) and build up a saved graph |
| `csr-lab analyze FILES... --strategy graph_assisted --out result.json` | Analyse a document bundle (`.txt`, text `.pdf`, `.json`, or a folder) |
| `csr-lab experiment dev --llm lmstudio` | Run a named experiment from `configs/experiments.yaml` with an LLM profile |
| `csr-lab evaluate result.json gold.json` | Score one result against one gold file |
| `csr-lab graph result.json --shared --conflicts --mermaid g.mmd --llm-text g.md` | Query, validate or export the relationship graph |
| `csr-lab analyze FILES... --graph PATH [--no-use-graph] [--no-update-graph]` | Also use and/or update a saved graph |
| `csr-lab graph-store info\|merge\|revisions\|prune\|remove-source\|export ... PATH` | Inspect, add to, prune and export a saved graph |
| `csr-lab strategies` | List strategies, including YAML-defined variants |
| `csr-lab workflow --strategy staged_verified` | Print a strategy's LangGraph workflow as Mermaid |
| `csr-lab generate-data [--scaled 60 --refs 3 --position end]` | Regenerate the dataset, or build scaled cases for ablations |
| `csr-lab generate-data --pdf` / `--chunked paragraph\|window` | Regenerate the PDF / pre-chunked variants of the dataset (same cases and gold) |
| `csr-lab sweep NAME` | Run an experiment over a grid of settings (`sweeps:` in `configs/experiments.yaml`) |
| `csr-lab check-text SOURCE CHUNKS.json` | Check chunks/quotes against a source: verbatim, reworded (with the edits), missing text. No model |
| `csr-lab chunk FILES --out-dir DIR` | Run only the model chunking stage and write chunk files |
| `csr-lab llm-check [--llm NAME \| --experiment NAME]` | Show the resolved endpoint, auth and served models; send one test request |
| `csr-lab schema --out schema.json` | JSON Schema of the exported result |
| `csr-lab history` | List experiments stored in SQLite |

`python -m csr_lab ...` works the same as `csr-lab ...`.

## Experimental conditions

Each condition adds one thing to the previous one, so adjacent pairs isolate one effect.

| Condition | Name | Adds | Comparison isolates |
|---|---|---|---|
| A | `single_pass` | One structured call per chunk returns requirements, references and conflicts | baseline |
| B | `staged` | Separate requirement, reference and pairwise-conflict stages | A→B: task decomposition |
| C | `staged_verified` | Evidence validation, distractor/duplicate checks, coverage audit, bounded targeted re-extraction, conflict-evidence check | B→C: explicit verification |
| D | `graph_assisted` | Graph-informed alias resolution, known-document audit, candidate selection, context retrieval, exclusion/precedence rules | C→D: graph-informed processing |

Important for interpretation:

- All conditions get the same chunks, the same model settings and the same scoring. Every
  condition exports a relationship graph, but only D uses it to make decisions. Graph use is
  logged (`metadata.graph_decisions`, `graph_informed` on each LLM call).
- Later conditions make more model calls. Cost is reported next to accuracy; an
  improvement may come from extra computation rather than structure.
- An improvement in D comes from specific graph *uses* (each can be switched off), not from
  LangGraph, which only orchestrates the steps. Ready-made ablations are in
  `configs/experiments.yaml` (`ablation_mock`, `candidate_selection_mock`).

A second family starts from **model-made chunks**, the common industry approach (fill a batch
to N words, ask the model for semantic chunks, extract references per chunk):

| Name | Chunks | Unit of extraction | Chunk text |
|---|---|---|---|
| `llm_chunked` | model | chunk | as returned |
| `llm_chunked_repaired` | model | chunk | replaced by the aligned source text when reworded |
| `llm_chunked_staged` | model | requirements inside each chunk | as returned |
| `llm_chunked_merge_llm` (variant) | model | chunk | as returned, plus model-only reference merging |

Pre-chunked input (a chunk file made by any tool) replaces the chunks of every strategy, so
the same comparisons run on another pipeline's chunks. See
[docs/experiment_design.md](docs/experiment_design.md) for metrics, matching rules and
statistics guidance, and [docs/integration.md](docs/integration.md) for reusing the pieces in
another evaluator.

## Transfer experiments: is each improvement worth it?

Each experiment changes one step and keeps everything else fixed. Run any of them offline
with the mock model, or with a real model by overriding `llm:`.

| Experiment / sweep | Question | Read in `report.md` |
|---|---|---|
| `practice_vs_pipeline_mock` | Industry-style pipeline vs staged + verified + guarded merging, end to end | all tables; cost per page / gold requirement |
| `chunking_mock`, sweep `chunk_size` | Per-chunk vs per-requirement extraction from the same model chunks | per-requirement vs unit-level F1, error causes |
| `external_chunks_mock` | The same on chunk files from a simulated external chunker (verbatim chunks) | gold coverage, verbatim rate, unit-level F1 |
| `merging_mock`, sweep `merge_errors` | Model-only vs guarded (hybrid) reference merging | reference errors by cause, merges applied/rejected |
| `pdf_mock`, sweeps `pdf_preprocessing`, `text_vs_pdf` | PDF reader and header/footer removal | requirement verbatim/location accuracy, spurious references |

```bash
csr-lab experiment practice_vs_pipeline_mock
csr-lab sweep merge_errors
csr-lab check-text data/synthetic/chunked/documents/dev_corvane_packaging/cvm_csr_200.txt \
                   data/synthetic/chunked/documents/dev_corvane_packaging/cvm_csr_200.chunks.json
```

**Knobs** (strategy options, set in a variant under `strategies:` or swept with `options.*`):
`chunking.method` (`deterministic`/`llm`), `chunking.unit` (`requirement`/`chunk`),
`chunking.repair` (`none`/`source_slice`), `chunking.batch_words`, `chunking.target_chunk_words`,
`reference_merging.method` (`deterministic`/`llm`/`hybrid`), `reference_merging.protect_revisions`,
`reference_merging.protect_identifiers`. Processing settings (shared by all strategies):
`processing.pdf_backend` (`pypdf`/`pypdf_layout`/`pymupdf`), `processing.pdf_strip_repeated_lines`.
Mock error rates for the new model tasks: `llm.mock.merge_revision_drop_rate`, `merge_confusion_rate`, `merge_alias_miss_rate`.

## Choosing the LLM

Any OpenAI-compatible chat-completions server works. Connections are named **profiles** in
[configs/llms.yaml](configs/llms.yaml): `mock` (offline), `lmstudio`, `openai`,
`azure`. Every command that calls a model takes `--llm NAME`:

```bash
csr-lab llms                          # list profiles; * marks the default
csr-lab llms models lmstudio          # what LM Studio has downloaded / loaded (+ quantisation)
csr-lab llm-check --llm lmstudio      # one test request
csr-lab experiment dev --llm lmstudio
csr-lab build-graph --llm lmstudio
csr-lab llms use lmstudio             # make it the default (or CSR_LAB_LLM=lmstudio in .env)
```

**LM Studio**: Developer tab > Start Server (or `lms server start`), load one model. The
`lmstudio` profile has `model: auto`, so whichever model is loaded is used and its real name is
stored with every result; switching models needs no config change. If several models are
loaded, `auto` stops and lists them: unload one, or pass `--model KEY`. LM Studio's default
context length is often too small for these prompts; load models with 16k+ context.

Which profile applies:

1. `--llm NAME` on the command line (for experiments it replaces the experiment's `llm:`);
2. otherwise the experiment's own `llm:` (a profile name like `llm: lmstudio`, or settings);
3. otherwise `$CSR_LAB_LLM`, else `default:` in `configs/llms.yaml`.

`--model` and `--base-url` adjust the chosen profile for one run. The experiments `dev` and
`heldout` have no `llm:` of their own, so they run with whatever profile you select.

**Commercial API**: key in `.env` as `OPENAI_API_KEY`, set `model:` in the
`openai` profile. **Azure OpenAI**: fill in the `azure` profile. Keyless (Microsoft Entra ID):
`auth: {method: azure_identity}` after `pip install azure-identity`; API key:
`auth: {method: env, header: api-key, scheme: null}`. `auth: {method: command, command: ...}`
uses any command that prints a token.

Structured output uses the server's JSON-schema mode when available. If the server rejects
it, the client switches to schema-in-prompt mode and records a warning; replies are always
validated against the Pydantic model, retried a bounded number of times, and recorded as
failures if still invalid. Token counts and cost are stored only when the server reports them.

Keep the model fixed when comparing strategies. A model change is a separate variable:
run it as a separate experiment.

## Adding a strategy

Without code, as a variant in `configs/experiments.yaml`:

```yaml
strategies:
  graph_assisted_no_context:
    base: graph_assisted
    description: Condition D without graph context in conflict prompts
    options: {graph: {context_retrieval: false}}
```

In Python, register a `StrategySpec` (optionally with a custom `builder` returning a compiled
LangGraph workflow) with `csr_lab.pipelines.registry.register_strategy`. The runner and
evaluator need no changes because every strategy returns an `AnalysisResult`.

## Output

Each analysis produces one JSON document (schema version `0.3.0`), validated by Pydantic:

- `requirements`: chapter, verbatim text, source offsets, `referenced_document_ids`, per-mention
  evidence, and `department: null` (assignment is deliberately deferred);
- `referenced_documents`: canonical identifier, raw spellings, revision, type, `content_available`;
- `relationships`: graph edges (`contains`, `references`, `explicitly_excludes`,
  `explicitly_takes_precedence_over`, `potentially_conflicts_with`, ...) with evidence;
- `potential_conflicts`: requirement pair, type, evidence from both sides,
  `review_status: requires_human_review` (the only allowed value);
- `coverage`: what was not examined, e.g. referenced documents whose content was not provided;
- `metadata`: strategy, options, model, prompt versions, seeds, initial vs verified references,
  graph decisions, every LLM call, cost.

Field-level documentation: [docs/output_schema.md](docs/output_schema.md). JSON Schema:
[docs/analysis_result.schema.json](docs/analysis_result.schema.json). Example (mock) output:
[docs/examples/](docs/examples/).

## Dataset

`data/synthetic/text/` holds 5 cases (7 documents) generated by
[src/csr_lab/datasets/synthetic.py](src/csr_lab/datasets/synthetic.py):

| Split | Cases | Requirements | References | Conflict pairs | Hard negatives |
|---|---|---|---|---|---|
| dev | 2 | 27 | 25 | 7 | 5 |
| heldout | 3 | 45 | 42 | 10 (2 ambiguous) | 8 |

Covered situations include multiple and repeated references, tables, revisions, spacing and
naming aliases, section-like identifiers, distractor numbers, long requirements, unavailable
external documents, nested references between provided documents, exclusions contradicted by a
referenced document, mutual precedence claims, cross-document conflicts and non-conflicting
overlaps. Ground truth is written by hand in the generator; offsets are computed while
rendering, so they always match the files.

Input variants of the same cases and gold, for parser and chunker experiments:
`data/synthetic/pdf/` (PDFs with running headers, page footers, page breaks inside
requirements and drawn tables), `data/synthetic/chunked/` and `data/synthetic/chunked_window/`
(chunk files from a simulated external chunker: verbatim paragraph packing, or
blind 60-word windows).

Use `dev` for debugging prompts and code. Do not edit `heldout` cases to suit a pipeline;
change `DATASET_VERSION` whenever content changes. The test suite fails if `data/synthetic/`
is out of date with the generator. See [data/README.md](data/README.md) for the folder layout
and for adding public datasets such as PURE.

## Results and storage

`csr-lab experiment NAME` writes to `outputs/<experiment-id>/`:

- `report.md`: concise tables (references, conflicts, requirements, graph, cost, paired
  per-case comparison against the baseline, failures);
- `summary.json`, `per_case_metrics.csv`, `per_run_metrics.csv`, `llm_calls.csv`;
- `predictions/<strategy>/rep<k>/<case>.json`: every result.

The same records go to SQLite (`outputs/experiments.sqlite`; tables `experiments`, `runs`,
`case_results`, `metrics`, `llm_calls`) together with the configuration, dataset hash, prompt
hashes, git commit and package versions. `outputs/` is git-ignored.

## Saved (persistent) graph

Every run builds a graph for its own documents. A **graph store** (one JSON file,
[src/csr_lab/relationships/store.py](src/csr_lab/relationships/store.py)) merges those graphs
so knowledge builds up across runs. It is kept apart from the SQLite file: SQLite records
experiment runs, while the store is reusable knowledge that a CSR evaluator can curate.

**Building a graph from your own PDFs.** Drop files into [input/](input/) (each subfolder is
analysed as one bundle, so a CSR and the documents it references belong together), then:

```bash
csr-lab build-graph --dry-run                         # what would run, what is already in the graph
csr-lab build-graph --llm lmstudio                    # input/ -> graphs/main.json
csr-lab build-graph ~/csr_docs --graph graphs/tessaly.json --llm lmstudio
csr-lab build-graph --graph graphs/main.json --no-update-graph   # use it, do not change it
```

| Setting | Flag | Config (`graph_store:` in default.yaml) |
|---|---|---|
| Which graph | `--graph PATH` | `path: graphs/main.json` |
| Graph-assisted strategies start from it | `--use-graph` / `--no-use-graph` | `use: true` |
| Merge the results into it | `--update-graph` / `--no-update-graph` | `update: true` |
| Write the views below | `--no-views` | `write_views: true` |

Files already merged (same bytes) are skipped, so re-running after adding files only analyses
the new ones (`--force` redoes them). The graph is saved after each bundle. Each build writes
`outputs/graph_builds/<time>-<graph>/build_report.md` (graph size before/after, and per bundle:
requirements, references, conflicts, nodes/edges added, model calls, tokens, seconds), plus each
bundle's result JSON. `input/` contents and `graphs/` are git-ignored.

**Graph formats.** JSON is the graph itself: lossless, validated, queryable, and what the
pipeline reads back. Next to `graphs/main.json`, every update writes:

- `main.overview.mmd`: Mermaid, one box per document; the one to show people;
- `main.detail.mmd`: Mermaid with every requirement;
- `main.llm.md`: Markdown for an LLM: documents by name and revision, each requirement's
  verbatim text with what it refers to, and potential conflicts kept separate as hypotheses.
  The same view: `csr-lab graph result.json --llm-text g.md`, `graph-store export PATH --llm-text g.md`.

Same graph, different rendering: for prompting, plain sentences or Markdown are usually better
than raw JSON (fewer tokens, no opaque IDs to join). Which rendering helps *your* model is
measurable: the `graph_context_local` experiment compares sentence, triple, JSON and ID
renderings of the graph context in conflict prompts.

**Curating.**

```bash
csr-lab analyze docs/rev_d/ --graph graphs/main.json              # use + merge one bundle
csr-lab graph-store info graphs/main.json                          # size, validation
csr-lab graph-store sources graphs/main.json                       # every analysis, with files and model
csr-lab graph-store revisions graphs/main.json DOC-CVM-CSR-100     # what changed
csr-lab graph-store remove-source graphs/main.json ANL-...         # undo one analysis
csr-lab graph-store prune graphs/main.json --superseded --mock     # clean up
```

- **Provenance:** every node and edge lists the analyses that produced it; `remove-source`
  undoes one.
- **Revisions:** revisions of a document are ordered by `set-revision-order`, else by ISO
  effective dates. Consecutive revisions are compared chapter by chapter (unchanged, changed,
  moved, removed, added). Older requirements become `superseded`, not deleted.
- **Re-analysis:** analysing a document revision again replaces its stored requirements
  (`merge --accumulate` keeps both readings).
- **`--use-graph`:** graph-assisted strategies may use the store's *documents*
  (identifiers, names, revisions) for alias resolution and the known-document audit. The
  exported result still contains only the run's own documents; `metadata.saved_graph` records
  which store (fingerprint) was used. Other strategies ignore it with a warning.
- The experiment runner never reads or writes a store, so experiment runs stay independent.

## Architecture

```
src/csr_lab/
  schemas.py        export schema (integration contract) + gold schema
  config.py         typed YAML/env configuration
  ingestion/        loader (txt/pdf/json/chunk files), segmentation, chunk units, model chunking
  llm/              provider interface + validated structured output, OpenAI-compatible client, auth, mock model
  prompts/          versioned prompt text + response models
  extraction/       grounding, fidelity check, normalisation, requirements, references, verification, merging, graph audit
  relationships/    JSON graph builder, queries and flagging rules, structural validation
  conflicts/        candidate selection, pairwise detector, evidence validation
  pipelines/        strategies, registry, LangGraph workflows, nodes, final assembly
  evaluation/       matching, metrics, gold re-anchoring, runner, sweeps, reports
  storage/          SQLite experiment store
  datasets/         synthetic generator, PDF and chunked variants, manifest loader
```

There are two different graphs. The **LangGraph workflow graph** orders processing steps. The
**document relationship graph** is the JSON artifact describing requirements, documents and
their relations; it is plain data, can be rebuilt from any result, and does not depend on
LangGraph. Details and diagrams: [docs/architecture.md](docs/architecture.md).

## Limitations and assumptions

- **Mock results are not evidence.** The mock model uses heuristics with seeded error
  injection applied per item, never per strategy. It cannot show benefits of task
  decomposition (A and B score identically by construction); it shows that verification and
  graph mechanisms work. Hypotheses must be tested with real models.
- Department assignment is deferred: the field exists and is always `null`.
- No OCR: scanned PDFs are reported as unsupported. PDF text is verbatim relative to the chosen
  reader's extraction. `pypdf` (BSD) is the default and the only PDF dependency; the optional
  `pymupdf` reader is AGPL-licensed, so check its licence before using it in an industrial setting.
- Mock error rates for chunking and merging are set by hand. Sweeps over them show how
  strategies react to those errors, not how often a real model makes them.
- Each revision of a document is its own node under a document-level node. An unrevisioned
  reference is resolved to a revision only when it is queried; with several provided revisions
  and no dates or declared order, no revision is treated as current.
- Conflicts are always potential conflicts for review. The system never decides precedence or
  resolves a conflict; graph rules only point at where exclusion or precedence statements clash.
- The dataset is small; per-case paired comparisons are descriptive, not significance tests.
