# CSR Relationship Lab

A research framework for measuring how structured, multi-stage processing and a JSON
document-relationship graph affect two tasks on customer-specific requirement (CSR)
documents:

1. **Referenced-document extraction**: finding every document each requirement references.
2. **Potential conflict detection**: flagging requirements that may be incompatible, with
   source evidence, for human review.

> **Research question.** To what extent do structured, multi-stage processing and JSON-based
> document relationship tracking improve the completeness and accuracy of referenced-document
> extraction and potential conflict identification in CSR documents, compared with a
> single-pass LLM approach?

This is an experimentation environment, not a single optimised evaluator. Strategies are
selectable, every strategy emits the same JSON schema, and all are scored by the same code
against hand-specified ground truth. Everything runs offline with a mock model; real models
are used through any OpenAI-compatible endpoint (a commercial API or a local `llama-server`).

All documents, companies and identifiers in this repository are fictional. No proprietary
CSR content, prompts or code are used (see [Academic separation](#academic-separation)).

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
pytest                                # 95 tests, ~5 s
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
| `csr-lab analyze FILES... --strategy graph_assisted --out result.json` | Analyse a document bundle (`.txt`, text `.pdf`, `.json`, or a folder) |
| `csr-lab experiment demo_mock` | Run a named experiment from `configs/experiments.yaml` |
| `csr-lab evaluate result.json gold.json` | Score one result against one gold file |
| `csr-lab graph result.json --shared --conflicts --mermaid g.mmd` | Query, validate or export the relationship graph |
| `csr-lab strategies` | List strategies, including YAML-defined variants |
| `csr-lab workflow --strategy staged_verified` | Print a strategy's LangGraph workflow as Mermaid |
| `csr-lab generate-data [--scaled 60 --refs 3 --position end]` | Regenerate the dataset, or build scaled cases for ablations |
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

See [docs/experiment_design.md](docs/experiment_design.md) for metrics, matching rules and
statistics guidance.

## Using a real model

Any OpenAI-compatible chat-completions server works. Settings live under `llm:` in
`configs/default.yaml` or in an experiment's `llm:` override.

**Local model with llama.cpp** (any GGUF model):

```bash
llama-server -m /path/to/model.gguf -c 16384 -ngl 99 --port 8080
csr-lab experiment local_llama_dev
```

**Commercial API**: put the key in `.env` (copy `.env.example`), set `model:` in the
`api_model_heldout` experiment, then `csr-lab experiment api_model_heldout`.

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

Each analysis produces one JSON document (schema version `0.1.0`), validated by Pydantic:

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
[data/examples/](data/examples/).

## Dataset

`data/synthetic/` holds 5 cases (7 documents) generated by
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

Use `dev` for debugging prompts and code. Do not edit `heldout` cases to suit a pipeline;
change `DATASET_VERSION` whenever content changes. The test suite fails if `data/synthetic/`
is out of date with the generator. See [data/public/README.md](data/public/README.md) for adding
public datasets such as PURE.

## Results and storage

`csr-lab experiment NAME` writes to `outputs/<experiment-id>/`:

- `report.md`: concise tables (references, conflicts, requirements, graph, cost, paired
  per-case comparison against the baseline, failures);
- `summary.json`, `per_case_metrics.csv`, `per_run_metrics.csv`, `llm_calls.csv`;
- `predictions/<strategy>/rep<k>/<case>.json`: every result.

The same records go to SQLite (`outputs/experiments.sqlite`; tables `experiments`, `runs`,
`case_results`, `metrics`, `llm_calls`) together with the configuration, dataset hash, prompt
hashes, git commit and package versions. `outputs/` is git-ignored.

## Architecture

```
src/csr_lab/
  schemas.py        export schema (integration contract) + gold schema
  config.py         typed YAML/env configuration
  ingestion/        loader (txt/pdf/json + header parsing), segmentation, chunking
  llm/              provider interface + validated structured output, OpenAI-compatible client, mock model
  prompts/          versioned prompt text + response models
  extraction/       grounding, normalisation, requirements, references, verification, graph audit
  relationships/    JSON graph builder, queries and flagging rules, structural validation
  conflicts/        candidate selection, pairwise detector, evidence validation
  pipelines/        strategies, registry, LangGraph workflows, nodes, final assembly
  evaluation/       matching, metrics, runner, reports
  storage/          SQLite experiment store
  datasets/         synthetic generator, manifest loader
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
- No OCR: scanned PDFs are reported as unsupported. PDF text is verbatim relative to `pypdf`'s
  extraction. `pypdf` (BSD) is used instead of PyMuPDF (AGPL) to keep a future industrial
  integration unencumbered.
- Revision-qualified references are separate documents from unqualified ones, and a reference
  to a provided document resolves only without a revision qualifier.
- Conflicts are always potential conflicts for review. The system never decides precedence or
  resolves a conflict; graph rules only point at where exclusion or precedence statements clash.
- The dataset is small; per-case paired comparisons are descriptive, not significance tests.

## Academic separation

This university project is independent of any employer's codebase. It contains no proprietary
CSR documents, customer data, employer prompts, department definitions, internal identifiers or
reproduced restricted standards. The architecture is designed independently so that a later
integration is possible but not required.
