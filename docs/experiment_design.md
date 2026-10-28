# Experiment design

## Hypotheses and the comparisons that test them

| Hypothesis | Comparison | Primary metrics |
|---|---|---|
| H1 multi-stage processing improves reference recall and F1 | A `single_pass` vs B `staged` | reference recall, F1, complete-set accuracy |
| H2 verification and targeted re-examination reduce missed and wrong references | B vs C `staged_verified` | reference precision, recall, complete-set accuracy; verification counts |
| H3 a relationship graph improves traceability and may improve conflict identification | C vs D `graph_assisted`, plus D ablations | conflict recall/F1, candidate-pair recall, evidence validity, node resolution |
| H4 gains cost more tokens and time | all | LLM calls, input chars/tokens, seconds, cost, next to the metrics above |

### Comparisons against common industry practice

A widespread way to build a CSR evaluator is: fill a batch to N words, ask the model for
semantic chunks, ask the model for the referenced documents of each chunk, then ask the model
once to deduplicate the reference names. The `llm_chunked*` conditions and the
`reference_merging` options reproduce each of those steps with csr-lab's own prompts, so each
step can be compared with its alternative while everything else is held fixed:

| Question | Comparison | Experiment / sweep | Metrics to read |
|---|---|---|---|
| Does per-requirement extraction beat per-chunk extraction? | `llm_chunked` vs `llm_chunked_staged` (same chunks); `chunk_units` vs `staged_verified` on external chunks | `chunking_mock`, `external_chunks_mock`, sweep `chunk_size` | per-requirement F1 vs unit-level F1, `wrong_requirement` errors, gold req. per unit |
| Does checking/repairing chunk text against the source help? | `llm_chunked` vs `llm_chunked_repaired`; `staged_verified` vs `staged_verified_repaired` | `chunking_mock`, `external_chunks_mock` | input verbatim rate, requirement verbatim rate, gold coverage, requirement-like gaps |
| Is model-only reference merging safe? Do guard rules help? | `staged_verified` vs `_merge_llm` vs `_merge_hybrid` (and D) | `merging_mock`, sweep `merge_errors` | reference errors by cause (revision, confused identifier, unresolved alias), merges applied/rejected |
| How much does PDF handling matter? | text vs PDF dataset; PDF reader; header/footer removal | `pdf_mock`, sweeps `pdf_preprocessing`, `text_vs_pdf` | requirement recall/verbatim/location, spurious references |
| Does the presentation of graph facts matter? | `graph_assisted` (sentences) vs `_ctx_triples` / `_ctx_json` / `_ctx_ids` (original), and `staged_verified` (no graph) | `graph_context_local`, sweep `graph_context` | conflict F1/precision/recall, hard-negative FPs, cross-document recall, conflict-stage tokens |
| End to end | `llm_chunked_merge_llm` vs `staged_verified_merge_hybrid` / `graph_assisted` | `practice_vs_pipeline_mock` | all of the above, next to cost per page and per gold requirement |

None of these is assumed. With the mock model, A and B are identical by construction and the
remaining differences come only from deterministic mechanisms; real models are required to
evaluate the hypotheses.

## Controls

- Same chunk plan, prompts per stage, model and generation settings for every condition in an
  experiment. Model changes are separate experiments.
- Same final assembly and scoring code for every condition.
- The exported graph is built after predictions are final; only `graph.enabled` strategies
  use the graph during processing, and every such use is logged.
- The evaluator never passes gold data to a pipeline; the test suite checks that pipeline
  modules do not import the gold loader.
- Fresh provider, recorder and options per run; seeds are `base_seed + repetition`.

Known confound to keep in mind: later conditions make more calls (B adds pairwise conflict
calls; C adds targeted checks; D adds graph-informed checks and longer conflict prompts). The
cost table and the ablations help separate "more computation" from "better structure".

## Matching rules (summary)

1. Requirements are aligned one-to-one per document by text similarity or span overlap
   (threshold 0.75, configurable).
2. Documents are compared by canonical key: identifier plus stated revision
   (`CVM-ES-210@REV-B`). Name-only predictions (`NAME:...`) never match an identifier.
3. References are scored as (gold requirement, document key) pairs. A correct document on the
   wrong requirement is one false positive and one false negative. Duplicates count once and
   are reported separately.
4. Conflicts are unordered requirement pairs; hard-negative pairs are tracked separately.
5. Canonicalisation is applied to every strategy and to the evaluator alike, so formatting
   differences are not counted as errors. Normalisation therefore shows up in graph metrics
   (duplicate or unresolved nodes) rather than in reference F1.

Full detail: `src/csr_lab/evaluation/matching.py`.

## Metrics

| Group | Metrics |
|---|---|
| References (primary) | precision, recall, F1, complete reference-set accuracy (over all gold requirements and over matched ones), duplicates, relation-qualifier accuracy, availability accuracy, document-level precision/recall |
| Requirements | precision, recall, chapter accuracy, verbatim exact rate, location accuracy |
| Graph | structural validity, correct/missing/unsupported reference edges, duplicate edges, invalid endpoints, node resolution rate, unresolved identifiers, explicit-relation edge F1 |
| Conflicts | precision, recall, F1, false positives/negatives, candidate-pair recall, evidence validity rate, hard-negative false positives, ambiguous-case recall, type accuracy, cross-document recall |
| Units (granularity-independent) | units, gold requirements per unit, share of gold requirements inside some unit, unit-level precision/recall/F1 |
| Reference error causes | revision, identifier variant, unresolved alias, confused identifier, wrong requirement, spurious, missed; merges applied/rejected |
| Input fidelity (chunk units only) | chunks, verbatim rate, mean word ratio, source coverage, requirement-like gaps |
| Operations | total and per-stage seconds, LLM calls, failed calls, retries, JSON validation failures, input/output characters, tokens and cost when reported, graph-informed calls; calls and tokens per stage, per page, per gold requirement and per correct reference (tokens estimated as characters / 4 when the server reports none) |

Undefined ratios are `null`. Run-level values are micro-averaged (counts summed over cases).
Across repetitions the report gives mean ± sample standard deviation.

**Unit-level scoring.** A strategy that extracts references per chunk cannot say which
requirement references a document, so its strict per-requirement score is near zero by
construction. The unit-level score is the fair complement: a predicted (unit, document) pair
is correct when any gold requirement inside the unit references that document. Report both;
the gap between them is what per-requirement attribution adds.

**Error causes.** Within one gold requirement, a wrong key and the missing key it stands for
are paired and counted once: same identifier, other revision → `revision`; a title not linked
to its identifier → `unresolved_alias`; same identifier spelled differently → `identifier_variant`;
one character different → `confused_identifier`. Unpaired wrong keys are `wrong_requirement`
(a document of the case on the wrong requirement) or `spurious`; unpaired missing keys are `missed`.

**Graph context formats.** `graph_assisted` adds up to `max_context_items` graph facts to each
conflict pair. All formats state the same facts in the same order, with A and B for the pair's
two requirements: `sentences` ("Requirement A states that document CVM-PKG-050 does not
apply"), `triples` (`A | explicitly_excludes | CVM-PKG-050 | ...`) and `json` (objects).
`ids` reproduces the original rendering (raw IDs of other requirements, relation codes, no
merging of a plain reference into a qualified one), so the effect of the rewrite itself can be
measured. The mock model ignores graph context: mock runs only confirm that every format
reaches the prompt and what it costs. `csr-lab graph-context RESULT.json` prints the context
per pair in every format.

## Sweeps

`csr-lab sweep NAME` runs one experiment for every combination in a grid (`sweeps:` in
`configs/experiments.yaml`) and writes `sweep_report.md` / `sweep.csv`. Grid keys may be
experiment settings (`llm.*`, `processing.*`, `evaluation.*`), strategy options (`options.*`,
registered as variants per value) or `dataset`/`split`.

With the mock model, sweeping an error-injection rate (`llm.mock.merge_revision_drop_rate`,
`merge_confusion_rate`, ...) shows how each strategy *reacts* to that kind of model error.
That is a robustness argument, not evidence of how often real models make the error; measure
the rate itself with a real model (e.g. `input.verbatim_rate` of `llm_chunked`, or the merge
decisions of `_merge_llm`).

## Statistics

- Use repeated runs (`repetitions`) with real models; temperature 0 is still not always
  deterministic.
- Paired comparisons use the same held-out cases. The report's paired table is descriptive.
  With enough cases, apply a paired test (for example Wilcoxon signed-rank on per-case F1, or a
  paired bootstrap) to `per_case_metrics.csv`; do not claim significance from one run or a
  handful of cases.
- Report failed runs: the strict aggregate scores them as empty predictions; `valid_only`
  aggregates exclude them and are stored separately.

## Datasets and splits

- `dev`: for prompt and code debugging.
- `heldout`: for reported results; do not edit to favour a pipeline. Bump `DATASET_VERSION`
  for any content change (the dataset SHA-256 is stored with every experiment).
- Scaled cases (`csr-lab generate-data --scaled N --refs K --filler F --position start|middle|end`)
  support the planned ablations on document length, reference density and position.
- Input variants with the same cases and gold: `data/synthetic_pdf/` (`generate-data --pdf`:
  running headers, page footers, page breaks inside requirements, drawn tables; gold offsets are
  re-anchored to the extracted text when scoring) and `data/synthetic_chunked/`,
  `data/synthetic_chunked_window/` (`generate-data --chunked paragraph|window`: chunk files from a
  simulated external chunker; chunk text is copied verbatim).

## Planned ablations already supported by configuration

- graph-assisted without verification (`graph_assisted_no_verification`);
- one versus two verification rounds (`staged_verified_single_attempt`);
- individual graph uses (`graph_assisted_no_alias`, `graph_assisted_no_rules`, or
  `graph: {context_retrieval: false}` etc.);
- candidate selection without all-pairs comparison (`*_topic_candidates`);
- context size (`processing.chunk_max_chars`) and reference density/position (scaled data).
