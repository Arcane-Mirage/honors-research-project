# Experiment design

## Hypotheses and the comparisons that test them

| Hypothesis | Comparison | Primary metrics |
|---|---|---|
| H1 multi-stage processing improves reference recall and F1 | A `single_pass` vs B `staged` | reference recall, F1, complete-set accuracy |
| H2 verification and targeted re-examination reduce missed and wrong references | B vs C `staged_verified` | reference precision, recall, complete-set accuracy; verification counts |
| H3 a relationship graph improves traceability and may improve conflict identification | C vs D `graph_assisted`, plus D ablations | conflict recall/F1, candidate-pair recall, evidence validity, node resolution |
| H4 gains cost more tokens and time | all | LLM calls, input chars/tokens, seconds, cost, next to the metrics above |

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
| Operations | total and per-stage seconds, LLM calls, failed calls, retries, JSON validation failures, input/output characters, tokens and cost when reported, graph-informed calls |

Undefined ratios are `null`. Run-level values are micro-averaged (counts summed over cases).
Across repetitions the report gives mean ± sample standard deviation.

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

## Planned ablations already supported by configuration

- graph-assisted without verification (`graph_assisted_no_verification`);
- one versus two verification rounds (`staged_verified_single_attempt`);
- individual graph uses (`graph_assisted_no_alias`, `graph_assisted_no_rules`, or
  `graph: {context_retrieval: false}` etc.);
- candidate selection without all-pairs comparison (`*_topic_candidates`);
- context size (`processing.chunk_max_chars`) and reference density/position (scaled data).
