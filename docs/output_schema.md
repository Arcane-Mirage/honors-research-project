# Output schema (version 0.3.0)

0.3.0 added `metadata.input_fidelity`, `metadata.reference_merges`, per-stage and per-unit
cost in `metadata.operational`, and the `merged` mention resolution. All additions are
optional fields; 0.2.0 consumers can ignore them.

The Pydantic models in `src/csr_lab/schemas.py` are the contract; the generated JSON Schema
is `docs/analysis_result.schema.json`. Unknown fields are rejected, and the top-level model
checks referential integrity (every referenced document, conflict requirement and edge
endpoint must exist).

## Identifiers

| Kind | Form | Example |
|---|---|---|
| Document (any revision) | `DOC-<identifier>` | `DOC-CVM-PKG-210` |
| One revision of a document | `DOC-<identifier>-REV-<rev>` | `DOC-CVM-PKG-210-REV-D`, `DOC-CVM-ES-210-REV-B` |
| Document known only by name | `DOC-NAME-<name>` | `DOC-NAME-CORVANE-SUPPLIER-QUALITY-MANUAL` |
| Requirement | `REQ-<document revision>-<chapter>-<text hash>` | `REQ-CVM-CSR-200-REV-A-1.3-4f240cf0` |
| Relationship | `REL-<hash of source, type, target>` | `REL-c829775f0e58` |
| Conflict | `CON-<hash of requirement pair>` | `CON-09f1275918` |

IDs are derived from content, so the same input and configuration give the same IDs.

**Documents and revisions.** An input document is the revision stated in its header
(`DOC-CVM-PKG-210-REV-D`); without a stated revision it gets the plain document ID and a
warning. A reference points at what the text names: "CVM-PKG-210" at the document node,
"CVM-PKG-210 Rev C" at that revision. The two levels are linked by `has_revision` edges, and
the document node's `current_revision` attribute says which revision an unrevisioned reference
currently means. A reference to the provided revision, or an unrevisioned one, has
`content_available: true`; a reference to a different revision does not.

**Requirements.** A requirement is identified by its source document revision, chapter and
verbatim text (whitespace-collapsed, hashed to 8 hex characters). The same requirement gets the
same ID in every run and every strategy. An exact duplicate within one chapter gets a `-2`
suffix. A missing chapter is written `NOCHAPTER`. Prompts show the model short keys (`R1`,
`A`/`B`), never these IDs.

## Top level

| Field | Meaning |
|---|---|
| `schema_version`, `analysis_id`, `status` | `status`: `completed`, `completed_with_errors` (some step failed; see `metadata.errors`), `failed` (no analysable input) |
| `input_documents` | Provided files: identifier and where it came from (`header`/`filename`/`manifest`), title, revision, hash, pages, parse status, warnings, extracted `text` |
| `requirements` | See below |
| `referenced_documents` | One entry per referenced document: `raw_identifiers` (spellings seen), `canonical_identifier`, `revision`, `document_name` (only if stated in the text), `document_type`, `content_available` |
| `relationships` | Graph edges (below) |
| `potential_conflicts` | See below |
| `coverage` | What was and was not examined (below) |
| `metadata` | Strategy, options, provider settings, prompt versions, seed, operational stats, verification report, conflict candidates, graph decisions, LLM calls |

## Requirement

- `chapter`: the label as printed (`4.2.1`, `TSA 2.1`); for a paragraph under a heading, the heading's label.
- `requirement_text`: the exact source slice when located (`source_location.match` is `exact`
  or `whitespace_normalized`); otherwise the model's text with `match: not_found` and no offsets.
- `referenced_document_ids`: always a list. Read it together with
  `extraction_metadata.reference_analysis_status`:
  - `verified` + `[]`: no explicit reference, and the result was checked;
  - `unverified`: extracted without a verification stage;
  - `failed`: reference analysis did not complete; an empty list means *unknown*, not *none*.
- `reference_mentions`: one per occurrence, with `raw_identifier`, `relation`
  (`references`, `excludes`, `supersedes`, `depends_on`, `takes_precedence_over`), `evidence`,
  `status` (`found_initially`, `added_during_verification`, `added_by_graph_audit`),
  `resolution` (`identifier`, `name_alias`, `unresolved_name`, `merged`) and `verified`.
  `merged` means the reference-merging stage renamed the mention (see `metadata.reference_merges`).
- `department`: `null` with `department_status: deferred`.

## Evidence

`quoted_text` plus `source_document_id`, `requirement_id`, offsets and `verified`. Verified
evidence was found in the extracted source text and its `quoted_text` is the exact slice.
`repaired: true` means a verification step replaced an unlocatable model quote with source text.
Model explanations are never stored as evidence.

## Relationships

| Type | From → to | Origin |
|---|---|---|
| `contains` | input document → requirement | `document_structure` |
| `references` | requirement → document (with mention evidence) | `extraction` |
| `explicitly_excludes` / `_supersedes` / `_depends_on` / `_takes_precedence_over` | requirement → document, only when the text states it | `extraction` |
| `references` (`derived: true`) | document → document, aggregated from its requirements | `aggregation` |
| `potentially_conflicts_with` | requirement → requirement (`analytical_hypothesis: true`) | `analysis` |
| `has_revision` | document → one of its revisions (`sequence` when the order is known) | `document_structure` |
| `superseded_by` | older → newer revision, and older → newer version of a requirement (`change`: unchanged / changed / moved). Only in a saved graph store. | `revision_tracking` |

`explicitly_supersedes` is only ever what the text states. `superseded_by` comes from revision
order (declared, or ISO effective dates), never from comparing revision labels as strings.

## Potential conflict

`related_requirement_ids` (two), `related_document_ids`, `related_relationship_ids` (graph
edges involved), `conflict_type`, a one-sentence `description`, `evidence` from both
requirements, `evidence_valid`, `assessment` (`apparent_incompatibility` or `ambiguous`),
`detection_methods` (e.g. `llm_pairwise`, `graph_applicability_rule`) and `review_status`, which
is always `requires_human_review`.

## Coverage

- `documents`: per input document, analysed / partially analysed / unsupported / failed, with
  character and segment counts.
- `referenced_documents`: `examined` when the content was provided, otherwise
  `not_examined_source_unavailable`. An unexamined document is never reported as conflict-free.
- `conflict_analysis.outcome`: `potential_conflicts_found`,
  `no_conflicts_detected_in_available_text`, or `not_examined`; plus pair counts and notes
  (e.g. when candidate selection skipped pairs).
- `limitations`: unavailable referenced documents, unresolved identifiers, truncated input,
  unsupported input, failed LLM calls, uncertain segmentation, oversized segments.
- `unresolved_identifiers`: names that could not be tied to an identifier.

## Input fidelity (`metadata.input_fidelity`)

Present when the analysed units were chunks (a chunk file, or `chunking.method: llm`); `null`
for csr-lab's own segmentation. Produced by `extraction/fidelity.py`, which needs no model.

- `fragments[]`: one per chunk: `match` (`exact`, `whitespace_normalized`, `fuzzy` = reworded,
  `not_found`), `fragment_ratio` (share of the chunk's words that are source words in order),
  `span_ratio` (share of the aligned source span the chunk kept), source offsets and pages,
  `reported_pages` and `pages_agree` when the chunker stated pages, and `differences` (up to 12
  edits: `replace`/`insert`/`delete` with the chunk text and the source text).
- `gaps[]`: source passages no chunk covered (`looks_like_requirement` when they contain
  shall/must/required).
- `summary`: counts per match kind, `verbatim_rate`, `mean_fragment_ratio`, `source_coverage`,
  gaps and page agreement.

Whether a reworded chunk was analysed as given or replaced by the aligned source text is the
strategy option `chunking.repair` (`none` / `source_slice`), recorded in `metadata.strategy_options`.

## Reference merges (`metadata.reference_merges`)

One entry per decision of the reference-merging stage (`reference_merging.method` `llm` or
`hybrid`): `method`, `members` (reference labels as extracted), `canonical` (what they were
renamed to), `applied`, `reason` (model reason, or the guard rule that vetoed it) and
`mention_ids`. Empty with the default `deterministic` method.

## Operational cost (`metadata.operational`)

Besides calls, retries, characters, tokens and seconds:

- `stage_costs`: per pipeline stage, calls, failed calls, characters, reported tokens and
  `estimated_tokens` (reported counts when the server sends them, else characters / 4);
- `token_source`: `reported` or `estimated_from_chars`;
- `pages` (PDF pages; for text input characters / 3000, see `pages_source`) and `requirements`
  (units in the result), with `calls_per_page`, `tokens_per_page`, `calls_per_requirement`,
  `tokens_per_requirement`.

## Pre-chunked input

A `.json` input with a `chunks` list is analysed with those chunks instead of csr-lab's
segmentation (format in `src/csr_lab/ingestion/provided_chunks.py`):

```json
{"source": {"file": "manual.pdf"},
 "chunks": [{"chunk_id": "c1", "title": "Scope", "content": "...", "source_pages": [1]}]}
```

`content`/`text`, `source_pages`/`pages` and `chunk_id`/`id` are accepted spellings. With a
source file the chunks are checked against it and requirement offsets refer to the source;
without one, the chunks themselves become the document text. The chunks are not repeated in
the exported `input_documents`.

## Integration notes

- The graph is plain JSON: nodes are implicit in `input_documents`, `requirements` and
  `referenced_documents`; edges are `relationships`. `csr-lab graph result.json --export g.json`
  writes an explicit `{nodes, edges}` file.
- `metadata` is large (it includes every LLM call); consumers that only need the content can
  ignore it. Set `processing.include_source_text: false` to omit document text.
