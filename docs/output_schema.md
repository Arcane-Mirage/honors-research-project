# Output schema (version 0.1.0)

The Pydantic models in `src/csr_lab/schemas.py` are the contract; the generated JSON Schema
is `docs/analysis_result.schema.json`. Unknown fields are rejected, and the top-level model
checks referential integrity (every referenced document, conflict requirement and edge
endpoint must exist).

## Identifiers

| Kind | Form | Example |
|---|---|---|
| Input or referenced document | `DOC-<identifier>[-REV-<rev>]` | `DOC-CVM-PKG-210`, `DOC-CVM-ES-210-REV-B` |
| Document known only by name | `DOC-NAME-<name>` | `DOC-NAME-CORVANE-SUPPLIER-QUALITY-MANUAL` |
| Requirement | `REQ-<document>-<nnn>` in source order | `REQ-CVM-CSR-200-008` |
| Relationship | `REL-<hash of source, type, target>` | `REL-c829775f0e58` |
| Conflict | `CON-<hash of requirement pair>` | `CON-09f1275918` |

IDs are derived from content, so the same input and configuration give the same IDs.

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
  `resolution` (`identifier`, `name_alias`, `unresolved_name`) and `verified`.
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

A newer revision is never taken to supersede an older one unless the text says so.

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

## Integration notes

- The graph is plain JSON: nodes are implicit in `input_documents`, `requirements` and
  `referenced_documents`; edges are `relationships`. `csr-lab graph result.json --export g.json`
  writes an explicit `{nodes, edges}` file.
- `metadata` is large (it includes every LLM call); consumers that only need the content can
  ignore it. Set `processing.include_source_text: false` to omit document text.
