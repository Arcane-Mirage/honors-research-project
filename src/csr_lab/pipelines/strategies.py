"""src/csr_lab/pipelines/strategies.py

Built-in experimental conditions.

The three staged conditions share one workflow and differ only in their options,
so the controlled differences are visible side by side in this file.

    condition          verification   conflict evidence check   graph use
    single_pass        -              -                         -
    staged             -              -                         -
    staged_verified    yes            yes                       -
    graph_assisted     yes            yes                       yes

Each condition adds one thing to the previous one, so adjacent conditions
can be compared to isolate decomposition (A->B), verification (B->C) and
graph-informed processing (C->D).

A second family starts from *model-made* chunks instead of chapters, the common industry approach
(fill a batch to N words, ask the model for semantic chunks, extract references
per chunk). It is the realistic baseline to compare the conditions above against:

    condition              units                chunk text      isolates
    llm_chunked            model chunks         as returned     baseline (per-chunk extraction)
    llm_chunked_repaired   model chunks         source slice    effect of the fidelity repair

All other conditions use whole chapters as units; every unit is one requirement.
"""

from __future__ import annotations

from csr_lab.config import ChunkingSettings, ConflictSettings, GraphSettings, PipelineOptions, VerificationSettings
from csr_lab.pipelines.base import StrategySpec

SINGLE_PASS = StrategySpec(
    name="single_pass",
    description="Condition A: one structured call per group of chapters returns references and conflicts.",
    options=PipelineOptions(
        workflow="single_pass",
        verification=VerificationSettings(enabled=False),
        conflicts=ConflictSettings(enabled=True, validate_evidence=False),
        graph=GraphSettings(enabled=False),
    ),
)

STAGED = StrategySpec(
    name="staged",
    description="Condition B: separate reference and pairwise conflict stages; no verification, no graph use.",
    options=PipelineOptions(
        workflow="staged",
        verification=VerificationSettings(enabled=False),
        conflicts=ConflictSettings(enabled=True, validate_evidence=False),
        graph=GraphSettings(enabled=False),
    ),
)

STAGED_VERIFIED = StrategySpec(
    name="staged_verified",
    description=(
        "Condition C: staged plus source-evidence validation, distractor and duplicate checks, "
        "coverage audit and bounded targeted re-extraction; no graph use."
    ),
    options=PipelineOptions(
        workflow="staged",
        verification=VerificationSettings(enabled=True),
        conflicts=ConflictSettings(enabled=True, validate_evidence=True),
        graph=GraphSettings(enabled=False),
    ),
)

GRAPH_ASSISTED = StrategySpec(
    name="graph_assisted",
    description=(
        "Condition D: staged_verified plus graph-informed alias resolution, known-document audit, "
        "conflict candidate selection, context retrieval and exclusion/precedence rules."
    ),
    options=PipelineOptions(
        workflow="staged",
        verification=VerificationSettings(enabled=True),
        conflicts=ConflictSettings(enabled=True, validate_evidence=True),
        graph=GraphSettings(enabled=True),
    ),
)

def _llm_chunked(repair: str) -> PipelineOptions:
    return PipelineOptions(
        workflow="staged",
        chunking=ChunkingSettings(method="llm", repair=repair),  # type: ignore[arg-type]
        verification=VerificationSettings(enabled=False),
        conflicts=ConflictSettings(enabled=True, validate_evidence=False),
        graph=GraphSettings(enabled=False),
    )


LLM_CHUNKED = StrategySpec(
    name="llm_chunked",
    description=(
        "Industry-style baseline: the model splits ~2500-word batches into ~500-word semantic chunks; "
        "each chunk is one requirement; chunk text is used as returned."
    ),
    options=_llm_chunked("none"),
)

LLM_CHUNKED_REPAIRED = StrategySpec(
    name="llm_chunked_repaired",
    description="llm_chunked, but chunks whose wording differs from the source are replaced by the aligned source text.",
    options=_llm_chunked("source_slice"),
)

BUILT_IN = [SINGLE_PASS, STAGED, STAGED_VERIFIED, GRAPH_ASSISTED, LLM_CHUNKED, LLM_CHUNKED_REPAIRED]
