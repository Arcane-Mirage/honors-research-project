"""Built-in experimental conditions.

The suggested ``staged.py`` / ``graph_assisted.py`` modules are combined here:
the three staged conditions share one workflow and differ only in options,
which makes the controlled differences easy to see side by side.

    condition          verification   conflict evidence check   graph use
    single_pass        -              -                         -
    staged             -              -                         -
    staged_verified    yes            yes                       -
    graph_assisted     yes            yes                       yes

Each condition adds one thing to the previous one, so adjacent conditions
can be compared to isolate decomposition (A->B), verification (B->C) and
graph-informed processing (C->D).
"""

from __future__ import annotations

from csr_lab.config import ConflictSettings, GraphSettings, PipelineOptions, VerificationSettings
from csr_lab.pipelines.base import StrategySpec

SINGLE_PASS = StrategySpec(
    name="single_pass",
    description="Condition A: one structured call per chunk extracts requirements, references and conflicts.",
    options=PipelineOptions(
        workflow="single_pass",
        verification=VerificationSettings(enabled=False),
        conflicts=ConflictSettings(enabled=True, validate_evidence=False),
        graph=GraphSettings(enabled=False),
    ),
)

STAGED = StrategySpec(
    name="staged",
    description="Condition B: separate requirement, reference and pairwise conflict stages; no verification, no graph use.",
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

BUILT_IN = [SINGLE_PASS, STAGED, STAGED_VERIFIED, GRAPH_ASSISTED]
