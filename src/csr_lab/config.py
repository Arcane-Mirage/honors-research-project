"""Typed configuration loaded from YAML files and environment variables.

All experimental knobs live here so a run's full configuration can be stored
with its results. ``PipelineOptions`` holds the per-strategy settings; the
strategy registry (``csr_lab.pipelines.registry``) defines named presets.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_EXPERIMENTS_PATH = PROJECT_ROOT / "configs" / "experiments.yaml"


class _Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------


class MockSettings(_Settings):
    """Error injection for the offline mock model.

    The mock applies these rates per *task* (for example per reference mention),
    never per strategy, so it cannot favour one strategy by design. Each model
    "look" is an independent seeded draw.
    """

    requirement_miss_rate: float = Field(default=0.0, ge=0, le=1)
    reference_miss_rate: float = Field(default=0.15, ge=0, le=1)
    hallucination_rate: float = Field(default=0.08, ge=0, le=1)
    quote_paraphrase_rate: float = Field(default=0.10, ge=0, le=1)
    conflict_miss_rate: float = Field(default=0.20, ge=0, le=1)
    conflict_false_positive_rate: float = Field(default=0.05, ge=0, le=1)
    invalid_json_rate: float = Field(default=0.0, ge=0, le=1)
    seed: int = 0


class PricingSettings(_Settings):
    input_usd_per_million_tokens: float | None = None
    output_usd_per_million_tokens: float | None = None


class LLMSettings(_Settings):
    provider: Literal["mock", "openai_compatible"] = "mock"
    model: str = "mock-heuristic-v1"
    base_url: str | None = None
    api_key_env: str | None = "OPENAI_API_KEY"
    temperature: float | None = 0.0
    max_output_tokens: int | None = 4096
    max_tokens_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    timeout_seconds: float = 120.0
    max_transport_retries: int = Field(default=2, ge=0)
    retry_backoff_seconds: float = Field(default=2.0, ge=0)
    max_validation_retries: int = Field(default=2, ge=0)
    seed: int | None = 1234
    send_seed: bool = True
    structured_output: Literal["json_schema", "json_object", "prompt"] = "json_schema"
    fallback_to_prompt_mode: bool = True
    max_concurrency: int = Field(default=1, ge=1)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    pricing: PricingSettings = Field(default_factory=PricingSettings)
    mock: MockSettings = Field(default_factory=MockSettings)

    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env) or None

    def public_dict(self) -> dict[str, Any]:
        """Settings safe to store with results (no secrets)."""
        data = self.model_dump(mode="json")
        data["api_key_present"] = self.api_key() is not None
        return data


# ---------------------------------------------------------------------------
# Processing and pipeline settings
# ---------------------------------------------------------------------------


class ProcessingSettings(_Settings):
    """Input preparation. Shared by every strategy in an experiment."""

    max_document_chars: int = Field(default=200_000, ge=1000)
    chunk_max_chars: int = Field(default=24_000, ge=500)
    chunk_overlap_segments: int = Field(default=1, ge=0)
    include_source_text: bool = True
    pdf_min_chars_per_page: int = 20


class VerificationSettings(_Settings):
    enabled: bool = False
    audit_enabled: bool = True
    max_attempts: int = Field(default=2, ge=0)
    max_candidates_per_requirement: int = Field(default=8, ge=1)
    repair_evidence: bool = True
    reject_distractors: bool = True


class ConflictSettings(_Settings):
    enabled: bool = True
    all_pairs_max: int = Field(
        default=400,
        ge=0,
        description="Compare every pair when the pair count is at most this; otherwise select candidates.",
    )
    pairs_per_call: int = Field(default=8, ge=1)
    attribute_min_score: float = Field(default=0.12, ge=0)
    max_candidate_pairs: int = Field(default=400, ge=1)
    validate_evidence: bool = False


class GraphSettings(_Settings):
    """Graph-informed processing. Only strategies with ``enabled: true`` may use the graph."""

    enabled: bool = False
    alias_resolution: bool = True
    reference_audit: bool = True
    candidate_selection: bool = True
    context_retrieval: bool = True
    applicability_rules: bool = True
    precedence_rules: bool = True
    related_document_hops: int = Field(default=1, ge=0)
    max_context_items: int = Field(default=6, ge=0)
    max_shared_reference_fanout: int = Field(default=8, ge=2)


class PipelineOptions(_Settings):
    workflow: Literal["single_pass", "staged"] = "staged"
    reference_batch_size: int = Field(default=8, ge=1)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    conflicts: ConflictSettings = Field(default_factory=ConflictSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)


class EvaluationSettings(_Settings):
    requirement_match_threshold: float = Field(default=0.75, ge=0, le=1)
    baseline_strategy: str = "single_pass"


class AppConfig(_Settings):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    output_dir: str = "outputs"
    database_path: str = "outputs/experiments.sqlite"

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT_ROOT / path


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------


class ExperimentConfig(_Settings):
    name: str
    description: str = ""
    dataset: str = "data/synthetic/manifest.json"
    split: Literal["dev", "heldout", "all"] = "heldout"
    strategies: list[str] = Field(default_factory=lambda: ["single_pass", "staged", "staged_verified", "graph_assisted"])
    repetitions: int = Field(default=1, ge=1)
    base_seed: int = 1000
    llm: dict[str, Any] = Field(default_factory=dict)
    processing: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    case_ids: list[str] | None = None


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict with ``override`` merged into ``base`` (dicts merge, other values replace)."""
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_dotenv(path: Path | None = None) -> None:
    """Minimal .env reader (KEY=VALUE lines). Existing environment variables win."""
    path = path or PROJECT_ROOT / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    return data


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> AppConfig:
    load_dotenv()
    data = _read_yaml(Path(path) if path else DEFAULT_CONFIG_PATH)
    return AppConfig.model_validate(deep_merge(data, overrides or {}))


def load_experiments_file(path: str | Path | None = None) -> dict[str, Any]:
    return _read_yaml(Path(path) if path else DEFAULT_EXPERIMENTS_PATH)


def load_experiment(name: str, path: str | Path | None = None) -> ExperimentConfig:
    data = load_experiments_file(path)
    experiments = data.get("experiments") or {}
    if name not in experiments:
        raise KeyError(f"unknown experiment {name!r}; available: {sorted(experiments)}")
    return ExperimentConfig.model_validate({"name": name, **experiments[name]})


def apply_experiment(config: AppConfig, experiment: ExperimentConfig) -> AppConfig:
    """Return a new AppConfig with the experiment's overrides applied."""
    merged = deep_merge(
        config.model_dump(),
        {"llm": experiment.llm, "processing": experiment.processing, "evaluation": experiment.evaluation},
    )
    return AppConfig.model_validate(merged)
