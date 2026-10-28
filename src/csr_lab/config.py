"""src/csr_lab/config.py

Typed configuration loaded from YAML files and environment variables.

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
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
DEFAULT_EXPERIMENTS_PATH = PROJECT_ROOT / "configs" / "experiments.yaml"
DEFAULT_LLMS_PATH = PROJECT_ROOT / "configs" / "llms.yaml"
LLM_ENV_VAR = "CSR_LAB_LLM"  # names the default LLM profile; wins over `default:` in llms.yaml
AUTO_MODEL = "auto"  # model name meaning "ask the server which model is loaded"


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
    # Reference merging (tasks: reference_merging, reference_alias_check).
    merge_confusion_rate: float = Field(default=0.3, ge=0, le=1, description="two similar identifiers merged as one document")
    merge_revision_drop_rate: float = Field(default=0.5, ge=0, le=1, description="revision variants merged under one name without the revision")
    merge_alias_miss_rate: float = Field(default=0.2, ge=0, le=1, description="a document name not linked to its identifier")
    seed: int = 0


class PricingSettings(_Settings):
    input_usd_per_million_tokens: float | None = None
    output_usd_per_million_tokens: float | None = None


class AuthSettings(_Settings):
    """How requests are authenticated. The default sends ``Authorization: Bearer <$api_key_env>``.

    * ``env``: the value of the ``api_key_env`` variable, sent as ``{header}: {scheme} {key}``.
      Azure OpenAI key auth: ``header: api-key``, ``scheme: null``.
    * ``command``: run ``command`` and send its standard output as the token (cached for
      ``refresh_seconds``), e.g. ``az account get-access-token --resource
      https://cognitiveservices.azure.com --query accessToken -o tsv``.
    * ``azure_identity``: Microsoft Entra ID token from ``azure.identity.DefaultAzureCredential``
      for ``scope`` (needs ``pip install azure-identity``; not a dependency of this package).
    * ``none``: no credentials (local servers).
    """

    method: Literal["env", "command", "azure_identity", "none"] = "env"
    header: str = "Authorization"
    scheme: str | None = "Bearer"
    command: list[str] | str | None = None
    scope: str = "https://cognitiveservices.azure.com/.default"
    refresh_seconds: float = Field(default=2700, ge=0)


class LLMSettings(_Settings):
    provider: Literal["mock", "openai_compatible"] = "mock"
    model: str = "mock-heuristic-v1"
    base_url: str | None = None
    api_key_env: str | None = "OPENAI_API_KEY"
    auth: AuthSettings = Field(default_factory=AuthSettings)
    chat_path: str = Field(default="/chat/completions", description="Appended to base_url.")
    query_params: dict[str, str] = Field(default_factory=dict, description="e.g. {api-version: 2024-10-21} for Azure OpenAI deployment URLs")
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
        if isinstance(data["auth"].get("command"), (list, str)):
            data["auth"]["command"] = "(set)"
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
    pdf_backend: Literal["pypdf", "pypdf_layout", "pymupdf"] = Field(
        default="pypdf", description="pypdf_layout keeps column positions; pymupdf needs `pip install pymupdf`."
    )
    pdf_strip_repeated_lines: bool = Field(
        default=False, description="Remove header/footer lines that repeat at the top or bottom of many pages."
    )
    pdf_repeated_line_min_share: float = Field(default=0.5, gt=0, le=1, description="Share of pages a line must appear on to count as a header/footer.")


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
    context_format: Literal["sentences", "triples", "json", "ids"] = Field(
        default="sentences",
        description=(
            "How graph context is shown to the conflict model: plain sentences, 'subject | relation | object' "
            "triples, JSON objects, or 'ids' (the original rendering with raw requirement IDs and relation codes)."
        ),
    )
    max_shared_reference_fanout: int = Field(default=8, ge=2)


class ChunkingSettings(_Settings):
    """What the analysed text units are and where they come from.

    * ``method``: ``deterministic`` uses csr-lab's section-aware segmenter; ``llm`` asks the
      model to split word-count batches into semantic chunks (a common industry approach).
      Pre-chunked input (a ``chunks.json`` file) always uses its own chunks.
    * ``unit``: ``requirement`` identifies individual requirements inside the units;
      ``chunk`` treats each chunk as one unit and extracts references per chunk.
    * ``repair``: ``source_slice`` replaces a chunk whose wording differs from the source
      with the source text it was aligned to; ``none`` analyses the chunk as given.
    """

    method: Literal["deterministic", "llm"] = "deterministic"
    unit: Literal["requirement", "chunk"] = "requirement"
    repair: Literal["none", "source_slice"] = "none"
    batch_words: int = Field(default=2500, ge=50, description="llm method: words per chunking request")
    target_chunk_words: int = Field(default=500, ge=20, description="llm method: requested chunk size")
    min_match_ratio: float = Field(default=0.6, ge=0, le=1, description="below this word ratio a chunk counts as not found in the source")


class MergingSettings(_Settings):
    """Reference merging after extraction (and verification).

    * ``deterministic``: identifier normalisation only (``extraction/normalization.py``); no model call.
    * ``llm``: one model call per document bundle groups all reference labels and picks a
      canonical name per group; the model's grouping is applied as given.
    * ``hybrid``: deterministic keys stay; the model is asked only about candidate groups
      (a document *name* that may belong to a known identifier, near-duplicate identifiers),
      and guard rules veto merges across different identifiers or revisions.
    """

    method: Literal["deterministic", "llm", "hybrid"] = "deterministic"
    protect_revisions: bool = Field(default=True, description="hybrid: never merge references with different explicit revisions")
    protect_identifiers: bool = Field(default=True, description="hybrid: never merge two different explicit identifiers")
    max_candidates_per_name: int = Field(default=5, ge=1)


class PipelineOptions(_Settings):
    workflow: Literal["single_pass", "staged"] = "staged"
    reference_batch_size: int = Field(default=8, ge=1)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    reference_merging: MergingSettings = Field(default_factory=MergingSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    conflicts: ConflictSettings = Field(default_factory=ConflictSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)


class GraphStoreSettings(_Settings):
    """The saved graph that ``csr-lab build-graph`` (and ``analyze --graph``) work on.

    * ``use``: graph-assisted strategies start from the saved graph's documents
      (identifiers, names, revisions). Other strategies ignore it.
    * ``update``: each run's result is merged into the saved graph.
    * ``write_views``: after an update, also write ``<graph>.mmd`` (Mermaid, for people)
      and ``<graph>.md`` (plain Markdown, for an LLM) next to the graph file.

    The experiment runner never reads or writes a saved graph.
    """

    path: str = "graphs/main.json"
    use: bool = True
    update: bool = True
    write_views: bool = True


class EvaluationSettings(_Settings):
    requirement_match_threshold: float = Field(default=0.75, ge=0, le=1)
    baseline_strategy: str = "single_pass"


class AppConfig(_Settings):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    processing: ProcessingSettings = Field(default_factory=ProcessingSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    graph_store: GraphStoreSettings = Field(default_factory=GraphStoreSettings)
    input: str = Field(default="input", description="Folder `csr-lab build-graph` reads when no folder is given.")
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
    dataset: str = "data/synthetic/text/manifest.json"
    split: Literal["dev", "heldout", "all"] = "heldout"
    strategies: list[str] = Field(default_factory=lambda: ["single_pass", "staged", "staged_verified", "graph_assisted"])
    repetitions: int = Field(default=1, ge=1)
    base_seed: int = 1000
    llm: dict[str, Any] = Field(
        default_factory=dict,
        description="LLM settings for this experiment: a profile name from llms.yaml, or a mapping "
        "(optionally with `profile:` plus overrides). Empty: the default profile.",
    )
    processing: dict[str, Any] = Field(default_factory=dict)
    evaluation: dict[str, Any] = Field(default_factory=dict)
    case_ids: list[str] | None = None

    @field_validator("llm", mode="before")
    @classmethod
    def _profile_name(cls, value: Any) -> Any:
        return {"profile": value} if isinstance(value, str) else (value or {})


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


# -- LLM profiles (configs/llms.yaml) ------------------------------------------


def load_llm_profiles(path: str | Path | None = None) -> dict[str, Any]:
    """``{"default": name or None, "profiles": {name: llm settings}}``; empty if the file is missing."""
    path = Path(path) if path else DEFAULT_LLMS_PATH
    if not path.is_file():
        return {"default": None, "profiles": {}}
    data = _read_yaml(path)
    profiles = data.get("profiles") or {}
    if not isinstance(profiles, dict):
        raise ValueError(f"{path}: `profiles` must be a mapping of name -> llm settings")
    return {"default": data.get("default"), "profiles": profiles}


def default_llm_profile(path: str | Path | None = None) -> str | None:
    """The profile used when none is named: ``$CSR_LAB_LLM`` (also read from .env), else ``default:``."""
    load_dotenv()
    return os.environ.get(LLM_ENV_VAR) or load_llm_profiles(path).get("default")


def llm_profile(name: str, path: str | Path | None = None) -> dict[str, Any]:
    profiles = load_llm_profiles(path)["profiles"]
    if name not in profiles:
        raise KeyError(f"unknown LLM profile {name!r}; available in configs/llms.yaml: {sorted(profiles)}")
    return copy.deepcopy(profiles[name] or {})


def expand_llm_block(block: dict[str, Any] | str | None, path: str | Path | None = None) -> dict[str, Any]:
    """Replace a ``profile:`` key (or a bare profile name) by that profile's settings, keeping overrides."""
    if not block:
        return {}
    if isinstance(block, str):
        block = {"profile": block}
    block = dict(block)
    name = block.pop("profile", None)
    return deep_merge(llm_profile(name, path), block) if name else block


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
    llm_profile: str | None = None,
) -> AppConfig:
    """``default.yaml`` (or ``path``), then the named LLM profile, then ``overrides``.

    No profile is applied unless one is named, so library callers and tests keep the
    ``llm:`` settings of the config file. The CLI names the selected profile.
    """
    load_dotenv()
    data = _read_yaml(Path(path) if path else DEFAULT_CONFIG_PATH)
    if llm_profile:
        data = deep_merge(data, {"llm": expand_llm_block(llm_profile)})
    overrides = dict(overrides or {})
    if "llm" in overrides:
        overrides["llm"] = expand_llm_block(overrides["llm"])
    return AppConfig.model_validate(deep_merge(data, overrides))


def load_experiments_file(path: str | Path | None = None) -> dict[str, Any]:
    return _read_yaml(Path(path) if path else DEFAULT_EXPERIMENTS_PATH)


def load_experiment(name: str, path: str | Path | None = None) -> ExperimentConfig:
    data = load_experiments_file(path)
    experiments = data.get("experiments") or {}
    if name not in experiments:
        raise KeyError(f"unknown experiment {name!r}; available: {sorted(experiments)}")
    return ExperimentConfig.model_validate({"name": name, **experiments[name]})


def apply_experiment(config: AppConfig, experiment: ExperimentConfig) -> AppConfig:
    """Return a new AppConfig with the experiment's overrides applied (``profile:`` names expanded)."""
    merged = deep_merge(
        config.model_dump(),
        {"llm": expand_llm_block(experiment.llm), "processing": experiment.processing, "evaluation": experiment.evaluation},
    )
    return AppConfig.model_validate(merged)
