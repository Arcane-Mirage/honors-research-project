"""src/csr_lab/evaluation/sweep.py

Parameter sweeps: run one experiment for every combination of settings in a grid.

Defined under ``sweeps:`` in ``configs/experiments.yaml``::

    sweeps:
      chunk_size:
        experiment: chunking_mock              # the experiment to vary
        strategies: [llm_chunked, staged]   # optional override
        repetitions: 3                         # optional override
        grid:
          options.chunking.target_chunk_words: [100, 250, 500, 1000]
        metrics: [references.f1, units.f1]     # optional; a useful default set otherwise

Grid keys:

* ``llm.*``, ``system_prompt.*``, ``processing.*``, ``evaluation.*``: experiment-level settings (shared by all
  strategies in the run), e.g. ``llm.mock.merge_confusion_rate`` or ``processing.pdf_backend``;
* ``options.*``: strategy options; every strategy is re-registered as a variant named
  ``<strategy>__<setting>`` for each value;
* ``dataset``, ``split``: run the same strategies on another dataset (e.g. the PDF variant);
* ``saved_graph``: compare saved graphs, e.g. ``[{path: graphs/a.json}, {path: graphs/b.json}]``.

**Saved graphs.** An experiment with ``saved_graph: {build_from: FOLDER}`` builds its graph
once for the whole sweep (``<sweep>/saved_graph.json``), so every setting is graded against
the same graph. It is rebuilt per setting only when the grid changes how it is built
(``llm.*``, ``system_prompt.*``, ``processing.*``) or which graph is used (``saved_graph``).

Each combination is a normal experiment (stored in SQLite, with its own report.md);
the sweep writes ``sweep_report.md`` and ``sweep.csv`` with the chosen metrics side by side.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from csr_lab.config import AppConfig, ExperimentConfig, apply_experiment, deep_merge
from csr_lab.evaluation.reporting import markdown_table, write_csv
from csr_lab.evaluation.runner import run_experiment
from csr_lab.pipelines.registry import derive_strategy, get_strategy, register_strategy

DEFAULT_METRICS = [
    "references.f1",
    "references.precision",
    "references.recall",
    "units.f1",
    "requirements.verbatim_exact_rate",
    "input.verbatim_rate",
    "conflicts.f1",
    "ops.tokens_per_gold_requirement",
]
_EXPERIMENT_SECTIONS = ("llm", "system_prompt", "processing", "evaluation")
_EXPERIMENT_FIELDS = ("dataset", "split", "saved_graph")
_BUILD_SECTIONS = ("llm", "system_prompt", "processing", "saved_graph")  # grid keys that change the saved graph


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    experiment: str
    description: str = ""
    strategies: list[str] | None = None
    repetitions: int | None = Field(default=None, ge=1)
    split: str | None = None
    grid: dict[str, list[Any]]
    metrics: list[str] = Field(default_factory=lambda: list(DEFAULT_METRICS))


def load_sweep(name: str, experiments_file: dict[str, Any]) -> SweepConfig:
    sweeps = experiments_file.get("sweeps") or {}
    if name not in sweeps:
        raise KeyError(f"unknown sweep {name!r}; available: {sorted(sweeps)}")
    return SweepConfig.model_validate({"name": name, **sweeps[name]})


def _nested(dotted: str, value: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    cursor = out
    parts = dotted.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    return out


def _label(key: str, value: Any) -> str:
    return f"{key.split('.')[-1]}={json.dumps(value) if not isinstance(value, str) else value}"


def combinations(grid: dict[str, list[Any]]) -> list[list[tuple[str, Any]]]:
    keys = list(grid)
    for key in keys:
        head = key.split(".", 1)[0]
        if head not in (*_EXPERIMENT_SECTIONS, "options", *_EXPERIMENT_FIELDS):
            raise ValueError(
                f"sweep key {key!r} must start with llm., system_prompt., processing., evaluation., options., or be dataset/split/saved_graph"
            )
        if not grid[key]:
            raise ValueError(f"sweep key {key!r} has no values")
    return [list(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def run_sweep(
    sweep: SweepConfig,
    base_experiment: ExperimentConfig,
    base_config: AppConfig,
    *,
    output_root: Path | None = None,
    database_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, list[dict[str, Any]]]:
    say = progress or (lambda _m: None)
    stamp = f"{datetime.now(UTC):%Y%m%dT%H%M%S}"
    out_dir = (output_root or base_config.resolve_path(base_config.output_dir)) / f"sweep-{sweep.name}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_strategies = sweep.strategies or base_experiment.strategies
    rows: list[dict[str, Any]] = []
    settings_done: list[dict[str, Any]] = []
    shared_graph = _build_shared_graph(sweep, base_experiment, base_config, out_dir, say)

    for index, combo in enumerate(combinations(sweep.grid), start=1):
        label = ", ".join(_label(k, v) for k, v in combo)
        say(f"sweep {sweep.name}: {label}")
        exp_data = base_experiment.model_dump()
        option_overrides: dict[str, Any] = {}
        for key, value in combo:
            head, _, rest = key.partition(".")
            if head in _EXPERIMENT_SECTIONS:
                exp_data[head] = deep_merge(exp_data.get(head) or {}, _nested(rest, value))
            elif head == "options":
                option_overrides = deep_merge(option_overrides, _nested(rest, value))
            else:
                exp_data[head] = value
        strategy_names = []
        suffix = "__" + "_".join(_label(k, v) for k, v in combo if k.startswith("options."))
        for name in base_strategies:
            if option_overrides:
                variant = derive_strategy(f"{name}{suffix}", name, option_overrides, f"{get_strategy(name).description} [{label}]")
                register_strategy(variant, replace=True)
                strategy_names.append(variant.name)
            else:
                strategy_names.append(name)
        exp_data.update(
            name=f"{sweep.name}-s{index:02d}",  # one experiment ID per setting
            description=f"sweep {sweep.name}: {label}",
            strategies=strategy_names,
            repetitions=sweep.repetitions or base_experiment.repetitions,
            split=sweep.split or exp_data["split"],
        )
        if shared_graph:
            exp_data["saved_graph"] = shared_graph
        experiment = ExperimentConfig.model_validate(exp_data)
        config = apply_experiment(base_config, experiment)
        outcome = run_experiment(config, experiment, output_root=out_dir, database_path=database_path)
        settings_done.append({"setting": label, "experiment_id": outcome.experiment_id, "report": str(outcome.output_dir / "report.md")})
        for base_name, run_name in zip(base_strategies, strategy_names):
            agg = outcome.summary["aggregates"].get(run_name, {})
            row: dict[str, Any] = {"setting": label, "strategy": base_name, "experiment_id": outcome.experiment_id}
            for metric in sweep.metrics:
                stat = agg.get(metric) or {}
                row[metric] = stat.get("mean")
                row[f"{metric}.std"] = stat.get("std")
            rows.append(row)

    _write(out_dir, sweep, rows, settings_done, base_experiment, base_config)
    return out_dir, rows


def _build_shared_graph(sweep: SweepConfig, base: ExperimentConfig, config: AppConfig, out_dir: Path, say) -> dict[str, Any] | None:
    """Build the experiment's saved graph once when no grid key changes how it is built."""
    from csr_lab.evaluation.runner import build_saved_graph

    spec = base.saved_graph or {}
    if not spec.get("build_from") or any(k.split(".", 1)[0] in _BUILD_SECTIONS for k in sweep.grid):
        return None
    say(f"sweep {sweep.name}: building the saved graph once for every setting")
    return build_saved_graph(
        config.resolve_path(spec["build_from"]),
        apply_experiment(config, base),
        out_dir / "saved_graph.json",
        seed=base.base_seed,
        output_root=out_dir,
        strategy=spec.get("strategy", "graph_assisted"),
    )


def _fmt(value: Any, std: Any) -> str:
    if value is None:
        return "–"
    text = f"{value:,.0f}" if abs(value) >= 100 else f"{value:.3f}"
    return text if std in (None, 0) else f"{text} ± {std:.3f}" if abs(value) < 100 else f"{text} ± {std:,.0f}"


def _write(out_dir: Path, sweep: SweepConfig, rows: list[dict], settings: list[dict], base: ExperimentConfig, config: AppConfig) -> None:
    write_csv(out_dir / "sweep.csv", rows)
    (out_dir / "sweep.json").write_text(json.dumps({"sweep": sweep.model_dump(), "settings": settings, "rows": rows}, indent=2, default=str), encoding="utf-8")
    is_mock = config.llm.provider == "mock" and all("llm.provider" not in k for k in sweep.grid)
    lines = [f"# Sweep report: {sweep.name}", ""]
    if is_mock:
        lines += [
            ("> **Mock provider.** The offline mock model's behaviour is set by its error-injection rates. "
            "A sweep over those rates shows how each strategy *reacts* to a given kind of model error; "
            "it is not evidence of how often a real model makes that error."),
            "",
        ]
    lines += [
        sweep.description or "",
        "",
        f"Base experiment `{sweep.experiment}` (dataset `{base.dataset}`, split `{sweep.split or base.split}`), "
        f"{sweep.repetitions or base.repetitions} repetitions per setting. Grid: "
        + "; ".join(f"`{k}` ∈ {v}" for k, v in sweep.grid.items()),
        "",
        markdown_table(
            ["Setting", "Strategy", *sweep.metrics],
            [[row["setting"], f"`{row['strategy']}`", *(_fmt(row[m], row[f"{m}.std"]) for m in sweep.metrics)] for row in rows],
        ),
        "",
        "Values are means over repetitions (± sample standard deviation). Full reports per setting:",
        "",
    ]
    lines += [f"- {s['setting']}: `{Path(s['report']).relative_to(out_dir)}`" for s in settings]
    lines.append("")
    (out_dir / "sweep_report.md").write_text("\n".join(lines), encoding="utf-8")
