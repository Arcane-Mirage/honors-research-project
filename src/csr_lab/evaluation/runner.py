"""src/csr_lab/evaluation/runner.py

Experiment runner: datasets x strategies x repetitions, with storage and reports.

For every (strategy, repetition, case) the runner builds a fresh provider and
pipeline context (``run_analysis``), so runs cannot share state. A failure in
one case is recorded and the runner continues. Failed cases are kept in the
results: the *strict* aggregate scores them as empty predictions, and the
*valid-only* aggregate excludes them; both are reported with the failure list.

**Saved graph** (``saved_graph:`` in an experiment). The graph is made the same way
as for real documents: ``build-graph`` on a folder (``build_from``), with the
experiment's model, into ``<output>/saved_graph.json``; or an existing file (``path``).
It is loaded once and only read: every graph-assisted strategy with
``graph.use_saved`` starts each case from it, the others analyse the case alone, and
all are graded against the same gold.
"""

from __future__ import annotations

import json
import platform
import subprocess
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from csr_lab.config import PROJECT_ROOT, AppConfig, ExperimentConfig
from csr_lab.datasets.loader import Case, load_dataset
from csr_lab.evaluation.metrics import CaseEvaluation, aggregate, describe, evaluate_case, flatten_case
from csr_lab.evaluation.reporting import write_reports
from csr_lab.ids import stable_hash
from csr_lab.pipelines.registry import get_strategy
from csr_lab.pipelines.run import load_inputs, run_analysis
from csr_lab.prompts import prompt_versions
from csr_lab.schemas import AnalysisResult, InputDocument
from csr_lab.storage.database import ExperimentStore

PAIRED_METRICS = [
    "references.f1",
    "references.precision",
    "references.recall",
    "references.complete_set_accuracy",
    "units.f1",
    "conflicts.f1",
    "conflicts.recall",
]


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _git_state() -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5, check=False)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5, check=False)
        return {"git_commit": commit.stdout.strip() or None, "git_dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty": None}


def environment() -> dict[str, Any]:
    def ver(pkg: str) -> str | None:
        try:
            return version(pkg)
        except PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "csr_relationship_lab": ver("csr-relationship-lab"),
        "langgraph": ver("langgraph"),
        "pydantic": ver("pydantic"),
        **_git_state(),
    }


@dataclass
class CaseRun:
    run_id: str
    strategy: str
    repetition: int
    seed: int
    case_id: str
    status: str
    evaluation: CaseEvaluation
    error: str | None = None
    output_path: str | None = None
    analysis_id: str | None = None


@dataclass
class ExperimentOutcome:
    experiment_id: str
    output_dir: Path
    summary: dict[str, Any]
    case_runs: list[CaseRun] = field(default_factory=list)


def validate_cases(cases: list[Case], config: AppConfig, registries: dict | None = None) -> tuple[dict[str, list[InputDocument]], list[str]]:
    """Load every case's documents and gold once, before any model call.

    ``registries`` (filled in when given): case ID -> the case's ``documents.yaml`` registry.
    """
    documents: dict[str, list[InputDocument]] = {}
    problems: list[str] = []
    for case in cases:
        missing = [str(p) for p in case.document_paths if not p.is_file()]
        if missing:
            problems.append(f"{case.case_id}: missing documents {missing}")
            continue
        docs, registry = load_inputs(list(case.document_paths), config)
        if registries is not None:
            registries[case.case_id] = registry
        documents[case.case_id] = docs
        gold = case.load_gold()
        if gold is None:
            problems.append(f"{case.case_id}: no ground truth; the case cannot be scored")
            continue
        loaded = {d.document_id for d in docs}
        expected = {d.document_id for d in gold.documents}
        if loaded != expected:
            problems.append(f"{case.case_id}: document IDs {sorted(loaded)} do not match gold {sorted(expected)}")
    return documents, problems


def run_experiment(
    config: AppConfig,
    experiment: ExperimentConfig,
    *,
    output_root: Path | None = None,
    database_path: Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> ExperimentOutcome:
    say = progress or (lambda _msg: None)
    dataset = load_dataset(experiment.dataset)
    cases = dataset.select(experiment.split, experiment.case_ids)
    if not cases:
        raise ValueError(f"dataset {dataset.name} has no cases for split={experiment.split!r}")
    strategies = [get_strategy(name) for name in experiment.strategies]
    registries: dict[str, Any] = {}
    documents, problems = validate_cases(cases, config, registries)
    fatal = [p for p in problems if "missing documents" in p or "do not match gold" in p or "no ground truth" in p]
    if fatal:
        raise ValueError("dataset validation failed:\n  " + "\n  ".join(fatal))

    created = _now()
    experiment_id = f"{experiment.name}-{datetime.now(UTC):%Y%m%dT%H%M%S}-{stable_hash(created, experiment.name, length=4)}"
    out_dir = (output_root or config.resolve_path(config.output_dir)) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    store = ExperimentStore(database_path or config.resolve_path(config.database_path))
    is_mock = config.llm.provider == "mock"
    env = environment()
    store.create_experiment(
        experiment_id=experiment_id,
        name=experiment.name,
        description=experiment.description,
        created_at=created,
        status="running",
        dataset_name=dataset.name,
        dataset_version=dataset.version,
        dataset_sha256=dataset.manifest_sha256,
        split=experiment.split,
        provider=config.llm.provider,
        model=config.llm.model,
        is_mock=int(is_mock),
        config_json=json.dumps({"app": config.model_dump(mode="json", exclude={"llm"}), "llm": config.llm.public_dict(), "experiment": experiment.model_dump(mode="json")}),
        environment_json=json.dumps(env),
    )

    saved, saved_info = _saved_graph(experiment, config, out_dir, say)
    saved_texts = {u.unit_id: (u.document_id, u.text) for u in saved.saved_units().values()} if saved else None

    gold = {c.case_id: c.load_gold() for c in cases}
    case_runs: list[CaseRun] = []
    run_records: list[dict[str, Any]] = []
    llm_rows: list[dict[str, Any]] = []
    threshold = config.evaluation.requirement_match_threshold

    for spec in strategies:
        for rep in range(experiment.repetitions):
            seed = experiment.base_seed + rep
            run_id = f"{experiment_id}:{spec.name}:r{rep}"
            store.start_run(
                run_id=run_id,
                experiment_id=experiment_id,
                strategy=spec.name,
                strategy_description=spec.description,
                strategy_options_json=spec.options.model_dump_json(),
                repetition=rep,
                seed=seed,
                status="running",
                started_at=_now(),
            )
            evaluations: list[CaseEvaluation] = []
            valid: list[CaseEvaluation] = []
            for case in cases:
                say(f"{spec.name} rep {rep + 1}/{experiment.repetitions} {case.case_id}")
                started = time.perf_counter()
                result: AnalysisResult | None = None
                error = None
                try:
                    use_saved = saved is not None and spec.options.graph.enabled and spec.options.graph.use_saved
                    result = run_analysis(
                        documents[case.case_id],
                        strategy=spec,
                        config=config,
                        case_id=case.case_id,
                        run_seed=seed,
                        saved_graph=saved if use_saved else None,
                        saved_graph_path=saved_info.get("path") if use_saved else None,
                        registry=registries.get(case.case_id),
                    )
                    status = result.status
                    if result.status == "failed":
                        error = "; ".join(result.metadata.errors)[:2000]
                except Exception as exc:  # noqa: BLE001 - recoverable at experiment level: record and continue
                    status = "error"
                    error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"
                evaluation = evaluate_case(
                    result if status != "error" else None, gold[case.case_id], threshold=threshold, status=status, saved_units=saved_texts, customer=config.evaluation.customer  # type: ignore[arg-type]
                )
                output_path = None
                if result is not None:
                    path = out_dir / "predictions" / spec.name / f"rep{rep}" / f"{case.case_id}.json"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
                    output_path = path.relative_to(out_dir).as_posix()
                    store.add_llm_calls(run_id, case.case_id, result.metadata.llm_calls)
                    for call in result.metadata.llm_calls:
                        llm_rows.append({"strategy": spec.name, "repetition": rep, "case_id": case.case_id, **call.model_dump(mode="json")})
                ops = result.metadata.operational if result else None
                store.add_case_result(
                    run_id=run_id,
                    case_id=case.case_id,
                    status=status,
                    error=error,
                    analysis_id=result.analysis_id if result else None,
                    output_path=output_path,
                    seconds=round(time.perf_counter() - started, 4),
                    llm_calls=ops.llm_calls if ops else None,
                    input_tokens=ops.input_tokens if ops else None,
                    output_tokens=ops.output_tokens if ops else None,
                    input_chars=ops.input_chars if ops else None,
                    output_chars=ops.output_chars if ops else None,
                )
                store.add_metrics(run_id, case.case_id, "case", flatten_case(evaluation))
                evaluations.append(evaluation)
                if status in ("completed", "completed_with_errors"):
                    valid.append(evaluation)
                case_runs.append(
                    CaseRun(run_id, spec.name, rep, seed, case.case_id, status, evaluation, error, output_path, result.analysis_id if result else None)
                )

            strict = aggregate(evaluations)
            valid_only = aggregate(valid) if valid else {}
            store.add_metrics(run_id, "*", "strict", strict)
            store.add_metrics(run_id, "*", "valid_only", valid_only)
            failed = len(evaluations) - len(valid)
            store.finish_run(
                run_id,
                status="completed" if not failed else "completed_with_failures",
                finished_at=_now(),
                cases_total=len(evaluations),
                cases_failed=failed,
                prompt_versions=prompt_versions(),
            )
            store.commit()
            run_records.append(
                {
                    "run_id": run_id,
                    "strategy": spec.name,
                    "repetition": rep,
                    "seed": seed,
                    "cases_total": len(evaluations),
                    "cases_failed": failed,
                    "metrics_strict": strict,
                    "metrics_valid_only": valid_only,
                }
            )

    summary = build_summary(
        experiment_id=experiment_id,
        experiment=experiment,
        config=config,
        dataset=dataset,
        cases=cases,
        strategies=strategies,
        run_records=run_records,
        case_runs=case_runs,
        created=created,
        env=env,
        problems=problems,
    )
    summary["saved_graph"] = saved_info or None
    write_reports(out_dir, summary, case_runs, llm_rows)
    store.finish_experiment(experiment_id, "completed", _now())
    store.close()
    return ExperimentOutcome(experiment_id=experiment_id, output_dir=out_dir, summary=summary, case_runs=case_runs)


def build_saved_graph(folder: Path, config: AppConfig, path: Path, *, seed: int, output_root: Path, strategy: str = "graph_assisted") -> dict[str, Any]:
    """``build-graph`` on ``folder`` into ``path``, as for real documents. Returns what to report."""
    from csr_lab.pipelines.graph_build import build_graph

    build_config = config.model_copy(update={"input_formats": [".pdf", ".txt", ".json"]})  # synthetic data is text
    outcome = build_graph(folder, config=build_config, strategy=strategy, graph_path=path, run_seed=seed, output_root=output_root)
    return {
        "built_from": str(folder),
        "path": str(path),
        "build_report": str(outcome.out_dir / "build_report.md"),
        "build_llm_calls": sum(r.get("llm_calls") or 0 for r in outcome.rows),
        "build_tokens": sum(r.get("tokens") or 0 for r in outcome.rows),
        "build_failures": [r["bundle"] for r in outcome.rows if r.get("status") in ("error", "failed")],
    }


def _saved_graph(experiment: ExperimentConfig, config: AppConfig, out_dir: Path, say: Callable[[str], None]):
    """Build or load the experiment's saved graph. Returns (store or None, info for the report).

    ``{build_from: FOLDER}`` builds into the output folder; ``{path: FILE}`` loads a graph (other
    keys, such as those a sweep adds after building once, are reported as given).
    """
    from csr_lab.relationships.store import GraphStore

    spec = experiment.saved_graph
    if not spec:
        return None, {}
    if spec.get("path"):
        path = config.resolve_path(spec["path"])
        if not path.is_file():
            raise ValueError(f"saved graph {path} not found; build it with: csr-lab build-graph FOLDER --graph {path}")
        info = {**spec, "path": str(path)}
    elif spec.get("build_from"):
        folder = config.resolve_path(spec["build_from"])
        say(f"building the saved graph from {folder}")
        info = build_saved_graph(
            folder, config, out_dir / "saved_graph.json", seed=experiment.base_seed, output_root=out_dir, strategy=spec.get("strategy", "graph_assisted")
        )
        path = Path(info["path"])
    else:
        raise ValueError("saved_graph needs build_from: FOLDER or path: FILE")
    store = GraphStore.load(path)
    summary = store.summary()
    info |= {"fingerprint": summary["fingerprint"], "documents": summary["documents"], "requirements": summary["requirements"]}
    return store, info


def build_summary(*, experiment_id, experiment, config, dataset, cases, strategies, run_records, case_runs, created, env, problems) -> dict[str, Any]:
    names = [s.name for s in strategies]

    def agg(scope: str) -> dict[str, dict[str, dict]]:
        out: dict[str, dict[str, dict]] = {}
        for name in names:
            records = [r[scope] for r in run_records if r["strategy"] == name and r[scope]]
            keys = sorted({k for r in records for k in r})
            out[name] = {k: describe([r.get(k) for r in records]) for k in keys}
        return out

    baseline = config.evaluation.baseline_strategy if config.evaluation.baseline_strategy in names else names[0]
    per_case: dict[str, dict[str, dict[str, list]]] = {}
    for cr in case_runs:
        flat = flatten_case(cr.evaluation)
        for metric in PAIRED_METRICS:
            per_case.setdefault(cr.strategy, {}).setdefault(cr.case_id, {}).setdefault(metric, []).append(flat.get(metric))

    def mean(xs: list) -> float | None:
        vals = [x for x in xs if x is not None]
        return sum(vals) / len(vals) if vals else None

    paired: dict[str, dict[str, dict]] = {}
    for name in names:
        if name == baseline:
            continue
        paired[name] = {}
        for metric in PAIRED_METRICS:
            deltas = []
            for case in cases:
                a = mean(per_case.get(name, {}).get(case.case_id, {}).get(metric, []))
                b = mean(per_case.get(baseline, {}).get(case.case_id, {}).get(metric, []))
                if a is not None and b is not None:
                    deltas.append(a - b)
            paired[name][metric] = {
                "n_cases": len(deltas),
                "mean_delta": sum(deltas) / len(deltas) if deltas else None,
                "better": sum(1 for d in deltas if d > 1e-9),
                "worse": sum(1 for d in deltas if d < -1e-9),
                "tied": sum(1 for d in deltas if abs(d) <= 1e-9),
            }

    return {
        "experiment_id": experiment_id,
        "name": experiment.name,
        "description": experiment.description,
        "created_at": created,
        "finished_at": _now(),
        "dataset": {
            "name": dataset.name,
            "version": dataset.version,
            "sha256": dataset.manifest_sha256,
            "manifest": str(dataset.manifest_path),
            "split": experiment.split,
            "cases": [c.case_id for c in cases],
        },
        "provider": {
            "provider": config.llm.provider,
            "model": config.llm.model,
            "is_mock": config.llm.provider == "mock",
            "temperature": config.llm.temperature,
            "structured_output": config.llm.structured_output,
        },
        "strategies": [{"name": s.name, "description": s.description, "base": s.base, "options": s.options.model_dump(mode="json")} for s in strategies],
        "repetitions": experiment.repetitions,
        "base_seed": experiment.base_seed,
        "baseline_strategy": baseline,
        "prompt_versions": prompt_versions(),
        "environment": env,
        "config": {
            "llm": config.llm.public_dict(),
            "system_prompt": config.system_prompt.model_dump(),
            "processing": config.processing.model_dump(),
            "evaluation": config.evaluation.model_dump(),
        },
        "validation_warnings": problems,
        "runs": run_records,
        "aggregates": agg("metrics_strict"),
        "aggregates_valid_only": agg("metrics_valid_only"),
        "paired_vs_baseline": paired,
        "failures": [
            {"strategy": c.strategy, "repetition": c.repetition, "case_id": c.case_id, "status": c.status, "error": (c.error or "")[:500]}
            for c in case_runs
            if c.status not in ("completed", "completed_with_errors")
        ],
        "partial_results": [
            {"strategy": c.strategy, "repetition": c.repetition, "case_id": c.case_id}
            for c in case_runs
            if c.status == "completed_with_errors"
        ],
    }
