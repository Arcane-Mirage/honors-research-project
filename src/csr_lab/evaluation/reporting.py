"""Write experiment reports: summary.json, CSV tables and a concise report.md."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from csr_lab.evaluation.metrics import flatten_case

if TYPE_CHECKING:
    from csr_lab.evaluation.runner import CaseRun

MOCK_WARNING = (
    "> **Mock provider.** These numbers come from the offline mock model, which uses simple "
    "heuristics with seeded error injection. They show that the pipeline, verification and "
    "evaluation work end to end. They are **not** evidence about real LLM performance and must "
    "not be used to accept or reject the research hypotheses."
)


def _fmt(stat: dict | None, digits: int = 3) -> str:
    if not stat or stat.get("mean") is None:
        return "–"
    mean = stat["mean"]
    if stat.get("std") is None:
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {stat['std']:.{digits}f}"


def _fmt_int(stat: dict | None) -> str:
    if not stat or stat.get("mean") is None:
        return "–"
    mean = stat["mean"]
    text = f"{mean:,.0f}" if abs(mean) >= 10 or float(mean).is_integer() else f"{mean:.1f}"
    return text if not stat.get("std") else f"{text} ± {stat['std']:,.1f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in row.items()})


def markdown_report(summary: dict[str, Any]) -> str:
    agg = summary["aggregates"]
    names = list(agg)
    baseline = summary["baseline_strategy"]
    ds, prov = summary["dataset"], summary["provider"]
    out: list[str] = [f"# Experiment report: {summary['name']}", ""]
    if prov["is_mock"]:
        out += [MOCK_WARNING, ""]
    out += [
        _table(
            ["Item", "Value"],
            [
                ["Experiment ID", f"`{summary['experiment_id']}`"],
                ["Dataset", f"{ds['name']} v{ds['version']} · split `{ds['split']}` · {len(ds['cases'])} cases · sha256 `{ds['sha256'][:12]}`"],
                ["Model", f"{prov['provider']} / `{prov['model']}` · temperature {prov['temperature']} · output mode {prov['structured_output']}"],
                ["Strategies", ", ".join(f"`{n}`" for n in names)],
                ["Repetitions", f"{summary['repetitions']} (seeds {summary['base_seed']}–{summary['base_seed'] + summary['repetitions'] - 1})"],
                ["Code", f"commit `{(summary['environment'].get('git_commit') or 'unknown')[:10]}`{' (uncommitted changes)' if summary['environment'].get('git_dirty') else ''}"],
            ],
        ),
        "",
        "Values are micro-averaged over cases within a run, then shown as mean ± sample standard "
        "deviation over repetitions. Failed cases are scored as empty predictions (strict scoring).",
        "",
        "## Reference extraction (primary outcome)",
        "",
        _table(
            ["Strategy", "Precision", "Recall", "F1", "Complete-set accuracy", "Duplicates", "Relation accuracy"],
            [
                [
                    f"`{n}`",
                    _fmt(agg[n].get("references.precision")),
                    _fmt(agg[n].get("references.recall")),
                    _fmt(agg[n].get("references.f1")),
                    _fmt(agg[n].get("references.complete_set_accuracy")),
                    _fmt_int(agg[n].get("references.duplicate_predictions")),
                    _fmt(agg[n].get("references.relation_accuracy")),
                ]
                for n in names
            ],
        ),
        "",
        "Scored per (requirement, document) pair: a correct document on the wrong requirement is an error. "
        "Complete-set accuracy is the share of gold requirements whose whole reference set is exactly right.",
        "",
        "## Potential conflicts",
        "",
        _table(
            ["Strategy", "Precision", "Recall", "F1", "Candidate-pair recall", "Evidence valid", "Hard-negative FPs", "Ambiguous recall", "Cross-doc recall"],
            [
                [
                    f"`{n}`",
                    _fmt(agg[n].get("conflicts.precision")),
                    _fmt(agg[n].get("conflicts.recall")),
                    _fmt(agg[n].get("conflicts.f1")),
                    _fmt(agg[n].get("conflicts.candidate_pair_recall")),
                    _fmt(agg[n].get("conflicts.evidence_validity_rate")),
                    _fmt_int(agg[n].get("conflicts.hard_negative_false_positives")),
                    _fmt(agg[n].get("conflicts.ambiguous_recall")),
                    _fmt(agg[n].get("conflicts.cross_document_recall")),
                ]
                for n in names
            ],
        ),
        "",
        "## Requirement extraction (context for the above)",
        "",
        _table(
            ["Strategy", "Precision", "Recall", "Chapter accuracy", "Verbatim exact", "Location accuracy"],
            [
                [
                    f"`{n}`",
                    _fmt(agg[n].get("requirements.precision")),
                    _fmt(agg[n].get("requirements.recall")),
                    _fmt(agg[n].get("requirements.chapter_accuracy")),
                    _fmt(agg[n].get("requirements.verbatim_exact_rate")),
                    _fmt(agg[n].get("requirements.location_accuracy")),
                ]
                for n in names
            ],
        ),
        "",
        "## Relationship graph",
        "",
        _table(
            ["Strategy", "Structurally valid", "Node resolution", "Unresolved identifiers", "Explicit-relation edge F1", "Unsupported edges", "Missing edges"],
            [
                [
                    f"`{n}`",
                    _fmt(agg[n].get("graph.structurally_valid_rate")),
                    _fmt(agg[n].get("graph.node_resolution_rate")),
                    _fmt_int(agg[n].get("graph.unresolved_identifiers")),
                    _fmt(agg[n].get("graph.qualified_edge_f1")),
                    _fmt_int(agg[n].get("graph.unsupported_reference_edges")),
                    _fmt_int(agg[n].get("graph.missing_reference_edges")),
                ]
                for n in names
            ],
        ),
        "",
        "Structural validity only means the JSON graph is internally consistent; correctness is in the tables above.",
        "",
        "## Cost",
        "",
        _table(
            ["Strategy", "LLM calls", "Input chars", "Input tokens", "Output tokens", "Seconds", "Retries", "JSON failures", "Graph-informed calls", "Est. cost (USD)"],
            [
                [
                    f"`{n}`",
                    _fmt_int(agg[n].get("ops.llm_calls")),
                    _fmt_int(agg[n].get("ops.input_chars")),
                    _fmt_int(agg[n].get("ops.input_tokens")),
                    _fmt_int(agg[n].get("ops.output_tokens")),
                    _fmt(agg[n].get("ops.total_seconds"), 2),
                    _fmt_int(agg[n].get("ops.retries")),
                    _fmt_int(agg[n].get("ops.json_validation_failures")),
                    _fmt_int(agg[n].get("ops.graph_informed_calls")),
                    _fmt(agg[n].get("ops.estimated_cost_usd"), 4),
                ]
                for n in names
            ],
        ),
        "",
        "Token counts and cost are shown only when the provider reports them; `–` means not measured.",
        "",
    ]

    paired = summary.get("paired_vs_baseline", {})
    if paired:
        rows = []
        for n, metrics in paired.items():
            for metric, d in metrics.items():
                delta = "–" if d["mean_delta"] is None else f"{d['mean_delta']:+.3f}"
                rows.append([f"`{n}`", metric, delta, str(d["better"]), str(d["worse"]), str(d["tied"]), str(d["n_cases"])])
        out += [
            f"## Paired per-case comparison against `{baseline}`",
            "",
            _table(["Strategy", "Metric", "Mean Δ", "Better", "Worse", "Tied", "Cases"], rows),
            "",
            "Each case's metric is averaged over repetitions and compared on the same case. This is "
            "descriptive: with few cases and repetitions it does not establish statistical significance.",
            "",
        ]

    failures = summary.get("failures", [])
    out += ["## Failed and partial runs", ""]
    if failures:
        out += [
            _table(
                ["Strategy", "Repetition", "Case", "Status", "Error"],
                [[f"`{f['strategy']}`", str(f["repetition"]), f["case_id"], f["status"], f["error"].splitlines()[0][:120] if f["error"] else ""] for f in failures],
            ),
            "",
        ]
    else:
        out += ["No case failed.", ""]
    partial = summary.get("partial_results", [])
    if partial:
        out += [f"{len(partial)} case runs completed with recoverable errors (see `summary.json`).", ""]

    out += ["## Strategies", ""]
    out += [f"- `{s['name']}`: {s['description']}" for s in summary["strategies"]]
    out.append("")
    return "\n".join(out)


def write_reports(out_dir: Path, summary: dict[str, Any], case_runs: list[CaseRun], llm_rows: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    case_rows = []
    for cr in case_runs:
        row = {"strategy": cr.strategy, "repetition": cr.repetition, "seed": cr.seed, **flatten_case(cr.evaluation)}
        row["status"] = cr.status
        row["error"] = (cr.error or "").splitlines()[0] if cr.error else ""
        row["output_path"] = cr.output_path
        case_rows.append(row)
    _write_csv(out_dir / "per_case_metrics.csv", case_rows)
    run_rows = []
    for r in summary["runs"]:
        for scope in ("metrics_strict", "metrics_valid_only"):
            if r[scope]:
                run_rows.append({"strategy": r["strategy"], "repetition": r["repetition"], "seed": r["seed"], "scope": scope.removeprefix("metrics_"), **r[scope]})
    _write_csv(out_dir / "per_run_metrics.csv", run_rows)
    if llm_rows:
        _write_csv(out_dir / "llm_calls.csv", llm_rows)
    (out_dir / "report.md").write_text(markdown_report(summary), encoding="utf-8")
