"""src/csr_lab/pipelines/graph_build.py

Build up a saved graph from a folder of documents (``csr-lab build-graph``).

Drop PDFs (or .txt / chunk .json files) into a folder, choose a graph file, run.
Each *bundle* of files is analysed together (so references between them resolve)
and merged into the graph. Files whose bytes were already merged are skipped, so
the same folder can be re-run after adding new files.

Bundles (``group``):

* ``folder`` (default): each subfolder is one bundle; loose files in the top folder
  form one more bundle. Put a CSR and the documents it references in one subfolder.
* ``file``: every file is analysed on its own.
* ``all``: everything is one bundle.

Every build writes ``outputs/graph_builds/<time>-<graph>/``: one result JSON and
graph views per bundle, ``summary.json`` and ``build_report.md`` with what each
bundle added to the graph, model calls, tokens and time.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal

from csr_lab.config import AppConfig
from csr_lab.ingestion.loader import SUPPORTED_SUFFIXES, load_bundle, processing_load_options
from csr_lab.relationships.builder import RelationshipGraph, graph_from_result
from csr_lab.relationships.queries import to_mermaid
from csr_lab.relationships.render import to_llm_markdown
from csr_lab.relationships.store import GraphStore

GroupMode = Literal["folder", "file", "all"]


@dataclass
class Bundle:
    name: str
    files: list[Path]


@dataclass
class BuildOutcome:
    out_dir: Path
    rows: list[dict[str, Any]]
    summary: dict[str, Any]
    view_paths: list[Path] = field(default_factory=list)


def _documents_in(folder: Path) -> list[Path]:
    return sorted(f for f in folder.iterdir() if f.is_file() and not f.name.startswith(".") and f.suffix.lower() in SUPPORTED_SUFFIXES)


def plan_bundles(folder: str | Path, group: GroupMode = "folder") -> list[Bundle]:
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"input folder not found: {folder}")
    loose = _documents_in(folder)
    subfolders = sorted(d for d in folder.iterdir() if d.is_dir() and not d.name.startswith("."))
    if group == "all":
        files = loose + [f for d in subfolders for f in _documents_in(d)]
        return [Bundle(folder.name, files)] if files else []
    if group == "file":
        return [Bundle(f.stem, [f]) for f in loose] + [Bundle(f"{d.name}/{f.stem}", [f]) for d in subfolders for f in _documents_in(d)]
    bundles = [Bundle(d.name, _documents_in(d)) for d in subfolders]
    if loose:
        bundles.insert(0, Bundle(folder.name, loose))
    return [b for b in bundles if b.files]


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


VIEW_SUFFIXES = (".overview.mmd", ".detail.mmd", ".llm.md")


def view_paths_for(path: Path) -> list[Path]:
    """``graphs/main.json`` -> ``graphs/main.overview.mmd``, ``main.detail.mmd``, ``main.llm.md``."""
    base = path.with_suffix("") if path.suffix == ".json" else path
    return [base.with_name(base.name + suffix) for suffix in VIEW_SUFFIXES]


def write_views(graph: RelationshipGraph, path: Path, title: str) -> list[Path]:
    """Overview Mermaid (one box per document), detailed Mermaid, and Markdown for an LLM."""
    paths = view_paths_for(path)
    paths[0].write_text(to_mermaid(graph, documents_only=True), encoding="utf-8")
    paths[1].write_text(to_mermaid(graph), encoding="utf-8")
    paths[2].write_text(to_llm_markdown(graph, title=title), encoding="utf-8")
    return paths


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name) or "bundle"


def build_graph(
    folder: str | Path,
    *,
    config: AppConfig,
    strategy: str,
    graph_path: str | Path,
    use: bool = True,
    update: bool = True,
    force: bool = False,
    group: GroupMode = "folder",
    run_seed: int | None = None,
    views: bool = True,
    output_root: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> BuildOutcome:
    from csr_lab.pipelines.run import run_analysis

    say = progress or (lambda _m: None)
    graph_path = Path(graph_path)
    store = GraphStore.load_or_new(graph_path)
    before = store.summary()
    known = store.known_file_hashes()
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = Path(output_root or config.resolve_path(config.output_dir)) / "graph_builds" / f"{stamp}-{graph_path.stem}"
    (out_dir / "results").mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for bundle in plan_bundles(folder, group):
        files = [{"name": str(f.relative_to(folder)) if f.is_relative_to(folder) else f.name, "sha256": file_sha256(f)} for f in bundle.files]
        row: dict[str, Any] = {"bundle": bundle.name, "files": len(files), "file_names": [f["name"] for f in files]}
        if not force and all(f["sha256"] in known for f in files):
            rows.append(row | {"status": "skipped (already in graph)"})
            say(f"{bundle.name}: skipped, already in the graph")
            continue
        say(f"{bundle.name}: analysing {len(files)} file(s)")
        started = time.perf_counter()
        try:
            documents = load_bundle(bundle.files, **processing_load_options(config.processing))
            result = run_analysis(
                documents,
                strategy=strategy,
                config=config,
                case_id=bundle.name,
                run_seed=run_seed,
                saved_graph=store if use and store.nodes else None,
                saved_graph_path=graph_path,
            )
        except Exception as exc:  # one bad bundle must not stop the build
            rows.append(row | {"status": "error", "error": f"{type(exc).__name__}: {exc}", "seconds": round(time.perf_counter() - started, 2)})
            say(f"{bundle.name}: error: {exc}")
            continue
        stem = _safe(bundle.name)
        (out_dir / "results" / f"{stem}.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
        if views:
            write_views(graph_from_result(result), out_dir / "results" / stem, title=f"Relationship graph: {bundle.name}")
        op = result.metadata.operational
        row |= {
            "status": result.status,
            "requirements": len(result.requirements),
            "referenced_documents": len(result.referenced_documents),
            "potential_conflicts": len(result.potential_conflicts),
            "used_saved_graph": result.metadata.saved_graph is not None,
            "llm_calls": op.llm_calls,
            "llm_failed_calls": op.llm_failed_calls,
            "tokens": op.estimated_tokens,
            "token_source": op.token_source,
            "seconds": op.total_seconds,
            "warnings": len(result.metadata.warnings),
            "errors": result.metadata.errors[:3],
        }
        if update and result.status != "failed":
            change = store.merge_result(result, input_files=files)
            store.save(graph_path)  # after every bundle, so an interrupted build keeps its progress
            known.update(f["sha256"] for f in files)
            row |= {"nodes_added": change.nodes_added, "edges_added": change.edges_added, "requirements_replaced": change.requirements_replaced}
        rows.append(row)
        say(f"{bundle.name}: {result.status}, {len(result.requirements)} requirements, {len(result.potential_conflicts)} potential conflicts")

    after = store.summary()
    view_paths: list[Path] = []
    changed = any("nodes_added" in r for r in rows)
    missing = not view_paths_for(graph_path)[-1].exists()
    if views and store.nodes and (changed or missing):
        view_paths = write_views(store.to_graph(include_superseded=False), graph_path, title=f"Relationship graph: {graph_path.stem}")
    llm = config.llm
    summary = {
        "graph": str(graph_path),
        "use_graph": use,
        "update_graph": update,
        "group": group,
        "force": force,
        "strategy": strategy,
        "llm": {"provider": llm.provider, "model": llm.model, "base_url": llm.base_url},
        "graph_before": before,
        "graph_after": after,
        "bundles": rows,
        "views": [str(p) for p in view_paths],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "build_report.md").write_text(_report(summary), encoding="utf-8")
    return BuildOutcome(out_dir=out_dir, rows=rows, summary=summary, view_paths=view_paths)


def _report(s: dict[str, Any]) -> str:
    b, a = s["graph_before"], s["graph_after"]
    mode = ("uses" if s["use_graph"] else "does not use") + " and " + ("updates" if s["update_graph"] else "does not update")
    lines = [
        f"# Graph build: {s['graph']}",
        "",
        f"- LLM: {s['llm']['provider']} / {s['llm']['model']}" + (f" at {s['llm']['base_url']}" if s["llm"]["base_url"] else ""),
        f"- Strategy: {s['strategy']}; bundles grouped by {s['group']}; the build {mode} the saved graph",
    ]
    if s["llm"]["provider"] == "mock":
        lines.append("- **Offline mock model: these results test the machinery and are not evidence.**")
    lines += [
        "",
        "## Graph before and after",
        "",
        "| | Documents | Analysed revisions | Requirements | Sources |",
        "|---|---|---|---|---|",
        f"| before | {b['documents']} | {b['analysed_document_revisions']} | {b['requirements']} | {b['sources']} |",
        f"| after | {a['documents']} | {a['analysed_document_revisions']} | {a['requirements']} | {a['sources']} |",
        "",
        "## Bundles",
        "",
        "| Bundle | Files | Status | Requirements | References | Conflicts | Saved graph used | +Nodes | +Edges | Replaced | LLM calls (failed) | Tokens | Seconds |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in s["bundles"]:
        get = lambda k, d="": r.get(k, d)  # noqa: E731
        calls = f"{get('llm_calls')} ({get('llm_failed_calls')})" if "llm_calls" in r else ""
        tokens = f"{get('tokens')}{'' if get('token_source') == 'reported' else ' est.'}" if "tokens" in r else ""
        used = {True: "yes", False: "no"}.get(get("used_saved_graph", None), "")
        lines.append(
            f"| {r['bundle']} | {r['files']} | {r['status']} | {get('requirements')} | {get('referenced_documents')} | "
            f"{get('potential_conflicts')} | {used} | {get('nodes_added')} | {get('edges_added')} | {get('requirements_replaced')} | "
            f"{calls} | {tokens} | {get('seconds')} |"
        )
    problems = [r for r in s["bundles"] if r.get("error") or r.get("errors")]
    if problems:
        lines += ["", "## Problems", ""]
        for r in problems:
            for msg in [r.get("error")] + list(r.get("errors") or []):
                if msg:
                    lines.append(f"- {r['bundle']}: {msg}")
    if s["views"]:
        lines += ["", "## Views of the saved graph", "", *[f"- {v}" for v in s["views"]]]
    return "\n".join(lines) + "\n"
