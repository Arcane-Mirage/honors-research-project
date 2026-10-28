"""src/csr_lab/cli.py

Command-line interface: `csr-lab <command>` (or `python -m csr_lab <command>`).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from csr_lab.config import (
    DEFAULT_LLMS_PATH,
    LLM_ENV_VAR,
    PROJECT_ROOT,
    AppConfig,
    ExperimentConfig,
    apply_experiment,
    deep_merge,
    default_llm_profile,
    load_config,
    load_experiment,
    load_experiments_file,
    load_llm_profiles,
)


def _register_yaml_strategies(experiments_path: str | None) -> None:
    from csr_lab.pipelines.registry import register_variants

    try:
        data = load_experiments_file(experiments_path)
    except FileNotFoundError:
        if experiments_path:
            raise
        return
    register_variants(data.get("strategies"))


def _llm_flags(args: argparse.Namespace) -> dict:
    """--provider / --model / --base-url: final tweaks on top of the selected profile."""
    return {key: value for key in ("provider", "model", "base_url") if (value := getattr(args, key, None))}


def _llm_overrides(args: argparse.Namespace) -> dict:
    llm = _llm_flags(args)
    return {"llm": llm} if llm else {}


def _resolve_model(config: AppConfig) -> AppConfig:
    """Replace `model: auto` by the model the server has loaded, so results record the real name."""
    from csr_lab.llm.interface import LLMProviderError
    from csr_lab.llm.provider import resolve_model

    try:
        settings = resolve_model(config.llm)
    except LLMProviderError as exc:
        raise ValueError(f"LLM {config.llm.base_url}: {exc}") from exc
    return config if settings is config.llm else config.model_copy(update={"llm": settings})


def _load(args: argparse.Namespace) -> AppConfig:
    """Config for single runs: default.yaml, then the --llm (or default) profile, then --model etc."""
    profile = getattr(args, "llm", None) or default_llm_profile()
    return _resolve_model(load_config(args.config, _llm_overrides(args), llm_profile=profile))


def _experiment_llm(args: argparse.Namespace, experiment: ExperimentConfig) -> ExperimentConfig:
    """--llm replaces the experiment's `llm:`; an experiment without one gets the default profile.
    `model: auto` is resolved here so the stored configuration names the real model."""
    block = dict(experiment.llm)
    if getattr(args, "llm", None):
        block = {"profile": args.llm}
    elif not block and (default := default_llm_profile()):
        block = {"profile": default}
    block = deep_merge(block, _llm_flags(args))
    experiment = experiment.model_copy(update={"llm": block})
    config = apply_experiment(load_config(args.config), experiment)
    resolved = _resolve_model(config)
    if resolved is not config:
        experiment = experiment.model_copy(update={"llm": {**block, "model": resolved.llm.model}})
    return experiment


def _print_llm(config: AppConfig) -> None:
    llm = config.llm
    where = f" at {llm.base_url}" if llm.provider != "mock" and llm.base_url else ""
    print(f"LLM: {llm.provider} / {llm.model}{where}", file=sys.stderr)


def _graph_choice(args: argparse.Namespace, config: AppConfig) -> tuple[str | None, bool, bool]:
    """(graph path, use it, update it) from --graph/--use-graph/--update-graph, falling back to
    `graph_store:` in the config. The older --graph-store/--use-saved-graph flags still work."""
    if getattr(args, "graph", None):
        use = config.graph_store.use if args.use_graph is None else args.use_graph
        update = config.graph_store.update if args.update_graph is None else args.update_graph
        return args.graph, use, update
    if getattr(args, "graph_store", None):
        return args.graph_store, bool(args.use_saved_graph or args.use_graph), args.update_graph is not False
    if getattr(args, "use_saved_graph", False) or getattr(args, "use_graph", None):
        raise ValueError("using a saved graph needs --graph PATH")
    return None, False, False


def cmd_generate_data(args: argparse.Namespace) -> int:
    from csr_lab.datasets.synthetic import generate_scaled_case, write_dataset
    from csr_lab.datasets.variants import write_chunked_dataset, write_pdf_dataset

    if args.pdf:
        manifest = write_pdf_dataset(args.out or PROJECT_ROOT / "data" / "synthetic_pdf", lines_per_page=args.lines_per_page)
    elif args.chunked:
        default = "synthetic_chunked" if args.chunked == "paragraph" else f"synthetic_chunked_{args.chunked}"
        manifest = write_chunked_dataset(args.out or PROJECT_ROOT / "data" / default, chunker=args.chunked, words=args.chunk_words)
    elif args.scaled:
        cases = [
            generate_scaled_case(
                f"scaled_n{args.scaled}_r{args.refs}_f{args.filler}_{args.position}_s{seed}",
                n_requirements=args.scaled,
                refs_per_requirement=args.refs,
                filler_sentences=args.filler,
                position=args.position,
                seed=seed,
            )
            for seed in range(args.cases)
        ]
        manifest = write_dataset(args.out or PROJECT_ROOT / "data" / "scaled", cases, name="synthetic_scaled")
    else:
        manifest = write_dataset(args.out or PROJECT_ROOT / "data" / "synthetic")
    print(f"wrote {manifest}")
    return 0


def cmd_strategies(args: argparse.Namespace) -> int:
    from csr_lab.pipelines.registry import list_strategies

    _register_yaml_strategies(args.experiments)
    for spec in list_strategies():
        base = f" (variant of {spec.base})" if spec.base else ""
        print(f"{spec.name}{base}\n    {spec.description}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from csr_lab.pipelines.run import analyze_files
    from csr_lab.relationships.builder import graph_from_result
    from csr_lab.relationships.queries import export_graph, to_mermaid
    from csr_lab.relationships.render import to_llm_markdown
    from csr_lab.relationships.store import GraphStore

    _register_yaml_strategies(args.experiments)
    config = _load(args)
    graph_path, use_graph, update_graph = _graph_choice(args, config)
    _print_llm(config)
    store = GraphStore.load_or_new(graph_path) if graph_path else None
    result = analyze_files(
        args.inputs,
        strategy=args.strategy,
        config=config,
        run_seed=args.seed,
        saved_graph=store if use_graph and store.nodes else None,
        saved_graph_path=graph_path,
    )
    text = result.model_dump_json(indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    graph = graph_from_result(result)
    if args.graph_out:
        Path(args.graph_out).write_text(json.dumps(export_graph(graph), indent=2), encoding="utf-8")
        print(f"wrote {args.graph_out}")
    if args.mermaid:
        Path(args.mermaid).write_text(to_mermaid(graph, documents_only=args.overview), encoding="utf-8")
        print(f"wrote {args.mermaid}")
    if args.llm_text:
        Path(args.llm_text).write_text(to_llm_markdown(graph), encoding="utf-8")
        print(f"wrote {args.llm_text}")
    if store is not None and update_graph and result.status != "failed":
        report = store.merge_result(result)
        store.save(graph_path)
        print(f"merged into {graph_path}: +{report.nodes_added} nodes, +{report.edges_added} edges, {report.requirements_replaced} requirements replaced", file=sys.stderr)
        if config.graph_store.write_views:
            from csr_lab.pipelines.graph_build import write_views

            views = write_views(store.to_graph(include_superseded=False), Path(graph_path), title=f"Relationship graph: {Path(graph_path).stem}")
            print(f"views: {', '.join(map(str, views))}", file=sys.stderr)
    elif store is not None:
        print(f"{graph_path} left unchanged (--no-update-graph)", file=sys.stderr)
    if result.metadata.provider_is_mock:
        print("note: produced by the offline mock model (not real LLM output)", file=sys.stderr)
    return 0 if result.status != "failed" else 1


def cmd_experiment(args: argparse.Namespace) -> int:
    from csr_lab.evaluation.runner import run_experiment

    _register_yaml_strategies(args.experiments)
    experiment = load_experiment(args.name, args.experiments)
    updates = {}
    if args.repetitions:
        updates["repetitions"] = args.repetitions
    if args.strategies:
        updates["strategies"] = [s.strip() for s in args.strategies.split(",") if s.strip()]
    if args.split:
        updates["split"] = args.split
    if updates:
        experiment = experiment.model_copy(update=updates)
    experiment = _experiment_llm(args, experiment)
    config = apply_experiment(load_config(args.config), experiment)
    _print_llm(config)
    outcome = run_experiment(
        config,
        experiment,
        output_root=Path(args.output_dir) if args.output_dir else None,
        database_path=Path(args.db) if args.db else None,
        progress=(lambda m: print(f"  running {m}", file=sys.stderr)) if not args.quiet else None,
    )
    print(f"experiment {outcome.experiment_id}")
    print(f"report:  {outcome.output_dir / 'report.md'}")
    print(f"summary: {outcome.output_dir / 'summary.json'}")
    failures = outcome.summary["failures"]
    if failures:
        print(f"{len(failures)} case runs failed (listed in the report)")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    from csr_lab.evaluation.metrics import evaluate_case
    from csr_lab.schemas import AnalysisResult, CaseGold

    result = AnalysisResult.model_validate_json(Path(args.prediction).read_text(encoding="utf-8"))
    gold = CaseGold.model_validate_json(Path(args.gold).read_text(encoding="utf-8"))
    evaluation = evaluate_case(result, gold, threshold=args.threshold)
    print(evaluation.model_dump_json(indent=2))
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    from csr_lab.relationships import queries
    from csr_lab.relationships.builder import graph_from_result
    from csr_lab.relationships.validation import validate_graph
    from csr_lab.schemas import AnalysisResult

    result = AnalysisResult.model_validate_json(Path(args.result).read_text(encoding="utf-8"))
    graph = graph_from_result(result)
    out: dict = {}
    if args.requirement:
        out["documents_referenced"] = queries.documents_referenced_by(graph, args.requirement)
        out["connected_requirements"] = queries.requirements_sharing_references(graph, args.requirement)
    if args.document:
        out["referencing_requirements"] = queries.requirements_referencing(graph, args.document)
        out["related_documents"] = sorted(queries.related_documents(graph, args.document))
    if args.shared:
        out["shared_documents"] = queries.shared_documents(graph)
    if args.unresolved:
        out["unresolved_documents"] = [n.node_id for n in queries.unresolved_documents(graph)]
        out["unavailable_documents"] = [n.node_id for n in queries.unavailable_documents(graph)]
    if args.conflicts:
        out["potential_conflicts"] = queries.potentially_conflicting_requirements(graph)
        out["applicability_flags"] = [f.__dict__ | {"evidence": len(f.evidence)} for f in queries.find_applicability_conflicts(graph)]
        out["precedence_flags"] = [f.__dict__ | {"evidence": len(f.evidence)} for f in queries.find_precedence_conflicts(graph)]
    if args.evidence:
        out["evidence"] = [e.model_dump(mode="json") for e in queries.evidence_for(graph, args.evidence)]
    if args.validate or not out:
        out["validation"] = validate_graph(graph).model_dump()
    if args.export:
        Path(args.export).write_text(json.dumps(queries.export_graph(graph), indent=2), encoding="utf-8")
        out["exported"] = args.export
    if args.mermaid:
        Path(args.mermaid).write_text(queries.to_mermaid(graph, documents_only=args.overview), encoding="utf-8")
        out["mermaid"] = args.mermaid
    if args.llm_text:
        from csr_lab.relationships.render import to_llm_markdown

        Path(args.llm_text).write_text(to_llm_markdown(graph, include_ids=args.with_ids), encoding="utf-8")
        out["llm_text"] = args.llm_text
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_graph_store(args: argparse.Namespace) -> int:
    from csr_lab.relationships.queries import export_graph, to_mermaid
    from csr_lab.relationships.store import GraphStore
    from csr_lab.relationships.validation import validate_graph
    from csr_lab.schemas import AnalysisResult

    path = Path(args.store)
    store = GraphStore.load_or_new(path) if args.action == "merge" else GraphStore.load(path)
    out: dict = {}
    changed = True
    if args.action == "info":
        out = store.summary() | {"validation": validate_graph(store.to_graph()).model_dump()}
        changed = False
    elif args.action == "sources":
        out = {"sources": [s.model_dump() for s in store.sources.values()]}
        changed = False
    elif args.action == "merge":
        reports = []
        for item in args.results:
            result = AnalysisResult.model_validate_json(Path(item).read_text(encoding="utf-8"))
            reports.append({"result": item, **store.merge_result(result, replace=not args.accumulate, include_conflicts=not args.no_conflicts).model_dump()})
        out = {"merged": reports}
    elif args.action == "remove-source":
        out = store.remove_source(args.source_id).model_dump()
    elif args.action == "remove-document":
        out = store.remove_document(args.document_id).model_dump()
    elif args.action == "prune":
        out = store.prune(
            drop_superseded=args.superseded,
            drop_unverified=args.unverified,
            drop_conflicts=args.conflicts,
            drop_mock=args.mock,
            min_sources=args.min_sources,
        ).model_dump()
    elif args.action == "set-revision-order":
        store.set_revision_order(args.document_id, args.revisions)
        out = {"document": args.document_id, "revision_order": store.revision_orders[args.document_id]}
    elif args.action == "revisions":
        out = {"document": args.document_id, "revisions": store.to_graph().revisions_of(args.document_id), "changes": store.revision_changes(args.document_id)}
        changed = False
    elif args.action == "export":
        graph = store.to_graph(include_superseded=not args.current_only)
        if args.json:
            Path(args.json).write_text(json.dumps(export_graph(graph), indent=2), encoding="utf-8")
            out["json"] = args.json
        if args.mermaid:
            Path(args.mermaid).write_text(to_mermaid(graph, documents_only=args.overview), encoding="utf-8")
            out["mermaid"] = args.mermaid
        if args.llm_text:
            from csr_lab.relationships.render import to_llm_markdown

            Path(args.llm_text).write_text(to_llm_markdown(graph, title=f"Relationship graph: {path.stem}", include_ids=args.with_ids), encoding="utf-8")
            out["llm_text"] = args.llm_text
        changed = False
    if changed:
        store.save(path)
        out["saved"] = str(path)
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_workflow(args: argparse.Namespace) -> int:
    from csr_lab.pipelines.registry import build_workflow, get_strategy

    _register_yaml_strategies(args.experiments)
    print(build_workflow(get_strategy(args.strategy)).get_graph().draw_mermaid())
    return 0


def cmd_schema(args: argparse.Namespace) -> int:
    from csr_lab.schemas import AnalysisResult

    text = json.dumps(AnalysisResult.model_json_schema(), indent=2)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


def cmd_sweep(args: argparse.Namespace) -> int:
    from csr_lab.evaluation.sweep import load_sweep, run_sweep

    _register_yaml_strategies(args.experiments)
    sweep = load_sweep(args.name, load_experiments_file(args.experiments))
    if args.repetitions:
        sweep = sweep.model_copy(update={"repetitions": args.repetitions})
    base = _experiment_llm(args, load_experiment(sweep.experiment, args.experiments))
    _print_llm(apply_experiment(load_config(args.config), base))
    out_dir, rows = run_sweep(
        sweep,
        base,
        load_config(args.config),
        output_root=Path(args.output_dir) if args.output_dir else None,
        database_path=Path(args.db) if args.db else None,
        progress=(lambda m: print(f"  {m}", file=sys.stderr)) if not args.quiet else None,
    )
    print(f"sweep report: {out_dir / 'sweep_report.md'}")
    print(f"sweep table:  {out_dir / 'sweep.csv'}")
    return 0


def cmd_check_text(args: argparse.Namespace) -> int:
    from csr_lab.extraction.fidelity import check_fragments, format_report
    from csr_lab.ingestion.loader import load_document, processing_load_options
    from csr_lab.ingestion.provided_chunks import is_chunk_payload, parse_chunks

    config = load_config(args.config)
    source = load_document(args.source, **processing_load_options(config.processing))
    data = json.loads(Path(args.fragments).read_text(encoding="utf-8"))
    if is_chunk_payload(data):
        fragments = [{"id": c.chunk_id, "title": c.title, "text": c.content, "source_pages": c.source_pages} for c in parse_chunks(data["chunks"])]
    elif isinstance(data, list):
        fragments = data
    else:
        raise ValueError("fragments must be a chunk file, or a JSON list of strings or {id, text} objects")
    report = check_fragments(source.text or "", fragments, document_id=source.document_id, pages=source.pages, min_ratio=args.min_ratio)
    print(format_report(report))
    if args.out:
        Path(args.out).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    import os

    from csr_lab.config import ChunkingSettings
    from csr_lab.extraction.fidelity import format_report
    from csr_lab.ingestion.loader import expand_inputs, load_bundle, processing_load_options
    from csr_lab.ingestion.provided_chunks import write_chunk_file
    from csr_lab.ingestion.units import model_chunks, segments_from_chunks
    from csr_lab.llm.interface import CallRecorder, StructuredLLM
    from csr_lab.llm.provider import create_provider
    from csr_lab.pipelines.assembly import fidelity_report

    config = _load(args)
    settings = ChunkingSettings(batch_words=args.batch_words, target_chunk_words=args.target_words)
    documents = load_bundle(args.inputs, **processing_load_options(config.processing))
    provider = create_provider(config.llm, run_seed=args.seed)
    recorder = CallRecorder()
    try:
        made = model_chunks(StructuredLLM(provider, config.llm, recorder), documents, settings, config.processing.max_document_chars)
    finally:
        provider.close()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_path = {p.name: p for p in expand_inputs(args.inputs)}
    for doc in documents:
        inputs = made.chunks.get(doc.document_id, [])
        _, checks = segments_from_chunks(doc, inputs, origin="llm", settings=settings)
        source = by_path.get(doc.filename)
        rel = os.path.relpath(source.resolve(), out_dir.resolve()) if source else None
        path = write_chunk_file(
            out_dir / f"{Path(doc.filename).stem}.chunks.json",
            [{"chunk_id": c.chunk_id, "title": c.title, "content": c.content} for c in inputs],
            source_file=rel,
            chunker=f"csr-lab semantic_chunking ({config.llm.provider}/{config.llm.model})",
        )
        report = fidelity_report({"chunk_checks": checks, "documents": [doc], "segments": []})
        print(f"== {path}")
        if report:
            print(format_report(report, max_rows=20))
    for doc_id, b, err in made.failed_batches:
        print(f"warning: {doc_id} batch {b} failed and was kept as one chunk: {err}", file=sys.stderr)
    if provider.is_mock:
        print("note: chunks made by the offline mock model (not real LLM output)", file=sys.stderr)
    return 0


def cmd_llm_check(args: argparse.Namespace) -> int:
    from pydantic import BaseModel

    from csr_lab.llm.interface import CallRecorder, StructuredLLM
    from csr_lab.llm.provider import create_provider

    class Pong(BaseModel):
        ok: bool
        message: str = ""

    _register_yaml_strategies(args.experiments)
    if args.experiment:
        experiment = load_experiment(args.experiment, args.experiments)
        try:
            config = apply_experiment(load_config(args.config), _experiment_llm(args, experiment))
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    else:
        profile = args.llm or default_llm_profile()
        print(f"profile: {profile or '(none: configs/default.yaml llm settings)'}")
        config = load_config(args.config, _llm_overrides(args), llm_profile=profile)
        if config.llm.provider != "mock" and _print_served_models(config) is None:
            return 1
        try:
            config = _resolve_model(config)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
    provider = create_provider(config.llm)
    print(json.dumps(provider.describe(), indent=2, default=str))
    if provider.is_mock:
        print("mock provider: nothing to connect to")
        return 0
    llm = StructuredLLM(provider, config.llm, CallRecorder())
    try:
        result = llm.call(
            task="llm_check",
            stage="llm_check",
            system="You are a connectivity check.",
            user='Reply with the JSON object {"ok": true, "message": "pong"}.\n<input_json>\n{}\n</input_json>',
            response_model=Pong,
            prompt_version="check",
        )
    finally:
        provider.close()
    call = llm.recorder.calls[-1]
    print(f"success: {result.ok}  attempts: {call.attempts}  latency: {call.latency_seconds}s  tokens in/out: {call.input_tokens}/{call.output_tokens}  mode: {llm.mode}")
    for err in result.errors:
        print(f"  {err}")
    return 0 if result.ok else 1


def _print_served_models(config: AppConfig) -> list | None:
    from csr_lab.llm.interface import LLMProviderError
    from csr_lab.llm.provider import list_models

    try:
        models, source = list_models(config.llm)
    except LLMProviderError as exc:
        print(f"error: {exc}\n  is the server running? (LM Studio: Developer tab > Start Server, or `lms server start`)", file=sys.stderr)
        return None
    print(f"models served at {config.llm.base_url} ({'LM Studio API' if source == 'lmstudio' else '/models'}):")
    for m in models:
        state = {True: "loaded", False: "not loaded", None: ""}[m.loaded]
        extra = ", ".join(f"{k}={v}" for k, v in m.details.items())
        kind = f" [{m.model_type}]" if m.model_type else ""
        print(f"  {m.model_id}{kind}  {state}  {extra}".rstrip())
    if not models:
        print("  (none)")
    return models


def cmd_llms(args: argparse.Namespace) -> int:
    data = load_llm_profiles(args.llms_file)
    profiles = data["profiles"]
    env = __import__("os").environ.get(LLM_ENV_VAR)
    default = env or data["default"]
    if args.action == "list":
        source = f"${LLM_ENV_VAR}" if env else "default: in configs/llms.yaml"
        print(f"default profile: {default or '(none)'}  (from {source})\n")
        for name, s in profiles.items():
            s = s or {}
            mark = "*" if name == default else " "
            target = s.get("base_url") or ""
            print(f" {mark} {name:12} {s.get('provider', '?'):18} {s.get('model', '?'):28} {target}")
        print("\nuse one: --llm NAME on any command; `csr-lab llms use NAME` makes it the default")
        return 0
    name = args.name or default
    if args.action == "models":
        if not name:
            raise ValueError("name a profile: csr-lab llms models NAME")
        config = load_config(args.config, llm_profile=name)
        if config.llm.provider == "mock":
            print("mock profile: no server")
            return 0
        return 0 if _print_served_models(config) is not None else 1
    # use
    if not args.name:
        raise ValueError("name a profile: csr-lab llms use NAME")
    if args.name not in profiles:
        raise KeyError(f"unknown LLM profile {args.name!r}; available: {sorted(profiles)}")
    import re

    path = Path(args.llms_file) if args.llms_file else DEFAULT_LLMS_PATH
    text = path.read_text(encoding="utf-8")
    text, n = re.subn(r"^default:.*$", f"default: {args.name}", text, count=1, flags=re.M)
    if not n:
        text = f"default: {args.name}\n" + text
    path.write_text(text, encoding="utf-8")
    print(f"default LLM profile is now {args.name} ({path})")
    if env:
        print(f"note: ${LLM_ENV_VAR}={env} is set and wins over the file", file=sys.stderr)
    return 0


def cmd_build_graph(args: argparse.Namespace) -> int:
    from csr_lab.pipelines.graph_build import build_graph, file_sha256, plan_bundles
    from csr_lab.relationships.store import GraphStore

    _register_yaml_strategies(args.experiments)
    config = load_config(args.config, _llm_overrides(args), llm_profile=args.llm or default_llm_profile())
    folder = Path(args.folder) if args.folder else config.resolve_path(config.input)
    graph = Path(args.graph) if args.graph else config.resolve_path(config.graph_store.path)
    use = config.graph_store.use if args.use_graph is None else args.use_graph
    update = config.graph_store.update if args.update_graph is None else args.update_graph
    bundles = plan_bundles(folder, args.group)
    if not bundles:
        print(f"no .pdf/.txt/.json files in {folder}", file=sys.stderr)
        return 1
    known = GraphStore.load(graph).known_file_hashes() if graph.is_file() else set()
    print(f"graph:  {graph} ({'exists' if graph.is_file() else 'new'}; {'used' if use else 'not used'} as context, {'updated' if update else 'NOT updated'})", file=sys.stderr)
    print(f"folder: {folder} ({len(bundles)} bundles, grouped by {args.group})", file=sys.stderr)
    if args.dry_run:
        for b in bundles:
            done = not args.force and all(file_sha256(f) in known for f in b.files)
            print(f"  {'skip' if done else 'run '}  {b.name}: {', '.join(f.name for f in b.files)}")
        return 0
    config = _resolve_model(config)
    _print_llm(config)
    outcome = build_graph(
        folder,
        config=config,
        strategy=args.strategy,
        graph_path=graph,
        use=use,
        update=update,
        force=args.force,
        group=args.group,
        run_seed=args.seed,
        views=not args.no_views,
        output_root=Path(args.output_dir) if args.output_dir else None,
        progress=(lambda m: print(f"  {m}", file=sys.stderr)) if not args.quiet else None,
    )
    print(f"report: {outcome.out_dir / 'build_report.md'}")
    for path in outcome.view_paths:
        print(f"view:   {path}")
    if config.llm.provider == "mock":
        print("note: offline mock model (not real LLM output); use --llm lmstudio for a real model", file=sys.stderr)
    return 0 if not any(r["status"] in ("error", "failed") for r in outcome.rows) else 1


def cmd_graph_context(args: argparse.Namespace) -> int:
    from csr_lab.conflicts.detector import CONTEXT_FORMATS, graph_context_for_pair
    from csr_lab.relationships.builder import graph_from_result
    from csr_lab.schemas import AnalysisResult

    result = AnalysisResult.model_validate_json(Path(args.result).read_text(encoding="utf-8"))
    graph = graph_from_result(result)
    texts = {r.requirement_id: r for r in result.requirements}
    pairs = [tuple(c.requirement_ids) for c in result.metadata.conflict_candidates]
    if args.requirement:
        pairs = [p for p in pairs if args.requirement in p]
    if not pairs:
        print("no conflict candidate pairs in this result (run a staged strategy with conflicts enabled)")
        return 1
    formats = CONTEXT_FORMATS if args.format == "all" else (args.format,)
    shown = 0
    for pair in pairs:
        rendered = {f: graph_context_for_pair(graph, pair, args.max_items, f) for f in formats}
        if not any(rendered.values()) and not args.include_empty:
            continue
        shown += 1
        if shown > args.limit:
            break
        print(f"=== pair {shown}")
        for letter, rid in zip("AB", pair):
            req = texts.get(rid)
            body = " ".join(req.requirement_text.split()) if req else "?"
            print(f"  {letter} = {rid}\n      {body[:160]}{'...' if len(body) > 160 else ''}")
        for fmt, items in rendered.items():
            print(f"  [{fmt}]")
            for item in items or ["(no graph context)"]:
                print(f"    {json.dumps(item) if isinstance(item, dict) else item}")
    print(
        "\nnote: rebuilt from the exported result's graph, which also holds the final conflict edges "
        "(those are never shown to the model); it can differ slightly from the graph used during the run."
    )
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    from csr_lab.storage.database import ExperimentStore

    config = load_config(args.config)
    with ExperimentStore(Path(args.db) if args.db else config.resolve_path(config.database_path)) as store:
        for row in store.experiments():
            print(f"{row['experiment_id']}  {row['status']:9}  {row['provider']}/{row['model']}  {row['dataset_name']} v{row['dataset_version']} ({row['split']})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="csr-lab", description="CSR reference extraction and conflict detection research framework")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, experiments: bool = True) -> None:
        p.add_argument("--config", help="configuration YAML (default: configs/default.yaml)")
        if experiments:
            p.add_argument("--experiments", help="experiments YAML (default: configs/experiments.yaml)")

    def llm_args(p: argparse.ArgumentParser) -> None:
        g = p.add_argument_group("LLM selection (profiles live in configs/llms.yaml)")
        g.add_argument("--llm", metavar="PROFILE", help=f"LLM profile, e.g. lmstudio or mock (default: ${LLM_ENV_VAR} or `default:` in llms.yaml)")
        g.add_argument("--model", help="model name on top of the profile ('auto': whatever the server has loaded)")
        g.add_argument("--base-url", help="server URL on top of the profile, e.g. http://localhost:1234/v1")
        g.add_argument("--provider", choices=["mock", "openai_compatible"], help=argparse.SUPPRESS)

    def graph_args(p: argparse.ArgumentParser, default_note: str) -> None:
        g = p.add_argument_group("saved graph")
        g.add_argument("--graph", metavar="PATH", help=f"saved graph JSON to work on ({default_note})")
        g.add_argument("--use-graph", action=argparse.BooleanOptionalAction, default=None,
                       help="graph-assisted strategies start from the saved graph's documents (default: graph_store.use)")
        g.add_argument("--update-graph", action=argparse.BooleanOptionalAction, default=None,
                       help="merge the results into the saved graph; --no-update-graph leaves it untouched (default: graph_store.update)")

    p = sub.add_parser("generate-data", help="write the synthetic dataset (or a scaled one)")
    p.add_argument("--out", help="output directory")
    p.add_argument("--scaled", type=int, metavar="N", help="generate scaled cases with N requirements each")
    p.add_argument("--refs", type=int, default=2, help="references per requirement (scaled)")
    p.add_argument("--filler", type=int, default=2, help="filler sentences per requirement (scaled)")
    p.add_argument("--position", default="mixed", choices=["start", "middle", "end", "mixed"])
    p.add_argument("--cases", type=int, default=3, help="number of scaled cases")
    p.add_argument("--pdf", action="store_true", help="write the PDF variant (headers, footers, page breaks, drawn tables)")
    p.add_argument("--lines-per-page", type=int, default=44, help="PDF variant: lines per page")
    p.add_argument("--chunked", choices=["paragraph", "window"], help="write a pre-chunked variant made by a simulated external chunker")
    p.add_argument("--chunk-words", type=int, default=120, help="chunked variant: words per chunk")
    p.set_defaults(func=cmd_generate_data)

    p = sub.add_parser("strategies", help="list available strategies")
    common(p)
    p.set_defaults(func=cmd_strategies)

    p = sub.add_parser("analyze", help="analyse input documents with one strategy")
    p.add_argument("inputs", nargs="+", help=".txt/.pdf/.json files or directories (one bundle)")
    p.add_argument("--strategy", default="graph_assisted")
    p.add_argument("--out", help="write the JSON result here (default: stdout)")
    p.add_argument("--graph-out", help="also write the relationship graph JSON")
    p.add_argument("--mermaid", help="also write a Mermaid diagram of the graph")
    p.add_argument("--overview", action="store_true", help="Mermaid: one box per document (easier for non-technical readers)")
    p.add_argument("--llm-text", metavar="PATH", help="also write the graph as Markdown for an LLM to read")
    p.add_argument("--seed", type=int)
    llm_args(p)
    graph_args(p, "no saved graph unless given")
    p.add_argument("--graph-store", metavar="PATH", help=argparse.SUPPRESS)  # older name: --graph PATH --no-use-graph
    p.add_argument("--use-saved-graph", action="store_true", help=argparse.SUPPRESS)
    common(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("build-graph", help="analyse a folder of PDFs (default: input/) and build up a saved graph")
    p.add_argument("folder", nargs="?", help="folder to read; subfolders are bundles (default: `input:` in the config)")
    graph_args(p, "default: graph_store.path in the config, graphs/main.json")
    p.add_argument("--strategy", default="graph_assisted")
    p.add_argument("--group", choices=["folder", "file", "all"], default="folder",
                   help="folder: each subfolder is one bundle, loose files another; file: one file at a time; all: one bundle")
    p.add_argument("--force", action="store_true", help="re-analyse files that are already in the graph")
    p.add_argument("--dry-run", action="store_true", help="show which bundles would run or be skipped; no model calls")
    p.add_argument("--no-views", action="store_true", help="do not write the .overview.mmd / .detail.mmd / .llm.md views")
    p.add_argument("--seed", type=int)
    p.add_argument("--output-dir", help="where the build report goes (default: outputs/graph_builds/)")
    p.add_argument("--quiet", action="store_true")
    llm_args(p)
    common(p)
    p.set_defaults(func=cmd_build_graph)

    p = sub.add_parser("llms", help="list LLM profiles, show a server's models, or set the default profile")
    p.add_argument("action", nargs="?", choices=["list", "models", "use"], default="list")
    p.add_argument("name", nargs="?", help="profile name (models: default profile if omitted)")
    p.add_argument("--llms-file", help="profiles YAML (default: configs/llms.yaml)")
    common(p, experiments=False)
    p.set_defaults(func=cmd_llms)

    p = sub.add_parser("experiment", help="run a named experiment from experiments.yaml")
    p.add_argument("name")
    p.add_argument("--repetitions", type=int)
    p.add_argument("--strategies", help="comma-separated list overriding the experiment's strategies")
    p.add_argument("--split", choices=["dev", "heldout", "all"])
    p.add_argument("--output-dir")
    p.add_argument("--db", help="SQLite file (default from config)")
    p.add_argument("--quiet", action="store_true")
    llm_args(p)
    common(p)
    p.set_defaults(func=cmd_experiment)

    p = sub.add_parser("evaluate", help="score one result JSON against one gold JSON")
    p.add_argument("prediction")
    p.add_argument("gold")
    p.add_argument("--threshold", type=float, default=0.75)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("graph", help="query or export the relationship graph of a result JSON")
    p.add_argument("result")
    p.add_argument("--requirement", help="documents referenced by / requirements connected to this requirement")
    p.add_argument("--document", help="requirements referencing this document node")
    p.add_argument("--shared", action="store_true", help="documents referenced by several requirements")
    p.add_argument("--unresolved", action="store_true", help="unresolved and unavailable documents")
    p.add_argument("--conflicts", action="store_true", help="flagged conflicts and graph rule flags")
    p.add_argument("--evidence", metavar="REL_ID", help="evidence for one relationship")
    p.add_argument("--validate", action="store_true")
    p.add_argument("--export", help="write the graph JSON")
    p.add_argument("--mermaid", help="write a Mermaid diagram")
    p.add_argument("--overview", action="store_true", help="Mermaid: one box per document")
    p.add_argument("--llm-text", metavar="PATH", help="write the graph as Markdown for an LLM to read")
    p.add_argument("--with-ids", action="store_true", help="LLM text: include node IDs so a model can cite them")
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("graph-store", help="inspect, add to and prune a saved (persistent) relationship graph")
    store_sub = p.add_subparsers(dest="action", required=True)
    s = store_sub.add_parser("info", help="size, revision status and structural validation")
    s.add_argument("store")
    s = store_sub.add_parser("sources", help="list the analyses and manual edits in the store")
    s.add_argument("store")
    s = store_sub.add_parser("merge", help="merge result JSON files into the store (created if missing)")
    s.add_argument("store")
    s.add_argument("results", nargs="+")
    s.add_argument("--accumulate", action="store_true", help="keep earlier readings of a document revision instead of replacing them")
    s.add_argument("--no-conflicts", action="store_true", help="do not store potential-conflict hypotheses")
    s = store_sub.add_parser("remove-source", help="undo one merged analysis")
    s.add_argument("store")
    s.add_argument("source_id")
    s = store_sub.add_parser("remove-document", help="remove a document revision (or a document with all revisions)")
    s.add_argument("store")
    s.add_argument("document_id")
    s = store_sub.add_parser("prune", help="remove outdated or low-confidence content")
    s.add_argument("store")
    s.add_argument("--superseded", action="store_true", help="requirements of older revisions")
    s.add_argument("--unverified", action="store_true", help="references that did not pass verification")
    s.add_argument("--conflicts", action="store_true", help="all potential-conflict hypotheses")
    s.add_argument("--mock", action="store_true", help="everything contributed by the offline mock model")
    s.add_argument("--min-sources", type=int, default=1, help="references/conflicts seen in fewer analyses")
    s = store_sub.add_parser("set-revision-order", help="declare a document's revision order, oldest first")
    s.add_argument("store")
    s.add_argument("document_id")
    s.add_argument("revisions", nargs="+")
    s = store_sub.add_parser("revisions", help="a document's revisions and what changed between them")
    s.add_argument("store")
    s.add_argument("document_id")
    s = store_sub.add_parser("export", help="write the stored graph as JSON and/or Mermaid")
    s.add_argument("store")
    s.add_argument("--json")
    s.add_argument("--mermaid")
    s.add_argument("--overview", action="store_true", help="Mermaid: one box per document")
    s.add_argument("--llm-text", metavar="PATH", help="Markdown for an LLM to read")
    s.add_argument("--with-ids", action="store_true", help="LLM text: include node IDs")
    s.add_argument("--current-only", action="store_true", help="leave out superseded requirements")
    p.set_defaults(func=cmd_graph_store)

    p = sub.add_parser("workflow", help="print a strategy's LangGraph workflow as Mermaid")
    p.add_argument("--strategy", default="graph_assisted")
    common(p)
    p.set_defaults(func=cmd_workflow)

    p = sub.add_parser("schema", help="print the JSON Schema of the exported result")
    p.add_argument("--out")
    p.set_defaults(func=cmd_schema)

    p = sub.add_parser("sweep", help="run an experiment over a grid of settings (sweeps: in experiments.yaml)")
    p.add_argument("name")
    p.add_argument("--repetitions", type=int)
    p.add_argument("--output-dir")
    p.add_argument("--db", help="SQLite file (default from config)")
    p.add_argument("--quiet", action="store_true")
    llm_args(p)
    common(p)
    p.set_defaults(func=cmd_sweep)

    p = sub.add_parser("check-text", help="check whether chunks/quotes occur verbatim in a source document")
    p.add_argument("source", help=".txt/.pdf/.json source document")
    p.add_argument("fragments", help="chunk file, or JSON list of strings / {id, text} objects")
    p.add_argument("--min-ratio", type=float, default=0.6, help="word ratio below which a fragment is 'not found'")
    p.add_argument("--out", help="also write the full report as JSON")
    common(p, experiments=False)
    p.set_defaults(func=cmd_check_text)

    p = sub.add_parser("chunk", help="run only the model chunking stage and write chunk files")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--batch-words", type=int, default=2500)
    p.add_argument("--target-words", type=int, default=500)
    p.add_argument("--seed", type=int)
    llm_args(p)
    common(p, experiments=False)
    p.set_defaults(func=cmd_chunk)

    p = sub.add_parser("llm-check", help="show the resolved endpoint/auth and send one test request")
    p.add_argument("--experiment", help="use this experiment's llm settings")
    llm_args(p)
    common(p)
    p.set_defaults(func=cmd_llm_check)

    p = sub.add_parser("graph-context", help="show the graph context a conflict prompt gets for each pair, per format")
    p.add_argument("result", help="result JSON (any strategy with conflict candidates)")
    p.add_argument("--format", choices=["all", "sentences", "triples", "json", "ids"], default="all")
    p.add_argument("--requirement", help="only pairs that include this requirement ID")
    p.add_argument("--max-items", type=int, default=6)
    p.add_argument("--limit", type=int, default=5, help="number of pairs to show")
    p.add_argument("--include-empty", action="store_true", help="also show pairs without graph context")
    p.set_defaults(func=cmd_graph_context)

    p = sub.add_parser("history", help="list experiments stored in SQLite")
    p.add_argument("--db")
    common(p, experiments=False)
    p.set_defaults(func=cmd_history)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
