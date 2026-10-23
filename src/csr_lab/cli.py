"""Command-line interface: `csr-lab <command>` (or `python -m csr_lab <command>`)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from csr_lab.config import (
    PROJECT_ROOT,
    apply_experiment,
    load_config,
    load_experiment,
    load_experiments_file,
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


def _llm_overrides(args: argparse.Namespace) -> dict:
    llm = {}
    for key in ("provider", "model", "base_url"):
        value = getattr(args, key, None)
        if value:
            llm[key] = value
    return {"llm": llm} if llm else {}


def cmd_generate_data(args: argparse.Namespace) -> int:
    from csr_lab.datasets.synthetic import generate_scaled_case, write_dataset

    if args.scaled:
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

    _register_yaml_strategies(args.experiments)
    config = load_config(args.config, _llm_overrides(args))
    result = analyze_files(args.inputs, strategy=args.strategy, config=config, run_seed=args.seed)
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
        Path(args.mermaid).write_text(to_mermaid(graph), encoding="utf-8")
        print(f"wrote {args.mermaid}")
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
    config = apply_experiment(load_config(args.config), experiment)
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
        Path(args.mermaid).write_text(queries.to_mermaid(graph), encoding="utf-8")
        out["mermaid"] = args.mermaid
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

    p = sub.add_parser("generate-data", help="write the synthetic dataset (or a scaled one)")
    p.add_argument("--out", help="output directory")
    p.add_argument("--scaled", type=int, metavar="N", help="generate scaled cases with N requirements each")
    p.add_argument("--refs", type=int, default=2, help="references per requirement (scaled)")
    p.add_argument("--filler", type=int, default=2, help="filler sentences per requirement (scaled)")
    p.add_argument("--position", default="mixed", choices=["start", "middle", "end", "mixed"])
    p.add_argument("--cases", type=int, default=3, help="number of scaled cases")
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
    p.add_argument("--seed", type=int)
    p.add_argument("--provider", choices=["mock", "openai_compatible"])
    p.add_argument("--model")
    p.add_argument("--base-url")
    common(p)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("experiment", help="run a named experiment from experiments.yaml")
    p.add_argument("name")
    p.add_argument("--repetitions", type=int)
    p.add_argument("--strategies", help="comma-separated list overriding the experiment's strategies")
    p.add_argument("--split", choices=["dev", "heldout", "all"])
    p.add_argument("--output-dir")
    p.add_argument("--db", help="SQLite file (default from config)")
    p.add_argument("--quiet", action="store_true")
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
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("workflow", help="print a strategy's LangGraph workflow as Mermaid")
    p.add_argument("--strategy", default="graph_assisted")
    common(p)
    p.set_defaults(func=cmd_workflow)

    p = sub.add_parser("schema", help="print the JSON Schema of the exported result")
    p.add_argument("--out")
    p.set_defaults(func=cmd_schema)

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
