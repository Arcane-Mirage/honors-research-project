"""Offline end-to-end demonstration (no API key needed).

1. Generates the synthetic dataset if it is missing.
2. Analyses one two-document bundle with every built-in strategy.
3. Runs the `demo_mock` experiment and prints where the report is.

With --write-examples it also refreshes the committed examples in data/examples/.
Everything uses the mock model: the output shows that the system works, not how
well a real LLM performs.
"""

from __future__ import annotations

import argparse
import json
import sys

from csr_lab.config import PROJECT_ROOT, apply_experiment, load_config, load_experiment, load_experiments_file
from csr_lab.datasets.loader import load_dataset
from csr_lab.datasets.synthetic import write_dataset
from csr_lab.evaluation.metrics import evaluate_case
from csr_lab.evaluation.runner import run_experiment
from csr_lab.ingestion.loader import load_bundle
from csr_lab.pipelines.registry import build_workflow, get_strategy, register_variants
from csr_lab.pipelines.run import run_analysis
from csr_lab.relationships.builder import graph_from_result
from csr_lab.relationships.queries import export_graph, to_mermaid

EXAMPLE_CASE = "dev_corvane_packaging"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-examples", action="store_true", help="refresh data/examples/")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)

    manifest = PROJECT_ROOT / "data" / "synthetic" / "manifest.json"
    if not manifest.is_file():
        write_dataset(manifest.parent)
        print(f"generated {manifest}")
    register_variants(load_experiments_file().get("strategies"))
    config = load_config()
    dataset = load_dataset(manifest)
    case = next(c for c in dataset.cases if c.case_id == EXAMPLE_CASE)
    documents = load_bundle(list(case.document_paths))
    gold = case.load_gold()

    print(f"\nOne bundle ({EXAMPLE_CASE}, {len(documents)} documents), mock model:")
    print(f"  {'strategy':18} {'reqs':>4} {'ref F1':>7} {'conflict F1':>11} {'LLM calls':>9}")
    results = {}
    for name in ("single_pass", "staged", "staged_verified", "graph_assisted"):
        result = run_analysis(documents, strategy=name, config=config, case_id=EXAMPLE_CASE, run_seed=1000)
        results[name] = result
        ev = evaluate_case(result, gold)
        f1 = lambda v: "-" if v is None else f"{v:.3f}"  # noqa: E731
        print(f"  {name:18} {len(result.requirements):>4} {f1(ev.references.f1):>7} {f1(ev.conflicts.f1):>11} {result.metadata.operational.llm_calls:>9}")

    if args.write_examples:
        out = PROJECT_ROOT / "data" / "examples"
        out.mkdir(parents=True, exist_ok=True)
        result = results["graph_assisted"]
        (out / f"{EXAMPLE_CASE}.graph_assisted.result.json").write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
        graph = graph_from_result(result)
        (out / f"{EXAMPLE_CASE}.graph_assisted.graph.json").write_text(json.dumps(export_graph(graph), indent=2) + "\n", encoding="utf-8")
        (out / f"{EXAMPLE_CASE}.graph_assisted.graph.mmd").write_text(to_mermaid(graph), encoding="utf-8")
        for name in ("single_pass", "staged_verified", "graph_assisted"):
            mermaid = build_workflow(get_strategy(name)).get_graph().draw_mermaid()
            (out / f"workflow_{name}.mmd").write_text(mermaid, encoding="utf-8")
        print(f"\nwrote examples to {out}")

    experiment = load_experiment("demo_mock").model_copy(update={"repetitions": args.repetitions})
    outcome = run_experiment(apply_experiment(config, experiment), experiment)
    print(f"\nExperiment {outcome.experiment_id} ({args.repetitions} repetitions, held-out split)")
    agg = outcome.summary["aggregates"]
    print(f"  {'strategy':18} {'ref P':>6} {'ref R':>6} {'ref F1':>7} {'conf F1':>8} {'calls':>6}")
    for name, metrics in agg.items():
        def m(key: str, metrics=metrics) -> float:
            return metrics[key]["mean"]

        print(f"  {name:18} {m('references.precision'):6.3f} {m('references.recall'):6.3f} {m('references.f1'):7.3f} {m('conflicts.f1'):8.3f} {m('ops.llm_calls'):6.0f}")
    print(f"\nreport: {outcome.output_dir / 'report.md'}")
    print("These are mock-model results: they demonstrate the pipeline, not real LLM performance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
