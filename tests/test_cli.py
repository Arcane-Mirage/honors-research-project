import json
import sqlite3

from csr_lab.cli import main


def test_generate_data(tmp_path, capsys):
    assert main(["generate-data", "--out", str(tmp_path / "d")]) == 0
    assert (tmp_path / "d" / "manifest.json").is_file()
    assert main(["generate-data", "--out", str(tmp_path / "s"), "--scaled", "12", "--cases", "1"]) == 0
    assert json.loads((tmp_path / "s" / "manifest.json").read_text())["name"] == "synthetic_scaled"


def test_analyze_graph_evaluate(tmp_path, dataset, capsys):
    case = next(c for c in dataset.cases if c.case_id == "dev_corvane_packaging")
    out, graph, mermaid = tmp_path / "r.json", tmp_path / "g.json", tmp_path / "g.mmd"
    args = ["analyze", *map(str, case.document_paths), "--strategy", "graph_assisted", "--out", str(out), "--graph-out", str(graph), "--mermaid", str(mermaid)]
    assert main(args) == 0
    result = json.loads(out.read_text())
    assert result["schema_version"] and result["requirements"]
    assert json.loads(graph.read_text())["edges"]
    assert mermaid.read_text().startswith("flowchart")
    assert "mock" in capsys.readouterr().err

    assert main(["evaluate", str(out), str(case.gold_path)]) == 0
    scores = json.loads(capsys.readouterr().out)
    assert scores["references"]["tp"] > 0

    assert main(["graph", str(out), "--shared", "--conflicts", "--unresolved", "--validate"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["validation"]["valid"] and info["applicability_flags"]


def test_experiment_writes_reports_and_sqlite(tmp_path, capsys):
    db = tmp_path / "exp.sqlite"
    args = ["experiment", "dev_mock", "--repetitions", "2", "--strategies", "single_pass,graph_assisted_no_alias", "--output-dir", str(tmp_path), "--db", str(db), "--quiet"]
    assert main(args) == 0
    exp_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    for name in ("report.md", "summary.json", "per_case_metrics.csv", "per_run_metrics.csv", "llm_calls.csv"):
        assert (exp_dir / name).is_file(), name
    report = (exp_dir / "report.md").read_text()
    assert "Mock provider" in report and "graph_assisted_no_alias" in report
    summary = json.loads((exp_dir / "summary.json").read_text())
    assert summary["repetitions"] == 2 and len(summary["runs"]) == 4
    assert summary["aggregates"]["single_pass"]["references.f1"]["n"] == 2
    con = sqlite3.connect(db)
    assert con.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 4
    assert con.execute("SELECT COUNT(*) FROM case_results").fetchone()[0] == 8
    assert con.execute("SELECT COUNT(*) FROM metrics WHERE case_id='*' AND metric='references.f1'").fetchone()[0] == 8
    assert con.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0] > 0
    assert main(["history", "--db", str(db)]) == 0


def test_listing_commands(capsys):
    assert main(["strategies"]) == 0
    out = capsys.readouterr().out
    assert "graph_assisted_no_alias (variant of graph_assisted)" in out
    assert main(["workflow", "--strategy", "staged_verified"]) == 0
    assert "verify_references" in capsys.readouterr().out
    assert main(["schema"]) == 0
    assert "AnalysisResult" in capsys.readouterr().out
    assert main(["experiment", "no_such_experiment"]) == 1
