"""SQLite storage for experiment records.

Tables:
* ``experiments``  one row per experiment run of the runner (config, dataset hash, environment)
* ``runs``         one row per strategy x repetition
* ``case_results`` one row per run x case (status, error, output file, cost)
* ``metrics``      long format: (run_id, case_id, metric, value); case_id '*' holds run aggregates
* ``llm_calls``    every model call (stage, prompt version, attempts, tokens, latency)

This is separate from LangGraph state; workflows never read from it.
Inspect it with Python's ``sqlite3`` module or any SQLite browser.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from csr_lab.schemas import LLMCallSummary

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    dataset_name TEXT,
    dataset_version TEXT,
    dataset_sha256 TEXT,
    split TEXT,
    provider TEXT,
    model TEXT,
    is_mock INTEGER,
    config_json TEXT,
    environment_json TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
    strategy TEXT NOT NULL,
    strategy_description TEXT,
    strategy_options_json TEXT,
    repetition INTEGER NOT NULL,
    seed INTEGER,
    prompt_versions_json TEXT,
    status TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT,
    cases_total INTEGER,
    cases_failed INTEGER
);
CREATE TABLE IF NOT EXISTS case_results (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    case_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    analysis_id TEXT,
    output_path TEXT,
    seconds REAL,
    llm_calls INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    input_chars INTEGER,
    output_chars INTEGER,
    PRIMARY KEY (run_id, case_id)
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    case_id TEXT NOT NULL,
    scope TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL,
    PRIMARY KEY (run_id, case_id, scope, metric)
);
CREATE TABLE IF NOT EXISTS llm_calls (
    run_id TEXT NOT NULL REFERENCES runs(run_id),
    case_id TEXT NOT NULL,
    call_id TEXT NOT NULL,
    stage TEXT,
    task TEXT,
    prompt_version TEXT,
    success INTEGER,
    attempts INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    input_chars INTEGER,
    output_chars INTEGER,
    latency_seconds REAL,
    graph_informed INTEGER,
    errors_json TEXT,
    PRIMARY KEY (run_id, case_id, call_id)
);
"""


def _num(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


class ExperimentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> ExperimentStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def create_experiment(self, **fields: Any) -> None:
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        self.conn.execute(f"INSERT INTO experiments ({cols}) VALUES ({marks})", list(fields.values()))
        self.conn.commit()

    def finish_experiment(self, experiment_id: str, status: str, finished_at: str) -> None:
        self.conn.execute("UPDATE experiments SET status=?, finished_at=? WHERE experiment_id=?", (status, finished_at, experiment_id))
        self.conn.commit()

    def start_run(self, **fields: Any) -> None:
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        self.conn.execute(f"INSERT INTO runs ({cols}) VALUES ({marks})", list(fields.values()))
        self.conn.commit()

    def finish_run(self, run_id: str, *, status: str, finished_at: str, cases_total: int, cases_failed: int, prompt_versions: dict) -> None:
        self.conn.execute(
            "UPDATE runs SET status=?, finished_at=?, cases_total=?, cases_failed=?, prompt_versions_json=? WHERE run_id=?",
            (status, finished_at, cases_total, cases_failed, json.dumps(prompt_versions), run_id),
        )
        self.conn.commit()

    def add_case_result(self, **fields: Any) -> None:
        cols = ", ".join(fields)
        marks = ", ".join("?" for _ in fields)
        self.conn.execute(f"INSERT OR REPLACE INTO case_results ({cols}) VALUES ({marks})", list(fields.values()))

    def add_metrics(self, run_id: str, case_id: str, scope: str, metrics: dict[str, Any]) -> None:
        rows = [(run_id, case_id, scope, k, _num(v)) for k, v in metrics.items() if k not in ("case_id", "status")]
        self.conn.executemany("INSERT OR REPLACE INTO metrics VALUES (?, ?, ?, ?, ?)", rows)

    def add_llm_calls(self, run_id: str, case_id: str, calls: Iterable[LLMCallSummary]) -> None:
        rows = [
            (
                run_id,
                case_id,
                c.call_id,
                c.stage,
                c.task,
                c.prompt_version,
                int(c.success),
                c.attempts,
                c.input_tokens,
                c.output_tokens,
                c.input_chars,
                c.output_chars,
                c.latency_seconds,
                int(c.graph_informed),
                json.dumps(c.errors),
            )
            for c in calls
        ]
        self.conn.executemany("INSERT OR REPLACE INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def commit(self) -> None:
        self.conn.commit()

    # -- reading ---------------------------------------------------------

    def experiments(self) -> list[dict[str, Any]]:
        cur = self.conn.execute("SELECT experiment_id, name, created_at, status, dataset_name, dataset_version, split, provider, model FROM experiments ORDER BY created_at")
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def run_metrics(self, experiment_id: str, metric: str, scope: str = "strict") -> list[dict[str, Any]]:
        cur = self.conn.execute(
            """SELECT r.strategy, r.repetition, m.value FROM metrics m JOIN runs r ON r.run_id = m.run_id
               WHERE r.experiment_id = ? AND m.case_id = '*' AND m.scope = ? AND m.metric = ?
               ORDER BY r.strategy, r.repetition""",
            (experiment_id, scope, metric),
        )
        return [{"strategy": s, "repetition": rep, "value": v} for s, rep, v in cur.fetchall()]
