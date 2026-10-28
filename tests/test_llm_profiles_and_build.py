"""tests/test_llm_profiles_and_build.py

LLM profiles (configs/llms.yaml), `model: auto` discovery, build-graph and the LLM-readable graph view.
"""

import json

import httpx
import pytest

from csr_lab.cli import main
from csr_lab.config import (
    AppConfig,
    ExperimentConfig,
    apply_experiment,
    default_llm_profile,
    expand_llm_block,
    load_config,
)
from csr_lab.llm.interface import LLMProviderError
from csr_lab.llm.provider import OpenAICompatibleProvider, choose_model, list_models, resolve_model
from csr_lab.pipelines.graph_build import build_graph, plan_bundles, view_paths_for
from csr_lab.pipelines.run import run_analysis
from csr_lab.relationships.builder import graph_from_result
from csr_lab.relationships.queries import to_mermaid
from csr_lab.relationships.render import to_llm_markdown
from csr_lab.relationships.store import GraphStore

from conftest import SIMPLE_DOC

SECOND_DOC = SIMPLE_DOC.replace("CVM-CSR-900", "CVM-CSR-901").replace("Test Requirements", "Other Requirements")


# -- profiles ----------------------------------------------------------------------


def test_profile_applies_on_top_of_default_yaml():
    config = load_config(llm_profile="lmstudio")
    assert config.llm.provider == "openai_compatible"
    assert config.llm.base_url == "http://localhost:1234/v1"
    assert config.llm.model == "auto"
    assert config.llm.auth.method == "none"
    assert config.llm.retry_backoff_seconds == 2  # untouched default.yaml value


def test_no_profile_keeps_config_file_settings():
    assert load_config().llm.provider == "mock"


def test_profile_block_with_overrides_and_unknown_name():
    block = expand_llm_block({"profile": "lmstudio", "model": "qwen/x", "max_output_tokens": 1000})
    assert block["base_url"] == "http://localhost:1234/v1" and block["model"] == "qwen/x" and block["max_output_tokens"] == 1000
    with pytest.raises(KeyError, match="unknown LLM profile"):
        expand_llm_block("no_such_profile")


def test_experiment_llm_may_name_a_profile():
    experiment = ExperimentConfig(name="x", llm="lmstudio")
    assert experiment.llm == {"profile": "lmstudio"}
    config = apply_experiment(load_config(), experiment)
    assert config.llm.base_url == "http://localhost:1234/v1"


def test_default_profile_from_environment(monkeypatch):
    monkeypatch.setenv("CSR_LAB_LLM", "openai")
    assert default_llm_profile() == "openai"


def test_llms_command_lists_and_sets_default(tmp_path, capsys):
    from csr_lab.config import DEFAULT_LLMS_PATH

    copy = tmp_path / "llms.yaml"
    copy.write_text(DEFAULT_LLMS_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["llms", "--llms-file", str(copy)]) == 0
    assert "lmstudio" in capsys.readouterr().out
    assert main(["llms", "use", "lmstudio", "--llms-file", str(copy)]) == 0
    assert "default: lmstudio" in copy.read_text(encoding="utf-8")
    assert "# Named LLM connections" in copy.read_text(encoding="utf-8")  # comments kept
    assert main(["llms", "use", "nope", "--llms-file", str(copy)]) == 1


def test_experiment_cli_accepts_llm(tmp_path):
    args = ["experiment", "dev", "--llm", "mock", "--quiet", "--output-dir", str(tmp_path), "--db", str(tmp_path / "e.sqlite")]
    assert main(args) == 0
    summary = json.loads(next(tmp_path.glob("*/summary.json")).read_text(encoding="utf-8"))
    assert "mock" in json.dumps(summary)


# -- model discovery -----------------------------------------------------------------


def _client(routes: dict[str, object]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        body = routes.get(request.url.path)
        if body is None:
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


LMSTUDIO_V0 = {
    "data": [
        {"id": "qwen/qwen3.5-9b", "type": "llm", "state": "loaded", "quantization": "Q4_K_M", "max_context_length": 32768},
        {"id": "openai/gpt-oss-20b", "type": "llm", "state": "not-loaded"},
        {"id": "text-embedding-nomic-embed-text-v1.5", "type": "embeddings", "state": "loaded"},
    ]
}


def _lmstudio_settings(**kw):
    return load_config(llm_profile="lmstudio", overrides={"llm": kw}).llm


def test_auto_picks_the_loaded_lmstudio_model():
    settings = _lmstudio_settings()
    models, source = list_models(settings, _client({"/api/v0/models": LMSTUDIO_V0}))
    assert source == "lmstudio"
    assert models[0].details["quantization"] == "Q4_K_M"
    assert resolve_model(settings, _client({"/api/v0/models": LMSTUDIO_V0})).model == "qwen/qwen3.5-9b"


def test_auto_with_newer_lmstudio_format():
    data = {"models": [{"key": "a/model", "type": "llm", "loaded_instances": [{"id": "a"}]}, {"key": "b/model", "type": "llm", "loaded_instances": []}]}
    assert resolve_model(_lmstudio_settings(), _client({"/api/v0/models": data})).model == "a/model"


def test_auto_with_plain_openai_server():
    client = _client({"/v1/models": {"data": [{"id": "my-model.gguf"}]}})
    assert resolve_model(_lmstudio_settings(base_url="http://localhost:8080/v1"), client).model == "my-model.gguf"


def test_auto_is_ambiguous_without_exactly_one_loaded_model():
    data = {"data": [{"id": "a", "type": "llm", "state": "loaded"}, {"id": "b", "type": "llm", "state": "loaded"}]}
    with pytest.raises(LLMProviderError, match="2 models are loaded"):
        resolve_model(_lmstudio_settings(), _client({"/api/v0/models": data}))
    from csr_lab.llm.provider import ServedModel

    with pytest.raises(LLMProviderError, match="none is loaded"):
        choose_model([ServedModel("a", loaded=False), ServedModel("b", loaded=False)])


def test_explicit_model_is_not_looked_up():
    settings = _lmstudio_settings(model="fixed/model")
    assert resolve_model(settings, _client({})) is settings


def test_provider_resolves_auto_itself():
    provider = OpenAICompatibleProvider(_lmstudio_settings(), client=_client({"/api/v0/models": LMSTUDIO_V0}))
    assert provider.settings.model == "qwen/qwen3.5-9b"
    assert provider.describe()["model"] == "qwen/qwen3.5-9b"


def test_unreachable_server_gives_a_clear_cli_error(capsys):
    assert main(["llm-check", "--llm", "lmstudio", "--base-url", "http://127.0.0.1:9/v1"]) == 1
    assert "is the server running" in capsys.readouterr().err


# -- build-graph -----------------------------------------------------------------------


@pytest.fixture
def input_dir(tmp_path):
    folder = tmp_path / "input"
    (folder / "bundle_a").mkdir(parents=True)
    (folder / "bundle_a" / "a.txt").write_text(SIMPLE_DOC, encoding="utf-8")
    (folder / "loose.txt").write_text(SECOND_DOC, encoding="utf-8")
    (folder / "README.md").write_text("ignored", encoding="utf-8")
    return folder


def test_plan_bundles(input_dir):
    assert [(b.name, len(b.files)) for b in plan_bundles(input_dir)] == [("input", 1), ("bundle_a", 1)]
    assert len(plan_bundles(input_dir, "file")) == 2
    assert [len(b.files) for b in plan_bundles(input_dir, "all")] == [2]


def test_build_graph_builds_skips_and_reports(input_dir, tmp_path, config: AppConfig):
    graph = tmp_path / "graphs" / "g.json"
    first = build_graph(input_dir, config=config, strategy="graph_assisted", graph_path=graph, output_root=tmp_path / "out")
    assert [r["status"] for r in first.rows] == ["completed", "completed"]
    assert first.rows[1]["used_saved_graph"] is True  # the second bundle saw the first one's documents
    store = GraphStore.load(graph)
    assert len(store.known_file_hashes()) == 2
    assert all(p.is_file() for p in view_paths_for(graph))
    assert view_paths_for(graph)[-1].name == "g.llm.md"
    report = (first.out_dir / "build_report.md").read_text(encoding="utf-8")
    assert "| bundle_a | 1 | completed |" in report and "not evidence" in report

    second = build_graph(input_dir, config=config, strategy="graph_assisted", graph_path=graph, output_root=tmp_path / "out")
    assert all(r["status"].startswith("skipped") for r in second.rows)
    forced = build_graph(input_dir, config=config, strategy="graph_assisted", graph_path=graph, force=True, output_root=tmp_path / "out")
    assert all(r["status"] == "completed" for r in forced.rows)


def test_build_graph_read_only_leaves_graph_untouched(input_dir, tmp_path, config: AppConfig):
    graph = tmp_path / "g.json"
    build_graph(input_dir, config=config, strategy="graph_assisted", graph_path=graph, group="all", output_root=tmp_path / "out")
    before = graph.read_bytes()
    (input_dir / "new.txt").write_text(SIMPLE_DOC.replace("CVM-CSR-900", "CVM-CSR-902"), encoding="utf-8")
    outcome = build_graph(input_dir, config=config, strategy="graph_assisted", graph_path=graph, update=False, group="file", output_root=tmp_path / "out")
    assert graph.read_bytes() == before
    assert any(r["status"] == "completed" and "nodes_added" not in r for r in outcome.rows)


def test_build_graph_cli(input_dir, tmp_path, capsys):
    graph = tmp_path / "g.json"
    common = [str(input_dir), "--graph", str(graph), "--output-dir", str(tmp_path / "out"), "--quiet"]
    assert main(["build-graph", *common, "--dry-run"]) == 0
    assert not graph.exists()
    assert main(["build-graph", *common]) == 0
    assert graph.is_file()
    capsys.readouterr()
    assert main(["build-graph", *common, "--dry-run"]) == 0
    assert capsys.readouterr().out.count("skip") == 2


def test_analyze_graph_flags(simple_doc_path, tmp_path):
    graph = tmp_path / "g.json"
    out = ["--out", str(tmp_path / "r.json")]
    assert main(["analyze", str(simple_doc_path), "--graph", str(graph), "--no-update-graph", *out]) == 0
    assert not graph.exists()
    assert main(["analyze", str(simple_doc_path), "--graph", str(graph), *out]) == 0
    assert graph.is_file() and view_paths_for(graph)[0].is_file()
    assert main(["analyze", str(simple_doc_path), "--use-graph", *out]) == 1  # needs --graph


# -- views ------------------------------------------------------------------------------


def test_llm_markdown_and_overview(simple_docs, config):
    result = run_analysis(simple_docs, strategy="graph_assisted", config=config)
    graph = graph_from_result(result)
    text = to_llm_markdown(graph)
    assert "**CVM-CSR-900" in text and "CVM-QS-110" in text
    assert "refers to CVM-QS-110" in text
    assert "states that CVM-PKG-050 does not apply" in text
    assert "## Potential conflicts" in text and "105 °C" in text
    req_id = result.requirements[0].requirement_id
    assert req_id not in text and req_id in to_llm_markdown(graph, include_ids=True)

    overview = to_mermaid(graph, documents_only=True)
    assert "requirement" not in overview.lower().replace("requirements", "")
    assert "explicitly_excludes" in overview
    assert "(not provided)" in overview
