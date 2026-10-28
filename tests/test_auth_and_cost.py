"""tests/test_auth_and_cost.py

Provider authentication options (llm/auth.py) and per-stage / per-unit cost.
"""

import importlib.util
import sys

import httpx
import pytest

from csr_lab.config import AuthSettings, LLMSettings
from csr_lab.llm.interface import CallRecorder, GenerationRequest, LLMProviderError
from csr_lab.llm.provider import OpenAICompatibleProvider
from csr_lab.pipelines.run import run_analysis


def _request():
    return GenerationRequest(messages=[{"role": "user", "content": "hi"}], task="t", schema_name="S", json_schema=None, mode="json_object")


def _ok():
    return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {"prompt_tokens": 7, "completion_tokens": 2}})


def _provider(handler, **settings):
    s = LLMSettings(provider="openai_compatible", model="dep", base_url="https://res.example/openai/deployments/dep", max_transport_retries=1, retry_backoff_seconds=0, **settings)
    return OpenAICompatibleProvider(s, client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_default_bearer_and_azure_key_header(monkeypatch):
    monkeypatch.setenv("CSR_TEST_KEY", "secret")
    seen = []
    p = _provider(lambda r: seen.append(r) or _ok(), api_key_env="CSR_TEST_KEY")
    p.generate(_request())
    assert seen[-1].headers["authorization"] == "Bearer secret"
    p = _provider(
        lambda r: seen.append(r) or _ok(),
        api_key_env="CSR_TEST_KEY",
        auth=AuthSettings(header="api-key", scheme=None),
        query_params={"api-version": "2024-10-21"},
    )
    p.generate(_request())
    req = seen[-1]
    assert req.headers["api-key"] == "secret" and "authorization" not in req.headers
    assert req.url.path == "/openai/deployments/dep/chat/completions"
    assert req.url.params["api-version"] == "2024-10-21"
    described = p.describe()
    assert described["auth"] == {"method": "env", "header": "api-key", "scheme": None, "api_key_env": "CSR_TEST_KEY", "key_present": True}
    assert "secret" not in str(described)


def test_command_token_is_cached_and_refreshed_after_401(tmp_path):
    counter = tmp_path / "n"
    counter.write_text("0")
    script = tmp_path / "token.py"
    script.write_text(
        "import pathlib,sys\np=pathlib.Path(sys.argv[1]);n=int(p.read_text())+1;p.write_text(str(n));print(f'tok{n}')\n"
    )
    seen = []

    def handler(r):
        seen.append(r.headers["authorization"])
        return httpx.Response(401, text="expired") if len(seen) == 2 else _ok()

    p = _provider(handler, auth=AuthSettings(method="command", command=[sys.executable, str(script), str(counter)]))
    p.generate(_request())
    p.generate(_request())  # 401 -> token refreshed -> retried
    assert seen == ["Bearer tok1", "Bearer tok1", "Bearer tok2"]


def test_no_auth_and_missing_azure_identity():
    seen = []
    p = _provider(lambda r: seen.append(r) or _ok(), auth=AuthSettings(method="none"))
    p.generate(_request())
    assert "authorization" not in seen[-1].headers
    if importlib.util.find_spec("azure") is None:  # only testable where azure-identity is absent
        p = _provider(lambda r: _ok(), auth=AuthSettings(method="azure_identity"))
        with pytest.raises(LLMProviderError, match="azure-identity"):
            p.generate(_request())


def test_public_settings_hide_the_auth_command():
    s = LLMSettings(auth=AuthSettings(method="command", command="get-token --secret-flag"))
    assert s.public_dict()["auth"]["command"] == "(set)"


def test_stage_costs_and_unit_costs(simple_docs, config):
    result = run_analysis(simple_docs, strategy="staged_verified", config=config)
    ops = result.metadata.operational
    assert set(ops.stage_costs) >= {"identify_requirements", "extract_references", "analyze_conflicts"}
    assert sum(sc.calls for sc in ops.stage_costs.values()) == ops.llm_calls
    assert ops.token_source == "estimated_from_chars"
    assert ops.estimated_tokens == sum(sc.estimated_tokens for sc in ops.stage_costs.values()) > 0
    assert ops.pages_source == "estimated" and ops.pages > 0
    assert ops.calls_per_page == pytest.approx(ops.llm_calls / ops.pages, rel=1e-2)
    assert ops.tokens_per_requirement == pytest.approx(ops.estimated_tokens / len(result.requirements), rel=1e-2)


def test_reported_tokens_are_used_when_available():
    rec = CallRecorder()
    from csr_lab.schemas import LLMCallSummary

    rec.add(LLMCallSummary(call_id="1", stage="a", task="t", prompt_version="v", success=True, attempts=1, input_tokens=100, output_tokens=20, input_chars=4000, output_chars=80, latency_seconds=0.1))
    stats = rec.stats(None, None)
    assert stats.token_source == "reported" and stats.estimated_tokens == 120
    assert stats.stage_costs["a"].input_tokens == 100
