import json

import httpx
import pytest
from pydantic import BaseModel

from csr_lab.config import LLMSettings
from csr_lab.llm.interface import CallRecorder, StructuredLLM, extract_json
from csr_lab.llm.mock_provider import MockProvider
from csr_lab.llm.provider import OpenAICompatibleProvider
from csr_lab.prompts import prompt_versions
from csr_lab.prompts import references as reference_prompt


class Answer(BaseModel):
    value: int


def test_extract_json_modes():
    assert extract_json('```json\n{"value": 1}\n```', lenient=False) == {"value": 1}
    assert extract_json('Here you go: {"value": 2} hope it helps', lenient=True) == {"value": 2}
    with pytest.raises(json.JSONDecodeError):
        extract_json('Here you go: {"value": 2}', lenient=False)


def _provider(handler, **settings):
    s = LLMSettings(
        provider="openai_compatible",
        model="test-model",
        base_url="http://llm.test/v1",
        api_key_env="CSR_TEST_KEY",
        max_transport_retries=1,
        retry_backoff_seconds=0,
        **settings,
    )
    return s, OpenAICompatibleProvider(s, client=httpx.Client(transport=httpx.MockTransport(handler)))


def _ok(content: str, usage=True):
    body = {"model": "test-model", "choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    if usage:
        body["usage"] = {"prompt_tokens": 11, "completion_tokens": 3}
    return httpx.Response(200, json=body)


def test_openai_compatible_payload_and_usage(monkeypatch):
    monkeypatch.setenv("CSR_TEST_KEY", "secret")
    seen = {}

    def handler(request: httpx.Request):
        seen["body"] = json.loads(request.content)
        seen["auth"] = request.headers.get("authorization")
        return _ok('{"value": 5}')

    settings, provider = _provider(handler)
    llm = StructuredLLM(provider, settings, CallRecorder(), seed=42)
    result = llm.call(task="t", stage="s", system="sys", user="usr", response_model=Answer, prompt_version="v1")
    assert result.ok and result.value.value == 5
    body = seen["body"]
    assert body["model"] == "test-model" and body["seed"] == 42 and body["temperature"] == 0.0
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"]["properties"]["value"]["type"] == "integer"
    assert seen["auth"] == "Bearer secret"
    stats = llm.recorder.stats(1.0, 2.0)
    assert (stats.input_tokens, stats.output_tokens) == (11, 3)
    assert stats.estimated_cost_usd == pytest.approx(11e-6 + 6e-6)


def test_transport_retry_and_fallback_to_prompt_mode():
    attempts = {"n": 0}

    def handler(request: httpx.Request):
        attempts["n"] += 1
        body = json.loads(request.content)
        if attempts["n"] == 1:
            return httpx.Response(503, text="busy")
        if "response_format" in body:
            return httpx.Response(400, text="response_format json_schema is not supported")
        assert "JSON Schema" in body["messages"][0]["content"]  # schema moved into the prompt
        return _ok('Result: {"value": 9}', usage=False)

    settings, provider = _provider(handler)
    recorder = CallRecorder()
    llm = StructuredLLM(provider, settings, recorder)
    result = llm.call(task="t", stage="s", system="sys", user="usr", response_model=Answer, prompt_version="v1")
    assert result.ok and result.value.value == 9
    assert llm.mode == "prompt" and recorder.warnings
    assert recorder.stats(None, None).input_tokens is None  # not reported -> not invented


def test_invalid_output_retries_are_bounded():
    responses = iter(['{"value": "x"}', '{"value": 3}'])
    settings, provider = _provider(lambda r: _ok(next(responses)))
    llm = StructuredLLM(provider, settings, CallRecorder())
    result = llm.call(task="t", stage="s", system="sys", user="usr", response_model=Answer, prompt_version="v1")
    assert result.ok and llm.recorder.calls[0].attempts == 2
    assert llm.recorder.stats(None, None).json_validation_failures == 1

    settings, provider = _provider(lambda r: _ok("not json"), max_validation_retries=1)
    llm = StructuredLLM(provider, settings, CallRecorder())
    result = llm.call(task="t", stage="s", system="sys", user="usr", response_model=Answer, prompt_version="v1")
    assert not result.ok and result.value is None and len(result.errors) == 2


def test_mock_provider_is_seeded_and_uses_prompt_input():
    from csr_lab.config import MockSettings
    from csr_lab.llm.interface import GenerationRequest

    payload = {"requirements": [{"requirement_id": "R1", "chapter": "1", "text": "The supplier shall follow CVM-QS-110 and DRW-045."}]}
    system, user = reference_prompt.SPEC.render(payload)
    request = GenerationRequest(messages=[{"role": "system", "content": system}, {"role": "user", "content": user}], task="reference_extraction", schema_name="x", json_schema=None, mode="json_schema")
    perfect = MockProvider(MockSettings(reference_miss_rate=0, hallucination_rate=0, quote_paraphrase_rate=0))
    refs = json.loads(perfect.generate(request).text)["results"][0]["references"]
    assert [r["identifier"] for r in refs] == ["CVM-QS-110", "DRW-045"]
    noisy = MockSettings(reference_miss_rate=0.5, hallucination_rate=0.5)
    assert MockProvider(noisy, run_seed=1).generate(request).text == MockProvider(noisy, run_seed=1).generate(request).text


def test_prompt_versions_include_content_hash():
    versions = prompt_versions()
    assert set(versions) == {"single_pass_extraction", "requirement_identification", "reference_extraction", "reference_verification", "conflict_analysis"}
    assert all("@" in v for v in versions.values())
