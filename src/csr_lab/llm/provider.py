"""OpenAI-compatible chat-completions provider and the provider factory.

One small HTTP adapter covers commercial APIs that speak the OpenAI
chat-completions protocol and local servers (llama.cpp ``llama-server``, vLLM,
Ollama). It is used instead of a LangChain chat-model wrapper so the exact
request, token counts and retries are visible and identical across providers.

Parameters differ between servers, so each one is configurable:
``max_tokens_parameter``, ``send_seed``, ``temperature: null`` (omit), and
``extra_body`` for anything server-specific.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from csr_lab.config import LLMSettings
from csr_lab.llm.interface import (
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    LLMProviderError,
    StructuredOutputUnsupported,
)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatibleProvider(LLMProvider):
    is_mock = False

    def __init__(self, settings: LLMSettings, client: httpx.Client | None = None):
        self.settings = settings
        self.base_url = (settings.base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        self._client = client or httpx.Client(timeout=settings.timeout_seconds)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "openai_compatible",
            "model": self.settings.model,
            "base_url": self.base_url,
            "structured_output": self.settings.structured_output,
            "temperature": self.settings.temperature,
            "max_output_tokens": self.settings.max_output_tokens,
            "seed_sent": self.settings.send_seed,
        }

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": self.settings.model, "messages": request.messages}
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_output_tokens is not None:
            payload[self.settings.max_tokens_parameter] = request.max_output_tokens
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.mode == "json_schema" and request.json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": request.schema_name, "schema": request.json_schema, "strict": False},
            }
        elif request.mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        payload.update(self.settings.extra_body)
        return payload

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        payload = self._payload(request)
        headers = {"Content-Type": "application/json"}
        if key := self.settings.api_key():
            headers["Authorization"] = f"Bearer {key}"
        url = f"{self.base_url}/chat/completions"

        last_error = "unknown error"
        for attempt in range(self.settings.max_transport_retries + 1):
            if attempt:
                time.sleep(min(self.settings.retry_backoff_seconds * 2 ** (attempt - 1), 30))
            try:
                response = self._client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                last_error = f"timeout: {exc}"
                continue
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc}"
                continue

            if response.status_code in _RETRY_STATUS:
                last_error = f"HTTP {response.status_code}: {response.text[:300]}"
                continue
            if response.status_code in (400, 422) and "response_format" in payload:
                body = response.text.lower()
                if "response_format" in body or "json_schema" in body or "schema" in body:
                    raise StructuredOutputUnsupported(f"HTTP {response.status_code}: {response.text[:300]}")
            if response.status_code >= 400:
                raise LLMProviderError(f"HTTP {response.status_code}: {response.text[:500]}")

            try:
                data = response.json()
                choice = data["choices"][0]
                text = choice["message"].get("content") or ""
            except (ValueError, KeyError, IndexError, TypeError) as exc:
                raise LLMProviderError(f"unexpected response format: {exc}") from exc
            usage = data.get("usage") or {}
            return GenerationResponse(
                text=text,
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                model=data.get("model"),
                finish_reason=choice.get("finish_reason"),
            )
        raise LLMProviderError(f"request failed after {self.settings.max_transport_retries + 1} attempts: {last_error}")

    def close(self) -> None:
        self._client.close()


def create_provider(settings: LLMSettings, run_seed: int | None = None) -> LLMProvider:
    """Build the configured provider. A new instance per run keeps runs isolated."""
    if settings.provider == "mock":
        from csr_lab.llm.mock_provider import MockProvider

        return MockProvider(settings.mock, run_seed=run_seed or 0, model=settings.model)
    if settings.provider == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    raise ValueError(f"unknown LLM provider {settings.provider!r}")
