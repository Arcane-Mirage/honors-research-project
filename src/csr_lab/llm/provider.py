"""src/csr_lab/llm/provider.py

The OpenAI-compatible provider, model discovery, and the provider factory.

**Chat completions.** One small HTTP adapter covers commercial APIs that speak the
OpenAI chat-completions protocol and local servers (llama.cpp ``llama-server``, vLLM,
Ollama, LM Studio). Plain HTTP rather than a LangChain wrapper keeps the exact request,
token counts and retries visible and identical across providers.

Parameters differ between servers, so each one is configurable:
``max_tokens_parameter``, ``send_seed``, ``temperature: null`` (omit), and
``extra_body`` for anything server-specific. ``chat_path`` and ``query_params``
cover deployment-style URLs (Azure OpenAI: ``.../openai/deployments/<name>`` with
``api-version``), and ``auth`` selects how credentials are sent (``llm/auth.py``).

**Model discovery** (``csr-lab llms`` and ``model: auto``): ask the server which models
it serves.

* Every OpenAI-compatible server answers ``GET {base_url}/models``.
* LM Studio also has a native REST API (``/api/v0/models``) that says which models
  are *loaded* (``/v1/models`` lists every downloaded model when just-in-time
  loading is on), plus quantisation and context length; record these with your results.

``model: auto`` picks the only model the server offers, else the only loaded LM
Studio model, else fails with the list to choose from. The resolved name replaces
``auto`` before a run, so results always record the real model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from csr_lab.config import AUTO_MODEL, LLMSettings
from csr_lab.llm.auth import AuthHeaders
from csr_lab.llm.interface import (
    GenerationRequest,
    GenerationResponse,
    LLMProvider,
    LLMProviderError,
    StructuredOutputUnsupported,
)

# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


class OpenAICompatibleProvider(LLMProvider):
    is_mock = False

    def __init__(self, settings: LLMSettings, client: httpx.Client | None = None):
        self.base_url = (settings.base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
        self._client = client or httpx.Client(timeout=settings.timeout_seconds)
        if settings.model == AUTO_MODEL:  # the CLI resolves this earlier; library callers may not
            settings = resolve_model(settings, self._client)
        self.settings = settings
        self.auth = AuthHeaders(settings)

    @property
    def url(self) -> str:
        path = self.settings.chat_path if self.settings.chat_path.startswith("/") else f"/{self.settings.chat_path}"
        return f"{self.base_url}{path}"

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "openai_compatible",
            "model": self.settings.model,
            "base_url": self.base_url,
            "url": self.url,
            "query_params": dict(self.settings.query_params),
            "auth": self.auth.describe(),
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
        url = self.url
        params = dict(self.settings.query_params) or None

        last_error = "unknown error"
        for attempt in range(self.settings.max_transport_retries + 1):
            if attempt:
                time.sleep(min(self.settings.retry_backoff_seconds * 2 ** (attempt - 1), 30))
            headers = {"Content-Type": "application/json", **self.auth.headers()}
            try:
                response = self._client.post(url, json=payload, headers=headers, params=params)
            except httpx.TimeoutException as exc:
                last_error = f"timeout: {exc}"
                continue
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc}"
                continue

            if response.status_code == 401 and self.settings.auth.method in ("command", "azure_identity"):
                self.auth.invalidate()  # expired token: fetch a new one on the next attempt
                last_error = f"HTTP 401: {response.text[:300]}"
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


# ---------------------------------------------------------------------------
# Model discovery
# ---------------------------------------------------------------------------

_NON_CHAT_TYPES = {"embeddings", "embedding"}


@dataclass
class ServedModel:
    model_id: str
    loaded: bool | None = None  # None: the server does not say
    model_type: str | None = None  # LM Studio: llm | vlm | embeddings
    details: dict[str, Any] = field(default_factory=dict)  # quantization, context length, ...

    @property
    def is_chat(self) -> bool:
        if self.model_type:
            return self.model_type not in _NON_CHAT_TYPES
        return "embed" not in self.model_id.lower()


def _root_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    return base[: -len("/v1")] if base.endswith("/v1") else base


def _get(client: httpx.Client, url: str, headers: dict[str, str]) -> Any | None:
    try:
        response = client.get(url, headers=headers)
    except httpx.HTTPError:
        return None
    if response.status_code >= 400:
        return None
    try:
        return response.json()
    except ValueError:
        return None


def _parse_lmstudio(data: Any) -> list[ServedModel]:
    """LM Studio native API: ``/api/v0/models`` (``data``/``state``) or newer ``models``/``loaded_instances``."""
    if not isinstance(data, dict):
        return []
    items = data.get("data") if isinstance(data.get("data"), list) else data.get("models")
    out = []
    for item in items or []:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id") or item.get("key")
        if not model_id:
            continue
        if "state" in item:
            loaded = item.get("state") == "loaded"
        elif "loaded_instances" in item:
            loaded = bool(item.get("loaded_instances"))
        else:
            loaded = None
        details = {
            k: item[k]
            for k in ("quantization", "max_context_length", "loaded_context_length", "arch", "publisher")
            if item.get(k) is not None
        }
        out.append(ServedModel(model_id=model_id, loaded=loaded, model_type=item.get("type"), details=details))
    return out


def list_models(settings: LLMSettings, client: httpx.Client | None = None) -> tuple[list[ServedModel], str]:
    """Models the server offers and where the list came from (``lmstudio`` or ``openai``).

    Raises ``LLMProviderError`` when the server cannot be reached.
    """
    if settings.provider != "openai_compatible":
        return [ServedModel(model_id=settings.model, loaded=True)], "mock"
    base = (settings.base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")
    own = client is None
    client = client or httpx.Client(timeout=min(settings.timeout_seconds, 15))
    try:
        headers = AuthHeaders(settings).headers()
        lmstudio = _parse_lmstudio(_get(client, f"{_root_url(base)}/api/v0/models", headers))
        if lmstudio:
            return lmstudio, "lmstudio"
        try:
            response = client.get(f"{base}/models", headers=headers, params=dict(settings.query_params) or None)
        except httpx.HTTPError as exc:
            raise LLMProviderError(f"cannot reach {base}: {exc}") from exc
        if response.status_code >= 400:
            raise LLMProviderError(f"{base}/models answered HTTP {response.status_code}: {response.text[:200]}")
        try:
            items = response.json().get("data") or []
        except (ValueError, AttributeError) as exc:
            raise LLMProviderError(f"{base}/models: unexpected response: {exc}") from exc
        return [ServedModel(model_id=i["id"]) for i in items if isinstance(i, dict) and i.get("id")], "openai"
    finally:
        if own:
            client.close()


def choose_model(models: list[ServedModel]) -> str:
    """The model ``auto`` means, or ``LLMProviderError`` explaining what to do."""
    chat = [m for m in models if m.is_chat]
    if len(chat) == 1:
        return chat[0].model_id
    loaded = [m for m in chat if m.loaded]
    if len(loaded) == 1:
        return loaded[0].model_id
    names = ", ".join(m.model_id + (" (loaded)" if m.loaded else "") for m in chat) or "none"
    if not chat:
        raise LLMProviderError("the server offers no chat model; load one (LM Studio: `lms load MODEL_KEY`)")
    if len(loaded) > 1:
        reason = f"{len(loaded)} models are loaded"
    else:
        reason = f"{len(chat)} models are available and none is loaded"
    raise LLMProviderError(
        f"model: auto is ambiguous ({reason}). Load exactly one model, or name it with --model "
        f"or `model:` in configs/llms.yaml. Models: {names}"
    )


def resolve_model(settings: LLMSettings, client: httpx.Client | None = None) -> LLMSettings:
    """Return ``settings`` with ``model: auto`` replaced by the served model's name."""
    if settings.provider != "openai_compatible" or settings.model != AUTO_MODEL:
        return settings
    models, _ = list_models(settings, client)
    return settings.model_copy(update={"model": choose_model(models)})


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------

def create_provider(settings: LLMSettings, run_seed: int | None = None) -> LLMProvider:
    """Build the configured provider. A new instance per run keeps runs isolated."""
    if settings.provider == "mock":
        from csr_lab.llm.mock_provider import MockProvider

        return MockProvider(settings.mock, run_seed=run_seed or 0, model=settings.model)
    if settings.provider == "openai_compatible":
        return OpenAICompatibleProvider(settings)
    raise ValueError(f"unknown LLM provider {settings.provider!r}")
