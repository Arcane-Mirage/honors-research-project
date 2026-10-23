"""Provider-independent LLM interface with validated structured output.

Providers only move text (``LLMProvider.generate``). ``StructuredLLM`` adds the
behaviour that must be identical for every provider and strategy:

1. ask for JSON (native JSON-schema mode, JSON-object mode, or schema-in-prompt);
2. parse and validate the reply against a Pydantic model;
3. on invalid output, send the validation error back and retry (bounded);
4. record every call (attempts, characters, tokens, latency, failures).

An invalid reply is never turned into a successful empty prediction: the
caller receives ``ok=False`` and decides how to record the failure.
"""

from __future__ import annotations

import json
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Literal, TypeVar

from pydantic import BaseModel, ValidationError

from csr_lab.config import LLMSettings
from csr_lab.schemas import LLMCallSummary, OperationalStats

T = TypeVar("T", bound=BaseModel)
R = TypeVar("R")
OutputMode = Literal["json_schema", "json_object", "prompt"]


class LLMProviderError(RuntimeError):
    """Transport or server failure after the provider's own retries."""


class StructuredOutputUnsupported(LLMProviderError):
    """The server rejected the requested response_format."""


@dataclass
class GenerationRequest:
    messages: list[dict[str, str]]
    task: str
    schema_name: str
    json_schema: dict[str, Any] | None
    mode: OutputMode
    temperature: float | None = None
    max_output_tokens: int | None = None
    seed: int | None = None


@dataclass
class GenerationResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None
    finish_reason: str | None = None


class LLMProvider(ABC):
    is_mock: bool = False

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse: ...

    @abstractmethod
    def describe(self) -> dict[str, Any]: ...

    def close(self) -> None:  # pragma: no cover - optional hook
        return None


@dataclass
class StructuredResult(Generic[T]):
    ok: bool
    value: T | None
    call_id: str
    errors: list[str] = field(default_factory=list)


class CallRecorder:
    """Thread-safe log of LLM calls for one analysis run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counter = 0
        self.calls: list[LLMCallSummary] = []
        self.warnings: list[str] = []

    def next_id(self) -> str:
        with self._lock:
            self._counter += 1
            return f"CALL-{self._counter:04d}"

    def add(self, summary: LLMCallSummary) -> None:
        with self._lock:
            self.calls.append(summary)

    def warn(self, message: str) -> None:
        with self._lock:
            if message not in self.warnings:
                self.warnings.append(message)

    def stats(self, pricing_in: float | None, pricing_out: float | None) -> OperationalStats:
        with self._lock:
            calls = list(self.calls)
        tokens_in = [c.input_tokens for c in calls]
        tokens_out = [c.output_tokens for c in calls]
        input_tokens = sum(tokens_in) if calls and all(t is not None for t in tokens_in) else None  # type: ignore[arg-type]
        output_tokens = sum(tokens_out) if calls and all(t is not None for t in tokens_out) else None  # type: ignore[arg-type]
        cost = None
        if input_tokens is not None and output_tokens is not None and pricing_in is not None and pricing_out is not None:
            cost = round(input_tokens / 1e6 * pricing_in + output_tokens / 1e6 * pricing_out, 6)
        return OperationalStats(
            llm_calls=len(calls),
            llm_failed_calls=sum(1 for c in calls if not c.success),
            json_validation_failures=sum(
                sum(1 for e in c.errors if e.startswith("invalid output")) for c in calls
            ),
            retries=sum(max(c.attempts - 1, 0) for c in calls),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            input_chars=sum(c.input_chars for c in calls),
            output_chars=sum(c.output_chars for c in calls),
            estimated_cost_usd=cost,
            graph_informed_calls=sum(1 for c in calls if c.graph_informed),
        )


_FENCE_RE = re.compile(r"^```(?:json)?\s*(?P<body>.*?)\s*```$", re.DOTALL | re.IGNORECASE)


def extract_json(text: str, *, lenient: bool) -> Any:
    """Parse a JSON object from model text.

    Code fences are always tolerated. With ``lenient`` (prompt mode) the first
    ``{`` ... last ``}`` span is also tried, because plain-text models often add
    a sentence around the JSON. The result must still validate against the schema.
    """
    stripped = text.strip()
    if m := _FENCE_RE.match(stripped):
        stripped = m.group("body")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        if not lenient:
            raise
        start, end = stripped.find("{"), stripped.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(stripped[start : end + 1])


def schema_instructions(response_model: type[BaseModel]) -> str:
    schema = json.dumps(response_model.model_json_schema(), separators=(",", ":"))
    return (
        "Respond with a single JSON object only, with no prose before or after it. "
        f"The object must validate against this JSON Schema:\n{schema}"
    )


def _short_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]]
        return "; ".join(parts)
    return str(exc)[:300]


class StructuredLLM:
    """Validated JSON calls on top of any ``LLMProvider``."""

    def __init__(self, provider: LLMProvider, settings: LLMSettings, recorder: CallRecorder, seed: int | None = None):
        self.provider = provider
        self.settings = settings
        self.recorder = recorder
        self.seed = seed if seed is not None else settings.seed
        self._mode: OutputMode = settings.structured_output
        self._mode_lock = threading.Lock()

    @property
    def mode(self) -> OutputMode:
        return self._mode

    def _messages(self, system: str, user: str, response_model: type[BaseModel], mode: OutputMode) -> list[dict[str, str]]:
        if mode == "json_schema":
            system = f"{system}\n\nRespond with JSON only."
        else:
            system = f"{system}\n\n{schema_instructions(response_model)}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def call(
        self,
        *,
        task: str,
        stage: str,
        system: str,
        user: str,
        response_model: type[T],
        prompt_version: str,
        graph_informed: bool = False,
    ) -> StructuredResult[T]:
        call_id = self.recorder.next_id()
        mode = self._mode
        messages = self._messages(system, user, response_model, mode)
        schema = response_model.model_json_schema()
        errors: list[str] = []
        attempts = 0
        input_chars = output_chars = 0
        tokens_in: list[int | None] = []
        tokens_out: list[int | None] = []
        started = time.perf_counter()
        value: T | None = None
        max_attempts = 1 + self.settings.max_validation_retries

        while attempts < max_attempts:
            attempts += 1
            request = GenerationRequest(
                messages=messages,
                task=task,
                schema_name=response_model.__name__,
                json_schema=schema if mode == "json_schema" else None,
                mode=mode,
                temperature=self.settings.temperature,
                max_output_tokens=self.settings.max_output_tokens,
                seed=self.seed if self.settings.send_seed else None,
            )
            input_chars += sum(len(m["content"]) for m in messages)
            try:
                response = self.provider.generate(request)
            except StructuredOutputUnsupported as exc:
                if self.settings.fallback_to_prompt_mode and mode != "prompt":
                    with self._mode_lock:
                        self._mode = "prompt"
                    mode = "prompt"
                    self.recorder.warn(f"server rejected response_format; switched to prompt mode ({exc})")
                    messages = self._messages(system, user, response_model, mode)
                    errors.append(f"structured output unsupported: {str(exc)[:200]}")
                    continue
                errors.append(f"provider error: {exc}")
                break
            except LLMProviderError as exc:
                errors.append(f"provider error: {exc}")
                break

            output_chars += len(response.text)
            tokens_in.append(response.input_tokens)
            tokens_out.append(response.output_tokens)
            try:
                data = extract_json(response.text, lenient=(mode == "prompt"))
                value = response_model.model_validate(data)
                break
            except (json.JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                errors.append(f"invalid output (attempt {attempts}): {_short_error(exc)}")
                messages = [
                    *messages,
                    {"role": "assistant", "content": response.text[:4000]},
                    {
                        "role": "user",
                        "content": (
                            "Your previous reply was not valid for the required JSON schema: "
                            f"{_short_error(exc)}. Reply again with only the corrected JSON object."
                        ),
                    },
                ]

        def total(values: list[int | None]) -> int | None:
            return sum(values) if values and all(v is not None for v in values) else None  # type: ignore[arg-type]

        self.recorder.add(
            LLMCallSummary(
                call_id=call_id,
                stage=stage,
                task=task,
                prompt_version=prompt_version,
                success=value is not None,
                attempts=attempts,
                input_tokens=total(tokens_in),
                output_tokens=total(tokens_out),
                input_chars=input_chars,
                output_chars=output_chars,
                latency_seconds=round(time.perf_counter() - started, 4),
                graph_informed=graph_informed,
                errors=errors,
            )
        )
        return StructuredResult(ok=value is not None, value=value, call_id=call_id, errors=errors)

    def map(self, fn: Callable[[Any], R], items: list[Any]) -> list[R]:
        """Apply ``fn`` to items, concurrently up to ``max_concurrency``; order is preserved."""
        if self.settings.max_concurrency <= 1 or len(items) <= 1:
            return [fn(item) for item in items]
        with ThreadPoolExecutor(max_workers=self.settings.max_concurrency) as pool:
            return list(pool.map(fn, items))
