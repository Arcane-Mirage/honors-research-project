"""src/csr_lab/llm/auth.py

Request credentials for OpenAI-compatible servers (``LLMSettings.auth``).

Kept separate from the HTTP client so the same client works against a local
llama-server (no credentials), a commercial API (bearer key), Azure OpenAI with an
API key (``api-key`` header) and Azure OpenAI with Microsoft Entra ID tokens.

Tokens from ``command`` and ``azure_identity`` are cached and refreshed after
``refresh_seconds`` (or before the reported expiry). Credentials are never logged
or stored with results; ``describe()`` reports only the method and header name.
"""

from __future__ import annotations

import shlex
import subprocess
import threading
import time
from typing import Any

from csr_lab.config import LLMSettings
from csr_lab.llm.interface import LLMProviderError

_EXPIRY_MARGIN = 120  # refresh this many seconds before a reported expiry


class AuthHeaders:
    def __init__(self, settings: LLMSettings):
        self.settings = settings
        self.auth = settings.auth
        self._lock = threading.Lock()
        self._token: str | None = None
        self._expires_at = 0.0
        self._credential: Any = None

    def describe(self) -> dict[str, Any]:
        info: dict[str, Any] = {"method": self.auth.method}
        if self.auth.method != "none":
            info["header"] = self.auth.header
            info["scheme"] = self.auth.scheme
        if self.auth.method == "env":
            info["api_key_env"] = self.settings.api_key_env
            info["key_present"] = self.settings.api_key() is not None
        if self.auth.method == "azure_identity":
            info["scope"] = self.auth.scope
        return info

    def headers(self) -> dict[str, str]:
        token = self._current_token()
        if not token:
            return {}
        value = f"{self.auth.scheme} {token}" if self.auth.scheme else token
        return {self.auth.header: value}

    def invalidate(self) -> None:
        """Forget a cached token (called after HTTP 401 so the next attempt fetches a new one)."""
        with self._lock:
            self._token = None
            self._expires_at = 0.0

    # -- token sources -------------------------------------------------------

    def _current_token(self) -> str | None:
        method = self.auth.method
        if method == "none":
            return None
        if method == "env":
            return self.settings.api_key()
        with self._lock:
            if self._token and time.time() < self._expires_at:
                return self._token
            token, expires_at = self._command_token() if method == "command" else self._azure_token()
            self._token = token
            self._expires_at = min(expires_at, time.time() + self.auth.refresh_seconds) if expires_at else time.time() + self.auth.refresh_seconds
            return token

    def _command_token(self) -> tuple[str, float | None]:
        command = self.auth.command
        if not command:
            raise LLMProviderError("auth.method 'command' needs auth.command")
        args = shlex.split(command) if isinstance(command, str) else list(command)
        try:
            done = subprocess.run(args, capture_output=True, text=True, timeout=60, check=False)
        except (OSError, subprocess.SubprocessError) as exc:
            raise LLMProviderError(f"auth command failed to run: {exc}") from exc
        token = done.stdout.strip()
        if done.returncode != 0 or not token:
            raise LLMProviderError(f"auth command exited with {done.returncode}: {done.stderr.strip()[:200]}")
        return token, None

    def _azure_token(self) -> tuple[str, float | None]:
        if self._credential is None:
            try:
                from azure.identity import DefaultAzureCredential  # type: ignore[import-not-found]
            except ImportError as exc:
                raise LLMProviderError("auth.method 'azure_identity' needs `pip install azure-identity`") from exc
            self._credential = DefaultAzureCredential()
        try:
            access = self._credential.get_token(self.auth.scope)
        except Exception as exc:  # azure.identity raises its own exception hierarchy
            raise LLMProviderError(f"could not get an Entra ID token for {self.auth.scope}: {exc}") from exc
        return access.token, float(access.expires_on) - _EXPIRY_MARGIN
