from __future__ import annotations

import base64
import mimetypes
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from vuc.config import VisionLLMConfig


class LLMError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatResult:
    message: dict[str, Any]
    usage: dict[str, Any]
    latency_s: float
    attempts: int = 1


def input_token_count(usage: dict[str, Any]) -> int:
    """Total prompt tokens. Providers include cached tokens in this figure."""
    return int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)


def cached_input_token_count(usage: dict[str, Any]) -> int:
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        return int(details.get("cached_tokens", 0) or 0)
    return int(usage.get("cached_tokens", 0) or 0)


def output_token_count(usage: dict[str, Any]) -> int:
    return int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)


def reasoning_token_count(usage: dict[str, Any]) -> int:
    """Reasoning tokens, already counted inside the completion total."""
    details = usage.get("completion_tokens_details")
    if isinstance(details, dict):
        return int(details.get("reasoning_tokens", 0) or 0)
    return 0


def estimate_vlm_cost_usd(
    config: VisionLLMConfig,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float | None:
    rates = (
        config.input_cost_per_million_usd,
        config.cached_input_cost_per_million_usd,
        config.output_cost_per_million_usd,
    )
    if all(rate <= 0 for rate in rates):
        return None
    cached = max(0, min(cached_input_tokens, input_tokens))
    uncached = input_tokens - cached
    return (
        uncached * config.input_cost_per_million_usd
        + cached * config.cached_input_cost_per_million_usd
        + output_tokens * config.output_cost_per_million_usd
    ) / 1_000_000


def image_content(path: Path) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


_UNSET = object()


class ChatCompletionsClient:
    def __init__(self, config: VisionLLMConfig) -> None:
        base_url = os.environ.get(config.base_url_env, "").strip()
        model = os.environ.get(config.model_env, "").strip()
        api_key = os.environ.get(config.api_key_env, "").strip()
        missing = [
            name
            for name, value in (
                (config.base_url_env, base_url),
                (config.model_env, model),
                (config.api_key_env, api_key),
            )
            if not value
        ]
        if missing:
            raise LLMError(f"missing environment variables: {', '.join(missing)}")
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/chat/completions"
        self.model = model
        self.max_retries = config.max_retries
        self.max_output_tokens = config.max_output_tokens
        self.temperature = config.temperature
        self._client = httpx.Client(
            timeout=config.timeout_s,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None | object = _UNSET,
    ) -> ChatResult:
        effective_temperature = (
            self.temperature if temperature is _UNSET else temperature
        )
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens or self.max_output_tokens,
        }
        if effective_temperature is not None:
            payload["temperature"] = effective_temperature
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(self.url, json=payload)
                response.raise_for_status()
                data = response.json()
                message = data["choices"][0]["message"]
                if not isinstance(message, dict):
                    raise LLMError("chat completion message is not an object")
                usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
                return ChatResult(
                    message=message,
                    usage=usage,
                    latency_s=time.monotonic() - started,
                    attempts=attempt + 1,
                )
            except (httpx.HTTPError, KeyError, TypeError, ValueError, LLMError) as exc:
                last_error = exc
                retryable = not isinstance(exc, httpx.HTTPStatusError) or (
                    exc.response.status_code == 429 or exc.response.status_code >= 500
                )
                if attempt >= self.max_retries or not retryable:
                    break
                time.sleep(min(2**attempt, 4))
        raise LLMError(f"chat completion failed: {last_error}") from last_error
