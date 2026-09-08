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


def image_content(path: Path) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


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
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.model = model
        self.max_retries = config.max_retries
        self.max_output_tokens = config.max_output_tokens
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
    ) -> ChatResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens or self.max_output_tokens,
        }
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
