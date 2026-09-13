from __future__ import annotations

from typing import Any

import httpx

from vuc.config import VisionLLMConfig
from vuc.llm import ChatCompletionsClient


def _stub_config(*, temperature: float | None = None) -> VisionLLMConfig:
    return VisionLLMConfig(
        base_url_env="TEST_VLM_BASE_URL",
        model_env="TEST_VLM_MODEL",
        api_key_env="TEST_VLM_API_KEY",
        input_cost_env="TEST_VLM_INPUT_COST",
        cached_input_cost_env="TEST_VLM_CACHED_INPUT_COST",
        output_cost_env="TEST_VLM_OUTPUT_COST",
        timeout_s=30.0,
        max_retries=1,
        max_output_tokens=1024,
        input_cost_per_million_usd=0.0,
        cached_input_cost_per_million_usd=0.0,
        output_cost_per_million_usd=0.0,
        temperature=temperature,
    )


def test_client_includes_configured_temperature(monkeypatch) -> None:
    monkeypatch.setenv("TEST_VLM_BASE_URL", "https://api.test/v1")
    monkeypatch.setenv("TEST_VLM_MODEL", "model-test")
    monkeypatch.setenv("TEST_VLM_API_KEY", "test-key")

    client = ChatCompletionsClient(_stub_config(temperature=0.2))
    captured_payload: dict[str, Any] = {}

    def mock_post(url, *, json: dict[str, Any]):
        captured_payload.update(json)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(client._client, "post", mock_post)
    client.complete([{"role": "user", "content": "hello"}])

    assert captured_payload["temperature"] == 0.2


def test_client_omits_temperature_when_none(monkeypatch) -> None:
    monkeypatch.setenv("TEST_VLM_BASE_URL", "https://api.test/v1")
    monkeypatch.setenv("TEST_VLM_MODEL", "model-test")
    monkeypatch.setenv("TEST_VLM_API_KEY", "test-key")

    client = ChatCompletionsClient(_stub_config(temperature=None))
    captured_payload: dict[str, Any] = {}

    def mock_post(url, *, json: dict[str, Any]):
        captured_payload.update(json)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(client._client, "post", mock_post)
    client.complete([{"role": "user", "content": "hello"}])

    assert "temperature" not in captured_payload


def test_complete_temperature_override(monkeypatch) -> None:
    monkeypatch.setenv("TEST_VLM_BASE_URL", "https://api.test/v1")
    monkeypatch.setenv("TEST_VLM_MODEL", "model-test")
    monkeypatch.setenv("TEST_VLM_API_KEY", "test-key")

    client = ChatCompletionsClient(_stub_config(temperature=0.2))
    captured_payload: dict[str, Any] = {}

    def mock_post(url, *, json: dict[str, Any]):
        captured_payload.clear()
        captured_payload.update(json)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(client._client, "post", mock_post)

    # Override to 0.0
    client.complete([{"role": "user", "content": "hello"}], temperature=0.0)
    assert captured_payload["temperature"] == 0.0

    # Override to None (omit)
    client.complete([{"role": "user", "content": "hello"}], temperature=None)
    assert "temperature" not in captured_payload
