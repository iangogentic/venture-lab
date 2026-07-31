"""Native OpenAI Responses API transport tests.

Every request runs through ``httpx.MockTransport``; no test reaches the network
or depends on a real credential.
"""

import json
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import httpx
import pytest
from openrouter.components import (
    ChatFormatJSONObjectConfig,
    ChatFormatJSONSchemaConfig,
    ChatJSONSchemaConfig,
)

import app.llm.adapters.gpt as gpt_module
import app.llm.openai_client as openai_module
from app.config import get_settings
from app.llm.adapters.gpt import GptAdapter
from app.llm.messages import ChatMessage
from app.llm.openai_client import OpenAIResponsesClient
from app.llm.provider import GenerationRequest, GenerationResult, Provider, TokenUsage
from app.utils.errors import ConfigurationError, LLMError

_TEST_KEY = "sk-test-native-openai-123456789"


def _completed(
    text: str = "native answer",
    *,
    model: str = "gpt-5.4-mini",
) -> dict[str, Any]:
    return {
        "id": "resp_test",
        "object": "response",
        "status": "completed",
        "model": model,
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": 11,
            "output_tokens": 7,
            "total_tokens": 18,
        },
    }


@contextmanager
def _client(
    handler: Callable[[httpx.Request], httpx.Response],
) -> Iterator[OpenAIResponsesClient]:
    client = OpenAIResponsesClient(
        api_key=_TEST_KEY,
        transport=httpx.MockTransport(handler),
    )
    try:
        yield client
    finally:
        client.close()


def _request(**overrides: Any) -> GenerationRequest:
    values: dict[str, Any] = {
        "messages": [
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Answer this."),
        ],
        "model": "openai/gpt-5.4-mini",
        "temperature": 0.2,
        "max_tokens": 321,
        "seed": 42,
        "fallbacks": ("openai/gpt-5.4",),
        "provider_sort": "latency",
    }
    values.update(overrides)
    return GenerationRequest(**values)


def test_native_markdown_request_and_result_are_normalised() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_completed())

    with _client(handler) as client:
        result = GptAdapter(openai_client=client).generate(_request())

    assert len(requests) == 1
    sent = requests[0]
    assert sent.url == httpx.URL("https://api.openai.com/v1/responses")
    assert sent.headers["authorization"] == f"Bearer {_TEST_KEY}"
    payload = sent.read().decode()
    decoded = json.loads(payload)
    assert decoded == {
        "model": "gpt-5.4-mini",
        "input": [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Answer this."},
        ],
        "store": False,
        "max_output_tokens": 321,
    }
    assert "temperature" not in decoded
    assert "seed" not in decoded
    assert "fallbacks" not in decoded

    assert result.text == "native answer"
    assert result.model == "openai/gpt-5.4-mini"
    assert result.provider is Provider.GPT
    assert result.finish_reason == "stop"
    assert result.usage == TokenUsage(
        prompt_tokens=11,
        completion_tokens=7,
        total_tokens=18,
        cost=None,
        is_byok=True,
    )
    assert result.latency_ms is not None
    assert result.raw is not None


def test_json_object_uses_responses_text_format() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
        return httpx.Response(200, json=_completed('{"ok":true}'))

    response_format = ChatFormatJSONObjectConfig(type="json_object")
    with _client(handler) as client:
        result = GptAdapter(openai_client=client).generate(
            _request(),
            response_format=response_format,
        )

    assert result.text == '{"ok":true}'
    assert sent[0]["text"] == {"format": {"type": "json_object"}}


def test_json_schema_is_flattened_for_responses_api() -> None:
    sent: list[dict[str, Any]] = []
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
    }
    response_format = ChatFormatJSONSchemaConfig(
        type="json_schema",
        json_schema=ChatJSONSchemaConfig(name="answer", schema_=schema),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
        return httpx.Response(200, json=_completed('{"value":7}'))

    with _client(handler) as client:
        GptAdapter(openai_client=client).generate(
            _request(),
            response_format=response_format,
        )

    assert sent[0]["text"] == {
        "format": {
            "type": "json_schema",
            "name": "answer",
            "schema": {
                **schema,
                "additionalProperties": False,
            },
            "strict": True,
        }
    }


def test_native_json_schema_requires_nullable_defaults_at_every_object_level() -> None:
    sent: list[dict[str, Any]] = []
    schema = {
        "type": "object",
        "properties": {
            "nested": {
                "type": "object",
                "properties": {
                    "optional": {
                        "anyOf": [{"type": "string"}, {"type": "null"}],
                        "default": None,
                    }
                },
            }
        },
    }
    response_format = ChatFormatJSONSchemaConfig(
        type="json_schema",
        json_schema=ChatJSONSchemaConfig(name="answer", schema_=schema),
    )

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.read()))
        return httpx.Response(200, json=_completed('{"nested":{"optional":null}}'))

    with _client(handler) as client:
        GptAdapter(openai_client=client).generate(
            _request(),
            response_format=response_format,
        )

    native = sent[0]["text"]["format"]["schema"]
    assert native["required"] == ["nested"]
    assert native["additionalProperties"] is False
    assert native["properties"]["nested"]["required"] == ["optional"]
    assert native["properties"]["nested"]["additionalProperties"] is False
    assert "default" not in native["properties"]["nested"]["properties"]["optional"]


def test_native_max_output_incomplete_maps_to_length() -> None:
    payload = _completed('{"value":')
    payload["status"] = "incomplete"
    payload["incomplete_details"] = {"reason": "max_output_tokens"}

    with _client(lambda _request: httpx.Response(200, json=payload)) as client:
        result = GptAdapter(openai_client=client).generate(_request())

    assert result.text == '{"value":'
    assert result.finish_reason == "length"


def test_top_level_output_text_is_accepted() -> None:
    payload = _completed()
    payload["output"] = []
    payload["output_text"] = "shortcut"

    with _client(lambda _request: httpx.Response(200, json=payload)) as client:
        result = GptAdapter(openai_client=client).generate(_request())

    assert result.text == "shortcut"


def test_transient_native_failure_retries_without_leaking_key(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    leaked = "sk-do-not-log-this-secret-123456"
    calls = 0
    delays: list[float] = []

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0"},
                json={"error": {"message": f"key {leaked} is rate limited"}},
            )
        return httpx.Response(200, json=_completed())

    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", "1")
    get_settings.cache_clear()
    monkeypatch.setattr(openai_module, "_sleep", delays.append)
    with caplog.at_level(logging.WARNING), _client(handler) as client:
        result = GptAdapter(openai_client=client).generate(_request())

    assert result.text == "native answer"
    assert calls == 2
    assert delays == [0.0]
    assert leaked not in caplog.text
    assert "[redacted]" in caplog.text


def test_fatal_native_failure_is_not_retried_and_is_redacted() -> None:
    leaked = "sk-do-not-return-this-secret-123456"
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            401,
            json={"error": {"message": f"invalid api key {leaked}"}},
        )

    with _client(handler) as client, pytest.raises(LLMError) as caught:
        GptAdapter(openai_client=client).generate(_request())

    assert calls == 1
    assert leaked not in str(caught.value)
    assert "[redacted]" in str(caught.value)


def test_auto_transport_prefers_native_openai(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[GenerationRequest] = []

    class StubNative:
        def generate(
            self,
            request: GenerationRequest,
            *,
            default_model: str,
            response_format: object | None = None,
        ) -> GenerationResult:
            calls.append(request)
            return GenerationResult(
                text="native",
                model=request.model or default_model,
                provider=Provider.GPT,
            )

    monkeypatch.setenv("LLM_TRANSPORT", "auto")
    monkeypatch.setenv("OPENAI_API_KEY", _TEST_KEY)
    get_settings.cache_clear()
    monkeypatch.setattr(gpt_module, "get_openai_client", StubNative)

    result = GptAdapter().generate(_request())

    assert result.text == "native"
    assert calls == [_request()]


def test_auto_transport_falls_back_to_openrouter_without_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sent: list[GenerationRequest] = []

    def fake_send(
        _adapter: GptAdapter,
        request: GenerationRequest,
        *,
        response_format: object | None = None,
    ) -> GenerationResult:
        sent.append(request)
        return GenerationResult(
            text="gateway",
            model=request.model or GptAdapter.default_model,
            provider=Provider.GPT,
        )

    monkeypatch.setenv("LLM_TRANSPORT", "auto")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(GptAdapter, "_send", fake_send)

    result = GptAdapter().generate(_request())

    assert result.text == "gateway"
    assert sent == [_request()]


def test_explicit_openai_transport_requires_its_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TRANSPORT", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    get_settings.cache_clear()
    openai_module.get_openai_client.cache_clear()

    with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
        GptAdapter().generate(_request())
