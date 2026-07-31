"""The LLM abstraction: provider adapters behind one façade.

The rule these tests exist to defend is that the rest of the application never
talks to a provider directly — it goes through `app.llm`.
"""

import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from openrouter import OpenRouter
from pydantic import BaseModel

import app.llm.provider as provider_module
from app.config import get_settings
from app.llm import (
    LLM,
    ChatMessage,
    GenerationRequest,
    GenerationResult,
    Provider,
    ProviderAdapter,
    TokenUsage,
    available,
    get_adapter,
)
from app.llm.provider import _is_transient, _retry_delay
from app.utils.errors import ConfigurationError, LLMError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------- the boundary


def test_no_module_outside_app_llm_imports_the_provider_sdk() -> None:
    """ "The rest of the application must never call the providers directly."

    Enforced mechanically rather than by convention: an import that sneaks in
    elsewhere is exactly the regression the adapter layer exists to prevent.
    """
    pattern = re.compile(r"^\s*(?:from|import)\s+openrouter\b", re.MULTILINE)
    offenders = [
        path.relative_to(PROJECT_ROOT)
        for path in (PROJECT_ROOT / "app").rglob("*.py")
        if "llm" not in path.relative_to(PROJECT_ROOT / "app").parts
        and pattern.search(path.read_text(encoding="utf-8"))
    ]

    assert offenders == [], f"provider SDK imported outside app/llm: {offenders}"


# ----------------------------------------------------------------- providers


def test_all_three_providers_are_supported() -> None:
    assert {p.value for p in Provider} == {"claude", "gpt", "gemini"}
    assert set(available()) == set(Provider)


@pytest.mark.parametrize(
    ("provider", "vendor"),
    [
        (Provider.CLAUDE, "anthropic/"),
        (Provider.GPT, "openai/"),
        (Provider.GEMINI, "google/"),
    ],
)
def test_each_adapter_defaults_to_its_own_vendor(provider: Provider, vendor: str) -> None:
    adapter = get_adapter(provider)
    assert adapter.provider is provider
    assert adapter.default_model.startswith(vendor)


def test_adapters_do_not_share_a_default_model() -> None:
    models = [get_adapter(p).default_model for p in Provider]
    assert len(set(models)) == len(models)


def test_unknown_provider_raises() -> None:
    with pytest.raises((ConfigurationError, ValueError)):
        get_adapter("nope")  # type: ignore[arg-type]


# ------------------------------------------------------------- fake transport


class FakeAdapter(ProviderAdapter):
    """A stand-in for a real provider, so the façade is testable without a network."""

    provider = Provider.CLAUDE
    default_model = "fake/model"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[GenerationRequest] = []
        self.formats: list[object] = []

    def generate(
        self,
        request: GenerationRequest,
        *,
        response_format: object | None = None,
    ) -> GenerationResult:
        self.calls.append(request)
        self.formats.append(response_format)
        return GenerationResult(
            text=self.reply,
            model=request.model or self.default_model,
            provider=self.provider,
            usage=TokenUsage(),
            finish_reason="stop",
        )


def llm_returning(reply: str) -> tuple[LLM, FakeAdapter]:
    adapter = FakeAdapter(reply)
    return LLM(adapter=adapter), adapter


# ------------------------------------------------------------------ markdown


def test_generate_markdown_returns_text() -> None:
    llm, _ = llm_returning("# Heading\n\nBody.")
    assert llm.generate_markdown([ChatMessage(role="user", content="hi")]) == "# Heading\n\nBody."


def test_a_bare_string_is_accepted_as_a_user_message() -> None:
    llm, adapter = llm_returning("ok")
    llm.generate_markdown("just a string")

    assert adapter.calls[0].messages[-1].content == "just a string"


# ---------------------------------------------------------------------- json


def test_generate_json_parses_an_object() -> None:
    llm, _ = llm_returning('{"verdict": "adopt", "score": 0.8}')
    assert llm.generate_json("go") == {"verdict": "adopt", "score": 0.8}


def test_generate_json_tolerates_a_fenced_code_block() -> None:
    """Models wrap JSON in fences constantly; the façade must not be defeated by it."""
    llm, _ = llm_returning('```json\n{"ok": true}\n```')
    assert llm.generate_json("go") == {"ok": True}


def test_generate_json_raises_llm_error_on_garbage() -> None:
    llm, _ = llm_returning("I am afraid I cannot do that.")
    with pytest.raises(LLMError):
        llm.generate_json("go")


# ---------------------------------------------------------------- truncation


class TruncatingAdapter(ProviderAdapter):
    """Cuts the first reply off at the cap, then answers properly."""

    provider = Provider.CLAUDE
    default_model = "fake/model"

    def __init__(self, *, always: bool = False) -> None:
        self.always = always
        self.calls: list[GenerationRequest] = []

    def generate(
        self,
        request: GenerationRequest,
        *,
        response_format: object | None = None,
    ) -> GenerationResult:
        self.calls.append(request)
        truncated = self.always or len(self.calls) == 1
        return GenerationResult(
            text='{"ok": tr' if truncated else '{"ok": true}',
            model=request.model or self.default_model,
            provider=self.provider,
            usage=TokenUsage(),
            finish_reason="length" if truncated else "stop",
        )


def test_a_truncated_json_reply_is_retried_at_the_ceiling() -> None:
    """A reply cut off by max_tokens is not "invalid JSON" — it needs a bigger cap.

    The retry asks for the whole ceiling rather than double the cap. Truncation
    says the answer is larger than the cap and nothing about how much larger, so
    doubling is a guess that can fail with headroom still unspent — and then
    report "raise MAX_OUTPUT_TOKENS" about a setting that was already high enough.
    """
    adapter = TruncatingAdapter()
    llm = LLM(adapter=adapter)
    ceiling = get_settings().max_output_tokens

    assert llm.generate_json("go", max_tokens=100) == {"ok": True}
    assert [call.max_tokens for call in adapter.calls] == [100, ceiling]


def test_the_truncation_error_names_both_ways_out() -> None:
    """Raising the cap is only half the remedy; the other half is asking for less."""
    adapter = TruncatingAdapter(always=True)
    llm = LLM(adapter=adapter)

    with pytest.raises(LLMError, match="COLLECTOR_LIMIT"):
        llm.generate_json("go", max_tokens=get_settings().max_output_tokens)


def test_truncation_with_no_headroom_names_the_real_cause() -> None:
    adapter = TruncatingAdapter(always=True)
    llm = LLM(adapter=adapter)
    ceiling = get_settings().max_output_tokens

    with pytest.raises(LLMError, match="cut off"):
        llm.generate_json("go", max_tokens=ceiling)
    assert len(adapter.calls) == 1  # no headroom, so no pointless second spend


# ---------------------------------------------------------------------- retry


class _TransientError(Exception):
    """Wire-shaped stand-in for an SDK rate-limit error."""

    status_code = 429
    headers = httpx.Headers({"retry-after": "0"})


class _FatalError(Exception):
    status_code = 401
    headers = httpx.Headers()


class _FlakyChat:
    """Fails the first `failures` sends, then replies."""

    def __init__(self, failures: int, error: Exception) -> None:
        self.failures = failures
        self.error = error
        self.calls = 0

    def send(self, **_kwargs: object) -> SimpleNamespace:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        message = SimpleNamespace(content="steady on", refusal=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")],
            usage=None,
            model="fake/raw",
        )


class _RawAdapter(ProviderAdapter):
    """Exercises the real `_send`, transport stubbed out."""

    provider = Provider.CLAUDE
    default_model = "fake/raw"

    def generate(
        self,
        request: GenerationRequest,
        *,
        response_format: object | None = None,
    ) -> GenerationResult:
        return self._send(request, response_format=None)


def _no_sleep(monkeypatch: pytest.MonkeyPatch, *, retries: int = 2) -> list[float]:
    """Pin the retry budget and capture the delays instead of sleeping them."""
    monkeypatch.setenv("LLM_RETRY_ATTEMPTS", str(retries))
    get_settings.cache_clear()
    delays: list[float] = []
    monkeypatch.setattr(provider_module, "_sleep", delays.append)
    return delays


def test_a_transient_failure_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    delays = _no_sleep(monkeypatch)
    chat = _FlakyChat(failures=1, error=_TransientError("busy"))
    adapter = _RawAdapter(client=cast(OpenRouter, SimpleNamespace(chat=chat)))

    result = adapter.generate(GenerationRequest(messages=[ChatMessage(role="user", content="hi")]))

    assert result.text == "steady on"
    assert chat.calls == 2
    assert len(delays) == 1


def test_retries_are_bounded_by_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_sleep(monkeypatch)
    chat = _FlakyChat(failures=99, error=_TransientError("busy"))
    adapter = _RawAdapter(client=cast(OpenRouter, SimpleNamespace(chat=chat)))

    with pytest.raises(LLMError):
        adapter.generate(GenerationRequest(messages=[ChatMessage(role="user", content="hi")]))
    assert chat.calls == 3  # the original call plus the two retries allowed


def test_a_fatal_failure_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad key fails identically every time; retrying it only spends patience."""
    _no_sleep(monkeypatch)
    chat = _FlakyChat(failures=99, error=_FatalError("bad key"))
    adapter = _RawAdapter(client=cast(OpenRouter, SimpleNamespace(chat=chat)))

    with pytest.raises(LLMError):
        adapter.generate(GenerationRequest(messages=[ChatMessage(role="user", content="hi")]))
    assert chat.calls == 1


def test_transience_is_read_off_the_status_code() -> None:
    assert _is_transient(_TransientError("x"))
    assert not _is_transient(_FatalError("x"))
    assert _is_transient(httpx.ConnectTimeout("slow"))
    assert not _is_transient(ValueError("not a transport problem"))


def test_retry_delay_honours_retry_after() -> None:
    assert _retry_delay(_TransientError("x"), attempt=1) == 0.0  # header says now
    assert _retry_delay(_FatalError("x"), attempt=1) == 1.0  # no header: backoff
    assert _retry_delay(_FatalError("x"), attempt=3) == 4.0  # and it grows


# ---------------------------------------------------------------- structured


class Verdict(BaseModel):
    verdict: str
    score: float


def test_generate_structured_validates_into_the_model() -> None:
    llm, _ = llm_returning(json.dumps({"verdict": "adopt", "score": 0.8}))
    result = llm.generate_structured("go", Verdict)

    assert isinstance(result, Verdict)
    assert result.verdict == "adopt"
    assert result.score == 0.8


def test_generate_structured_raises_llm_error_on_schema_mismatch() -> None:
    llm, _ = llm_returning(json.dumps({"verdict": "adopt"}))
    with pytest.raises(LLMError):
        llm.generate_structured("go", Verdict)


def test_generate_structured_sends_the_models_schema() -> None:
    """The Pydantic model is the contract; it must reach the provider as a schema."""
    llm, adapter = llm_returning(json.dumps({"verdict": "adopt", "score": 0.1}))
    llm.generate_structured("go", Verdict)

    assert adapter.formats, "no response_format was passed to the adapter"
    assert "json_schema" in json.dumps(adapter.formats[-1], default=str)
