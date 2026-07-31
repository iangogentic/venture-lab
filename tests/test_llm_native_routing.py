"""Native OpenAI routing must not depend on OpenRouter metadata."""

from collections.abc import Iterator

import pytest

from app.config import get_settings
from app.llm.catalog import ModelCatalog
from app.llm.provider import Provider
from app.llm.routing import ModelRouter, get_catalogue


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _all_gpt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_TRANSPORT", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-not-a-real-key")
    monkeypatch.setenv(
        "LLM_CAPABILITIES",
        (
            '{"fast_extract":"gpt","summarize":"gpt","synthesis":"gpt",'
            '"deep_reasoning":"gpt","market_reasoning":"gpt","second_opinion":"gpt"}'
        ),
    )


def test_explicit_native_transport_never_loads_openrouter_catalogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _all_gpt(monkeypatch)

    def fail_if_called(*_args: object, **_kwargs: object) -> ModelCatalog:
        raise AssertionError("OpenRouter catalogue must not be loaded")

    monkeypatch.setattr(ModelCatalog, "load", fail_if_called)

    assert get_catalogue() is None


def test_native_gpt_route_retains_schema_capabilities_without_catalogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _all_gpt(monkeypatch)
    router = ModelRouter.from_settings(catalogue=None)

    route = router.resolve("discover-opportunities")

    assert route.provider is Provider.GPT
    assert route.supports_response_format
    assert route.supports_structured_outputs
    assert route.supports_prompt_caching
    assert not route.needs_explicit_cache_write


def test_openrouter_gpt_without_catalogue_stays_conservative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TRANSPORT", "openrouter")
    monkeypatch.setenv("LLM_CAPABILITIES", '{"deep_reasoning":"gpt"}')
    get_settings.cache_clear()
    router = ModelRouter.from_settings(catalogue=None)

    route = router.resolve("discover-opportunities")

    assert route.provider is Provider.GPT
    assert not route.supports_response_format
    assert not route.supports_structured_outputs
