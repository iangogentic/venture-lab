"""CLI diagnostics for native OpenAI stay local and never expose credentials."""

import json

import pytest
import typer
from typer.testing import CliRunner, Result

import app.cli.commands.benchmark as benchmark_module
import app.cli.commands.doctor as doctor_module
from app.cli.main import app
from app.config import get_settings
from app.llm.catalog import ModelCatalog, ModelInfo
from app.llm.roles import Capability
from app.utils.paths import WorkspacePaths

runner = CliRunner()
_SECRET = "sk-test-native-cli-do-not-print-123456"


def _configure_all_gpt(
    monkeypatch: pytest.MonkeyPatch,
    *,
    transport: str,
    openai_key: str | None = _SECRET,
    openrouter_key: str | None = None,
) -> None:
    monkeypatch.setenv(
        "LLM_CAPABILITIES",
        json.dumps({capability.value: "gpt" for capability in Capability}),
    )
    monkeypatch.setenv("LLM_STAGE_CAPABILITIES", "{}")
    monkeypatch.setenv("LLM_FALLBACK_TIERS", "[]")
    monkeypatch.setenv("LLM_TRANSPORT", transport)
    if openai_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", openai_key)
    if openrouter_key is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", openrouter_key)
    get_settings.cache_clear()


def _checks(result: Result) -> dict[str, dict[str, str]]:
    payload = json.loads(result.stdout)
    return {check["name"]: check for check in payload["checks"]}


def test_doctor_native_openai_skips_gateway_catalogue_and_credit(
    workspace: WorkspacePaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_all_gpt(monkeypatch, transport="openai")

    def unexpected(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("native doctor contacted an OpenRouter-only endpoint")

    monkeypatch.setattr(doctor_module, "get_catalogue", unexpected)
    monkeypatch.setattr(doctor_module, "_fetch_key_limits", unexpected)

    result = runner.invoke(app, ["doctor", "--no-probe", "--json"])
    checks = _checks(result)

    assert checks["api key"]["status"] == "ok"
    assert checks["api key"]["detail"] == "OPENAI_API_KEY is set"
    assert checks["key limits"]["status"] == "ok"
    assert "not applicable" in checks["key limits"]["detail"]
    assert "native OpenAI" in checks["routing"]["detail"]
    assert "catalogue not applicable" in checks["routing"]["detail"]
    assert "OPENROUTER_API_KEY" not in result.output
    assert _SECRET not in result.output


def test_doctor_auto_uses_native_openai_when_its_key_is_present(
    workspace: WorkspacePaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_all_gpt(monkeypatch, transport="auto")
    monkeypatch.setattr(
        doctor_module,
        "get_catalogue",
        lambda: pytest.fail("auto-native doctor must not fetch OpenRouter catalogue"),
    )
    monkeypatch.setattr(
        doctor_module,
        "_fetch_key_limits",
        lambda _key: pytest.fail("auto-native doctor must not fetch OpenRouter credit"),
    )

    result = runner.invoke(app, ["doctor", "--no-probe", "--json"])
    checks = _checks(result)

    assert checks["api key"]["status"] == "ok"
    assert checks["api key"]["detail"] == "OPENAI_API_KEY is set"
    assert "not applicable" in checks["key limits"]["detail"]


def test_doctor_mixed_routes_name_the_openrouter_requirement(
    workspace: WorkspacePaths,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TRANSPORT", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", _SECRET)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    get_settings.cache_clear()

    result = runner.invoke(app, ["doctor", "--no-probe", "--json"])
    check = _checks(result)["api key"]

    assert check["status"] == "fail"
    assert "OPENROUTER_API_KEY is not set" in check["detail"]
    assert "OpenRouter routes:" in check["detail"]
    assert _SECRET not in result.output


def test_doctor_does_not_apply_openrouter_credit_to_native_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_all_gpt(monkeypatch, transport="openai")
    settings = get_settings()
    routes, unroutable = doctor_module._resolve_routes()
    assert not unroutable

    catalogue = ModelCatalog(
        [
            ModelInfo(
                id=next(iter(routes.values())).model,
                pricing={"completion": "1.0"},
            )
        ]
    )
    check = doctor_module._output_check(
        settings,
        routes,
        catalogue,
        remaining=0.01,
    )

    assert check.status is doctor_module.Status.OK
    assert "credit" not in check.detail
    assert "credit" not in check.remedy


def test_benchmark_preflight_accepts_direct_openai_without_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_all_gpt(monkeypatch, transport="openai")

    benchmark_module._require_api_key()


def test_benchmark_auto_accepts_direct_openai_without_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_all_gpt(monkeypatch, transport="auto")

    benchmark_module._require_api_key()


def test_benchmark_auto_falls_back_to_openrouter_without_openai_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _configure_all_gpt(
        monkeypatch,
        transport="auto",
        openai_key=None,
        openrouter_key=_SECRET,
    )

    benchmark_module._require_api_key()


def test_benchmark_explicit_openrouter_does_not_accept_an_openai_key_substitute(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _configure_all_gpt(
        monkeypatch,
        transport="openrouter",
        openai_key=_SECRET,
        openrouter_key=None,
    )

    with pytest.raises(typer.Exit):
        benchmark_module._require_api_key()

    captured = capsys.readouterr()
    assert "OPENROUTER_API_KEY" in captured.err
    assert _SECRET not in captured.err


def test_benchmark_native_preflight_rejects_an_openrouter_key_substitute(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _configure_all_gpt(
        monkeypatch,
        transport="openai",
        openai_key=None,
        openrouter_key=_SECRET,
    )

    with pytest.raises(typer.Exit):
        benchmark_module._require_api_key()

    captured = capsys.readouterr()
    assert "OPENAI_API_KEY" in captured.err
    assert "OPENROUTER_API_KEY" not in captured.err
    assert _SECRET not in captured.err


def test_benchmark_native_cost_language_describes_byok_billing(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _configure_all_gpt(monkeypatch, transport="openai")

    benchmark_module._announce_cost("native-run")

    captured = capsys.readouterr()
    message = " ".join(captured.err.split())
    assert "billed directly by OpenAI to your own API key" in message
    assert "OPENAI_API_KEY" in message
    assert "OPENROUTER" not in message
    assert _SECRET not in message
