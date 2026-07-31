"""The `op` command surface."""

import json

import pytest
from typer.testing import CliRunner

from app import __version__
from app.artifacts import ArtifactKind, ArtifactRegistry
from app.cli.commands.stages import STAGE_COMMANDS
from app.cli.main import app
from app.config import get_settings
from app.pipeline import STAGE_ORDER
from app.utils.paths import WorkspacePaths, get_workspace_paths
from tests.factories import make

runner = CliRunner()

REQUIRED_COMMANDS = (
    "init",
    "collect",
    "brief",
    "cluster",
    "discover",
    "market",
    "competition",
    "contradiction",
    "decision",
    "interview",
    "pipeline",
    "inspect",
    "list",
    "show",
    "recall",
    "leads",
)


def test_every_required_command_exists() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for command in REQUIRED_COMMANDS:
        assert command in result.output, f"missing `op {command}`"


@pytest.mark.parametrize("command", list(REQUIRED_COMMANDS))
def test_command_has_help(command: str) -> None:
    result = runner.invoke(app, [command, "--help"])
    assert result.exit_code == 0


def test_stage_commands_cover_the_pipeline() -> None:
    """One command per stage, in pipeline order — no stage unreachable from the CLI."""
    assert tuple(stage for _, stage, _ in STAGE_COMMANDS) == STAGE_ORDER


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


# ------------------------------------------------------------------- init


def test_init_creates_the_workspace() -> None:
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0

    for directory in get_workspace_paths().all_dirs():
        assert directory.is_dir()


def test_init_seeds_a_question() -> None:
    result = runner.invoke(app, ["init", "Where do teams lose time?", "--run", "r1"])
    assert result.exit_code == 0

    questions = ArtifactRegistry().find_by_type(ArtifactKind.QUESTION, run_id="r1")
    assert len(questions) == 1
    assert questions[0].text == "Where do teams lose time?"  # type: ignore[attr-defined]


def test_init_can_emit_the_question_as_json() -> None:
    result = runner.invoke(app, ["init", "Why?", "--json"])
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload[0]["kind"] == "question"


# ------------------------------------------------------------------- list


def test_list_summarises_every_kind(workspace: WorkspacePaths) -> None:
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "evidence" in result.output


def test_list_of_a_kind_shows_rows(workspace: WorkspacePaths) -> None:
    registry = ArtifactRegistry(workspace)
    registry.save(make(ArtifactKind.EVIDENCE))

    result = runner.invoke(app, ["list", "evidence"])

    assert result.exit_code == 0
    assert "ev_" in result.stdout


def test_list_json_emits_artifacts(workspace: WorkspacePaths) -> None:
    registry = ArtifactRegistry(workspace)
    saved = make(ArtifactKind.OPPORTUNITY)
    registry.save(saved)

    result = runner.invoke(app, ["list", "opportunity", "--json"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)[0]["id"] == saved.id


def test_list_of_an_empty_kind_is_not_an_error(workspace: WorkspacePaths) -> None:
    result = runner.invoke(app, ["list", "report"])
    assert result.exit_code == 0


# ------------------------------------------------------------------- show


def test_show_prints_one_artifact_as_a_json_object(workspace: WorkspacePaths) -> None:
    """`op show` emits an object, not a one-element array — it is pipeable to jq."""
    registry = ArtifactRegistry(workspace)
    saved = make(ArtifactKind.PAIN_CLUSTER)
    registry.save(saved)

    result = runner.invoke(app, ["show", saved.id])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert isinstance(payload, dict)
    assert payload["id"] == saved.id
    assert payload["kind"] == "pain_cluster"


def test_show_finds_an_artifact_without_being_told_its_kind(
    workspace: WorkspacePaths,
) -> None:
    registry = ArtifactRegistry(workspace)
    saved = make(ArtifactKind.DECISION)
    registry.save(saved)

    assert runner.invoke(app, ["show", saved.id]).exit_code == 0


def test_show_unknown_id_exits_non_zero(workspace: WorkspacePaths) -> None:
    result = runner.invoke(app, ["show", "nope_1"])
    assert result.exit_code == 1


def test_show_can_read_an_archived_version(workspace: WorkspacePaths) -> None:
    registry = ArtifactRegistry(workspace)
    original = make(ArtifactKind.QUESTION, text="first")
    registry.save(original)
    registry.version(original, text="second")

    result = runner.invoke(app, ["show", original.id, "--version", "1"])

    assert result.exit_code == 0
    assert json.loads(result.stdout)["text"] == "first"


# ---------------------------------------------------------------- inspect


def test_inspect_lists_every_stage(workspace: WorkspacePaths) -> None:
    result = runner.invoke(app, ["inspect", "--run", "r1"])
    assert result.exit_code == 0
    for stage in STAGE_ORDER:
        assert stage in result.output


def test_inspect_json_reports_pending_stages(workspace: WorkspacePaths) -> None:
    result = runner.invoke(app, ["inspect", "--run", "r1", "--json"])
    assert result.exit_code == 0

    payload = json.loads(result.stdout)
    assert payload["run_id"] == "r1"
    assert payload["pending"] == list(STAGE_ORDER)
    assert all(stage["complete"] is False for stage in payload["stages"])


def test_inspect_reflects_completed_work(workspace: WorkspacePaths) -> None:
    registry = ArtifactRegistry(workspace)
    registry.save(make(ArtifactKind.EVIDENCE, run_id="r1"))

    payload = json.loads(runner.invoke(app, ["inspect", "--run", "r1", "--json"]).stdout)

    assert "collect-evidence" not in payload["pending"]
    assert payload["pending"][0] == "research-brief"


# --------------------------------------------------------------- config


def test_config_redacts_the_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-do-not-leak-me")
    get_settings.cache_clear()

    result = runner.invoke(app, ["config"])

    assert result.exit_code == 0
    assert "sk-do-not-leak-me" not in result.output
    assert "openrouter_api_key" in result.output


# ---------------------------------------------- stages without an LLM key


def test_a_stage_with_no_inputs_is_blocked_not_crashed(workspace: WorkspacePaths) -> None:
    """No question seeded, so `op collect` has nothing to read — a clean failure."""
    result = runner.invoke(app, ["collect", "--run", "empty"])
    assert result.exit_code == 1
    assert "blocked" in result.output.lower() or "no question" in result.output.lower()
