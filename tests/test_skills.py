"""The Skill interface: artifacts in, validated artifacts out, prompts on disk."""

import ast
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest
from pydantic import Field

from app.artifacts import (
    Artifact,
    ArtifactKind,
    ArtifactRegistry,
    Question,
)
from app.llm import LLM, GenerationRequest, GenerationResult, Provider, ProviderAdapter, TokenUsage
from app.pipeline import STAGE_ORDER
from app.prompts import load_prompt
from app.skills import SKILLS, Skill, SkillInput, SkillOutput, SkillRequest, get_skill
from app.utils.errors import SkillError
from app.utils.paths import WorkspacePaths

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SKILLS_DIR = PROJECT_ROOT / "app" / "skills"

OFF_PIPELINE_SKILLS = ("harvest-leads", "compose-report")
"""Skills registered like the rest but deliberately absent from STAGE_ORDER —
run on demand by their own command (`op leads harvest`, `op report`), not by
the engine."""

ALL_SKILLS = (*STAGE_ORDER, *OFF_PIPELINE_SKILLS)


# ------------------------------------------------------------ the registry


def test_every_pipeline_stage_has_a_skill() -> None:
    for stage in STAGE_ORDER:
        assert get_skill(stage).name == stage


def test_no_skills_beyond_the_pipeline_and_the_declared_exceptions() -> None:
    assert set(SKILLS) == set(ALL_SKILLS)


def test_off_pipeline_skills_stay_off_the_pipeline() -> None:
    """`harvest-leads` is human-gated and may legitimately produce nothing —
    the stage-resume machinery would misread that as unfinished work."""
    for name in OFF_PIPELINE_SKILLS:
        assert name not in STAGE_ORDER
        assert get_skill(name).name == name


@pytest.mark.parametrize("stage", list(ALL_SKILLS))
def test_skill_declares_the_full_contract(stage: str) -> None:
    skill = get_skill(stage)

    assert skill.description
    assert skill.prompt_name == stage, "prompt file is named after the skill"
    assert isinstance(skill.consumes, tuple)
    assert isinstance(skill.produces, ArtifactKind)
    assert issubclass(skill.input_schema, SkillInput)
    assert issubclass(skill.output_schema, SkillOutput)


@pytest.mark.parametrize("stage", list(ALL_SKILLS))
def test_schemas_render_as_json_schema(stage: str) -> None:
    """`output_schema` becomes the provider's response schema, so it must serialise."""
    skill = get_skill(stage)
    assert skill.input_json_schema()["type"] == "object"
    assert skill.output_json_schema()["type"] == "object"


def test_each_stage_produces_a_distinct_kind() -> None:
    produced = [get_skill(stage).produces for stage in ALL_SKILLS]
    assert len(set(produced)) == len(produced)


def test_stage_inputs_are_available_from_earlier_stages() -> None:
    """Every stage must consume something an earlier stage produced, or the Question."""
    seen = {ArtifactKind.QUESTION}
    for stage in STAGE_ORDER:
        skill = get_skill(stage)
        missing = set(skill.consumes) - seen
        assert not missing, f"{stage} consumes {missing}, which nothing upstream produces"
        seen.add(skill.produces)


# --------------------------------------- prompts and payloads cannot drift


@pytest.mark.parametrize("stage", list(ALL_SKILLS))
def test_prompt_placeholders_match_the_input_schema(stage: str) -> None:
    """The prompt's `${...}` names and the payload's fields are one contract.

    They are written in different files — and, here, by different authors — so
    this is the test that stops them drifting apart.
    """
    skill = get_skill(stage)
    fields = frozenset(skill.input_schema.model_fields)
    placeholders = load_prompt(stage).placeholders

    assert placeholders == fields, (
        f"{stage}.md interpolates {sorted(placeholders)} "
        f"but {skill.input_schema.__name__} declares {sorted(fields)}"
    )


# ------------------------------------------------- skills never call skills


def _module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    return imported


def test_no_skill_imports_another_skill() -> None:
    """ "Do not call Skills from Skills" — composition belongs to the pipeline.

    A skill reaching for a sibling would be work the engine cannot see, skip or
    resume, so the ban is enforced rather than documented.
    """
    offenders: list[str] = []
    for path in SKILLS_DIR.glob("*.py"):
        if path.stem in {"__init__", "base"}:
            continue
        for module in _module_imports(path):
            if module.startswith("app.skills.") and not module.endswith(".base"):
                offenders.append(f"{path.name} imports {module}")

    assert offenders == [], f"skill-to-skill imports: {offenders}"


def test_no_skill_looks_up_another_skill_at_runtime() -> None:
    calls: list[str] = []
    for path in SKILLS_DIR.glob("*.py"):
        if path.stem in {"__init__", "base"}:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "get_skill"
            ):
                calls.append(f"{path.name} calls get_skill()")

    assert calls == [], f"runtime skill lookup inside a skill: {calls}"


# ------------------------------------------------ the execute() template


class _Payload(SkillInput):
    question: dict[str, Any]
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class _Reply(SkillOutput):
    label: str


class _EchoAdapter(ProviderAdapter):
    """Returns a fixed JSON reply, so a real skill can run with no network."""

    provider: ClassVar[Provider] = Provider.CLAUDE
    default_model: ClassVar[str] = "fake/model"

    def __init__(self, reply: dict[str, Any]) -> None:
        self.reply = reply
        self.messages: list[list[Any]] = []

    def generate(
        self,
        request: GenerationRequest,
        *,
        response_format: object | None = None,
    ) -> GenerationResult:
        self.messages.append(list(request.messages))
        return GenerationResult(
            text=json.dumps(self.reply),
            model=self.default_model,
            provider=self.provider,
            usage=TokenUsage(),
            finish_reason="stop",
        )


class _ProbeSkill(Skill):
    """A minimal real skill, used to exercise the base template."""

    name: ClassVar[str] = "probe"
    description: ClassVar[str] = "test double"
    prompt_name: ClassVar[str] = STAGE_ORDER[0]
    consumes: ClassVar[tuple[ArtifactKind, ...]] = (ArtifactKind.QUESTION,)
    produces: ClassVar[ArtifactKind] = ArtifactKind.EVIDENCE
    input_schema: ClassVar[type[SkillInput]] = _Payload
    output_schema: ClassVar[type[SkillOutput]] = _Reply

    def gather(self, request: SkillRequest) -> SkillInput:
        questions = request.of_kind(ArtifactKind.QUESTION)
        if not questions:
            raise SkillError("probe needs a question")
        return _Payload(question=questions[0].model_dump(mode="json"), candidates=[])

    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        from app.artifacts import Evidence

        assert isinstance(output, _Reply)
        return [
            Evidence(
                id=Evidence.make_id(),
                run_id=request.run_id,
                collector="probe",
                excerpt=output.label,
            )
        ]


@pytest.fixture
def adapter() -> _EchoAdapter:
    return _EchoAdapter({"label": "a captured observation"})


@pytest.fixture
def probe(workspace: WorkspacePaths, adapter: _EchoAdapter) -> tuple[_ProbeSkill, ArtifactRegistry]:
    registry = ArtifactRegistry(workspace)
    return _ProbeSkill(llm=LLM(adapter=adapter), registry=registry), registry


def _question(run_id: str = "run_1") -> Question:
    return Question(id=Question.make_id(), run_id=run_id, text="Where does time go?")


def test_execute_returns_artifacts(probe: tuple[_ProbeSkill, ArtifactRegistry]) -> None:
    skill, _ = probe
    question = _question()

    result = skill.execute(SkillRequest(run_id="run_1", artifacts=[question]))

    assert result.skill == "probe"
    assert len(result.artifacts) == 1
    assert result.artifacts[0].excerpt == "a captured observation"  # type: ignore[attr-defined]


def test_execute_writes_the_artifacts(probe: tuple[_ProbeSkill, ArtifactRegistry]) -> None:
    """ "writes artifact" — the skill persists, the caller does not have to."""
    skill, registry = probe
    question = _question()

    result = skill.execute(SkillRequest(run_id="run_1", artifacts=[question]))

    stored = registry.load(ArtifactKind.EVIDENCE, result.artifacts[0].id)
    assert stored.id == result.artifacts[0].id


def test_execute_stamps_provenance(probe: tuple[_ProbeSkill, ArtifactRegistry]) -> None:
    """The template sets parents, so no skill can produce an orphan."""
    skill, _ = probe
    question = _question()

    result = skill.execute(SkillRequest(run_id="run_1", artifacts=[question]))

    assert result.artifacts[0].parents == [question.ref]
    assert result.artifacts[0].run_id == "run_1"


def test_execute_uses_the_markdown_prompt(
    probe: tuple[_ProbeSkill, ArtifactRegistry], adapter: _EchoAdapter
) -> None:
    """The messages the model saw must come from the .md file, not from code."""
    skill, _ = probe

    skill.execute(SkillRequest(run_id="run_1", artifacts=[_question()]))

    system = adapter.messages[0][0].content
    assert system == load_prompt(STAGE_ORDER[0]).system


def test_execute_refuses_missing_inputs(probe: tuple[_ProbeSkill, ArtifactRegistry]) -> None:
    skill, _ = probe
    with pytest.raises(SkillError, match="question"):
        skill.execute(SkillRequest(run_id="run_1", artifacts=[]))
