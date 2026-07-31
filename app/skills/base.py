"""The Skill interface: artifacts in, validated artifacts out.

A skill is one cognitive step. It reads artifacts, asks a model exactly one
question about them, validates the reply against a schema, and writes the result
back to the workspace as new artifacts.

`execute()` is deliberately concrete — a template method rather than an abstract
hook. Every skill must read, prompt, validate, persist and record provenance in
the same order, and centralising that here is what makes those guarantees
uniform: a subclass cannot forget to set `parents`, cannot skip validation, and
cannot smuggle in a hand-built prompt. Subclasses supply only the two ends that
are genuinely skill-specific:

* `gather()`   — turn input artifacts into the placeholder values a prompt wants;
* `assemble()` — turn the model's validated output into artifacts.

Skills never call other skills. Composition is the pipeline's job; a skill that
reached for another would hide a dependency the orchestrator cannot see or resume.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import Artifact, ArtifactKind, ArtifactRef, ArtifactRegistry
from app.config import get_settings
from app.llm import LLM
from app.llm.telemetry import CallRecord, default_sink
from app.prompts.loader import load_prompt
from app.utils.errors import SkillError
from app.utils.logging import get_logger

logger = get_logger(__name__)


class Batching(StrEnum):
    """How many calls one stage makes, and what each call is shown.

    The distinction is about the shape of the work, not performance. A brief must
    read many pieces of evidence at once to synthesise across them — that is a
    fan-in and cannot be split. A decision rules on one opportunity at a time, and
    showing it every other opportunity's analysis is noise that grows with the run.
    """

    FAN_IN = "fan_in"
    """One call over everything the stage consumes. For genuine synthesis steps."""

    PER_ITEM = "per_item"
    """One call per `primary_kind` artifact, given only that artifact's lineage."""


SKILLS: dict[str, type["Skill"]] = {}
"""Every registered skill, keyed by `Skill.name`."""


class SkillInput(BaseModel):
    """Base for the placeholder payload a skill sends to its prompt.

    Extras are forbidden: a payload carrying fields the prompt has no placeholder
    for is a caller bug, not something to quietly ship to a model.
    """

    model_config = ConfigDict(extra="forbid")


class SkillOutput(BaseModel):
    """Base for the structured reply a skill requires back from the model.

    Concrete subclasses become the JSON Schema handed to the provider, so keep
    them strict and JSON-Schema friendly.
    """

    model_config = ConfigDict(extra="forbid")


class SkillRequest(BaseModel):
    """What a skill is asked to do: a run, and the artifacts to work from."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    run_id: str
    artifacts: list[Artifact] = Field(default_factory=list)
    question: Artifact | None = Field(
        default=None,
        description="The Question seeding the run, for prompts that need the original ask.",
    )

    def of_kind(self, kind: ArtifactKind) -> list[Artifact]:
        """The input artifacts of one kind, in the order supplied."""
        return [artifact for artifact in self.artifacts if type(artifact).kind is kind]


class SkillResult(BaseModel):
    """What a skill produced, plus what it cost."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    skill: str
    artifacts: list[Artifact] = Field(default_factory=list)
    model: str | None = None
    fingerprint: str | None = None

    @property
    def refs(self) -> list[ArtifactRef]:
        """References to everything produced."""
        return [artifact.ref for artifact in self.artifacts]


class Skill(ABC):
    """One cognitive step of the pipeline."""

    name: ClassVar[str]
    """Registry key, matching the pipeline stage that runs it, e.g. `collect-evidence`."""

    description: ClassVar[str]
    """One line, shown by `op list skills`."""

    prompt_name: ClassVar[str]
    """Stem of the Markdown prompt in `app/prompts/`. Prompt text never lives in code."""

    consumes: ClassVar[tuple[ArtifactKind, ...]]
    """Artifact kinds this skill reads. Empty means it is seeded by the Question alone."""

    batching: ClassVar[Batching] = Batching.FAN_IN
    """Whether this stage runs once over everything, or once per item."""

    primary_kind: ClassVar[ArtifactKind | None] = None
    """For `PER_ITEM`, the kind iterated over. Each call sees one of these plus its
    lineage. Ignored for `FAN_IN`."""

    produces: ClassVar[ArtifactKind]
    """The artifact kind this skill writes."""

    input_schema: ClassVar[type[SkillInput]]
    """Payload model describing what the prompt is given."""

    output_schema: ClassVar[type[SkillOutput]]
    """Reply model the provider is constrained to. Becomes the JSON Schema."""

    def __init__(
        self,
        llm: LLM | None = None,
        registry: ArtifactRegistry | None = None,
    ) -> None:
        """Build a skill.

        Both collaborators are injected so a test can drive a real skill with a
        fake model and a temporary workspace.
        """
        self._llm = llm
        self._registry = registry

    # ------------------------------------------------------------ collaborators

    @property
    def llm(self) -> LLM:
        """The model façade, resolved on first use so construction needs no API key."""
        if self._llm is None:
            self._llm = LLM()
        return self._llm

    @property
    def registry(self) -> ArtifactRegistry:
        """The artifact store, resolved on first use."""
        if self._registry is None:
            self._registry = ArtifactRegistry()
        return self._registry

    # --------------------------------------------------------------- extension

    @abstractmethod
    def gather(self, request: SkillRequest) -> SkillInput:
        """Reduce the input artifacts to the values this skill's prompt asks for.

        Return an instance of `input_schema`. Raise `SkillError` if the request
        does not carry enough to proceed.
        """

    @abstractmethod
    def assemble(self, output: SkillOutput, request: SkillRequest) -> list[Artifact]:
        """Turn the model's validated reply into artifacts.

        Return unsaved artifacts; `execute` stamps provenance and persists them.
        Ids come from `type(artifact).make_id()`.
        """

    # ----------------------------------------------------------------- template

    def fingerprint(self, request: SkillRequest) -> str:
        """Digest of everything that determines this call's output.

        Four inputs, each chosen because changing it genuinely changes the answer:

        * the **skill** and its **output schema** — a changed contract is a changed job;
        * the **prompt file**, hashed by content, so editing a SKILL.md invalidates
          exactly the artifacts that prompt produced and nothing else;
        * the **resolved model slug**, so a vendor release regenerates only the stages
          routed to that tier, rather than everything or nothing;
        * the **question and input artifact ids**, sorted so ordering is not mistaken
          for a difference.

        Deliberately not included: timestamps, run ids, and anything else that differs
        between two runs which ought to produce the same result.
        """
        prompt = load_prompt(self.prompt_name)
        parts = (
            f"skill={self.name}",
            f"schema={_digest(json.dumps(self.output_json_schema(), sort_keys=True))}",
            f"prompt={_digest(prompt.system + prompt.user)}",
            f"model={self.llm.resolve(self.name).model}",
            f"question={request.question.id if request.question else '-'}",
            "inputs=" + ",".join(sorted(a.id for a in request.artifacts)),
        )
        return _digest("\n".join(parts))

    def execute(self, request: SkillRequest) -> SkillResult:
        """Read artifacts, ask the model, validate, persist, return the artifacts.

        The fixed order every skill obeys:

        1. check the request carries the kinds this skill `consumes`;
        2. `gather()` the prompt values, validated against `input_schema`;
        3. render the Markdown prompt — never assembled in code;
        4. ask the model for `output_schema`, which validates the reply;
        5. `assemble()` artifacts and stamp `parents` from the inputs;
        6. write each artifact to the workspace.

        Deciding whether the work is already done is the engine's job, not the
        skill's: the engine skips completed stages and completed per-item
        primaries before a request ever reaches here.

        Raises:
            SkillError: If required inputs are missing, or the model's reply
                cannot be turned into valid artifacts.
        """
        self._require_inputs(request)

        digest = self.fingerprint(request)
        payload = self.gather(request)
        if not isinstance(payload, self.input_schema):
            raise SkillError(
                f"{type(self).__name__}.gather returned {type(payload).__name__}, "
                f"expected {self.input_schema.__name__}"
            )

        messages = load_prompt(self.prompt_name).render(payload.model_dump(mode="json"))
        # Routed by stage name, so a cheap model can serve extraction while the
        # decision gets an expensive one — configured once, not per call site.
        output = self.llm.generate_structured(messages, self.output_schema, task=self.name)

        artifacts = self.assemble(output, request)

        persisted: list[Artifact] = []
        for artifact in artifacts:
            stamped = self._stamp(artifact, request, digest)
            self.registry.save(stamped)
            persisted.append(stamped)

        logger.debug("%s produced %d artifact(s)", self.name, len(persisted))
        self._record(request, digest, artifacts=persisted)
        return SkillResult(
            skill=self.name,
            artifacts=persisted,
            model=self.llm.model,
            fingerprint=digest,
        )

    # ---------------------------------------------------------------- internals

    def _record(
        self,
        request: SkillRequest,
        fingerprint: str,
        *,
        artifacts: Sequence[Artifact],
    ) -> None:
        """Log what this call cost. Never lets telemetry break a run."""
        settings = get_settings()
        if not settings.telemetry_enabled:
            return

        try:
            route = self.llm.resolve(self.name)
            usage = self.llm.last_call
            default_sink(self.registry.paths).record(
                CallRecord(
                    run_id=request.run_id,
                    skill=self.name,
                    capability=route.capability.value,
                    tier=route.tier.value,
                    # What answered, not what was asked for: a gateway fallback means
                    # those differ, and the log should say which one ran.
                    model=(usage.model if usage else route.model),
                    prompt_name=self.prompt_name,
                    prompt_digest=_digest_of_prompt(self.prompt_name),
                    fingerprint=fingerprint,
                    input_artifacts=[a.id for a in request.artifacts],
                    output_artifacts=[a.id for a in artifacts],
                    cached=False,
                    latency_ms=usage.latency_ms if usage else None,
                    prompt_tokens=usage.usage.prompt_tokens if usage else None,
                    completion_tokens=usage.usage.completion_tokens if usage else None,
                    total_tokens=usage.usage.total_tokens if usage else None,
                    cost=usage.usage.cost if usage else None,
                    is_byok=bool(usage and usage.usage.is_byok),
                )
            )
        except Exception as exc:  # a measurement must never cost you the research
            logger.debug("could not record telemetry for %s: %s", self.name, exc)

    def _require_inputs(self, request: SkillRequest) -> None:
        """Fail early when the orchestrator handed over the wrong kinds."""
        for kind in self.consumes:
            if not request.of_kind(kind):
                raise SkillError(
                    f"{self.name} needs at least one {kind.value} artifact, none supplied"
                )

    def _stamp(self, artifact: Artifact, request: SkillRequest, fingerprint: str) -> Artifact:
        """Attach run and provenance, and check the skill produced what it declared.

        Provenance is applied here rather than in `assemble` so no skill can
        produce an orphan artifact. An `assemble` that already set `parents`
        (because it knows which specific inputs a given output came from) is left
        alone; otherwise every input becomes a parent.
        """
        if type(artifact).kind is not self.produces:
            raise SkillError(
                f"{self.name} declares it produces {self.produces.value} "
                f"but assembled a {type(artifact).kind.value}"
            )

        data = artifact.model_dump()
        data["run_id"] = request.run_id
        data["fingerprint"] = fingerprint
        if not artifact.parents:
            data["parents"] = [ref.model_dump() for ref in self._parent_refs(request)]

        return type(artifact).model_validate(data)

    @staticmethod
    def _parent_refs(request: SkillRequest) -> Sequence[ArtifactRef]:
        """Every input artifact, as references."""
        return [artifact.ref for artifact in request.artifacts]

    # ---------------------------------------------------------------- metadata

    @classmethod
    def input_json_schema(cls) -> dict[str, Any]:
        """JSON Schema of the prompt payload, for docs and `op inspect`."""
        return cls.input_schema.model_json_schema()

    @classmethod
    def output_json_schema(cls) -> dict[str, Any]:
        """JSON Schema the provider is constrained to."""
        return cls.output_schema.model_json_schema()


def register[SkillT: Skill](cls: type[SkillT]) -> type[SkillT]:
    """Register a skill under its `name`; usable as a class decorator."""
    SKILLS[cls.name] = cls
    return cls


def get_skill(name: str) -> type[Skill]:
    """Look up a registered skill class.

    Raises:
        SkillError: If no skill is registered under `name`.
    """
    try:
        return SKILLS[name]
    except KeyError as exc:
        known = ", ".join(available()) or "<none>"
        raise SkillError(f"Unknown skill {name!r}; available: {known}") from exc


def available() -> tuple[str, ...]:
    """Every registered skill name, sorted."""
    return tuple(sorted(SKILLS))


def _digest_of_prompt(name: str) -> str | None:
    """Content hash of a prompt, or None if it cannot be read."""
    try:
        prompt = load_prompt(name)
    except Exception:
        return None
    return _digest(prompt.system + prompt.user)


def _digest(text: str) -> str:
    """SHA-256 of a string. One place, so every part of a fingerprint is hashed alike."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "SKILLS",
    "Batching",
    "Skill",
    "SkillInput",
    "SkillOutput",
    "SkillRequest",
    "SkillResult",
    "available",
    "get_skill",
    "register",
]
