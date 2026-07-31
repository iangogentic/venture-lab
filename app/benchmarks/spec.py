"""A benchmark: a research question, plus what a healthy run of it looks like.

The point of a benchmark is *not* a pass/fail gate. It is a fixed question that can
be re-run after a prompt or model change, so the two runs can be compared and the
question "did that make it better, or merely different?" has an answer.

That is why the expectations here are ranges rather than exact numbers. A run that
produces nine pain clusters instead of eight has not regressed; a run that produces
zero, or eighty, has. `Expectation` states the band a healthy run sits in and says
nothing about the band's interior — anything stricter would measure the wording of a
prompt rather than the health of the pipeline.

On disk::

    benchmarks/<name>/benchmark.json

One directory per benchmark, so seed data (a fixture corpus, a recorded fetch) can
live beside the definition later without changing this schema. The file is JSON
rather than TOML or YAML because every other durable record in this project is JSON,
and a benchmark is read by the same people who read artifacts.
"""

import json
from pathlib import Path
from typing import Final, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from app.artifacts import ArtifactKind
from app.utils.errors import ConfigurationError

BENCHMARKS_DIRNAME: Final[str] = "benchmarks"
"""Where benchmarks live, relative to the project root."""

BENCHMARK_FILENAME: Final[str] = "benchmark.json"


def project_root() -> Path:
    """The directory holding `pyproject.toml`.

    Resolved by walking up from this module rather than from the working directory,
    so a benchmark is found the same way whichever directory `op` was run from.
    Falls back to the current directory when the package is installed outside a
    checkout, in which case there are no benchmarks to find anyway.

    Deliberately a copy of `app.prompts.loader.project_root` rather than an import of
    it: that module reaches the LLM client through `app.llm.messages`, and nothing in
    this package should need a model provider to read a JSON file.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def benchmarks_root() -> Path:
    """The `benchmarks/` directory."""
    return project_root() / BENCHMARKS_DIRNAME


def benchmark_dir(name: str) -> Path:
    """The directory holding one benchmark's definition and any seed data."""
    return benchmarks_root() / name


def benchmark_path(name: str) -> Path:
    """Where one benchmark's `benchmark.json` lives."""
    return benchmark_dir(name) / BENCHMARK_FILENAME


class Expectation(BaseModel):
    """How many artifacts of one kind a healthy run produces.

    Both bounds are optional and both are inclusive. An absent bound is not a bound:
    `{"min": 1}` says only that the stage must produce something, which is the most
    useful thing a benchmark can assert about a stage whose output volume depends on
    what the sources happened to contain that day.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    min: int | None = Field(default=None, ge=0, description="Fewest artifacts. None = no floor.")
    max: int | None = Field(default=None, ge=0, description="Most artifacts. None = no ceiling.")

    @model_validator(mode="after")
    def _bounds_must_be_orderable(self) -> Self:
        """An inverted range can never be satisfied, so it is a mistake in the file."""
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) is above max ({self.max})")
        return self

    @property
    def is_empty(self) -> bool:
        """Whether this states no bound at all, and so asserts nothing."""
        return self.min is None and self.max is None

    def accepts(self, count: int) -> bool:
        """Whether `count` falls inside this expectation."""
        if self.min is not None and count < self.min:
            return False
        return not (self.max is not None and count > self.max)


class Benchmark(BaseModel):
    """A question worth re-asking, and the shape of a healthy answer to it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(description="Directory name under `benchmarks/`. Also the report key.")
    question: str = Field(description="The question text a run is seeded with, verbatim.")
    scope: str | None = Field(
        default=None,
        description="Boundary passed through to the Question — what counts as an answer.",
    )
    tags: list[str] = Field(default_factory=list, description="Labels for grouping benchmarks.")
    expect: dict[ArtifactKind, Expectation] = Field(
        default_factory=dict,
        description="Per-kind volume bands, keyed by the artifact kind's value, e.g. 'evidence'. "
        "A kind that is absent is simply not asserted about.",
    )
    themes: list[str] = Field(
        default_factory=list,
        description="Subjects a good run's clusters and opportunities should mention. Coarse by "
        "design: they catch a run that has wandered off the question entirely, not one that "
        "phrased its findings differently.",
    )

    @field_validator("name", "question")
    @classmethod
    def _require_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("scope")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return value.strip() or None

    @field_validator("tags")
    @classmethod
    def _normalise_tags(cls, value: list[str]) -> list[str]:
        """Fold case, drop blanks, de-duplicate — as `Question.tags` is normalised."""
        cleaned = (tag.strip().lower() for tag in value)
        return list(dict.fromkeys(tag for tag in cleaned if tag))

    @field_validator("themes")
    @classmethod
    def _normalise_themes(cls, value: list[str]) -> list[str]:
        """Keep the author's spelling — scoring normalises both sides at match time."""
        seen: set[str] = set()
        cleaned: list[str] = []
        for theme in (entry.strip() for entry in value):
            key = theme.casefold()
            if not theme or key in seen:
                continue
            seen.add(key)
            cleaned.append(theme)
        return cleaned

    def expectation_for(self, kind: ArtifactKind) -> Expectation | None:
        """The band declared for one kind, or `None` when the benchmark is silent."""
        return self.expect.get(kind)


def load_benchmark(name: str) -> Benchmark:
    """Load one benchmark by directory name.

    Raises:
        ConfigurationError: If the file is missing, is not JSON, is not an object, or
            does not validate. Every message names the path, because the reader's next
            action is always to open that file.
    """
    path = benchmark_path(name)
    if not path.is_file():
        raise ConfigurationError(f"No benchmark {name!r} at {path}")

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"{path} is not valid JSON: {exc}") from exc
    except OSError as exc:
        raise ConfigurationError(f"{path} could not be read: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigurationError(f"{path} does not contain a JSON object")

    # The directory is the benchmark's identity, so the file need not repeat it.
    data.setdefault("name", name)

    try:
        benchmark = Benchmark.model_validate(data)
    except ValidationError as exc:
        raise ConfigurationError(f"{path} is not a valid benchmark: {exc}") from exc

    if benchmark.name != name:
        raise ConfigurationError(
            f"{path} declares name {benchmark.name!r} but sits in the {name!r} directory"
        )
    return benchmark


def available() -> tuple[str, ...]:
    """Every benchmark that has a definition on disk, sorted."""
    root = benchmarks_root()
    if not root.is_dir():
        return ()
    return tuple(
        sorted(child.name for child in root.iterdir() if (child / BENCHMARK_FILENAME).is_file())
    )


def load_all() -> list[Benchmark]:
    """Load every benchmark, in name order.

    One malformed file fails the whole call rather than being skipped: a benchmark
    that silently disappeared from a comparison is worse than one that refused to load.
    """
    return [load_benchmark(name) for name in available()]


__all__ = [
    "BENCHMARKS_DIRNAME",
    "BENCHMARK_FILENAME",
    "Benchmark",
    "Expectation",
    "available",
    "benchmark_dir",
    "benchmark_path",
    "benchmarks_root",
    "load_all",
    "load_benchmark",
    "project_root",
]
