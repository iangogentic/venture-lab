"""Load skill prompts from `.claude/skills/<name>/SKILL.md`.

Prompt text lives on disk as Markdown, never in Python. That keeps prompts
reviewable as prose in a diff, editable by someone who does not read Python, and
impossible to build accidentally by string concatenation at a call site.

Two files make up a prompt:

* `CLAUDE.md` at the project root — the shared philosophy, sent as the system
  message on **every** skill call. It is the one place the project's standing
  rules live, so a change there applies to all nine stages at once.
* `.claude/skills/<name>/SKILL.md` — the stage's own instructions, sent as the
  user message. Optional YAML frontmatter is stripped before sending.

Placeholders use shell-style `${name}` (`string.Template`) rather than
`str.format`'s `{name}`. Prompt files routinely contain literal braces — example
JSON especially — and `str.format` would force every one to be doubled.

Values are serialised to JSON by `render`, so artifacts reach the model as JSON
and no module ever concatenates a prompt by hand.
"""

import json
import re
from functools import cache
from pathlib import Path
from string import Template
from typing import Any, Final, cast

from pydantic import BaseModel, ConfigDict

from app.llm.messages import ChatMessage, Role
from app.utils.errors import ConfigurationError

SKILLS_DIRNAME: Final[str] = ".claude/skills"
"""Where skill prompts live, relative to the project root."""

SKILL_FILENAME: Final[str] = "SKILL.md"

PHILOSOPHY_FILENAME: Final[str] = "CLAUDE.md"
"""The project philosophy, used as the shared system message."""

_FRONTMATTER_FENCE: Final[str] = "---"


class _BracedOnly(Template):
    """`${name}` is the only placeholder form; any other `$` is literal prose.

    The default `Template` also treats `$name` as a placeholder and raises on a
    bare dollar sign, which would turn "costs $5/seat" in a prompt into a crash
    and `$PATH` into a surprise required value. Prompts are prose first; only
    the documented `${...}` form is special.
    """

    pattern = r"""
    \$(?:
      \{(?P<braced>[_a-z][_a-z0-9]*)\}   # ${name} — the documented form
      | (?P<escaped>(?!))                # no escape form; $$ stays literal
      | (?P<named>(?!))                  # $name is prose, not a placeholder
      | (?P<invalid>(?!))                # nothing is invalid; a bare $ passes through
    )
    """  # type: ignore[assignment]  # the metaclass compiles this str to a Pattern


_PLACEHOLDER_PATTERN: Final[re.Pattern[str]] = cast("re.Pattern[str]", _BracedOnly.pattern)


def project_root() -> Path:
    """The directory holding `pyproject.toml`.

    Resolved by walking up from this module rather than from the working
    directory, so `op` behaves the same run from a subdirectory. Falls back to
    the current directory when the package is installed outside a checkout.
    """
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def skills_root() -> Path:
    """The `.claude/skills/` directory."""
    return project_root() / SKILLS_DIRNAME


def skill_dir(name: str) -> Path:
    """The directory holding one skill's prompt."""
    return skills_root() / name


def prompt_path(name: str) -> Path:
    """Where a skill's `SKILL.md` lives."""
    return skill_dir(name) / SKILL_FILENAME


def philosophy_path() -> Path:
    """Where the shared system message lives."""
    return project_root() / PHILOSOPHY_FILENAME


def strip_frontmatter(text: str) -> str:
    """Remove a leading `---` YAML block, if present.

    The frontmatter names and describes the skill for tooling; it is metadata
    about the prompt, not part of what the model should be told.

    Raises:
        ConfigurationError: If the block is opened but never closed. Falling
            through silently would send the metadata to the model as prompt text.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_FENCE:
        return text

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONTMATTER_FENCE:
            return "\n".join(lines[index + 1 :]).lstrip("\n")
    raise ConfigurationError(
        "Frontmatter opened with '---' but never closed — the metadata would be "
        "sent to the model as prompt text."
    )


def _read(path: Path, what: str) -> str:
    if not path.is_file():
        raise ConfigurationError(f"No {what} at {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ConfigurationError(f"{path} is empty")
    return text


def read_skill_file(name: str) -> str:
    """Read one skill's prompt body, frontmatter removed."""
    try:
        return strip_frontmatter(_read(prompt_path(name), f"SKILL.md for {name!r}")).strip()
    except ConfigurationError as exc:
        raise ConfigurationError(f"SKILL.md for {name!r}: {exc}") from exc


def read_philosophy() -> str:
    """Read the shared system message."""
    return _read(philosophy_path(), "project philosophy (CLAUDE.md)")


class PromptTemplate(BaseModel):
    """A prompt loaded from disk, with its placeholders still unfilled."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    system: str
    user: str

    @property
    def placeholders(self) -> frozenset[str]:
        """Every `${name}` the skill's prompt expects to be given."""
        return frozenset(_placeholders(self.user))

    def render(self, values: dict[str, Any]) -> list[ChatMessage]:
        """Fill the placeholders and return the messages for a completion.

        Args:
            values: One entry per placeholder. Anything that is not already a
                string is serialised to indented JSON — this is how artifacts
                reach the model.

        Raises:
            ConfigurationError: If a placeholder has no value, or a value has no
                placeholder. Both mean the prompt and its skill have drifted
                apart, which must fail loudly rather than send a half-filled prompt.
        """
        expected = self.placeholders
        supplied = frozenset(values)

        if missing := expected - supplied:
            raise ConfigurationError(
                f"Prompt {self.name!r} needs values for: {', '.join(sorted(missing))}"
            )
        if extra := supplied - expected:
            raise ConfigurationError(
                f"Prompt {self.name!r} has no placeholder for: {', '.join(sorted(extra))}"
            )

        rendered = _BracedOnly(self.user).substitute(
            {key: _as_text(value) for key, value in values.items()}
        )
        return [
            ChatMessage(role=Role.SYSTEM, content=self.system),
            ChatMessage(role=Role.USER, content=rendered),
        ]


def _as_text(value: Any) -> str:
    """Serialise a placeholder value. Non-strings become indented JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _placeholders(template: str) -> set[str]:
    return {
        match.group("braced")
        for match in _PLACEHOLDER_PATTERN.finditer(template)
        if match.group("braced")
    }


@cache
def load_prompt(name: str) -> PromptTemplate:
    """Load a skill's prompt, with the project philosophy as its system message.

    Cached: prompt files do not change within a process. Call
    `load_prompt.cache_clear()` after editing one.
    """
    return PromptTemplate(
        name=name,
        system=read_philosophy(),
        user=read_skill_file(name),
    )


def render_prompt(name: str, values: dict[str, Any]) -> list[ChatMessage]:
    """Load a prompt and fill it in one call — the only way skills build messages."""
    return load_prompt(name).render(values)


def available() -> tuple[str, ...]:
    """Every skill that has a prompt on disk, sorted."""
    root = skills_root()
    if not root.is_dir():
        return ()
    return tuple(
        sorted(child.name for child in root.iterdir() if (child / SKILL_FILENAME).is_file())
    )


__all__ = [
    "PHILOSOPHY_FILENAME",
    "SKILLS_DIRNAME",
    "SKILL_FILENAME",
    "PromptTemplate",
    "available",
    "load_prompt",
    "philosophy_path",
    "project_root",
    "prompt_path",
    "read_philosophy",
    "read_skill_file",
    "render_prompt",
    "skill_dir",
    "skills_root",
    "strip_frontmatter",
]
