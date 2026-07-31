"""Prompts live in Markdown, are filled with JSON artifacts, and never built in code."""

import ast
import json
from itertools import pairwise
from pathlib import Path

import pytest

from app.pipeline import STAGE_ORDER
from app.prompts import (
    PromptTemplate,
    available,
    load_prompt,
    philosophy_path,
    prompt_path,
    render_prompt,
    skills_root,
    strip_frontmatter,
)
from app.utils.errors import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MAX_INLINE_STRING = 200
"""Longer than any legitimate message; a prompt smuggled into code would exceed it."""

PROMPTED_SKILLS = (*STAGE_ORDER, "harvest-leads", "compose-report")
"""Every skill owning a prompt: the nine stages plus the on-demand lead harvest
and report composition."""


# ------------------------------------------------------ prompts live on disk


def test_every_stage_has_a_skill_file() -> None:
    for stage in PROMPTED_SKILLS:
        assert prompt_path(stage).is_file(), f"missing .claude/skills/{stage}/SKILL.md"


def test_project_philosophy_is_the_system_message() -> None:
    assert philosophy_path().is_file()
    assert load_prompt(STAGE_ORDER[0]).system == philosophy_path().read_text().strip()


def test_available_lists_every_skill() -> None:
    assert set(PROMPTED_SKILLS) <= set(available())


def test_prompts_live_outside_the_python_package() -> None:
    assert skills_root().is_dir()
    assert not list((skills_root().parent.parent / "app" / "prompts").glob("*.md"))


def test_frontmatter_is_stripped_before_sending() -> None:
    """Frontmatter names the skill for tooling; the model should not see it."""
    assert strip_frontmatter("---\nname: x\n---\n\nBody") == "Body"
    assert strip_frontmatter("No frontmatter") == "No frontmatter"
    assert not load_prompt(STAGE_ORDER[0]).user.startswith("---")


# ------------------------------------------------------------- the contract


def test_system_prompt_has_no_placeholders() -> None:
    """It is shared by every skill, so it cannot reference any one skill's inputs."""
    assert "${" not in load_prompt(STAGE_ORDER[0]).system


@pytest.mark.parametrize("stage", list(PROMPTED_SKILLS))
def test_stage_prompt_declares_placeholders(stage: str) -> None:
    assert load_prompt(stage).placeholders, f"{stage}.md interpolates nothing"


def test_missing_prompt_raises() -> None:
    with pytest.raises(ConfigurationError, match=r"No SKILL\.md"):
        load_prompt("no-such-prompt")


# --------------------------------------------------------- JSON injection


def _template(user: str) -> PromptTemplate:
    return PromptTemplate(name="t", system="sys", user=user)


def test_artifacts_are_injected_as_json() -> None:
    """ "The application injects JSON artifacts" — values are serialised, not str()'d."""
    messages = _template("Evidence:\n${evidence}").render(
        {"evidence": [{"id": "ev_1", "excerpt": "CI is slow"}]}
    )

    body = messages[-1].content
    assert json.loads(body.split("Evidence:", 1)[1]) == [{"id": "ev_1", "excerpt": "CI is slow"}]


def test_strings_are_injected_verbatim() -> None:
    messages = _template("Q: ${question}").render({"question": "Why?"})
    assert messages[-1].content == "Q: Why?"


def test_render_produces_a_system_and_a_user_message() -> None:
    messages = _template("body ${x}").render({"x": 1})
    assert [m.role.value for m in messages] == ["system", "user"]
    assert messages[0].content == "sys"


def test_literal_braces_survive() -> None:
    """`${name}` was chosen over `{name}` precisely so JSON examples need no escaping."""
    messages = _template('Example: {"ok": true}\n${payload}').render({"payload": {"a": 1}})
    assert '{"ok": true}' in messages[-1].content


def test_a_literal_dollar_is_prose_not_a_crash() -> None:
    """Prompts are prose first: "costs $5/seat" must survive rendering untouched."""
    messages = _template("It costs $5/seat or $CHEAP. ${x}").render({"x": "ok"})
    assert messages[-1].content == "It costs $5/seat or $CHEAP. ok"


def test_a_bare_dollar_name_is_not_a_placeholder() -> None:
    """Only the documented `${name}` form binds; `$name` is text, not a contract."""
    template = _template("Set $PATH before ${real}")
    assert template.placeholders == frozenset({"real"})


def test_unclosed_frontmatter_is_refused() -> None:
    """Falling through silently would send the YAML to the model as prompt text."""
    with pytest.raises(ConfigurationError, match="never closed"):
        strip_frontmatter("---\nname: x\ndescription: never closed\n\nBody")


def test_a_thematic_break_is_not_frontmatter() -> None:
    assert strip_frontmatter("----\nBody") == "----\nBody"


def test_a_missing_value_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="needs values for"):
        _template("${a} ${b}").render({"a": 1})


def test_an_unused_value_is_refused() -> None:
    """A value with no placeholder means the caller and prompt have drifted apart."""
    with pytest.raises(ConfigurationError, match="no placeholder for"):
        _template("${a}").render({"a": 1, "stray": 2})


def test_render_prompt_helper_matches_load_and_render() -> None:
    stage = STAGE_ORDER[0]
    values = {name: {"stub": True} for name in load_prompt(stage).placeholders}

    assert render_prompt(stage, values) == load_prompt(stage).render(values)


# ---------------------------------------------- no prompt text lives in code


def _is_string_expr(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _inline_strings(path: Path) -> list[str]:
    """Every string literal in a module that is not a docstring.

    "Docstring" includes the attribute form this codebase uses everywhere — a bare
    string directly after an assignment documents that constant. Missing those made
    an earlier version of this guard flag well-documented constants as smuggled
    prompts.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()

    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            continue
        # The leading string of a module, class or function.
        if body and _is_string_expr(body[0]) and not isinstance(node, ast.If | ast.Try):
            docstrings.add(id(body[0].value))
        # A string documenting the assignment above it.
        for previous, statement in pairwise(body):
            if isinstance(previous, ast.Assign | ast.AnnAssign) and _is_string_expr(statement):
                docstrings.add(id(statement.value))

    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


@pytest.mark.parametrize(
    "package", ["skills", "pipeline", "cli"], ids=["skills", "pipeline", "cli"]
)
def test_no_prompt_text_is_embedded_in_python(package: str) -> None:
    """ "Do not hardcode prompts" — enforced, not merely agreed.

    A prompt smuggled into a module would be a long string literal outside a
    docstring. Nothing legitimate in these packages needs one.
    """
    offenders = [
        f"{path.relative_to(PROJECT_ROOT)}: {text[:60]!r}…"
        for path in (PROJECT_ROOT / "app" / package).rglob("*.py")
        for text in _inline_strings(path)
        if len(text) > MAX_INLINE_STRING
    ]

    assert offenders == [], f"prompt-sized string literals found in code: {offenders}"
