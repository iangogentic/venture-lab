"""Skill prompts, loaded from `.claude/skills/<name>/SKILL.md`.

The shared system message is the project philosophy in `CLAUDE.md`; each skill's
own instructions are its `SKILL.md`. Placeholders are `${name}` and are filled
with the JSON of the artifacts the skill gathered, so no code ever concatenates a
prompt string.
"""

from app.prompts.loader import (
    PHILOSOPHY_FILENAME,
    SKILL_FILENAME,
    SKILLS_DIRNAME,
    PromptTemplate,
    available,
    load_prompt,
    philosophy_path,
    project_root,
    prompt_path,
    read_philosophy,
    read_skill_file,
    render_prompt,
    skill_dir,
    skills_root,
    strip_frontmatter,
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
