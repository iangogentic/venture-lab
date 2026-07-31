"""Guard the agreed project layout.

The folder structure is part of the spec, not an accident of how the code grew.
This test fails loudly if a package or workspace directory is renamed or dropped.
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

APP_PACKAGES = (
    "cli",
    "artifacts",
    "pipeline",
    "skills",
    "llm",
    "storage",
    "collectors",
    "prompts",
    "models",
    "utils",
)

WORKSPACE_DIRS = (
    "questions",
    "evidence",
    "briefs",
    "clusters",
    "opportunities",
    "market",
    "competition",
    "contradictions",
    "decisions",
    "interviews",
    "leads",
    "reports",
)

TOP_LEVEL_FILES = ("pyproject.toml", "README.md", "CLAUDE.md", "LICENSE")


@pytest.mark.parametrize("package", APP_PACKAGES)
def test_app_package_exists(package: str) -> None:
    directory = PROJECT_ROOT / "app" / package
    assert directory.is_dir(), f"missing package app/{package}/"
    assert (directory / "__init__.py").is_file(), f"app/{package}/ is not a package"


@pytest.mark.parametrize("name", WORKSPACE_DIRS)
def test_workspace_directory_exists(name: str) -> None:
    assert (PROJECT_ROOT / "workspace" / name).is_dir(), f"missing workspace/{name}/"


@pytest.mark.parametrize("filename", TOP_LEVEL_FILES)
def test_top_level_file_exists(filename: str) -> None:
    assert (PROJECT_ROOT / filename).is_file(), f"missing {filename}"


def test_tests_directory_is_this_one() -> None:
    assert (PROJECT_ROOT / "tests").is_dir()


def test_no_stale_src_layout() -> None:
    """The project uses app/ at the root, not a src/ layout."""
    assert not (PROJECT_ROOT / "src").exists()


def test_workspace_directories_match_the_artifact_kinds() -> None:
    """The declared layout and the code's own idea of it must not drift apart."""
    from app.utils.paths import STAGE_DIRECTORIES

    assert set(STAGE_DIRECTORIES) == set(WORKSPACE_DIRS)
