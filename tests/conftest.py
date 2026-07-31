"""Shared fixtures.

Every test runs against a throwaway workspace and ledger, so nothing touches the
developer's real `workspace/` directory.
"""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy.engine import Engine
from sqlmodel import Session

from app.config import get_settings
from app.storage.engine import create_db_engine
from app.storage.schema import create_all
from app.utils.paths import WorkspacePaths, get_workspace_paths


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Point every test at a throwaway workspace/ledger and clear the settings cache."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path / "workspace"))
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'engine.db'}")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Nothing here may touch the network. Model tiers resolve from pinned slugs
    # instead of the live Models API, and retrieval is pointed at the one collector
    # that needs no remote host — left unconfigured, so it reports itself
    # unavailable and `collect-evidence` fails locally and deterministically.
    # A suite that searches Hacker News is slow, flaky, and rude to the source.
    monkeypatch.setenv("LLM_USE_CATALOGUE", "false")
    monkeypatch.setenv("COLLECTORS", '["filesystem"]')
    monkeypatch.setenv("CORPUS_PATHS", "[]")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def workspace() -> WorkspacePaths:
    """A materialised, empty workspace tree."""
    paths = get_workspace_paths()
    paths.ensure()
    return paths


@pytest.fixture
def engine() -> Engine:
    """A migrated, test-scoped engine."""
    eng = create_db_engine()
    create_all(eng)
    return eng


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    """A session bound to the test engine."""
    with Session(engine) as sess:
        yield sess
