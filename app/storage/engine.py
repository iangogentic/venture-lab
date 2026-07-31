"""SQLModel/SQLAlchemy engine construction."""

from functools import lru_cache
from pathlib import Path

from sqlalchemy.engine import Engine, make_url
from sqlmodel import create_engine

from app.config import get_settings


def _ensure_sqlite_parent_dir(url: str) -> None:
    """Create the parent directory of a file-backed SQLite database, if needed.

    Parsed with `make_url` rather than `urlparse`: SQLAlchemy distinguishes
    `sqlite:///relative/path` from `sqlite:////absolute/path` by slash count, and
    hand-rolled stripping turns absolute paths into relative ones.
    """
    parsed = make_url(url)
    if not parsed.drivername.startswith("sqlite"):
        return
    database = parsed.database
    if not database or database == ":memory:":
        return
    Path(database).parent.mkdir(parents=True, exist_ok=True)


def create_db_engine(url: str | None = None, *, echo: bool | None = None) -> Engine:
    """Build a new engine. Pass `url` to point at a different database (e.g. in tests)."""
    settings = get_settings()
    resolved_url = url if url is not None else settings.database_url
    resolved_echo = settings.database_echo if echo is None else echo

    _ensure_sqlite_parent_dir(resolved_url)

    connect_args: dict[str, object] = {}
    if resolved_url.startswith("sqlite"):
        # Sessions may be handed between threads by the CLI; SQLite guards
        # against that by default.
        connect_args["check_same_thread"] = False

    return create_engine(resolved_url, echo=resolved_echo, connect_args=connect_args)


@lru_cache(maxsize=8)
def _engine_for(url: str, echo: bool) -> Engine:
    """One engine per database URL, built once and shared."""
    return create_db_engine(url, echo=echo)


def get_engine() -> Engine:
    """Return the shared engine for the currently configured database.

    Cached by URL rather than by process, so repointing `DATABASE_URL` — which a
    test does for every test — gets the database it asked for. Caching the first
    engine forever instead would silently write every later run's ledger into
    whichever database happened to be configured first.
    """
    settings = get_settings()
    return _engine_for(settings.database_url, settings.database_echo)


def reset_engines() -> None:
    """Forget every cached engine. For tests, and for a settings reload."""
    _engine_for.cache_clear()


__all__ = ["create_db_engine", "get_engine", "reset_engines"]
