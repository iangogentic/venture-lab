"""Session helpers."""

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy.engine import Engine
from sqlmodel import Session

from app.storage.engine import get_engine


def get_session(engine: Engine | None = None) -> Session:
    """Return an unmanaged session. Prefer `session_scope` unless you need manual control."""
    return Session(engine or get_engine())


@contextmanager
def session_scope(engine: Engine | None = None) -> Iterator[Session]:
    """Yield a session, committing on success and rolling back on error."""
    session = get_session(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


__all__ = ["get_session", "session_scope"]
