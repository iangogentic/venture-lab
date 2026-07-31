"""Generic repository base.

Repositories stage work on a session and never commit. Committing is the
caller's decision because a run's worth of ledger writes has to land or not land
together — `session_scope` is where that happens, and a repository that
committed on its own would quietly take that choice away.
"""

from typing import Any

from sqlalchemy import func
from sqlmodel import Session, SQLModel, select

from app.models.base import TimestampMixin
from app.utils.errors import StorageError


class BaseRepository[ModelT: SQLModel]:
    """CRUD surface shared by every repository."""

    model: type[ModelT]

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, obj_id: Any) -> ModelT | None:
        """Fetch one row by primary key, or None."""
        return self.session.get(self.model, obj_id)

    def require(self, obj_id: Any) -> ModelT:
        """Fetch one row by primary key.

        Raises:
            StorageError: If no such row exists.
        """
        found = self.get(obj_id)
        if found is None:
            raise StorageError(f"no {self.model.__name__} row with id {obj_id!r}")
        return found

    def list(self, *, limit: int = 50, offset: int = 0) -> list[ModelT]:
        """Fetch a page of rows, in insertion order."""
        return list(self.session.exec(select(self.model).offset(offset).limit(limit)).all())

    def add(self, obj: ModelT) -> ModelT:
        """Stage a new row and flush it, so a generated primary key is readable."""
        self.session.add(obj)
        self.session.flush()
        return obj

    def update(self, obj_id: Any, **fields: Any) -> ModelT:
        """Apply a partial update to a row, by primary key.

        Raises:
            StorageError: If no such row exists, or a field is not on the model.
        """
        return self.apply(self.require(obj_id), **fields)

    def apply(self, obj: ModelT, **fields: Any) -> ModelT:
        """Apply a partial update to a row already in hand.

        An unknown field name raises rather than being dropped: a typo in a
        ledger write would otherwise look exactly like a successful one.

        Raises:
            StorageError: If a field is not on the model.
        """
        for name, value in fields.items():
            if not hasattr(obj, name):
                raise StorageError(f"{type(obj).__name__} has no field {name!r}")
            setattr(obj, name, value)
        if isinstance(obj, TimestampMixin):
            obj.touch()
        self.session.add(obj)
        self.session.flush()
        return obj

    def delete(self, obj_id: Any) -> None:
        """Remove a row. A missing row is not an error — deleting is idempotent."""
        found = self.get(obj_id)
        if found is None:
            return
        self.session.delete(found)
        self.session.flush()

    def count(self) -> int:
        """Total number of rows."""
        return int(self.session.scalar(select(func.count()).select_from(self.model)) or 0)


__all__ = ["BaseRepository"]
