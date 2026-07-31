"""Schema lifecycle.

Importing `app.models` here is what registers the table classes on
`SQLModel.metadata`; without it `create_all` would be a silent no-op.

There is no migration tool here on purpose. The ledger is an index over the
artifacts, and its schema only ever grows — so bringing an old database up to
date means creating the tables that are new and adding the columns that are new,
which is a page of code rather than a dependency and a versions directory. If a
change ever needs more than that, the honest move is to drop the tables and
`op runs sync`, because everything except the run history is derivable.
"""

from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.schema import Table
from sqlmodel import SQLModel

import app.models  # noqa: F401  (registers tables on the metadata)
from app.storage.engine import get_engine
from app.utils.logging import get_logger

logger = get_logger(__name__)


def create_all(engine: Engine | None = None) -> None:
    """Bring the database up to the current schema.

    Creates tables that do not exist, then adds columns that do not exist.
    `metadata.create_all` alone does only the first, so a workspace whose
    `engine.db` predates a new column would keep loading and fail on the first
    query that selected it — the failure arriving later, and nowhere near the
    upgrade that caused it.
    """
    target = engine if engine is not None else get_engine()
    SQLModel.metadata.create_all(target)
    _add_missing_columns(target)


def _add_missing_columns(engine: Engine) -> None:
    """Add any column the models declare and the database has not got yet."""
    inspector = inspect(engine)
    for table in SQLModel.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        present = {column["name"] for column in inspector.get_columns(table.name)}
        added = [
            column.name
            for column in table.columns
            if column.name not in present and _add_column(engine, table, column.name)
        ]
        if added:
            logger.info("ledger: added %s to %s", ", ".join(added), table.name)
            _create_indexes(engine, table)


def _add_column(engine: Engine, table: Table, name: str) -> bool:
    """Add one column. Returns whether it was added.

    Deliberately without `NOT NULL`: SQLite rejects adding a non-null column to a
    table that already has rows and no default for it, and refusing the whole
    upgrade over a constraint the ORM enforces anyway would be the worse trade. A
    database created fresh gets the full constraint from `create_all`.
    """
    column = table.columns[name]
    if column.primary_key:
        logger.warning(
            "ledger: cannot add primary key %s.%s to an existing table; "
            "drop the table and re-run `op runs sync`",
            table.name,
            name,
        )
        return False
    ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{name}" {column.type.compile(engine.dialect)}'
    with engine.begin() as connection:
        connection.exec_driver_sql(ddl)
    return True


def _create_indexes(engine: Engine, table: Table) -> None:
    """Create any index the table is missing. New columns are often indexed ones."""
    for index in table.indexes:
        index.create(bind=engine, checkfirst=True)


def drop_all(engine: Engine | None = None) -> None:
    """Drop every known table. Destructive."""
    SQLModel.metadata.drop_all(engine if engine is not None else get_engine())


__all__ = ["create_all", "drop_all"]
