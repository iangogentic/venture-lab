"""Verify the run-ledger plumbing holds together."""

from pathlib import Path

import pytest
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlmodel import Session

from app.models import Run, RunStatus
from app.storage.engine import create_db_engine
from app.storage.schema import create_all


def test_schema_creates_expected_tables(engine: Engine) -> None:
    tables = set(inspect(engine).get_table_names())
    assert {"runs", "sources", "evidence", "opportunities"} <= tables


def test_absolute_sqlite_url_stays_absolute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An absolute SQLite URL must not be created relative to the cwd.

    Regression: naive `urlparse(...).path.lstrip("/")` turns SQLAlchemy's
    four-slash absolute form into a relative path and scatters directories
    into the working tree.
    """
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    target = tmp_path / "nested" / "abs.db"
    create_all(create_db_engine(f"sqlite:///{target}"))

    assert target.exists()
    assert list(cwd.iterdir()) == []


def test_relative_sqlite_url_creates_parent_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A relative SQLite URL resolves against the cwd."""
    monkeypatch.chdir(tmp_path)

    create_all(create_db_engine("sqlite:///./ledger/rel.db"))

    assert (tmp_path / "ledger" / "rel.db").exists()


def test_an_older_database_gains_the_columns_it_is_missing(tmp_path: Path) -> None:
    """Regression: `metadata.create_all` creates missing tables, never missing
    columns. A workspace whose `engine.db` predated a new column kept loading and
    then failed on the first query that selected it — an error arriving days
    after the upgrade that caused it, and nowhere near it.
    """
    url = f"sqlite:///{tmp_path / 'old.db'}"
    engine = create_db_engine(url)
    with engine.begin() as connection:
        # The ledger as it was before runs carried a question or a cost.
        connection.exec_driver_sql(
            "CREATE TABLE runs (id VARCHAR PRIMARY KEY, status VARCHAR, "
            "created_at DATETIME, updated_at DATETIME)"
        )
        connection.exec_driver_sql("INSERT INTO runs (id, status) VALUES ('old', 'COMPLETED')")

    create_all(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("runs")}
    assert {"question", "question_id", "auto", "cost", "report_id"} <= columns
    assert "stage_runs" in inspect(engine).get_table_names(), "new tables are created too"

    # The row that was already there survives, and is now queryable.
    with Session(engine) as session:
        preserved = session.get(Run, "old")
        assert preserved is not None
        assert preserved.status is RunStatus.COMPLETED, "the old row is intact"
        assert preserved.question is None, "an added column reads as null, not an error"


def test_bringing_a_database_up_to_date_is_idempotent(tmp_path: Path) -> None:
    engine = create_db_engine(f"sqlite:///{tmp_path / 'twice.db'}")
    create_all(engine)
    create_all(engine)

    assert {"runs", "stage_runs"} <= set(inspect(engine).get_table_names())
