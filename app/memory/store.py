"""The evidence memory: one SQLite file holding every kept excerpt and its vector.

Storage is `<workspace>/memory.db` — beside the stage directories, not inside
one, because memory is deliberately cross-run: the artifact tree answers "what
did this run keep", memory answers "have I ever seen this complaint before".

Vector search comes from the sqlite-vec extension, loaded into the ordinary
`sqlite3` connection at construction. That requires an interpreter built with
extension loading enabled: uv-managed interpreters (which this project uses)
support it; some system Pythons — notably Apple's — compile it out. When the
extension cannot load, construction raises `MemoryUnavailableError` and the
caller runs with memory off; a vector store that cannot do vectors is not a
degraded store, it is no store at all.
"""

import sqlite3
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from app.artifacts import Evidence
from app.utils.errors import MemoryUnavailableError

EMBEDDING_DIMENSIONS: Final[int] = 256
"""Width of every stored vector — what potion-base-8M emits.

Fixed at schema level because vec0 tables are fixed-width, and vectors from
differently-sized models are not comparable anyway. Changing the embedding
model to one with another width means starting a new memory file.
"""

_SCHEMA: Final[str] = f"""
CREATE TABLE IF NOT EXISTS evidence_memory (
    id INTEGER PRIMARY KEY,
    evidence_id TEXT UNIQUE,
    run_id TEXT,
    question_text TEXT,
    collector TEXT,
    url TEXT,
    author TEXT,
    published_at TEXT,
    excerpt TEXT,
    model TEXT
);
CREATE VIRTUAL TABLE IF NOT EXISTS evidence_vectors USING vec0(
    embedding float[{EMBEDDING_DIMENSIONS}]
);
"""


class RecallHit(BaseModel):
    """One remembered piece of evidence, with its distance from the query."""

    model_config = ConfigDict(extra="forbid")

    distance: float = Field(description="L2 distance from the query vector; smaller is closer.")
    evidence_id: str
    run_id: str | None = None
    question_text: str | None = None
    collector: str | None = None
    url: str | None = None
    author: str | None = None
    published_at: str | None = None
    excerpt: str | None = None
    model: str | None = Field(default=None, description="Embedding model that made the vector.")


class MemoryStore:
    """Vectors and their provenance, keyed by evidence id.

    Raises:
        MemoryUnavailableError: At construction, when the sqlite-vec extension
            cannot be loaded into this interpreter's sqlite3.
    """

    def __init__(self, db_path: Path) -> None:
        try:
            import sqlite_vec
        except ImportError as exc:
            raise MemoryUnavailableError(
                "sqlite-vec is not installed; evidence memory is off"
            ) from exc

        self._serialize = sqlite_vec.serialize_float32
        self._conn = sqlite3.connect(db_path)
        try:
            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
        except Exception as exc:
            # AttributeError on interpreters compiled without extension loading,
            # OperationalError when loading is refused. Same meaning either way.
            self._conn.close()
            raise MemoryUnavailableError(
                f"sqlite-vec could not load into this interpreter: {exc}"
            ) from exc
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def index(
        self,
        evidence: Sequence[Evidence],
        vectors: Sequence[list[float]],
        *,
        run_id: str,
        question_text: str,
        model: str,
    ) -> int:
        """Remember evidence with its vectors. Returns how many were new.

        Upsert by `evidence_id` via INSERT OR IGNORE: re-running a stage
        re-assembles the same artifacts, and remembering them twice would make
        recall return the same excerpt as two hits. The vector row shares the
        metadata row's id, so an ignored insert also writes no vector.
        """
        if len(evidence) != len(vectors):
            raise ValueError(f"{len(evidence)} evidence rows but {len(vectors)} vectors")

        inserted = 0
        for item, vector in zip(evidence, vectors, strict=True):
            if len(vector) != EMBEDDING_DIMENSIONS:
                raise ValueError(
                    f"vector for {item.id} has {len(vector)} dimensions, "
                    f"expected {EMBEDDING_DIMENSIONS}"
                )
            cursor = self._conn.execute(
                "INSERT OR IGNORE INTO evidence_memory "
                "(evidence_id, run_id, question_text, collector, url, author, "
                " published_at, excerpt, model) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    item.id,
                    run_id,
                    question_text,
                    item.collector,
                    str(item.source_url) if item.source_url else None,
                    item.author,
                    item.published_at.isoformat() if item.published_at else None,
                    item.excerpt,
                    model,
                ),
            )
            if cursor.rowcount == 0:
                continue  # already remembered; the stored vector stands
            self._conn.execute(
                "INSERT INTO evidence_vectors (rowid, embedding) VALUES (?, ?)",
                (cursor.lastrowid, self._serialize(vector)),
            )
            inserted += 1
        self._conn.commit()
        return inserted

    def recall(self, vector: list[float], *, limit: int) -> list[RecallHit]:
        """The `limit` remembered items nearest to `vector`, nearest first."""
        rows = self._conn.execute(
            "SELECT knn.distance, m.evidence_id, m.run_id, m.question_text, "
            "       m.collector, m.url, m.author, m.published_at, m.excerpt, m.model "
            "FROM (SELECT rowid, distance FROM evidence_vectors "
            "      WHERE embedding MATCH ? AND k = ?) AS knn "
            "JOIN evidence_memory m ON m.id = knn.rowid "
            "ORDER BY knn.distance",
            (self._serialize(vector), limit),
        ).fetchall()
        return [
            RecallHit(
                distance=row[0],
                evidence_id=row[1],
                run_id=row[2],
                question_text=row[3],
                collector=row[4],
                url=row[5],
                author=row[6],
                published_at=row[7],
                excerpt=row[8],
                model=row[9],
            )
            for row in rows
        ]

    def count(self) -> int:
        """How many pieces of evidence are remembered."""
        row = self._conn.execute("SELECT COUNT(*) FROM evidence_memory").fetchone()
        return int(row[0])

    def close(self) -> None:
        """Release the connection. The store is not usable afterwards."""
        self._conn.close()


__all__ = ["EMBEDDING_DIMENSIONS", "MemoryStore", "RecallHit"]
