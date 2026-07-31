"""Immutable, content-addressed storage for raw source captures."""

import os
from pathlib import Path
from typing import Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from app.venture.core.ids import safe_join, sha256_bytes, validate_relative_path, validate_sha256
from app.venture.core.models import FrozenModel


class SnapshotError(RuntimeError):
    """Base failure for the raw snapshot store."""


class SnapshotIntegrityError(SnapshotError):
    """Stored bytes no longer match their content-addressed key."""


class SnapshotRecord(FrozenModel):
    """Reference returned by ``SnapshotStore.put``."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        str_strip_whitespace=True,
        arbitrary_types_allowed=True,
    )

    sha256: str
    size_bytes: int = Field(ge=0)
    relative_path: str
    path: Path

    @field_validator("sha256")
    @classmethod
    def _valid_digest(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("relative_path")
    @classmethod
    def _safe_relative_path(cls, value: str) -> str:
        return validate_relative_path(value)

    @model_validator(mode="after")
    def _path_matches_relative_path(self) -> Self:
        if self.path.name != self.sha256[2:] or self.path.parent.name != self.sha256[:2]:
            raise ValueError("snapshot path does not match sha256 key")
        if self.path.as_posix().endswith(self.relative_path) is False:
            raise ValueError("path does not end with relative_path")
        return self

    @property
    def content_hash(self) -> str:
        """Alias matching ``Measurement.source.raw_content_hash`` terminology."""
        return self.sha256

    @property
    def raw_content_hash(self) -> str:
        """Direct alias for wiring the record into a Measurement source."""
        return self.sha256


class SnapshotStore:
    """Write-once raw-byte store keyed by SHA-256.

    A digest is sharded as ``<first two>/<remaining 62>``.  Existing files are
    verified and returned, never overwritten.  New files are created with
    ``O_EXCL``, flushed, fsynced, and made read-only.
    """

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    def put(self, content: bytes) -> SnapshotRecord:
        """Store exact bytes once and return their durable reference."""
        if not isinstance(content, bytes):
            raise TypeError("SnapshotStore.put accepts bytes only")

        digest = sha256_bytes(content)
        path = self.path_for(digest)
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        except FileExistsError:
            return self._record_for_existing(digest, path, expected=content)

        try:
            with os.fdopen(descriptor, "wb", closefd=True) as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(path, 0o444)
            _fsync_directory(path.parent)
        except BaseException:
            # Only remove the inode this call exclusively created.  A pre-existing
            # immutable snapshot is never touched.
            try:
                path.unlink(missing_ok=True)
            finally:
                raise

        return SnapshotRecord(
            sha256=digest,
            size_bytes=len(content),
            relative_path=self.relative_path_for(digest),
            path=path,
        )

    def relative_path_for(self, digest: str) -> str:
        """Return the deterministic sharded path for ``digest``."""
        valid = validate_sha256(digest)
        return f"{valid[:2]}/{valid[2:]}"

    def path_for(self, digest: str) -> Path:
        """Resolve a digest to a safe path under this store."""
        return safe_join(self.root, self.relative_path_for(digest))

    def read(self, snapshot: SnapshotRecord | str) -> bytes:
        """Read and verify a snapshot, raising on absence or corruption."""
        digest = snapshot.sha256 if isinstance(snapshot, SnapshotRecord) else snapshot
        path = self.path_for(digest)
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise SnapshotError(f"snapshot {digest!r} could not be read: {exc}") from exc
        if sha256_bytes(content) != digest:
            raise SnapshotIntegrityError(f"snapshot {digest!r} does not match its SHA-256 key")
        if isinstance(snapshot, SnapshotRecord) and len(content) != snapshot.size_bytes:
            raise SnapshotIntegrityError(
                f"snapshot {digest!r} has size {len(content)}, expected {snapshot.size_bytes}"
            )
        return content

    def verify(self, snapshot: SnapshotRecord | str) -> bool:
        """Return whether a snapshot exists and still matches its reference."""
        try:
            self.read(snapshot)
        except SnapshotError:
            return False
        return True

    def _record_for_existing(
        self,
        digest: str,
        path: Path,
        *,
        expected: bytes,
    ) -> SnapshotRecord:
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise SnapshotError(f"existing snapshot {digest!r} could not be read: {exc}") from exc
        if existing != expected or sha256_bytes(existing) != digest:
            raise SnapshotIntegrityError(
                f"existing snapshot {digest!r} is corrupt; immutable content was not overwritten"
            )
        return SnapshotRecord(
            sha256=digest,
            size_bytes=len(existing),
            relative_path=self.relative_path_for(digest),
            path=path,
        )


def _fsync_directory(path: Path) -> None:
    """Persist a directory entry on filesystems that support directory fsync."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        # Some platforms/filesystems reject directory fsync.  The file itself
        # has still been fsynced, which is the strongest portable guarantee.
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "SnapshotError",
    "SnapshotIntegrityError",
    "SnapshotRecord",
    "SnapshotStore",
]
