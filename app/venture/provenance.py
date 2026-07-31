"""Deterministic implementation snapshots for reproducible venture pilot runs."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Self

from pydantic import Field, field_validator, model_validator

from app.venture.core import (
    FrozenModel,
    sha256_bytes,
    validate_relative_path,
    validate_sha256,
)


class ImplementationFile(FrozenModel):
    """One repository-relative source file captured by a pilot run."""

    path: str
    size_bytes: int = Field(ge=0)
    sha256: str

    @field_validator("path")
    @classmethod
    def _safe_path(cls, value: str) -> str:
        return validate_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return validate_sha256(value)


class ImplementationManifest(FrozenModel):
    """Exact source inventory and its deterministic uncompressed tar archive."""

    schema_version: str = "venture-implementation-manifest-v1"
    archive_format: str = "ustar-uncompressed"
    source_tar_sha256: str
    source_tar_size_bytes: int = Field(ge=0)
    files: tuple[ImplementationFile, ...] = Field(min_length=1)

    @field_validator("source_tar_sha256")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return validate_sha256(value)

    @model_validator(mode="after")
    def _canonical_inventory(self) -> Self:
        if self.schema_version != "venture-implementation-manifest-v1":
            raise ValueError("unsupported implementation-manifest schema")
        if self.archive_format != "ustar-uncompressed":
            raise ValueError("implementation archive must be deterministic uncompressed USTAR")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)):
            raise ValueError("implementation-manifest paths must be sorted")
        if len(paths) != len(set(paths)):
            raise ValueError("implementation-manifest paths must be unique")
        return self


def build_implementation_bundle(
    repo_root: Path | None = None,
) -> tuple[ImplementationManifest, bytes]:
    """Capture the relevant implementation into a byte-stable source archive."""
    root = (repo_root or Path(__file__).resolve().parents[2]).resolve()
    source_paths = _implementation_source_paths(root)
    sources: list[tuple[str, bytes]] = []
    for relative_path in source_paths:
        absolute_path = root / relative_path
        try:
            content = absolute_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"could not read implementation source {relative_path!r}") from exc
        sources.append((relative_path, content))

    source_tar = _deterministic_tar(sources)
    manifest = ImplementationManifest(
        source_tar_sha256=sha256_bytes(source_tar),
        source_tar_size_bytes=len(source_tar),
        files=tuple(
            ImplementationFile(
                path=relative_path,
                size_bytes=len(content),
                sha256=sha256_bytes(content),
            )
            for relative_path, content in sources
        ),
    )
    return manifest, source_tar


def verify_implementation_bundle(
    manifest: ImplementationManifest,
    source_tar: bytes,
) -> None:
    """Verify archive metadata, members, and file hashes without extracting it."""
    if len(source_tar) != manifest.source_tar_size_bytes:
        raise ValueError("implementation source tar size does not match its manifest")
    if sha256_bytes(source_tar) != manifest.source_tar_sha256:
        raise ValueError("implementation source tar hash does not match its manifest")

    try:
        with tarfile.open(fileobj=io.BytesIO(source_tar), mode="r:") as archive:
            members = archive.getmembers()
            if tuple(member.name for member in members) != tuple(
                item.path for item in manifest.files
            ):
                raise ValueError("implementation source tar members do not match its manifest")
            for member, expected in zip(members, manifest.files, strict=True):
                if (
                    not member.isfile()
                    or member.mode != 0o644
                    or member.uid != 0
                    or member.gid != 0
                    or member.uname != ""
                    or member.gname != ""
                    or member.mtime != 0
                    or member.size != expected.size_bytes
                ):
                    raise ValueError(
                        f"implementation source tar metadata differs for {expected.path!r}"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"implementation source tar cannot read {expected.path!r}")
                content = extracted.read()
                if len(content) != expected.size_bytes or sha256_bytes(content) != expected.sha256:
                    raise ValueError(
                        f"implementation source tar content differs for {expected.path!r}"
                    )
    except (tarfile.TarError, OSError) as exc:
        raise ValueError("implementation source tar is unreadable") from exc


def _implementation_source_paths(root: Path) -> tuple[str, ...]:
    required = {
        "app/cli/commands/__init__.py",
        "app/cli/commands/venture.py",
        "app/cli/main.py",
        "app/config.py",
        "pyproject.toml",
        "uv.lock",
    }
    for source_root in ("app/venture", "app/llm"):
        directory = root / source_root
        if not directory.is_dir():
            raise ValueError(f"implementation source directory is missing: {source_root!r}")
        required.update(
            path.relative_to(root).as_posix() for path in directory.rglob("*.py") if path.is_file()
        )
    missing = tuple(path for path in sorted(required) if not (root / path).is_file())
    if missing:
        raise ValueError(f"required implementation sources are missing: {', '.join(missing)}")
    return tuple(sorted(validate_relative_path(path) for path in required))


def _deterministic_tar(sources: list[tuple[str, bytes]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:", format=tarfile.USTAR_FORMAT) as archive:
        for relative_path, content in sources:
            info = tarfile.TarInfo(name=relative_path)
            info.size = len(content)
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            info.mtime = 0
            info.type = tarfile.REGTYPE
            archive.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


__all__ = [
    "ImplementationFile",
    "ImplementationManifest",
    "build_implementation_bundle",
    "verify_implementation_bundle",
]
