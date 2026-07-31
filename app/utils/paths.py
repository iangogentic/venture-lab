"""Workspace layout.

`workspace/` is the artifact store of record: one directory per pipeline stage,
holding durable JSON artifacts. Resolving those paths in exactly one place keeps
the directory names from drifting apart across the pipeline, the CLI, and tests.
"""

from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings

STAGE_DIRECTORIES: tuple[str, ...] = (
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
"""Workspace subdirectories, in pipeline order — `leads` sits after the stages
because it is filled on demand by `op leads harvest`, not by a pipeline stage."""


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """Absolute paths to every workspace directory."""

    root: Path
    questions: Path
    evidence: Path
    briefs: Path
    clusters: Path
    opportunities: Path
    market: Path
    competition: Path
    contradictions: Path
    decisions: Path
    interviews: Path
    leads: Path
    reports: Path

    @classmethod
    def from_root(cls, root: Path) -> "WorkspacePaths":
        """Derive every stage directory from a workspace root."""
        resolved = root.expanduser().resolve()
        return cls(
            root=resolved,
            questions=resolved / "questions",
            evidence=resolved / "evidence",
            briefs=resolved / "briefs",
            clusters=resolved / "clusters",
            opportunities=resolved / "opportunities",
            market=resolved / "market",
            competition=resolved / "competition",
            contradictions=resolved / "contradictions",
            decisions=resolved / "decisions",
            interviews=resolved / "interviews",
            leads=resolved / "leads",
            reports=resolved / "reports",
        )

    @classmethod
    def from_settings(cls) -> "WorkspacePaths":
        """Derive the workspace layout from the configured `workspace_dir`."""
        return cls.from_root(get_settings().workspace_dir)

    @property
    def memory_db(self) -> Path:
        """The semantic-memory database — beside the stage directories, not in one.

        A file rather than a directory because it is not an artifact store:
        memory spans runs, and the stage tree is deliberately per-stage.
        """
        return self.root / "memory.db"

    def all_dirs(self) -> tuple[Path, ...]:
        """Every stage directory, in pipeline order."""
        return tuple(self.root / name for name in STAGE_DIRECTORIES)

    def for_directory(self, name: str) -> Path:
        """Resolve a stage directory by its workspace name."""
        return self.root / name

    def ensure(self) -> None:
        """Create the workspace root and every stage directory."""
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in self.all_dirs():
            directory.mkdir(parents=True, exist_ok=True)


def get_workspace_paths() -> WorkspacePaths:
    """Return the workspace layout for the current settings.

    Deliberately uncached: tests and the `--workspace` flag repoint it at runtime.
    """
    return WorkspacePaths.from_settings()


__all__ = ["STAGE_DIRECTORIES", "WorkspacePaths", "get_workspace_paths"]
