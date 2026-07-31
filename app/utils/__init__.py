"""Cross-cutting helpers: console, logging, errors, ids, workspace paths."""

from app.utils.console import console, err_console
from app.utils.errors import (
    ArtifactError,
    CollectorError,
    ConfigurationError,
    LLMError,
    OpportunityEngineError,
    PipelineError,
    SkillError,
    StorageError,
)
from app.utils.ids import new_id, slugify
from app.utils.logging import configure_logging, get_logger
from app.utils.paths import WorkspacePaths, get_workspace_paths

__all__ = [
    "ArtifactError",
    "CollectorError",
    "ConfigurationError",
    "LLMError",
    "OpportunityEngineError",
    "PipelineError",
    "SkillError",
    "StorageError",
    "WorkspacePaths",
    "configure_logging",
    "console",
    "err_console",
    "get_logger",
    "get_workspace_paths",
    "new_id",
    "slugify",
]
