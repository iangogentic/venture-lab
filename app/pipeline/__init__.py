"""The pipeline engine.

Nine stages, in order, each implemented by the skill of the same name. The engine
owns composition and resumability; skills own the individual steps.
"""

from app.pipeline.auto import AutoResult, AutoRunner, RunObserver, Spend, StageAttempt
from app.pipeline.engine import (
    STAGE_ORDER,
    PipelineEngine,
    PipelineRun,
    StageOutcome,
    StageStatus,
)
from app.pipeline.reporting import Composition, ReportUnavailableError, compose_report

__all__ = [
    "STAGE_ORDER",
    "AutoResult",
    "AutoRunner",
    "Composition",
    "PipelineEngine",
    "PipelineRun",
    "ReportUnavailableError",
    "RunObserver",
    "Spend",
    "StageAttempt",
    "StageOutcome",
    "StageStatus",
    "compose_report",
]
