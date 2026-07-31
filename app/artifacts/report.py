"""Report: the human-facing rendering of a run's decisions."""

from datetime import datetime
from enum import StrEnum
from typing import ClassVar, Self

from pydantic import Field, field_validator, model_validator

from app.artifacts.base import Artifact, ArtifactKind
from app.artifacts.decision import Verdict


class ReportFormat(StrEnum):
    """How `Report.body` is encoded."""

    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"


class Report(Artifact):
    """The final readable output of a run, assembled from its decisions.

    A report is a derived view and is safe to regenerate: everything it states must be
    traceable through `parents` back to the evidence.
    """

    kind: ClassVar[ArtifactKind] = ArtifactKind.REPORT
    id_prefix: ClassVar[str] = "rep"

    title: str
    executive_summary: str | None = Field(
        default=None,
        description="The reader-who-stops-here summary: the findings without the argument.",
    )

    body: str = Field(
        default="",
        description="The rendered report, encoded per `format`. Empty while still DRAFT.",
    )
    format: ReportFormat = ReportFormat.MARKDOWN

    period_start: datetime | None = Field(
        default=None,
        description="Start of the window covered. None = everything up to `period_end`.",
    )
    period_end: datetime | None = Field(
        default=None,
        description="End of the window covered. None = up to when the report was rendered.",
    )

    highlights: list[str] = Field(
        default_factory=list,
        description="The few lines worth reading if nothing else is.",
    )
    # Frozen at render time rather than derived on read: a report is a snapshot, and it
    # must keep saying what was true when it was published even as decisions move on.
    verdict_counts: dict[Verdict, int] = Field(
        default_factory=dict,
        description="How many covered decisions landed on each verdict, as rendered.",
    )

    @field_validator("title")
    @classmethod
    def _require_title(cls, value: str) -> str:
        """An untitled report cannot be referred to, so refuse to make one."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned

    @field_validator("executive_summary")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        """Collapse whitespace-only text to None so "missing" has one representation."""
        if value is None:
            return None
        return value.strip() or None

    @field_validator("highlights")
    @classmethod
    def _clean_entries(cls, value: list[str]) -> list[str]:
        """Drop blank entries so a list's length is a usable count."""
        return [entry for item in value if (entry := item.strip())]

    @field_validator("verdict_counts")
    @classmethod
    def _counts_are_not_negative(cls, value: dict[Verdict, int]) -> dict[Verdict, int]:
        """A negative tally means the roll-up was computed wrong."""
        negative = sorted(verdict for verdict, count in value.items() if count < 0)
        if negative:
            raise ValueError(f"counts must not be negative: {negative}")
        return value

    @model_validator(mode="after")
    def _period_runs_forwards(self) -> Self:
        """A backwards window silently selects nothing, so reject it at the boundary."""
        if (
            self.period_start is not None
            and self.period_end is not None
            and self.period_end < self.period_start
        ):
            raise ValueError("period_end must not precede period_start")
        return self


__all__ = ["Report", "ReportFormat"]
