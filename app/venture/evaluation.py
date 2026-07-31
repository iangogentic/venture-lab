"""Prospective forecasting and calibration metrics.

Metrics operate only on predictions whose outcomes matured after their
information cutoff.  Rejected and killed opportunities belong in the cohort too;
callers must not filter them out before passing observations here.
"""

import math
from collections import defaultdict
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BinaryForecast(BaseModel):
    """A probability recorded before a precisely defined binary outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_id: str = Field(min_length=1)
    opportunity_id: str = Field(min_length=1)
    target_definition: str = Field(min_length=1)
    probability: float = Field(ge=0.0, le=1.0)
    predicted_at: datetime
    information_cutoff: datetime
    matures_at: datetime
    outcome: bool | None = None
    adjudicated_at: datetime | None = None

    @model_validator(mode="after")
    def _temporal_order_and_outcome(self) -> "BinaryForecast":
        if self.information_cutoff > self.predicted_at:
            raise ValueError("information_cutoff cannot be after predicted_at")
        if self.matures_at <= self.predicted_at:
            raise ValueError("matures_at must be after predicted_at")
        if (self.outcome is None) != (self.adjudicated_at is None):
            raise ValueError("outcome and adjudicated_at must be set together")
        if self.adjudicated_at is not None and self.adjudicated_at < self.matures_at:
            raise ValueError("an outcome cannot be adjudicated before it matures")
        return self


class ReliabilityBin(BaseModel):
    """One probability bin, always carrying its sample size."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lower: float
    upper: float
    count: int
    mean_probability: float
    observed_frequency: float


class CalibrationReport(BaseModel):
    """Proper scoring rules plus reliability, with no small-sample overclaim."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    count: int
    brier_score: float | None
    log_loss: float | None
    reliability: tuple[ReliabilityBin, ...]
    warning: str | None = None


def calibration_report(
    forecasts: list[BinaryForecast],
    *,
    bins: int = 10,
    minimum_claimable_n: int = 30,
    probability_clip: float = 1e-6,
) -> CalibrationReport:
    """Compute Brier, log loss and a reliability table on matured forecasts."""
    if bins < 1:
        raise ValueError("bins must be positive")
    if not 0 < probability_clip < 0.5:
        raise ValueError("probability_clip must be between 0 and 0.5")

    matured = [forecast for forecast in forecasts if forecast.outcome is not None]
    if not matured:
        return CalibrationReport(
            count=0,
            brier_score=None,
            log_loss=None,
            reliability=(),
            warning="no matured predictions",
        )

    outcomes = [1.0 if forecast.outcome else 0.0 for forecast in matured]
    squared_errors = (
        (forecast.probability - outcome) ** 2
        for forecast, outcome in zip(matured, outcomes, strict=True)
    )
    brier = sum(squared_errors) / len(matured)
    losses: list[float] = []
    grouped: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for forecast, outcome in zip(matured, outcomes, strict=True):
        probability = min(max(forecast.probability, probability_clip), 1 - probability_clip)
        loss = -(outcome * math.log(probability) + (1 - outcome) * math.log(1 - probability))
        losses.append(loss)
        index = min(int(forecast.probability * bins), bins - 1)
        grouped[index].append((forecast.probability, outcome))

    reliability = tuple(
        ReliabilityBin(
            lower=index / bins,
            upper=(index + 1) / bins,
            count=len(values),
            mean_probability=sum(value[0] for value in values) / len(values),
            observed_frequency=sum(value[1] for value in values) / len(values),
        )
        for index, values in sorted(grouped.items())
    )
    warning = (
        f"n={len(matured)} is below the preregistered {minimum_claimable_n}; "
        "scores are descriptive only"
        if len(matured) < minimum_claimable_n
        else None
    )
    return CalibrationReport(
        count=len(matured),
        brier_score=brier,
        log_loss=sum(losses) / len(losses),
        reliability=reliability,
        warning=warning,
    )


class NumericIntervalForecast(BaseModel):
    """A central prediction interval recorded before a numeric outcome."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    prediction_id: str = Field(min_length=1)
    lower: float
    upper: float
    coverage: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> "NumericIntervalForecast":
        if self.lower > self.upper:
            raise ValueError("lower cannot exceed upper")
        return self


def interval_score(forecast: NumericIntervalForecast, outcome: float) -> float:
    """Gneiting-Raftery interval score; lower is better."""
    alpha = 1.0 - forecast.coverage
    width = forecast.upper - forecast.lower
    below = (2.0 / alpha) * (forecast.lower - outcome) if outcome < forecast.lower else 0.0
    above = (2.0 / alpha) * (outcome - forecast.upper) if outcome > forecast.upper else 0.0
    return width + below + above


__all__ = [
    "BinaryForecast",
    "CalibrationReport",
    "NumericIntervalForecast",
    "ReliabilityBin",
    "calibration_report",
    "interval_score",
]
