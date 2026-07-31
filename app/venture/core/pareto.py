"""Scenario-specific Pareto comparison in natural units.

There is deliberately no weighted master score.  A candidate dominates another
only when it is at least as good on every axis and strictly better on one.
Candidates missing any scenario axis are reported as ineligible rather than
having unknowns silently replaced with zero.
"""

import math
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Self

from pydantic import Field, field_validator, model_validator

from app.venture.core.ids import validate_identifier
from app.venture.core.models import FrozenModel


class Scenario(StrEnum):
    """Capital and operating lenses that use different natural axes."""

    BOOTSTRAPPED = "bootstrapped"
    OPERATOR_HEAVY = "operator_heavy"
    REAL_ESTATE_CAPITAL = "real_estate_capital"
    VENTURE_SCALE = "venture_scale"


class AxisDirection(StrEnum):
    """Whether more or less of a natural-unit metric is preferred."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class ParetoAxis(FrozenModel):
    """One scenario comparison axis in an explicit natural unit."""

    key: str
    label: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    direction: AxisDirection

    @field_validator("key")
    @classmethod
    def _safe_key(cls, value: str) -> str:
        return validate_identifier(value, field="axis key")


class ScenarioDefinition(FrozenModel):
    """Named set of axes; weights are intentionally absent."""

    scenario: Scenario
    axes: tuple[ParetoAxis, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_axes(self) -> Self:
        keys = [axis.key for axis in self.axes]
        if len(set(keys)) != len(keys):
            raise ValueError("scenario axis keys must be unique")
        return self


class ParetoCandidate(FrozenModel):
    """Opportunity metrics; ``None`` is an unknown, never a zero."""

    opportunity_id: str
    metrics: dict[str, float | int | None]

    @field_validator("opportunity_id")
    @classmethod
    def _safe_id(cls, value: str) -> str:
        return validate_identifier(value, field="opportunity_id")

    @field_validator("metrics")
    @classmethod
    def _finite_metrics(cls, value: dict[str, float | int | None]) -> dict[str, float | int | None]:
        for key, measurement in value.items():
            validate_identifier(key, field="metric key")
            if isinstance(measurement, bool):
                raise ValueError(f"metric {key!r} must be numeric, not boolean")
            if isinstance(measurement, float) and not math.isfinite(measurement):
                raise ValueError(f"metric {key!r} must be finite")
        return value


class ParetoResult(FrozenModel):
    """Nondominated candidates plus candidates excluded for missing axes."""

    scenario: Scenario
    axes: tuple[ParetoAxis, ...]
    frontier: tuple[ParetoCandidate, ...]
    ineligible: tuple[ParetoCandidate, ...]

    @property
    def frontier_ids(self) -> tuple[str, ...]:
        return tuple(candidate.opportunity_id for candidate in self.frontier)

    @property
    def ineligible_ids(self) -> tuple[str, ...]:
        return tuple(candidate.opportunity_id for candidate in self.ineligible)


_SCENARIO_DEFINITIONS: Final[MappingProxyType[Scenario, ScenarioDefinition]] = MappingProxyType(
    {
        Scenario.BOOTSTRAPPED: ScenarioDefinition(
            scenario=Scenario.BOOTSTRAPPED,
            axes=(
                ParetoAxis(
                    key="capital_required_usd",
                    label="Owner capital required",
                    unit="USD",
                    direction=AxisDirection.MINIMIZE,
                ),
                ParetoAxis(
                    key="time_to_cash_flow_positive_days",
                    label="Time to positive cash flow",
                    unit="days",
                    direction=AxisDirection.MINIMIZE,
                ),
                ParetoAxis(
                    key="stressed_contribution_margin_pct",
                    label="Stressed contribution margin",
                    unit="percent",
                    direction=AxisDirection.MAXIMIZE,
                ),
                ParetoAxis(
                    key="owner_hours_per_week",
                    label="Owner operating load",
                    unit="hours/week",
                    direction=AxisDirection.MINIMIZE,
                ),
            ),
        ),
        Scenario.OPERATOR_HEAVY: ScenarioDefinition(
            scenario=Scenario.OPERATOR_HEAVY,
            axes=(
                ParetoAxis(
                    key="operator_hours_per_week",
                    label="Specialist operator load",
                    unit="hours/week",
                    direction=AxisDirection.MINIMIZE,
                ),
                ParetoAxis(
                    key="staffing_fte_per_revenue_million",
                    label="Staffing intensity",
                    unit="FTE/USD million revenue",
                    direction=AxisDirection.MINIMIZE,
                ),
                ParetoAxis(
                    key="stressed_contribution_margin_pct",
                    label="Stressed contribution margin",
                    unit="percent",
                    direction=AxisDirection.MAXIMIZE,
                ),
                ParetoAxis(
                    key="time_to_cash_flow_positive_days",
                    label="Time to positive cash flow",
                    unit="days",
                    direction=AxisDirection.MINIMIZE,
                ),
            ),
        ),
        Scenario.REAL_ESTATE_CAPITAL: ScenarioDefinition(
            scenario=Scenario.REAL_ESTATE_CAPITAL,
            axes=(
                ParetoAxis(
                    key="equity_required_usd",
                    label="Equity required",
                    unit="USD",
                    direction=AxisDirection.MINIMIZE,
                ),
                ParetoAxis(
                    key="stabilized_yield_pct",
                    label="Stabilized yield",
                    unit="percent",
                    direction=AxisDirection.MAXIMIZE,
                ),
                ParetoAxis(
                    key="debt_service_coverage_ratio",
                    label="Debt service coverage",
                    unit="ratio",
                    direction=AxisDirection.MAXIMIZE,
                ),
                ParetoAxis(
                    key="months_to_stabilization",
                    label="Time to stabilization",
                    unit="months",
                    direction=AxisDirection.MINIMIZE,
                ),
            ),
        ),
        Scenario.VENTURE_SCALE: ScenarioDefinition(
            scenario=Scenario.VENTURE_SCALE,
            axes=(
                ParetoAxis(
                    key="annual_revenue_growth_pct",
                    label="Annual revenue growth",
                    unit="percent",
                    direction=AxisDirection.MAXIMIZE,
                ),
                ParetoAxis(
                    key="gross_margin_pct",
                    label="Gross margin",
                    unit="percent",
                    direction=AxisDirection.MAXIMIZE,
                ),
                ParetoAxis(
                    key="cac_payback_months",
                    label="Customer acquisition payback",
                    unit="months",
                    direction=AxisDirection.MINIMIZE,
                ),
                ParetoAxis(
                    key="capital_efficiency_ratio",
                    label="Capital efficiency",
                    unit="revenue/invested capital",
                    direction=AxisDirection.MAXIMIZE,
                ),
            ),
        ),
    }
)


def scenario_definition(scenario: Scenario) -> ScenarioDefinition:
    """Return the immutable default definition for a scenario."""
    return _SCENARIO_DEFINITIONS[scenario]


def is_eligible(candidate: ParetoCandidate, axes: tuple[ParetoAxis, ...]) -> bool:
    """Whether every comparison axis has an observed value."""
    return all(candidate.metrics.get(axis.key) is not None for axis in axes)


def dominates(
    left: ParetoCandidate,
    right: ParetoCandidate,
    axes: tuple[ParetoAxis, ...],
) -> bool:
    """Return whether ``left`` Pareto-dominates ``right``.

    A missing value on either side makes the comparison ineligible.  It does
    not become zero, positive infinity, or a penalty score.
    """
    if not is_eligible(left, axes) or not is_eligible(right, axes):
        return False

    strictly_better = False
    for axis in axes:
        left_value = left.metrics[axis.key]
        right_value = right.metrics[axis.key]
        assert left_value is not None and right_value is not None
        if axis.direction is AxisDirection.MAXIMIZE:
            if left_value < right_value:
                return False
            strictly_better = strictly_better or left_value > right_value
        else:
            if left_value > right_value:
                return False
            strictly_better = strictly_better or left_value < right_value
    return strictly_better


def pareto_analysis(
    candidates: list[ParetoCandidate] | tuple[ParetoCandidate, ...],
    *,
    scenario: Scenario,
    definition: ScenarioDefinition | None = None,
) -> ParetoResult:
    """Compute a deterministic scenario frontier and missing-data exclusions."""
    selected = definition if definition is not None else scenario_definition(scenario)
    if selected.scenario is not scenario:
        raise ValueError("definition scenario does not match requested scenario")

    ordered = tuple(sorted(candidates, key=lambda candidate: candidate.opportunity_id))
    ids = [candidate.opportunity_id for candidate in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate opportunity_id values must be unique")

    eligible = tuple(candidate for candidate in ordered if is_eligible(candidate, selected.axes))
    ineligible = tuple(
        candidate for candidate in ordered if not is_eligible(candidate, selected.axes)
    )
    frontier = tuple(
        candidate
        for candidate in eligible
        if not any(
            other.opportunity_id != candidate.opportunity_id
            and dominates(other, candidate, selected.axes)
            for other in eligible
        )
    )
    return ParetoResult(
        scenario=scenario,
        axes=selected.axes,
        frontier=frontier,
        ineligible=ineligible,
    )


def pareto_frontier(
    candidates: list[ParetoCandidate] | tuple[ParetoCandidate, ...],
    *,
    scenario: Scenario,
    definition: ScenarioDefinition | None = None,
) -> tuple[ParetoCandidate, ...]:
    """Return only the nondominated, fully observed candidates."""
    return pareto_analysis(candidates, scenario=scenario, definition=definition).frontier


__all__ = [
    "AxisDirection",
    "ParetoAxis",
    "ParetoCandidate",
    "ParetoResult",
    "Scenario",
    "ScenarioDefinition",
    "dominates",
    "is_eligible",
    "pareto_analysis",
    "pareto_frontier",
    "scenario_definition",
]
