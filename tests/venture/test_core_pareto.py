"""Scenario Pareto frontiers without weighted-score shortcuts."""

from app.venture.core import (
    AxisDirection,
    ParetoAxis,
    ParetoCandidate,
    Scenario,
    ScenarioDefinition,
    dominates,
    pareto_analysis,
    scenario_definition,
)


def _definition() -> ScenarioDefinition:
    return ScenarioDefinition(
        scenario=Scenario.BOOTSTRAPPED,
        axes=(
            ParetoAxis(
                key="capital_usd",
                label="Capital required",
                unit="USD",
                direction=AxisDirection.MINIMIZE,
            ),
            ParetoAxis(
                key="margin_pct",
                label="Contribution margin",
                unit="percent",
                direction=AxisDirection.MAXIMIZE,
            ),
        ),
    )


def test_pareto_dominance_requires_no_worse_and_at_least_one_better() -> None:
    axes = _definition().axes
    strong = ParetoCandidate(
        opportunity_id="strong",
        metrics={"capital_usd": 10, "margin_pct": 40},
    )
    weak = ParetoCandidate(
        opportunity_id="weak",
        metrics={"capital_usd": 20, "margin_pct": 30},
    )
    tradeoff = ParetoCandidate(
        opportunity_id="tradeoff",
        metrics={"capital_usd": 5, "margin_pct": 20},
    )

    assert dominates(strong, weak, axes)
    assert not dominates(strong, tradeoff, axes)
    assert not dominates(tradeoff, strong, axes)


def test_missing_values_are_ineligible_not_zero_filled_or_nondominated() -> None:
    definition = _definition()
    strong = ParetoCandidate(
        opportunity_id="strong",
        metrics={"capital_usd": 10, "margin_pct": 40},
    )
    weak = ParetoCandidate(
        opportunity_id="weak",
        metrics={"capital_usd": 20, "margin_pct": 30},
    )
    unknown = ParetoCandidate(
        opportunity_id="unknown",
        metrics={"capital_usd": None, "margin_pct": 100},
    )

    result = pareto_analysis(
        [unknown, weak, strong],
        scenario=Scenario.BOOTSTRAPPED,
        definition=definition,
    )

    assert result.frontier_ids == ("strong",)
    assert result.ineligible_ids == ("unknown",)
    assert not dominates(unknown, weak, definition.axes)
    assert not dominates(weak, unknown, definition.axes)


def test_scenarios_have_distinct_natural_axes_and_no_weights() -> None:
    definitions = [scenario_definition(scenario) for scenario in Scenario]
    axis_sets = [{axis.key for axis in definition.axes} for definition in definitions]

    assert len({frozenset(keys) for keys in axis_sets}) == len(Scenario)
    assert "weights" not in ScenarioDefinition.model_fields
    assert "score" not in ParetoCandidate.model_fields
