"""Deterministic G0-G7 policy gates with visible predicates.

Unknown facts are represented by ``None`` and always produce ``HOLD``.  A
``KILL`` requires an explicit disqualifier such as illegality, an unfinanceable
capital requirement, negative stressed contribution, or a preregistered field
test kill result.
"""

from enum import StrEnum
from typing import Self

from pydantic import Field, model_validator

from app.venture.core.models import ClaimEvidenceAssessment, FrozenModel, GateDecision
from app.venture.core.pareto import Scenario


class GateId(StrEnum):
    """Ordered methodology gates."""

    G0 = "G0"
    G1 = "G1"
    G2 = "G2"
    G3 = "G3"
    G4 = "G4"
    G5 = "G5"
    G6 = "G6"
    G7 = "G7"


class GatePredicate(FrozenModel):
    """One human-readable predicate and its three-valued result."""

    name: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    satisfied: bool | None
    observed: str | None = None
    explicit_disqualifier: bool = False

    @model_validator(mode="after")
    def _disqualifier_requires_explicit_failure(self) -> Self:
        if self.explicit_disqualifier and self.satisfied is not False:
            raise ValueError("an explicit disqualifier must be an observed false predicate")
        return self


class GateResult(FrozenModel):
    """Decision at one gate with every predicate exposed."""

    gate: GateId
    decision: GateDecision
    predicates: tuple[GatePredicate, ...] = Field(min_length=1)

    @property
    def unknown_predicates(self) -> tuple[str, ...]:
        return tuple(predicate.name for predicate in self.predicates if predicate.satisfied is None)

    @property
    def failed_predicates(self) -> tuple[str, ...]:
        return tuple(
            predicate.name for predicate in self.predicates if predicate.satisfied is False
        )


class GateEvaluation(FrozenModel):
    """Complete ordered gate evaluation."""

    results: tuple[GateResult, ...] = Field(min_length=8, max_length=8)
    decision: GateDecision

    @model_validator(mode="after")
    def _exact_order(self) -> Self:
        expected = tuple(GateId)
        actual = tuple(result.gate for result in self.results)
        if actual != expected:
            raise ValueError(f"gate results must be ordered {expected!r}")
        expected_decision = (
            GateDecision.KILL
            if any(result.decision is GateDecision.KILL for result in self.results)
            else (
                GateDecision.PASS
                if all(result.decision is GateDecision.PASS for result in self.results)
                else GateDecision.HOLD
            )
        )
        if self.decision is not expected_decision:
            raise ValueError(
                f"evaluation decision must be {expected_decision.value!r} for its gate results"
            )
        return self

    def for_gate(self, gate: GateId) -> GateResult:
        return self.results[tuple(GateId).index(gate)]


class GatePolicy(FrozenModel):
    """Configurable numerical defaults for otherwise fixed predicates."""

    minimum_independent_source_families: int = Field(default=2, ge=2)


class GateContext(FrozenModel):
    """Facts consumed by G0-G7; absence remains explicitly unknown."""

    # G0 — charter
    charter_customer_defined: bool | None = None
    charter_payer_defined: bool | None = None
    charter_geography_defined: bool | None = None
    founder_constraints_defined: bool | None = None
    capital_scenario_defined: bool | None = None
    outcome_horizons_defined: bool | None = None

    # G1 — critical claim evidence
    critical_claim_evidence: tuple[ClaimEvidenceAssessment, ...] = ()

    # G2 — reproduction and contradictions
    critical_quantitative_claims_reproduced: bool | None = None
    contradictions_resolved_or_bounded: bool | None = None

    # G3 — independent falsification
    substitution_checked: bool | None = None
    latent_competition_checked: bool | None = None
    regulatory_low_supply_checked: bool | None = None
    mandated_vs_contestable_spend_checked: bool | None = None
    demand_without_wtp_checked: bool | None = None
    stressed_economics_checked: bool | None = None

    # G4 — stress tests in natural units
    staffing_stress_survives: bool | None = None
    acquisition_cost_stress_survives: bool | None = None
    utilization_stress_survives: bool | None = None
    working_capital_stress_survives: bool | None = None
    time_to_cash_stress_survives: bool | None = None
    illegal: bool | None = None
    unfinanceable: bool | None = None
    negative_stressed_contribution: bool | None = None

    # G5 — preregistration
    riskiest_assumption_preregistered: bool | None = None
    behavioral_metric_preregistered: bool | None = None
    sample_preregistered: bool | None = None
    threshold_preregistered: bool | None = None
    budget_preregistered: bool | None = None
    stop_rule_preregistered: bool | None = None
    kill_rule_preregistered: bool | None = None

    # G6 — primary result; exploratory work must stay labeled post-hoc
    primary_result: GateDecision | None = None
    post_hoc_results_labeled_exploratory: bool | None = None

    # G7 — scenario-specific comparison
    scenario: Scenario | None = None
    scenario_metrics_complete: bool | None = None
    pareto_ready: bool | None = None


def evaluate_gate(gate: GateId, predicates: tuple[GatePredicate, ...]) -> GateResult:
    """Reduce typed predicates to PASS/HOLD/KILL without numeric coercion."""
    if not predicates:
        raise ValueError("a gate must expose at least one predicate")
    if any(predicate.explicit_disqualifier for predicate in predicates):
        decision = GateDecision.KILL
    elif all(predicate.satisfied is True for predicate in predicates):
        decision = GateDecision.PASS
    else:
        # Both a known unmet prerequisite and an unknown fact remain a hold.
        decision = GateDecision.HOLD
    return GateResult(gate=gate, decision=decision, predicates=predicates)


def evaluate_gates(
    context: GateContext,
    *,
    policy: GatePolicy | None = None,
) -> GateEvaluation:
    """Evaluate the full fixed G0-G7 sequence."""
    selected = policy if policy is not None else GatePolicy()

    g0 = evaluate_gate(
        GateId.G0,
        (
            _required("customer", "charter names the customer", context.charter_customer_defined),
            _required("payer", "charter names the payer", context.charter_payer_defined),
            _required(
                "geography",
                "charter fixes the geographic scope",
                context.charter_geography_defined,
            ),
            _required(
                "founder_constraints",
                "charter states founder/operator constraints",
                context.founder_constraints_defined,
            ),
            _required(
                "capital_scenario",
                "charter selects a capital scenario",
                context.capital_scenario_defined,
            ),
            _required(
                "outcome_horizons",
                "charter defines outcome horizons",
                context.outcome_horizons_defined,
            ),
        ),
    )

    g1_predicates: list[GatePredicate] = []
    if not context.critical_claim_evidence:
        g1_predicates.append(
            _required(
                "critical_claims_present",
                "critical claims have explicit evidence assessments",
                None,
            )
        )
    for assessment in context.critical_claim_evidence:
        family_count = assessment.independent_source_families
        g1_predicates.append(
            GatePredicate(
                name=f"{assessment.claim_id}_source_families",
                requirement=(
                    "critical claim has at least "
                    f"{selected.minimum_independent_source_families} independent source families"
                ),
                satisfied=(
                    None
                    if family_count is None
                    else family_count >= selected.minimum_independent_source_families
                ),
                observed=None if family_count is None else str(family_count),
            )
        )
        g1_predicates.append(
            _required(
                f"{assessment.claim_id}_primary_record",
                "critical claim has a primary administrative or behavioral record",
                assessment.has_primary_administrative_or_behavioral,
            )
        )
    g1 = evaluate_gate(GateId.G1, tuple(g1_predicates))

    g2 = evaluate_gate(
        GateId.G2,
        (
            _required(
                "quantitative_reproduction",
                "all critical quantitative claims are reproduced",
                context.critical_quantitative_claims_reproduced,
            ),
            _required(
                "contradictions",
                "contradictions are resolved or explicitly bounded",
                context.contradictions_resolved_or_bounded,
            ),
        ),
    )

    g3 = evaluate_gate(
        GateId.G3,
        (
            _required(
                "substitution",
                "independent falsifier found no substantive substitution contradiction",
                context.substitution_checked,
            ),
            _required(
                "latent_competition",
                "independent falsifier found no substantive latent-competition contradiction",
                context.latent_competition_checked,
            ),
            _required(
                "regulatory_low_supply",
                "independent falsifier found no unresolved regulatory low-supply problem",
                context.regulatory_low_supply_checked,
            ),
            _required(
                "mandated_vs_contestable_spend",
                "mandates are separated from demonstrably contestable outsourced spend",
                context.mandated_vs_contestable_spend_checked,
            ),
            _required(
                "demand_without_wtp",
                "independent falsifier found no unresolved demand-without-WTP problem",
                context.demand_without_wtp_checked,
            ),
            _required(
                "stressed_economics",
                "independent falsifier found no unresolved stressed-economics problem",
                context.stressed_economics_checked,
            ),
        ),
    )

    g4 = evaluate_gate(
        GateId.G4,
        (
            _required(
                "staffing_stress",
                "economics survive predefined staffing stress",
                context.staffing_stress_survives,
            ),
            _required(
                "acquisition_cost_stress",
                "economics survive predefined acquisition-cost stress",
                context.acquisition_cost_stress_survives,
            ),
            _required(
                "utilization_stress",
                "economics survive predefined utilization stress",
                context.utilization_stress_survives,
            ),
            _required(
                "working_capital_stress",
                "economics survive predefined working-capital stress",
                context.working_capital_stress_survives,
            ),
            _required(
                "time_to_cash_stress",
                "economics survive predefined time-to-cash stress",
                context.time_to_cash_stress_survives,
            ),
            _not_disqualified("legality", "opportunity is legal", context.illegal),
            _not_disqualified(
                "financeability",
                "capital requirement is financeable in the selected scenario",
                context.unfinanceable,
            ),
            _not_disqualified(
                "stressed_contribution",
                "stressed contribution is not negative",
                context.negative_stressed_contribution,
            ),
        ),
    )

    g5 = evaluate_gate(
        GateId.G5,
        (
            _required(
                "riskiest_assumption",
                "riskiest assumption is preregistered",
                context.riskiest_assumption_preregistered,
            ),
            _required(
                "behavioral_metric",
                "behavioral primary metric is preregistered",
                context.behavioral_metric_preregistered,
            ),
            _required("sample", "sample is preregistered", context.sample_preregistered),
            _required("threshold", "threshold is preregistered", context.threshold_preregistered),
            _required("budget", "budget is preregistered", context.budget_preregistered),
            _required("stop_rule", "stop rule is preregistered", context.stop_rule_preregistered),
            _required("kill_rule", "kill rule is preregistered", context.kill_rule_preregistered),
        ),
    )

    primary_satisfied = (
        None if context.primary_result is None else context.primary_result is GateDecision.PASS
    )
    g6 = evaluate_gate(
        GateId.G6,
        (
            GatePredicate(
                name="preregistered_primary_result",
                requirement="preregistered primary result meets its decision rule",
                satisfied=primary_satisfied,
                observed=None if context.primary_result is None else context.primary_result.value,
                explicit_disqualifier=context.primary_result is GateDecision.KILL,
            ),
            _required(
                "post_hoc_labeling",
                "post-hoc results are labeled exploratory",
                context.post_hoc_results_labeled_exploratory,
            ),
        ),
    )

    g7 = evaluate_gate(
        GateId.G7,
        (
            _required(
                "scenario_selected",
                "a capital/operating scenario is selected",
                None if context.scenario is None else True,
                observed=None if context.scenario is None else context.scenario.value,
            ),
            _required(
                "scenario_metrics",
                "all scenario-specific natural-unit axes are measured",
                context.scenario_metrics_complete,
            ),
            _required(
                "pareto_readiness",
                "candidate is ready for scenario-specific Pareto comparison",
                context.pareto_ready,
            ),
        ),
    )

    results = (g0, g1, g2, g3, g4, g5, g6, g7)
    if any(result.decision is GateDecision.KILL for result in results):
        decision = GateDecision.KILL
    elif all(result.decision is GateDecision.PASS for result in results):
        decision = GateDecision.PASS
    else:
        decision = GateDecision.HOLD
    return GateEvaluation(results=results, decision=decision)


def _required(
    name: str,
    requirement: str,
    value: bool | None,
    *,
    observed: str | None = None,
) -> GatePredicate:
    return GatePredicate(
        name=name,
        requirement=requirement,
        satisfied=value,
        observed=observed,
    )


def _not_disqualified(
    name: str,
    requirement: str,
    disqualifier_observed: bool | None,
) -> GatePredicate:
    return GatePredicate(
        name=name,
        requirement=requirement,
        satisfied=None if disqualifier_observed is None else not disqualifier_observed,
        observed=(
            None
            if disqualifier_observed is None
            else ("disqualifier observed" if disqualifier_observed else "not observed")
        ),
        explicit_disqualifier=disqualifier_observed is True,
    )


__all__ = [
    "GateContext",
    "GateEvaluation",
    "GateId",
    "GatePolicy",
    "GatePredicate",
    "GateResult",
    "evaluate_gate",
    "evaluate_gates",
]
