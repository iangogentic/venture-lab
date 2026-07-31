"""G0-G7 decisions keep unknowns distinct from explicit disqualifiers."""

from app.venture.core import (
    ClaimEvidenceAssessment,
    GateContext,
    GateDecision,
    GateId,
    Scenario,
    evaluate_gates,
)


def _passing_context() -> GateContext:
    return GateContext(
        charter_customer_defined=True,
        charter_payer_defined=True,
        charter_geography_defined=True,
        founder_constraints_defined=True,
        capital_scenario_defined=True,
        outcome_horizons_defined=True,
        critical_claim_evidence=(
            ClaimEvidenceAssessment(
                claim_id="claim_1",
                independent_source_families=2,
                has_primary_administrative_or_behavioral=True,
            ),
        ),
        critical_quantitative_claims_reproduced=True,
        contradictions_resolved_or_bounded=True,
        substitution_checked=True,
        latent_competition_checked=True,
        regulatory_low_supply_checked=True,
        mandated_vs_contestable_spend_checked=True,
        demand_without_wtp_checked=True,
        stressed_economics_checked=True,
        staffing_stress_survives=True,
        acquisition_cost_stress_survives=True,
        utilization_stress_survives=True,
        working_capital_stress_survives=True,
        time_to_cash_stress_survives=True,
        illegal=False,
        unfinanceable=False,
        negative_stressed_contribution=False,
        riskiest_assumption_preregistered=True,
        behavioral_metric_preregistered=True,
        sample_preregistered=True,
        threshold_preregistered=True,
        budget_preregistered=True,
        stop_rule_preregistered=True,
        kill_rule_preregistered=True,
        primary_result=GateDecision.PASS,
        post_hoc_results_labeled_exploratory=True,
        scenario=Scenario.BOOTSTRAPPED,
        scenario_metrics_complete=True,
        pareto_ready=True,
    )


def test_all_gate_predicates_are_visible_and_a_complete_case_passes() -> None:
    evaluation = evaluate_gates(_passing_context())

    assert evaluation.decision is GateDecision.PASS
    assert tuple(result.gate for result in evaluation.results) == tuple(GateId)
    assert all(result.decision is GateDecision.PASS for result in evaluation.results)
    assert all(result.predicates for result in evaluation.results)


def test_missing_values_hold_instead_of_becoming_zero_or_kill() -> None:
    evaluation = evaluate_gates(GateContext())

    assert evaluation.decision is GateDecision.HOLD
    assert all(result.decision is GateDecision.HOLD for result in evaluation.results)
    assert evaluation.for_gate(GateId.G0).unknown_predicates
    assert evaluation.for_gate(GateId.G1).predicates[0].satisfied is None


def test_unmet_evidence_requirement_holds_but_illegality_kills() -> None:
    thin = _passing_context().model_copy(
        update={
            "critical_claim_evidence": (
                ClaimEvidenceAssessment(
                    claim_id="claim_1",
                    independent_source_families=1,
                    has_primary_administrative_or_behavioral=True,
                ),
            )
        }
    )
    thin_evaluation = evaluate_gates(GateContext.model_validate(thin.model_dump()))
    assert thin_evaluation.for_gate(GateId.G1).decision is GateDecision.HOLD
    assert thin_evaluation.decision is GateDecision.HOLD

    illegal = _passing_context().model_copy(update={"illegal": True})
    illegal_evaluation = evaluate_gates(GateContext.model_validate(illegal.model_dump()))
    assert illegal_evaluation.for_gate(GateId.G4).decision is GateDecision.KILL
    assert illegal_evaluation.decision is GateDecision.KILL


def test_preregistered_kill_result_is_an_explicit_disqualifier() -> None:
    context = _passing_context().model_copy(update={"primary_result": GateDecision.KILL})
    evaluation = evaluate_gates(GateContext.model_validate(context.model_dump()))
    result = evaluation.for_gate(GateId.G6)

    assert result.decision is GateDecision.KILL
    assert result.predicates[0].explicit_disqualifier
