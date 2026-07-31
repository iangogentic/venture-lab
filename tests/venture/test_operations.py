from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.venture.evaluation import (
    BinaryForecast,
    NumericIntervalForecast,
    calibration_report,
    interval_score,
)
from app.venture.operations import (
    BudgetExceededError,
    BudgetGuard,
    BudgetPolicy,
    BudgetUsage,
    ExternalAction,
    KillSwitch,
    PacketRole,
    authorize_action,
    make_blind_packet,
)


def test_budget_fails_before_crossing_a_limit() -> None:
    guard = BudgetGuard(BudgetPolicy(max_model_calls=2, max_cost_usd=1.0))
    guard.reserve(BudgetUsage(model_calls=2, cost_usd=0.5))

    with pytest.raises(BudgetExceededError, match="model_calls"):
        guard.reserve(BudgetUsage(model_calls=1))

    assert guard.usage.model_calls == 2


def test_budget_reports_every_violation() -> None:
    guard = BudgetGuard(BudgetPolicy(max_model_calls=1, max_source_requests=2))
    violations = guard.violations(BudgetUsage(model_calls=2, source_requests=3))

    assert len(violations) == 2


def test_commercial_actions_require_a_recorded_human_approval() -> None:
    denied = authorize_action(ExternalAction.SEND_OUTREACH)
    allowed = authorize_action(
        ExternalAction.SEND_OUTREACH,
        approval_event_id="approval-123",
    )

    assert denied.allowed is False
    assert allowed.allowed is True
    assert allowed.approval_event_id == "approval-123"


def test_local_research_actions_are_allowed_without_approval() -> None:
    decision = authorize_action(ExternalAction.READ_PUBLIC_SOURCE)

    assert decision.allowed is True
    assert decision.requires_human_approval is False


def test_kill_switch_stops_the_next_operation(tmp_path: Path) -> None:
    switch = KillSwitch(tmp_path)
    switch.assert_clear()
    switch.path.touch()

    with pytest.raises(BudgetExceededError, match="kill switch"):
        switch.assert_clear()


def test_falsifier_packet_does_not_inherit_rank_or_proponent_rationale() -> None:
    packet = make_blind_packet(
        packet_id="packet-1",
        role=PacketRole.FALSIFIER,
        information_cutoff="2026-07-31T00:00:00Z",
        available={
            "anonymized_thesis": {"customer": "facility operators"},
            "claims": [{"statement": "capacity is constrained"}],
            "evidence_refs": ["evidence-1"],
            "current_rank": 1,
            "proponent_rationale": "we love this idea",
            "researcher_identity": "hidden",
        },
    )

    assert "current_rank" not in packet.payload
    assert "proponent_rationale" not in packet.payload
    assert "researcher_identity" not in packet.payload


def test_outcome_adjudicator_cannot_see_probability() -> None:
    packet = make_blind_packet(
        packet_id="packet-2",
        role=PacketRole.OUTCOME_ADJUDICATOR,
        information_cutoff="2026-07-31T00:00:00Z",
        available={
            "target_definition": "paid pilots >= 3",
            "outcome_source": "signed pilot ledger",
            "eligible_population": "all invited operators",
            "matures_at": "2026-10-31T00:00:00Z",
            "predicted_probability": 0.8,
        },
    )

    assert "predicted_probability" not in packet.payload


def _forecast(index: int, probability: float, outcome: bool) -> BinaryForecast:
    predicted = datetime(2026, 1, 1, tzinfo=UTC)
    return BinaryForecast(
        prediction_id=f"pred-{index}",
        opportunity_id=f"opp-{index}",
        target_definition="field test passes",
        probability=probability,
        information_cutoff=predicted - timedelta(days=1),
        predicted_at=predicted,
        matures_at=predicted + timedelta(days=30),
        outcome=outcome,
        adjudicated_at=predicted + timedelta(days=31),
    )


def test_calibration_scores_known_forecasts_and_warns_on_small_n() -> None:
    report = calibration_report(
        [_forecast(1, 0.9, True), _forecast(2, 0.2, False)],
        minimum_claimable_n=30,
    )

    assert report.count == 2
    assert report.brier_score == pytest.approx(0.025)
    assert report.log_loss is not None
    assert report.warning is not None
    assert sum(bucket.count for bucket in report.reliability) == 2


def test_unmatured_predictions_are_not_scored() -> None:
    predicted = datetime(2026, 1, 1, tzinfo=UTC)
    open_forecast = BinaryForecast(
        prediction_id="open",
        opportunity_id="opp",
        target_definition="first revenue",
        probability=0.5,
        information_cutoff=predicted,
        predicted_at=predicted,
        matures_at=predicted + timedelta(days=90),
    )

    report = calibration_report([open_forecast])

    assert report.count == 0
    assert report.brier_score is None


def test_interval_score_penalizes_misses_more_than_width() -> None:
    forecast = NumericIntervalForecast(
        prediction_id="interval-1",
        lower=10,
        upper=20,
        coverage=0.9,
    )

    assert interval_score(forecast, 15) == 10
    assert interval_score(forecast, 25) > 10
