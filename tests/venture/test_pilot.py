"""Deterministic pilot orchestration, persistence, safety, and reporting."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import BaseModel

from app.llm.messages import ChatMessage
from app.utils.errors import LLMError
from app.venture.analysts import StructuredGenerator
from app.venture.core import GateDecision, GateId, Scenario, make_content_id
from app.venture.discovery import (
    BusinessArchetype,
    ClassificationReviewOutcome,
    ClassificationStatus,
    EvidencePacket,
    FalsificationDimension,
    FalsificationFinding,
    FalsificationOutcome,
    FalsificationReport,
    HypothesisBatch,
    HypothesisDraft,
    MarketTopic,
    PacketMeasurement,
    materialize_hypothesis,
)
from app.venture.operations import BudgetPolicy, ExternalAction
from app.venture.pilot import (
    ClassificationReviewFailureKind,
    FalsificationFailureKind,
    OfflineCandidateFixture,
    OfflineClassificationReview,
    OfflineFalsification,
    OfflinePilotFixture,
    PilotIntegrityError,
    PilotMode,
    report_path,
    run_pilot,
    verify_pilot_run,
)

NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)
COMPARISON_SCOPE_REFS = (
    "naics22-811210-scope",
    "naics22-541990-scope",
)


def _packet() -> EvidencePacket:
    return EvidencePacket(
        packet_id="packet-pilot",
        as_of=NOW,
        measurements=(
            PacketMeasurement(
                measurement_id="demand-1",
                metric="federal contract obligations growth",
                value=51.1,
                unit="percent",
                geography="United States",
                observed_period="2025-02 through 2026-07",
                source_family="usaspending",
                source_url="https://api.usaspending.gov/",
                caveat="Federal obligations are not private willingness to pay.",
            ),
            PacketMeasurement(
                measurement_id="supply-1",
                metric="employer establishment growth",
                value=-4.4,
                unit="percent",
                geography="United States",
                observed_period="2022 through 2023",
                source_family="census-cbp",
                source_url="https://www.census.gov/programs-surveys/cbp.html",
                caveat="Employer establishments exclude nonemployers.",
            ),
            PacketMeasurement(
                measurement_id="naics22-811210-scope",
                metric="2022 NAICS scope for electronic and precision equipment repair",
                value="Repair and maintenance of electronic and precision equipment.",
                unit="official scope text",
                geography="United States",
                observed_period="2022",
                source_family="Census NAICS",
                source_url="https://www.census.gov/naics/",
                caveat="Classification scope is not market-demand evidence.",
            ),
            PacketMeasurement(
                measurement_id="naics22-541990-scope",
                metric="2022 NAICS scope for other professional technical services",
                value="Other professional, scientific, and technical services.",
                unit="official scope text",
                geography="United States",
                observed_period="2022",
                source_family="Census NAICS",
                source_url="https://www.census.gov/naics/",
                caveat="Classification scope is not market-demand evidence.",
            ),
        ),
        allowed_geographies=("United States",),
        allowed_scenarios=(Scenario.BOOTSTRAPPED,),
        allowed_naics_codes=("811210", "541990", "621111"),
        source_policy="Official public records only; proxies stay labeled.",
    )


def _draft(
    *,
    title: str = "Regulated equipment uptime service",
    provider_code: str = "811210",
    offer_market_topic: MarketTopic = MarketTopic.EQUIPMENT_SERVICE,
    context_market_topics: tuple[MarketTopic, ...] = (),
) -> HypothesisDraft:
    scope_ref = f"naics22-{provider_code}-scope"
    return HypothesisDraft(
        title=title,
        customer="Independent outpatient clinics using regulated equipment",
        payer="Clinic operations leader",
        problem="Equipment downtime can delay billable care.",
        mechanism="Coordinate preventive maintenance and qualified repair vendors.",
        business_model="Recurring managed-service fee plus disclosed service charges.",
        geography=("United States",),
        naics_codes=(provider_code,),
        customer_naics_codes=("621111",) if provider_code == "811210" else (),
        naics_basis=f"{provider_code} is proposed for the revenue-producing service.",
        classification_status=ClassificationStatus.PROVISIONAL,
        adjacent_market_exclusions=(
            "Consumer electronics repair",
            "Original-equipment manufacturing",
        ),
        entity_scope="Employer establishments serving independent outpatient clinics",
        contestable_spend_basis=(
            "Only outsourced maintenance coordination is proposed; no claim is made "
            "that all equipment spending is contestable."
        ),
        scenario=Scenario.BOOTSTRAPPED,
        archetype=BusinessArchetype.MANAGED_SERVICE,
        offer_market_topic=offer_market_topic,
        context_market_topics=context_market_topics,
        evidence_refs=("demand-1", "supply-1", scope_ref),
        reason_for_now="One demand proxy rose while an employer-supply proxy declined.",
        critical_assumptions=(
            "Clinics outsource the coordination problem.",
            "Qualified vendors can meet required service levels.",
        ),
        disconfirming_observations=(
            "Most target clinics are locked into comprehensive OEM contracts.",
            "Observed downtime has no material operating cost.",
        ),
    )


def _findings(
    *,
    invented_ref: str | None = None,
    outcome: FalsificationOutcome = FalsificationOutcome.NO_CONTRADICTION_FOUND,
) -> tuple[FalsificationFinding, ...]:
    return tuple(
        FalsificationFinding(
            dimension=dimension,
            outcome=outcome,
            analysis=(
                f"No contradiction was found for {dimension.value}."
                if outcome is FalsificationOutcome.NO_CONTRADICTION_FOUND
                else f"The review returned {outcome.value} for {dimension.value}."
            ),
            evidence_refs=((invented_ref,) if invented_ref is not None and index == 0 else ()),
            missing_evidence=(
                ()
                if outcome is FalsificationOutcome.NO_CONTRADICTION_FOUND
                else (f"Direct evidence for {dimension.value}",)
            ),
        )
        for index, dimension in enumerate(FalsificationDimension)
    )


def _fixture(*, allegation: bool = False) -> OfflinePilotFixture:
    return OfflinePilotFixture(
        candidates=(
            OfflineCandidateFixture(
                hypothesis=_draft(),
                classification_review=OfflineClassificationReview(
                    outcome=ClassificationReviewOutcome.FIT,
                    analysis="The core repair offer fits the official 811210 scope.",
                    plausible_naics_codes=("811210",),
                ),
                falsification=OfflineFalsification(
                    findings=_findings(),
                    explicit_unfinanceable_found=allegation,
                    kill_recommendation=allegation,
                    kill_basis="Unverified capital allegation" if allegation else None,
                    critical_unknowns=("Direct willingness to pay", "Stressed unit economics"),
                ),
            ),
        )
    )


def test_offline_pilot_writes_verifiable_artifacts_without_a_master_score(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pilot"
    execution = run_pilot(
        packet=_packet(),
        output_root=root,
        run_id="nightly-001",
        fixture=_fixture(),
        now=NOW,
    )

    candidate = execution.result.candidates[0]
    assert execution.result.schema_version == "venture-pilot-v6"
    assert execution.manifest.schema_version == "venture-pilot-manifest-v6"
    assert candidate.classification_assignment.visible_fields == (
        "anonymized_offer",
        "official_scope_measurements",
    )
    assert candidate.gates.decision is GateDecision.HOLD
    assert candidate.gates.for_gate(GateId.G0).decision is GateDecision.HOLD
    assert candidate.gates.for_gate(GateId.G3).decision is GateDecision.PASS
    assert all(
        decision.allowed is False
        for decision in execution.result.action_policy
        if decision.action
        not in {
            ExternalAction.READ_PUBLIC_SOURCE,
            ExternalAction.MODEL_CALL,
            ExternalAction.WRITE_LOCAL_ARTIFACT,
        }
    )
    assert candidate.classification_review is not None
    assert candidate.classification_review.outcome is ClassificationReviewOutcome.FIT
    artifact_kinds = [pointer.kind for pointer in execution.manifest.artifacts]
    assert len(execution.manifest.artifacts) == 15
    assert artifact_kinds.count("classification_input") == 1
    assert artifact_kinds.count("classification_comparison") == 1
    assert artifact_kinds.count("falsification_evidence_packet") == 1
    assert artifact_kinds.count("pilot_configuration") == 1
    assert artifact_kinds.count("implementation_manifest") == 1
    assert artifact_kinds.count("implementation_source_tar") == 1
    assert artifact_kinds.count("run_provenance") == 1
    verification = verify_pilot_run(output_root=root, run_id="nightly-001")
    assert verification.valid
    assert verification.artifact_count == len(execution.manifest.artifacts)

    markdown = report_path(output_root=root, run_id="nightly-001").read_text()
    assert "No master score or single-winner ranking is computed." in markdown
    assert "G0:HOLD" in markdown
    assert "G3:PASS" in markdown
    assert "willingness to pay" in markdown
    assert "Critical assumptions:" in markdown
    assert "Disconfirming observations:" in markdown
    assert "Independent provider-classification gate" in markdown
    assert "Offer market topic: `equipment_service`" in markdown
    assert "Context market topics: none" in markdown


def test_same_run_and_inputs_are_idempotent_but_different_inputs_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pilot"
    first = run_pilot(
        packet=_packet(),
        output_root=root,
        run_id="same",
        fixture=_fixture(),
        now=NOW,
    )
    second = run_pilot(
        packet=_packet(),
        output_root=root,
        run_id="same",
        fixture=_fixture(),
        now=NOW,
    )
    assert second.completion_event_id == first.completion_event_id
    assert second.ledger_head_hash == first.ledger_head_hash

    with pytest.raises(PilotIntegrityError, match="different immutable inputs"):
        run_pilot(
            packet=_packet(),
            output_root=root,
            run_id="same",
            fixture=OfflinePilotFixture(
                candidates=(
                    OfflineCandidateFixture(
                        hypothesis=_draft(title="Different candidate"),
                        classification_review=OfflineClassificationReview(
                            outcome=ClassificationReviewOutcome.UNRESOLVED,
                            analysis="The frozen fixture does not resolve classification.",
                            missing_evidence=("Independent offer-to-scope review",),
                        ),
                    ),
                )
            ),
            now=NOW,
        )


def test_unverified_model_disqualifier_cannot_execute_a_kill(tmp_path: Path) -> None:
    execution = run_pilot(
        packet=_packet(),
        output_root=tmp_path / "pilot",
        run_id="allegation",
        fixture=_fixture(allegation=True),
        now=NOW,
    )
    candidate = execution.result.candidates[0]
    assert candidate.falsification is not None
    assert candidate.falsification.kill_recommendation is True
    assert candidate.gates.decision is GateDecision.HOLD
    assert candidate.unverified_disqualifier_allegations == ("unfinanceable capital requirement",)


@pytest.mark.parametrize(
    "outcome",
    [
        FalsificationOutcome.WEAKENS,
        FalsificationOutcome.CONTRADICTS,
        FalsificationOutcome.UNRESOLVED,
    ],
)
def test_g3_requires_no_contradiction_found_not_merely_completed_coverage(
    tmp_path: Path,
    outcome: FalsificationOutcome,
) -> None:
    fixture = OfflinePilotFixture(
        candidates=(
            OfflineCandidateFixture(
                hypothesis=_draft(),
                classification_review=OfflineClassificationReview(
                    outcome=ClassificationReviewOutcome.FIT,
                    analysis="The provider classification fits.",
                    plausible_naics_codes=("811210",),
                ),
                falsification=OfflineFalsification(
                    findings=_findings(outcome=outcome),
                    critical_unknowns=("A substantive challenge was not cleared.",),
                ),
            ),
        )
    )

    execution = run_pilot(
        packet=_packet(),
        output_root=tmp_path / outcome.value,
        run_id=f"g3-{outcome.value}",
        fixture=fixture,
        now=NOW,
    )

    g3 = execution.result.candidates[0].gates.for_gate(GateId.G3)
    assert g3.decision is GateDecision.HOLD
    assert all(predicate.satisfied is False for predicate in g3.predicates)


def test_ledger_events_use_injected_actual_event_clock_not_run_start(
    tmp_path: Path,
) -> None:
    tick = 0

    def clock() -> datetime:
        nonlocal tick
        tick += 1
        return datetime(2026, 7, 31, 13, 0, tick, tzinfo=UTC)

    root = tmp_path / "pilot"
    run_pilot(
        packet=_packet(),
        output_root=root,
        run_id="clocked",
        fixture=_fixture(),
        now=NOW,
        clock=clock,
    )

    events = [json.loads(line) for line in (root / "ledger.jsonl").read_text().splitlines() if line]
    recorded = [datetime.fromisoformat(event["recorded_at"]) for event in events]
    assert recorded == sorted(recorded)
    assert all(value > NOW for value in recorded)
    assert len(set(recorded)) == len(recorded)


def test_integrity_check_detects_a_changed_run_file(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    execution = run_pilot(
        packet=_packet(),
        output_root=root,
        run_id="tamper",
        fixture=_fixture(),
        now=NOW,
    )
    target = root / execution.manifest.artifacts[0].run_relative_path
    os.chmod(target, 0o644)
    target.write_bytes(b"changed")

    with pytest.raises(PilotIntegrityError, match="no longer matches"):
        verify_pilot_run(output_root=root, run_id="tamper")


def test_integrity_check_binds_manifest_to_ledger_completion(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    run_pilot(
        packet=_packet(),
        output_root=root,
        run_id="manifest-tamper",
        fixture=_fixture(),
        now=NOW,
    )
    manifest_path = root / "runs" / "manifest-tamper" / "manifest.json"
    os.chmod(manifest_path, 0o644)
    payload = json.loads(manifest_path.read_text())
    payload["artifacts"] = payload["artifacts"][:-1]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(PilotIntegrityError, match="manifest"):
        verify_pilot_run(output_root=root, run_id="manifest-tamper")


class _FakeAnalyst:
    def __init__(
        self,
        packet: EvidencePacket,
        run_id: str,
        *,
        falsification_error: LLMError | None = None,
    ) -> None:
        assignment = make_content_id(
            "assignment",
            {"run_id": run_id, "role": "researcher", "packet_id": packet.packet_id},
            digest_length=32,
        )
        self.candidate = materialize_hypothesis(
            _draft(),
            packet=packet,
            assignment_id=assignment,
            created_at=NOW,
        )
        self.calls: list[str] = []
        self.falsification_error = falsification_error
        self.run_id = run_id

    def generate_structured[T: BaseModel](
        self,
        messages: str | Sequence[ChatMessage],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        task: str | None = None,
        seed: int | None = None,
    ) -> T:
        del messages, model, temperature, max_tokens, seed
        self.calls.append(task or "")
        if task == "discover-opportunities":
            return cast(T, HypothesisBatch(hypotheses=(_draft(),)))
        if task == "classification-review":
            assignment = make_content_id(
                "assignment",
                {
                    "run_id": self.run_id,
                    "role": "classification-reviewer",
                    "candidate": self.candidate.opportunity_id,
                    "ordinal": 1,
                },
                digest_length=32,
            )
            return schema.model_validate(
                {
                    "schema_version": "classification-review-v2",
                    "opportunity_id": self.candidate.opportunity_id,
                    "assignment_id": assignment,
                    "naics_code": "811210",
                    "scope_measurement_ref": "naics22-811210-scope",
                    "compared_scope_refs": COMPARISON_SCOPE_REFS,
                    "plausible_naics_codes": ("811210",),
                    "outcome": ClassificationReviewOutcome.FIT,
                    "analysis": "The repair offer fits the official provider scope.",
                }
            )
        if self.falsification_error is not None:
            raise self.falsification_error
        assignment = make_content_id(
            "assignment",
            {
                "run_id": self.run_id,
                "role": "falsifier",
                "candidate": self.candidate.opportunity_id,
                "ordinal": 1,
            },
            digest_length=32,
        )
        return cast(
            T,
            FalsificationReport(
                opportunity_id=self.candidate.opportunity_id,
                assignment_id=assignment,
                findings=_findings(),
                explicit_illegality_found=False,
                explicit_unfinanceable_found=False,
                explicit_negative_stressed_contribution_found=False,
                kill_recommendation=False,
                critical_unknowns=("Direct willingness to pay",),
            ),
        )


def test_llm_mode_uses_existing_generator_and_falsifier_seams(tmp_path: Path) -> None:
    packet = _packet()
    fake: StructuredGenerator = _FakeAnalyst(packet, "model-run")
    execution = run_pilot(
        packet=packet,
        output_root=tmp_path / "pilot",
        run_id="model-run",
        mode=PilotMode.LLM,
        llm=fake,
        max_hypotheses=1,
        budget_policy=BudgetPolicy(max_model_calls=3, max_hypotheses=1),
        now=NOW,
    )

    assert cast(_FakeAnalyst, fake).calls == [
        "discover-opportunities",
        "classification-review",
        "contradiction-analysis",
    ]
    assert execution.result.budget_usage.model_calls == 3
    assert execution.result.budget_usage.hypotheses == 1
    assert execution.result.candidates[0].gates.for_gate(GateId.G3).decision is GateDecision.PASS


class _MixedReferenceAnalyst:
    def __init__(self, packet: EvidencePacket, run_id: str) -> None:
        assignment = make_content_id(
            "assignment",
            {"run_id": run_id, "role": "researcher", "packet_id": packet.packet_id},
            digest_length=32,
        )
        self.drafts = (
            _draft(title="Candidate with invented falsifier reference"),
            _draft(title="Candidate with valid falsification").model_copy(
                update={
                    "customer": "Regional clinical engineering departments",
                    "payer": "Clinical engineering director",
                    "problem": "Unscheduled device failures interrupt clinical workflows.",
                    "mechanism": (
                        "Dispatch qualified repair technicians under a response-time agreement."
                    ),
                    "business_model": "Monthly availability fee plus per-repair charges.",
                    "critical_assumptions": (
                        "Clinical engineering teams will pay to shorten repair response time.",
                    ),
                    "disconfirming_observations": (
                        "OEM contracts already meet the promised response time.",
                    ),
                }
            ),
        )
        self.candidates = tuple(
            materialize_hypothesis(
                draft,
                packet=packet,
                assignment_id=assignment,
                created_at=NOW,
            )
            for draft in self.drafts
        )
        self.classification_calls = 0
        self.falsifier_calls = 0

    def generate_structured[T: BaseModel](
        self,
        messages: str | Sequence[ChatMessage],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        task: str | None = None,
        seed: int | None = None,
    ) -> T:
        del messages, model, temperature, max_tokens, seed
        if task == "discover-opportunities":
            return cast(T, HypothesisBatch(hypotheses=self.drafts))
        if task == "classification-review":
            index = self.classification_calls
            self.classification_calls += 1
            candidate = self.candidates[index]
            assignment = make_content_id(
                "assignment",
                {
                    "run_id": "mixed-run",
                    "role": "classification-reviewer",
                    "candidate": candidate.opportunity_id,
                    "ordinal": index + 1,
                },
                digest_length=32,
            )
            return schema.model_validate(
                {
                    "schema_version": "classification-review-v2",
                    "opportunity_id": candidate.opportunity_id,
                    "assignment_id": assignment,
                    "naics_code": candidate.naics_codes[0],
                    "scope_measurement_ref": f"naics22-{candidate.naics_codes[0]}-scope",
                    "compared_scope_refs": COMPARISON_SCOPE_REFS,
                    "plausible_naics_codes": (candidate.naics_codes[0],),
                    "outcome": ClassificationReviewOutcome.FIT,
                    "analysis": "The frozen independent review found provider-scope fit.",
                }
            )
        index = self.falsifier_calls
        self.falsifier_calls += 1
        candidate = self.candidates[index]
        assignment = make_content_id(
            "assignment",
            {
                "run_id": "mixed-run",
                "role": "falsifier",
                "candidate": candidate.opportunity_id,
                "ordinal": index + 1,
            },
            digest_length=32,
        )
        return cast(
            T,
            FalsificationReport(
                opportunity_id=candidate.opportunity_id,
                assignment_id=assignment,
                findings=_findings(invented_ref="invented-measurement" if index == 0 else None),
                explicit_illegality_found=False,
                explicit_unfinanceable_found=False,
                explicit_negative_stressed_contribution_found=False,
                kill_recommendation=False,
                critical_unknowns=("Direct willingness to pay",),
            ),
        )


def test_invented_falsifier_ref_is_quarantined_and_cohort_continues(
    tmp_path: Path,
) -> None:
    packet = _packet()
    fake = _MixedReferenceAnalyst(packet, "mixed-run")
    execution = run_pilot(
        packet=packet,
        output_root=tmp_path / "pilot",
        run_id="mixed-run",
        mode=PilotMode.LLM,
        llm=fake,
        max_hypotheses=2,
        budget_policy=BudgetPolicy(max_model_calls=5, max_hypotheses=2),
        now=NOW,
    )

    failed, succeeded = execution.result.candidates
    assert failed.falsification is None
    assert failed.falsification_failure is not None
    assert failed.falsification_failure.kind is FalsificationFailureKind.EVIDENCE_REFERENCE
    assert failed.gates.for_gate(GateId.G3).decision is GateDecision.HOLD
    assert all(
        predicate.satisfied is None for predicate in failed.gates.for_gate(GateId.G3).predicates
    )
    assert succeeded.falsification is not None
    assert succeeded.falsification_failure is None
    assert succeeded.gates.for_gate(GateId.G3).decision is GateDecision.PASS
    classification_inputs = tuple(
        pointer
        for pointer in execution.manifest.artifacts
        if pointer.kind == "classification_input"
    )
    scoped_packets = tuple(
        pointer
        for pointer in execution.manifest.artifacts
        if pointer.kind == "falsification_evidence_packet"
    )
    assert len(classification_inputs) == 2
    assert len(scoped_packets) == 2
    assert scoped_packets[0].sha256 == scoped_packets[1].sha256
    assert verify_pilot_run(output_root=tmp_path / "pilot", run_id="mixed-run").valid
    markdown = report_path(output_root=tmp_path / "pilot", run_id="mixed-run").read_text()
    assert r"Quarantined failure (evidence\_reference)" in markdown
    assert "cohort continued" in markdown


class _ClassificationCohortAnalyst:
    def __init__(self, packet: EvidencePacket, *, invented_binding: bool = False) -> None:
        assignment = make_content_id(
            "assignment",
            {
                "run_id": "classification-run",
                "role": "researcher",
                "packet_id": packet.packet_id,
            },
            digest_length=32,
        )
        self.drafts = (
            _draft(title="Fire training and delivery logistics"),
            _draft(
                title="Waterless suppression inspection",
                provider_code="541990",
                offer_market_topic=MarketTopic.FIRE_LIFE_SAFETY,
            ),
        )
        self.candidates = tuple(
            materialize_hypothesis(
                draft,
                packet=packet,
                assignment_id=assignment,
                created_at=NOW,
            )
            for draft in self.drafts
        )
        self.invented_binding = invented_binding
        self.classification_calls = 0
        self.falsifier_calls = 0
        self.calls: list[str] = []

    def generate_structured[T: BaseModel](
        self,
        messages: str | Sequence[ChatMessage],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        task: str | None = None,
        seed: int | None = None,
    ) -> T:
        del messages, model, temperature, max_tokens, seed
        self.calls.append(task or "")
        if task == "discover-opportunities":
            return cast(T, HypothesisBatch(hypotheses=self.drafts))
        if task == "classification-review":
            index = self.classification_calls
            self.classification_calls += 1
            candidate = self.candidates[index]
            assignment = make_content_id(
                "assignment",
                {
                    "run_id": "classification-run",
                    "role": "classification-reviewer",
                    "candidate": candidate.opportunity_id,
                    "ordinal": index + 1,
                },
                digest_length=32,
            )
            if index == 0 and self.invented_binding:
                return schema.model_validate(
                    {
                        "schema_version": "classification-review-v2",
                        "opportunity_id": "invented-opportunity",
                        "assignment_id": assignment,
                        "naics_code": candidate.naics_codes[0],
                        "scope_measurement_ref": "invented-scope-ref",
                        "compared_scope_refs": COMPARISON_SCOPE_REFS,
                        "plausible_naics_codes": (),
                        "outcome": ClassificationReviewOutcome.UNRESOLVED,
                        "analysis": "The returned binding is invalid.",
                        "missing_evidence": ("A valid official scope reference",),
                    }
                )
            if index == 0:
                return schema.model_validate(
                    {
                        "schema_version": "classification-review-v2",
                        "opportunity_id": candidate.opportunity_id,
                        "assignment_id": assignment,
                        "naics_code": candidate.naics_codes[0],
                        "scope_measurement_ref": "naics22-811210-scope",
                        "compared_scope_refs": COMPARISON_SCOPE_REFS,
                        "plausible_naics_codes": ("541990",),
                        "outcome": ClassificationReviewOutcome.CONTRADICTS,
                        "analysis": (
                            "Training and delivery logistics do not make the "
                            "revenue-producing offer an equipment-repair provider."
                        ),
                        "mismatches": ("No electronic or precision equipment repair is sold.",),
                    }
                )
            return schema.model_validate(
                {
                    "schema_version": "classification-review-v2",
                    "opportunity_id": candidate.opportunity_id,
                    "assignment_id": assignment,
                    "naics_code": candidate.naics_codes[0],
                    "scope_measurement_ref": "naics22-541990-scope",
                    "compared_scope_refs": COMPARISON_SCOPE_REFS,
                    "plausible_naics_codes": ("541990",),
                    "outcome": ClassificationReviewOutcome.FIT,
                    "analysis": "The inspection offer fits the selected provider scope.",
                }
            )

        self.falsifier_calls += 1
        candidate = self.candidates[1]
        assignment = make_content_id(
            "assignment",
            {
                "run_id": "classification-run",
                "role": "falsifier",
                "candidate": candidate.opportunity_id,
                "ordinal": 2,
            },
            digest_length=32,
        )
        return cast(
            T,
            FalsificationReport(
                opportunity_id=candidate.opportunity_id,
                assignment_id=assignment,
                findings=_findings(),
                explicit_illegality_found=False,
                explicit_unfinanceable_found=False,
                explicit_negative_stressed_contribution_found=False,
                kill_recommendation=False,
                critical_unknowns=("Direct willingness to pay",),
            ),
        )


def test_classification_contradiction_skips_falsification_and_cohort_continues(
    tmp_path: Path,
) -> None:
    fake = _ClassificationCohortAnalyst(_packet())
    execution = run_pilot(
        packet=_packet(),
        output_root=tmp_path / "pilot",
        run_id="classification-run",
        mode=PilotMode.LLM,
        llm=fake,
        max_hypotheses=2,
        budget_policy=BudgetPolicy(max_model_calls=4, max_hypotheses=2),
        now=NOW,
    )

    contradicted, fitted = execution.result.candidates
    assert contradicted.classification_review is not None
    assert contradicted.classification_review.outcome is ClassificationReviewOutcome.CONTRADICTS
    assert contradicted.falsification is None
    assert all(
        predicate.satisfied is None
        for predicate in contradicted.gates.for_gate(GateId.G3).predicates
    )
    assert fitted.classification_review is not None
    assert fitted.classification_review.outcome is ClassificationReviewOutcome.FIT
    assert fitted.falsification is not None
    assert fake.falsifier_calls == 1
    markdown = report_path(output_root=tmp_path / "pilot", run_id="classification-run").read_text()
    assert "Gate outcome: **CONTRADICTS**" in markdown
    assert "market falsification was skipped" in markdown


def test_invented_classification_ids_and_refs_fail_closed_and_cohort_continues(
    tmp_path: Path,
) -> None:
    fake = _ClassificationCohortAnalyst(_packet(), invented_binding=True)
    execution = run_pilot(
        packet=_packet(),
        output_root=tmp_path / "pilot",
        run_id="classification-run",
        mode=PilotMode.LLM,
        llm=fake,
        max_hypotheses=2,
        budget_policy=BudgetPolicy(max_model_calls=4, max_hypotheses=2),
        now=NOW,
    )

    failed, fitted = execution.result.candidates
    assert failed.classification_review is None
    assert failed.classification_review_failure is not None
    assert (
        failed.classification_review_failure.kind
        is ClassificationReviewFailureKind.EVIDENCE_REFERENCE
    )
    assert failed.falsification is None
    assert fitted.classification_review is not None
    assert fitted.falsification is not None
    assert any(
        pointer.kind == "classification_review_failure" for pointer in execution.manifest.artifacts
    )
    assert fake.falsifier_calls == 1


def test_llm_error_is_quarantined_as_transport_without_retry(tmp_path: Path) -> None:
    packet = _packet()
    fake = _FakeAnalyst(
        packet,
        "transport-run",
        falsification_error=LLMError("provider unavailable"),
    )
    execution = run_pilot(
        packet=packet,
        output_root=tmp_path / "pilot",
        run_id="transport-run",
        mode=PilotMode.LLM,
        llm=fake,
        max_hypotheses=1,
        budget_policy=BudgetPolicy(max_model_calls=3, max_hypotheses=1),
        now=NOW,
    )

    candidate = execution.result.candidates[0]
    assert fake.calls == [
        "discover-opportunities",
        "classification-review",
        "contradiction-analysis",
    ]
    assert candidate.falsification is None
    assert candidate.falsification_failure is not None
    assert candidate.falsification_failure.kind is FalsificationFailureKind.TRANSPORT
    assert candidate.gates.for_gate(GateId.G3).decision is GateDecision.HOLD
    assert execution.result.budget_usage.model_calls == 3


def test_kill_switch_stops_before_any_artifact_or_model_call(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    root.mkdir()
    (root / "STOP").write_text("stop")
    with pytest.raises(Exception, match="kill switch engaged"):
        run_pilot(
            packet=_packet(),
            output_root=root,
            run_id="stopped",
            fixture=_fixture(),
            now=NOW,
        )
    assert not (root / "ledger.jsonl").exists()


def test_result_json_has_no_composite_score_field(tmp_path: Path) -> None:
    execution = run_pilot(
        packet=_packet(),
        output_root=tmp_path / "pilot",
        run_id="json-shape",
        fixture=_fixture(),
        now=NOW,
    )
    payload: dict[str, Any] = execution.result.model_dump(mode="json")
    encoded = json.dumps(payload)
    assert "master_score" not in encoded
    assert '"score"' not in encoded
