"""Implementation and role-input provenance for venture pilot schemas v5-v6."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from app.llm.messages import ChatMessage
from app.venture.analysts import (
    StructuredGenerator,
    classification_input_payload,
    classification_input_payload_v2,
)
from app.venture.core import FrozenModel, canonical_json, sha256_bytes
from app.venture.discovery import (
    ClassificationComparisonReview,
    EvidencePacket,
    classification_scope_measurement,
    classification_scope_measurements,
    scope_falsification_packet,
)
from app.venture.operations import BudgetPolicy
from app.venture.pilot import (
    CandidateInputProvenance,
    CandidateInputProvenanceV2,
    LegacyPilotRunResult,
    PilotArtifactPointer,
    PilotConfiguration,
    PilotIntegrityError,
    PilotManifest,
    PilotMode,
    PilotRunProvenance,
    PilotRunProvenanceV2,
    PilotRunResult,
    _load_pilot_result,
    _verify_semantic_artifacts,
    run_pilot,
    verify_pilot_run,
)
from app.venture.provenance import (
    ImplementationManifest,
    build_implementation_bundle,
    verify_implementation_bundle,
)
from tests.venture.test_pilot import (
    NOW,
    _ClassificationCohortAnalyst,
    _draft,
    _FakeAnalyst,
    _fixture,
    _packet,
)


def test_v6_persists_exact_configuration_source_and_candidate_inputs(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pilot"
    execution = run_pilot(
        packet=_packet(),
        output_root=root,
        run_id="provenance",
        fixture=_fixture(),
        now=NOW,
    )
    result = execution.result
    assert isinstance(result, PilotRunResult)
    pointers = {pointer.kind: pointer for pointer in execution.manifest.artifacts}

    configuration_pointer = pointers["pilot_configuration"]
    configuration_content = (root / configuration_pointer.run_relative_path).read_bytes()
    configuration = PilotConfiguration.model_validate_json(configuration_content)
    assert sha256_bytes(configuration_content) == result.configuration_sha256
    assert result.configuration_sha256 == execution.manifest.configuration_sha256
    assert configuration.mode is PilotMode.OFFLINE
    assert configuration.fixture == _fixture()

    implementation_pointer = pointers["implementation_manifest"]
    implementation_content = (root / implementation_pointer.run_relative_path).read_bytes()
    implementation = ImplementationManifest.model_validate_json(implementation_content)
    source_pointer = pointers["implementation_source_tar"]
    source_tar = (root / source_pointer.run_relative_path).read_bytes()
    verify_implementation_bundle(implementation, source_tar)
    source_paths = {item.path for item in implementation.files}
    assert {
        "app/venture/pilot.py",
        "app/venture/provenance.py",
        "app/cli/commands/venture.py",
        "app/cli/main.py",
        "app/llm/openai_client.py",
        "app/llm/routing.py",
        "app/llm/provider.py",
        "app/llm/adapters/gpt.py",
        "pyproject.toml",
        "uv.lock",
    } <= source_paths

    repo_root = Path(__file__).resolve().parents[2]
    rebuilt_manifest, rebuilt_tar = build_implementation_bundle(repo_root)
    assert canonical_json(rebuilt_manifest) == implementation_content
    assert rebuilt_tar == source_tar

    provenance_pointer = pointers["run_provenance"]
    provenance_content = (root / provenance_pointer.run_relative_path).read_bytes()
    provenance = PilotRunProvenanceV2.model_validate_json(provenance_content)
    assert provenance.schema_version == "venture-pilot-run-provenance-v2"
    assert provenance.provider_usage_tracking == "pending"
    assert result.run_provenance_sha256 == provenance_pointer.sha256
    assert execution.manifest.run_provenance_sha256 == provenance_pointer.sha256
    assert provenance.implementation_manifest_sha256 == implementation_pointer.sha256
    assert provenance.implementation_source_tar_sha256 == source_pointer.sha256

    candidate_result = result.candidates[0]
    scopes = classification_scope_measurements(_packet())
    expected_classification = classification_input_payload_v2(
        candidate=candidate_result.hypothesis,
        scopes=scopes,
        assignment_id=candidate_result.classification_assignment.assignment_id,
    )
    classification_pointer = pointers["classification_input"]
    classification_content = (root / classification_pointer.run_relative_path).read_bytes()
    actual_classification = TypeAdapter(dict[str, object]).validate_json(classification_content)
    assert actual_classification == expected_classification
    assert actual_classification["schema_version"] == "classification-input-v2"
    assert len(actual_classification["official_scope_measurements"]) == len(scopes)  # type: ignore[arg-type]
    assert candidate_result.classification_assignment.visible_fields == (
        "anonymized_offer",
        "official_scope_measurements",
    )

    scoped_pointer = pointers["falsification_evidence_packet"]
    scoped_content = (root / scoped_pointer.run_relative_path).read_bytes()
    assert EvidencePacket.model_validate_json(scoped_content) == scope_falsification_packet(
        candidate_result.hypothesis,
        packet=_packet(),
    )
    assert provenance.candidate_inputs[0].classification_input_sha256 == (
        classification_pointer.sha256
    )
    assert provenance.candidate_inputs[0].falsification_evidence_packet_sha256 == (
        scoped_pointer.sha256
    )
    comparison_pointer = pointers["classification_comparison"]
    comparison_content = (root / comparison_pointer.run_relative_path).read_bytes()
    comparison = ClassificationComparisonReview.model_validate_json(comparison_content)
    assert comparison.schema_version == "classification-review-v2"
    assert comparison.compared_scope_refs == tuple(scope.measurement_id for scope in scopes)
    assert comparison.plausible_naics_codes == (candidate_result.hypothesis.naics_codes[0],)
    assert comparison.legacy_review() == candidate_result.classification_review
    assert provenance.candidate_inputs[0].classification_response_sha256 == (
        comparison_pointer.sha256
    )
    assert provenance.candidate_inputs[0].classification_failure_sha256 is None


def test_v5_semantics_reconstruct_the_legacy_single_scope_input(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pilot"
    execution = run_pilot(
        packet=_packet(),
        output_root=root,
        run_id="legacy-semantic",
        fixture=_fixture(),
        now=NOW,
    )
    manifest = execution.manifest
    contents = {
        pointer.run_relative_path: (root / pointer.run_relative_path).read_bytes()
        for pointer in manifest.artifacts
    }

    def reseal(
        current: PilotManifest,
        kind: str,
        content: bytes,
    ) -> tuple[PilotManifest, PilotArtifactPointer]:
        pointer = next(item for item in current.artifacts if item.kind == kind)
        replacement = pointer.model_copy(
            update={"sha256": sha256_bytes(content), "size_bytes": len(content)}
        )
        contents[pointer.run_relative_path] = content
        return (
            current.model_copy(
                update={
                    "artifacts": tuple(
                        replacement if item == pointer else item for item in current.artifacts
                    )
                }
            ),
            replacement,
        )

    candidate_result = execution.result.candidates[0]
    candidate = candidate_result.hypothesis
    legacy_assignment = candidate_result.classification_assignment.model_copy(
        update={"visible_fields": ("anonymized_offer", "official_scope_measurement")}
    )
    legacy_candidate_result = candidate_result.model_copy(
        update={"classification_assignment": legacy_assignment}
    )
    scope = classification_scope_measurement(candidate, packet=_packet())
    legacy_input = canonical_json(
        classification_input_payload(
            candidate=candidate,
            scope=scope,
            assignment_id=legacy_assignment.assignment_id,
        )
    )
    manifest, input_pointer = reseal(manifest, "classification_input", legacy_input)
    manifest, _ = reseal(
        manifest,
        "classification_assignment",
        canonical_json(legacy_assignment),
    )
    manifest = manifest.model_copy(
        update={
            "artifacts": tuple(
                pointer
                for pointer in manifest.artifacts
                if pointer.kind != "classification_comparison"
            )
        }
    )

    current_provenance_pointer = next(
        pointer for pointer in manifest.artifacts if pointer.kind == "run_provenance"
    )
    current_provenance = PilotRunProvenanceV2.model_validate_json(
        contents[current_provenance_pointer.run_relative_path]
    )
    scoped_pointer = next(
        pointer for pointer in manifest.artifacts if pointer.kind == "falsification_evidence_packet"
    )
    legacy_provenance = PilotRunProvenance(
        configuration_sha256=current_provenance.configuration_sha256,
        evidence_packet_sha256=current_provenance.evidence_packet_sha256,
        implementation_manifest_sha256=(current_provenance.implementation_manifest_sha256),
        implementation_source_tar_sha256=(current_provenance.implementation_source_tar_sha256),
        candidate_inputs=(
            CandidateInputProvenance(
                opportunity_id=candidate.opportunity_id,
                classification_assignment_id=legacy_assignment.assignment_id,
                classification_input_sha256=input_pointer.sha256,
                falsification_evidence_packet_sha256=scoped_pointer.sha256,
            ),
        ),
    )
    manifest, provenance_pointer = reseal(
        manifest,
        "run_provenance",
        canonical_json(legacy_provenance),
    )
    legacy_result = execution.result.model_copy(
        update={
            "schema_version": "venture-pilot-v5",
            "run_provenance_sha256": provenance_pointer.sha256,
            "candidates": (legacy_candidate_result,),
        }
    )
    manifest, _ = reseal(manifest, "result", canonical_json(legacy_result))

    from app.venture.reporting import render_pilot_report

    manifest, _ = reseal(
        manifest,
        "report",
        render_pilot_report(legacy_result).encode("utf-8"),
    )
    manifest = PilotManifest.model_validate(
        {
            **manifest.model_dump(mode="json"),
            "schema_version": "venture-pilot-manifest-v5",
            "run_provenance_sha256": provenance_pointer.sha256,
        }
    )

    legacy_payload = TypeAdapter(dict[str, object]).validate_json(legacy_input)
    assert "official_scope_measurement" in legacy_payload
    assert "official_scope_measurements" not in legacy_payload
    _verify_semantic_artifacts(manifest, contents)


class _PersistenceCheckingAnalyst(_FakeAnalyst):
    def __init__(self, packet: EvidencePacket, run_id: str, root: Path) -> None:
        super().__init__(packet, run_id)
        self.root = root

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
        artifact_root = self.root / "runs" / self.run_id / "artifacts"
        if task == "discover-opportunities":
            assert (artifact_root / "pilot-configuration.json").is_file()
            assert (artifact_root / "implementation-manifest.json").is_file()
            assert (artifact_root / "implementation-source.tar").is_file()
            assert (artifact_root / "evidence-packet.json").is_file()
        if task == "classification-review":
            suffix = f"001-{self.candidate.opportunity_id}"
            assert (artifact_root / f"classification-input-{suffix}.json").is_file()
            assert (artifact_root / f"falsification-evidence-packet-{suffix}.json").is_file()
        return super().generate_structured(
            messages,
            schema,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            task=task,
            seed=seed,
        )


def test_inputs_are_durable_before_the_corresponding_model_calls(tmp_path: Path) -> None:
    root = tmp_path / "pilot"
    packet = _packet()
    analyst: StructuredGenerator = _PersistenceCheckingAnalyst(
        packet,
        "call-order",
        root,
    )
    run_pilot(
        packet=packet,
        output_root=root,
        run_id="call-order",
        mode=PilotMode.LLM,
        llm=analyst,
        max_hypotheses=1,
        budget_policy=BudgetPolicy(max_model_calls=3, max_hypotheses=1),
        now=NOW,
    )


def test_every_candidate_gets_both_inputs_even_when_falsification_is_skipped(
    tmp_path: Path,
) -> None:
    packet = _packet()
    execution = run_pilot(
        packet=packet,
        output_root=tmp_path / "pilot",
        run_id="classification-run",
        mode=PilotMode.LLM,
        llm=_ClassificationCohortAnalyst(packet),
        max_hypotheses=2,
        budget_policy=BudgetPolicy(max_model_calls=4, max_hypotheses=2),
        now=NOW,
    )
    assert (
        sum(pointer.kind == "classification_input" for pointer in execution.manifest.artifacts) == 2
    )
    assert (
        sum(
            pointer.kind == "falsification_evidence_packet"
            for pointer in execution.manifest.artifacts
        )
        == 2
    )
    provenance_pointer = next(
        pointer for pointer in execution.manifest.artifacts if pointer.kind == "run_provenance"
    )
    provenance = PilotRunProvenanceV2.model_validate_json(
        (tmp_path / "pilot" / provenance_pointer.run_relative_path).read_bytes()
    )
    assert len(provenance.candidate_inputs) == 2
    assert execution.result.candidates[0].falsification is None


def test_v6_accepts_identical_scoped_packet_hashes_for_distinct_candidates(
    tmp_path: Path,
) -> None:
    original = _fixture().candidates[0]
    fixture = _fixture().model_copy(
        update={
            "candidates": (
                original,
                original.model_copy(
                    update={"hypothesis": _draft(title="Alternate equipment uptime service")}
                ),
            )
        }
    )
    root = tmp_path / "pilot"
    execution = run_pilot(
        packet=_packet(),
        output_root=root,
        run_id="duplicate-scoped-packets",
        fixture=fixture,
        max_hypotheses=2,
        now=NOW,
    )

    scoped = tuple(
        pointer
        for pointer in execution.manifest.artifacts
        if pointer.kind == "falsification_evidence_packet"
    )
    assert len(scoped) == 2
    assert scoped[0].sha256 == scoped[1].sha256
    assert verify_pilot_run(
        output_root=root,
        run_id="duplicate-scoped-packets",
    ).valid


def test_v6_models_require_provenance_but_legacy_results_remain_loadable(
    tmp_path: Path,
) -> None:
    execution = run_pilot(
        packet=_packet(),
        output_root=tmp_path / "pilot",
        run_id="schema",
        fixture=_fixture(),
        now=NOW,
    )
    result_payload = execution.result.model_dump(mode="json")
    result_payload["run_provenance_sha256"] = None
    with pytest.raises(ValidationError, match="requires run provenance"):
        PilotRunResult.model_validate(result_payload)

    manifest_payload = execution.manifest.model_dump(mode="json")
    manifest_payload["run_provenance_sha256"] = None
    with pytest.raises(ValidationError, match="requires run provenance"):
        PilotManifest.model_validate(manifest_payload)

    legacy_payload = execution.result.model_dump(mode="json")
    legacy_payload["schema_version"] = "venture-pilot-v3"
    legacy_payload.pop("run_provenance_sha256")
    legacy_path = tmp_path / "legacy-result.json"
    legacy_path.write_bytes(canonical_json(legacy_payload))
    legacy = _load_pilot_result(legacy_path)
    assert isinstance(legacy, LegacyPilotRunResult)
    assert legacy.schema_version == "venture-pilot-v3"


@pytest.mark.parametrize(
    ("response_hash", "failure_hash"),
    [(None, None), ("1" * 64, "2" * 64)],
)
def test_v2_candidate_provenance_requires_response_xor_failure(
    response_hash: str | None,
    failure_hash: str | None,
) -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        CandidateInputProvenanceV2(
            opportunity_id="opportunity-test",
            classification_assignment_id="assignment-test",
            classification_input_sha256="3" * 64,
            falsification_evidence_packet_sha256="4" * 64,
            classification_response_sha256=response_hash,
            classification_failure_sha256=failure_hash,
        )


def test_semantic_verifier_rejects_resealed_cross_artifact_inconsistencies(
    tmp_path: Path,
) -> None:
    root = tmp_path / "pilot"
    execution = run_pilot(
        packet=_packet(),
        output_root=root,
        run_id="semantic",
        fixture=_fixture(),
        now=NOW,
    )
    manifest = execution.manifest
    contents = {
        pointer.run_relative_path: (root / pointer.run_relative_path).read_bytes()
        for pointer in manifest.artifacts
    }
    _verify_semantic_artifacts(manifest, contents)

    bad_manifest = manifest.model_copy(update={"configuration_sha256": "0" * 64})
    with pytest.raises(PilotIntegrityError, match="configuration hash binding"):
        _verify_semantic_artifacts(bad_manifest, contents)

    report_pointer = next(pointer for pointer in manifest.artifacts if pointer.kind == "report")
    bad_report_contents = dict(contents)
    bad_report_contents[report_pointer.run_relative_path] += b"\nresealed change\n"
    with pytest.raises(PilotIntegrityError, match="exactly rerender"):
        _verify_semantic_artifacts(manifest, bad_report_contents)

    candidate_pointer = next(
        pointer for pointer in manifest.artifacts if pointer.kind == "candidate"
    )
    result = execution.result
    assert isinstance(result, PilotRunResult)
    changed_candidate: FrozenModel = result.candidates[0].hypothesis.model_copy(
        update={"title": "Resealed but inconsistent title"}
    )
    bad_candidate_contents = dict(contents)
    bad_candidate_contents[candidate_pointer.run_relative_path] = canonical_json(changed_candidate)
    with pytest.raises(PilotIntegrityError, match="child artifacts"):
        _verify_semantic_artifacts(manifest, bad_candidate_contents)

    classification_pointer = next(
        pointer for pointer in manifest.artifacts if pointer.kind == "classification_input"
    )
    classification_input = TypeAdapter(dict[str, object]).validate_json(
        contents[classification_pointer.run_relative_path]
    )
    classification_input["required_naics_code"] = "999999"
    bad_input_contents = dict(contents)
    bad_input_contents[classification_pointer.run_relative_path] = canonical_json(
        classification_input
    )
    with pytest.raises(PilotIntegrityError, match="candidate input provenance"):
        _verify_semantic_artifacts(manifest, bad_input_contents)

    comparison_pointer = next(
        pointer for pointer in manifest.artifacts if pointer.kind == "classification_comparison"
    )
    comparison = TypeAdapter(dict[str, object]).validate_json(
        contents[comparison_pointer.run_relative_path]
    )
    comparison.update(
        {
            "outcome": "unresolved",
            "plausible_naics_codes": ["541990", "811210"],
            "missing_evidence": ["Primary revenue activity and mix"],
        }
    )
    bad_comparison_contents = dict(contents)
    bad_comparison_contents[comparison_pointer.run_relative_path] = canonical_json(comparison)
    with pytest.raises(PilotIntegrityError, match="classification comparison does not bind"):
        _verify_semantic_artifacts(manifest, bad_comparison_contents)


def test_implementation_verifier_rejects_archive_tampering() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest, source_tar = build_implementation_bundle(repo_root)
    changed = bytearray(source_tar)
    changed[600] ^= 1
    with pytest.raises(ValueError, match="hash does not match"):
        verify_implementation_bundle(manifest, bytes(changed))
