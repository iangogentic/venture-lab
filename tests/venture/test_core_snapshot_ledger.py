"""Raw snapshot immutability and append-only ledger integrity."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from app.venture.core import (
    Ledger,
    LedgerCorruptionError,
    SnapshotIntegrityError,
    SnapshotStore,
    sha256_bytes,
)

NOW = datetime(2026, 7, 31, 12, tzinfo=UTC)


def test_snapshot_store_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "raw")
    content = b'{"source":"exact raw bytes"}\n'

    first = store.put(content)
    second = store.put(content)

    assert first == second
    assert first.sha256 == sha256_bytes(content)
    assert first.content_hash == first.sha256
    assert first.path.read_bytes() == content
    assert first.relative_path == f"{first.sha256[:2]}/{first.sha256[2:]}"
    assert store.verify(first)


def test_snapshot_store_detects_corruption_and_never_overwrites_it(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "raw")
    original = b"immutable evidence"
    record = store.put(original)
    os.chmod(record.path, 0o644)
    record.path.write_bytes(b"tampered evidence")

    with pytest.raises(SnapshotIntegrityError, match="corrupt"):
        store.put(original)

    assert record.path.read_bytes() == b"tampered evidence"
    assert not store.verify(record)


def test_snapshot_hash_input_cannot_be_used_for_path_traversal(tmp_path: Path) -> None:
    store = SnapshotStore(tmp_path / "raw")

    with pytest.raises(ValueError, match="64 lowercase"):
        store.path_for("../outside")


def test_ledger_appends_hash_chain_and_correction_without_mutation(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = Ledger(path)
    created = ledger.append(
        "claim_created",
        {"claim_id": "claim_1", "value": 12},
        subject_id="claim_1",
        actor_id="assignment_research",
        recorded_at=NOW,
    )
    before_correction = path.read_bytes()

    corrected = ledger.correct(
        created.event_id,
        {"claim_id": "claim_1", "value": 13},
        reason="source issued a corrected release",
        aggregate_id="claim_1",
        actor_id="assignment_verify",
        recorded_at=NOW,
    )

    events = ledger.events()
    assert path.read_bytes().startswith(before_correction)
    assert events == (created, corrected)
    assert corrected.sequence == 2
    assert corrected.previous_hash == created.event_hash
    assert corrected.supersedes_event_id == created.event_id
    assert ledger.validate().head_hash == corrected.event_hash


def test_ledger_detects_payload_tampering_before_any_new_append(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = Ledger(path)
    ledger.append("recorded", {"value": 1}, recorded_at=NOW)
    ledger.append("recorded", {"value": 2}, recorded_at=NOW)

    lines = path.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    first["payload"]["value"] = 999
    lines[0] = json.dumps(first, separators=(",", ":"), sort_keys=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(LedgerCorruptionError, match="event_hash"):
        ledger.validate()

    with pytest.raises(LedgerCorruptionError):
        ledger.append("would_not_write", {"value": 3}, recorded_at=NOW)
    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


def test_ledger_detects_partial_tail_and_rejects_unknown_correction(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    ledger = Ledger(path)
    event = ledger.append("recorded", {"value": 1}, recorded_at=NOW)

    with pytest.raises(ValueError, match="not earlier"):
        ledger.correct(
            "event_missing",
            {"value": 2},
            reason="not a valid correction",
            recorded_at=NOW,
        )

    path.write_bytes(path.read_bytes() + b'{"partial":')
    with pytest.raises(LedgerCorruptionError, match="partial"):
        ledger.events()

    assert event.event_id


def test_ledger_rejects_unsafe_caller_supplied_event_id(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "events.jsonl")

    with pytest.raises(ValueError, match="path segments"):
        ledger.append(
            "recorded",
            {"value": 1},
            event_id="../escape",
            recorded_at=NOW,
        )
