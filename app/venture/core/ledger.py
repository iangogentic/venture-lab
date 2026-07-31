"""Append-only, hash-chained JSONL event ledger.

Every append validates the existing chain while holding an exclusive advisory
lock, writes one canonical JSON line, flushes, and fsyncs.  There is no update
or delete API: corrections are new events that name the event they supersede.
"""

import json
import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Self

from pydantic import Field, field_validator, model_validator

from app.venture.core.ids import (
    canonical_json,
    make_content_id,
    sha256_bytes,
    validate_identifier,
    validate_sha256,
)
from app.venture.core.models import FrozenModel

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows portability
    fcntl = None  # type: ignore[assignment]

GENESIS_HASH: Final[str] = "0" * 64
_EVENT_TYPE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)*$")


class LedgerError(RuntimeError):
    """Base ledger failure."""


class LedgerCorruptionError(LedgerError):
    """The JSONL file is malformed or its hash chain does not validate."""


class LedgerEvent(FrozenModel):
    """One immutable event in the append-only chain."""

    sequence: int = Field(ge=1)
    event_id: str
    event_type: str
    aggregate_id: str | None = None
    recorded_at: datetime
    actor_id: str | None = None
    payload: dict[str, Any]
    supersedes_event_id: str | None = None
    previous_hash: str
    event_hash: str

    @field_validator("event_id", "aggregate_id", "actor_id", "supersedes_event_id")
    @classmethod
    def _safe_ids(cls, value: str | None) -> str | None:
        return None if value is None else validate_identifier(value)

    @field_validator("event_type")
    @classmethod
    def _valid_event_type(cls, value: str) -> str:
        return _validate_event_type(value)

    @field_validator("previous_hash", "event_hash")
    @classmethod
    def _valid_hashes(cls, value: str) -> str:
        return validate_sha256(value)

    @field_validator("recorded_at")
    @classmethod
    def _aware_recorded_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recorded_at must include a timezone")
        return value

    @field_validator("payload")
    @classmethod
    def _json_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        try:
            canonical_json(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"payload must be canonical JSON data: {exc}") from exc
        return value

    @model_validator(mode="after")
    def _correction_is_not_self_referential(self) -> Self:
        if self.supersedes_event_id == self.event_id:
            raise ValueError("an event cannot supersede itself")
        return self

    @property
    def subject_id(self) -> str | None:
        """Alias for callers that use subject rather than aggregate terminology."""
        return self.aggregate_id

    @property
    def prev_hash(self) -> str:
        """Compact alias used by some hash-chain readers."""
        return self.previous_hash

    @property
    def hash(self) -> str:
        """Content hash alias."""
        return self.event_hash


class LedgerValidation(FrozenModel):
    """Successful validation summary."""

    event_count: int = Field(ge=0)
    head_hash: str

    @field_validator("head_hash")
    @classmethod
    def _valid_hash(cls, value: str) -> str:
        return validate_sha256(value)

    @property
    def valid(self) -> bool:
        """A summary exists only after successful validation."""
        return True


class Ledger:
    """Filesystem-backed append-only event ledger."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    def append(
        self,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        aggregate_id: str | None = None,
        subject_id: str | None = None,
        actor_id: str | None = None,
        recorded_at: datetime | None = None,
        event_id: str | None = None,
        supersedes_event_id: str | None = None,
    ) -> LedgerEvent:
        """Validate the chain and durably append exactly one event."""
        safe_type = _validate_event_type(event_type)
        if aggregate_id is not None and subject_id is not None and aggregate_id != subject_id:
            raise ValueError("aggregate_id and subject_id disagree")
        resolved_aggregate = aggregate_id if aggregate_id is not None else subject_id
        if resolved_aggregate is not None:
            resolved_aggregate = validate_identifier(resolved_aggregate, field="aggregate_id")
        if actor_id is not None:
            actor_id = validate_identifier(actor_id, field="actor_id")
        if supersedes_event_id is not None:
            supersedes_event_id = validate_identifier(
                supersedes_event_id, field="supersedes_event_id"
            )

        event_payload = json.loads(canonical_json(dict(payload)))
        if not isinstance(event_payload, dict):  # defensive; Mapping guarantees this shape
            raise TypeError("event payload must be a JSON object")
        timestamp = recorded_at if recorded_at is not None else datetime.now(UTC)
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("recorded_at must include a timezone")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.path.exists()
        try:
            with self.path.open("a+b") as handle, _file_lock(handle, exclusive=True):
                handle.seek(0)
                events = _parse_and_validate(handle.read(), source=self.path)
                known_ids = {event.event_id for event in events}
                if supersedes_event_id is not None and supersedes_event_id not in known_ids:
                    raise ValueError(
                        f"superseded event {supersedes_event_id!r} is not earlier in this ledger"
                    )

                sequence = len(events) + 1
                previous_hash = events[-1].event_hash if events else GENESIS_HASH
                identity_body = _event_body(
                    sequence=sequence,
                    event_type=safe_type,
                    aggregate_id=resolved_aggregate,
                    recorded_at=timestamp,
                    actor_id=actor_id,
                    payload=event_payload,
                    supersedes_event_id=supersedes_event_id,
                    previous_hash=previous_hash,
                )
                resolved_event_id = (
                    validate_identifier(event_id, field="event_id")
                    if event_id is not None
                    else make_content_id("event", identity_body)
                )
                if resolved_event_id in known_ids:
                    raise ValueError(f"duplicate event_id {resolved_event_id!r}")
                event_hash = _hash_event(resolved_event_id, identity_body)
                event = LedgerEvent(
                    sequence=sequence,
                    event_id=resolved_event_id,
                    event_type=safe_type,
                    aggregate_id=resolved_aggregate,
                    recorded_at=timestamp,
                    actor_id=actor_id,
                    payload=event_payload,
                    supersedes_event_id=supersedes_event_id,
                    previous_hash=previous_hash,
                    event_hash=event_hash,
                )
                line = canonical_json(event.model_dump(mode="json")) + b"\n"
                handle.seek(0, os.SEEK_END)
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise LedgerError(f"could not append to ledger {self.path}: {exc}") from exc

        if not existed:
            _fsync_directory(self.path.parent)
        return event

    def correct(
        self,
        supersedes_event_id: str,
        payload: Mapping[str, Any],
        *,
        reason: str,
        aggregate_id: str | None = None,
        actor_id: str | None = None,
        recorded_at: datetime | None = None,
    ) -> LedgerEvent:
        """Append a correction; the earlier bytes remain untouched."""
        cleaned_reason = reason.strip()
        if not cleaned_reason:
            raise ValueError("correction reason must not be blank")
        correction_payload: dict[str, Any] = {
            "reason": cleaned_reason,
            "replacement": dict(payload),
        }
        return self.append(
            "correction",
            correction_payload,
            aggregate_id=aggregate_id,
            actor_id=actor_id,
            recorded_at=recorded_at,
            supersedes_event_id=supersedes_event_id,
        )

    def events(self) -> tuple[LedgerEvent, ...]:
        """Read and validate the complete chain."""
        if not self.path.exists():
            return ()
        try:
            with self.path.open("rb") as handle, _file_lock(handle, exclusive=False):
                return _parse_and_validate(handle.read(), source=self.path)
        except OSError as exc:
            raise LedgerError(f"could not read ledger {self.path}: {exc}") from exc

    def validate(self) -> LedgerValidation:
        """Validate structure, sequence, links, hashes, and corrections."""
        events = self.events()
        return LedgerValidation(
            event_count=len(events),
            head_hash=events[-1].event_hash if events else GENESIS_HASH,
        )

    def verify(self) -> LedgerValidation:
        """Alias for ``validate`` used by integrity-checking callers."""
        return self.validate()

    def event(self, event_id: str) -> LedgerEvent | None:
        """Find one event without bypassing chain validation."""
        safe_id = validate_identifier(event_id, field="event_id")
        return next((event for event in self.events() if event.event_id == safe_id), None)

    def __iter__(self) -> Iterator[LedgerEvent]:
        return iter(self.events())


def _event_body(
    *,
    sequence: int,
    event_type: str,
    aggregate_id: str | None,
    recorded_at: datetime,
    actor_id: str | None,
    payload: dict[str, Any],
    supersedes_event_id: str | None,
    previous_hash: str,
) -> dict[str, Any]:
    return {
        "sequence": sequence,
        "event_type": event_type,
        "aggregate_id": aggregate_id,
        "recorded_at": recorded_at,
        "actor_id": actor_id,
        "payload": payload,
        "supersedes_event_id": supersedes_event_id,
        "previous_hash": previous_hash,
    }


def _hash_event(event_id: str, body: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json({"event_id": event_id, **body}))


def _validate_event_type(value: str) -> str:
    if len(value) > 128 or _EVENT_TYPE.fullmatch(value) is None:
        raise ValueError("event_type must be a dotted, dashed, or underscored ASCII event name")
    return value


def _parse_and_validate(raw: bytes, *, source: Path) -> tuple[LedgerEvent, ...]:
    if not raw:
        return ()
    if not raw.endswith(b"\n"):
        raise LedgerCorruptionError(f"{source} ends with a partial event")

    events: list[LedgerEvent] = []
    known_ids: set[str] = set()
    expected_previous = GENESIS_HASH
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise LedgerCorruptionError(f"{source}:{index} is an empty event")
        try:
            decoded = json.loads(line)
            event = LedgerEvent.model_validate(decoded)
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            raise LedgerCorruptionError(f"{source}:{index} is not a valid event: {exc}") from exc

        if event.sequence != index:
            raise LedgerCorruptionError(
                f"{source}:{index} has sequence {event.sequence}, expected {index}"
            )
        if event.event_id in known_ids:
            raise LedgerCorruptionError(f"{source}:{index} repeats event_id {event.event_id!r}")
        if event.previous_hash != expected_previous:
            raise LedgerCorruptionError(
                f"{source}:{index} previous_hash does not match the preceding event"
            )
        if event.supersedes_event_id is not None and event.supersedes_event_id not in known_ids:
            raise LedgerCorruptionError(
                f"{source}:{index} supersedes a missing or future event "
                f"{event.supersedes_event_id!r}"
            )

        body = _event_body(
            sequence=event.sequence,
            event_type=event.event_type,
            aggregate_id=event.aggregate_id,
            recorded_at=event.recorded_at,
            actor_id=event.actor_id,
            payload=event.payload,
            supersedes_event_id=event.supersedes_event_id,
            previous_hash=event.previous_hash,
        )
        expected_hash = _hash_event(event.event_id, body)
        if event.event_hash != expected_hash:
            raise LedgerCorruptionError(f"{source}:{index} event_hash does not match its content")

        known_ids.add(event.event_id)
        expected_previous = event.event_hash
        events.append(event)
    return tuple(events)


@contextmanager
def _file_lock(handle: Any, *, exclusive: bool) -> Iterator[None]:
    if fcntl is None:
        yield
        return
    operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(handle.fileno(), operation)
    try:
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


__all__ = [
    "GENESIS_HASH",
    "Ledger",
    "LedgerCorruptionError",
    "LedgerError",
    "LedgerEvent",
    "LedgerValidation",
]
