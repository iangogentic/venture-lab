"""Run budgets and role-separated blind packets.

The model is an untrusted analyst, not the workflow controller.  These objects
keep spend limits and information barriers deterministic and testable before a
prompt is constructed or a paid request is made.
"""

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class BudgetExceededError(RuntimeError):
    """A proposed operation would cross an explicit run budget."""


class BudgetPolicy(BaseModel):
    """Hard ceilings for one unattended research run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_model_calls: int = Field(default=80, ge=0)
    max_input_tokens: int = Field(default=1_500_000, ge=0)
    max_output_tokens: int = Field(default=250_000, ge=0)
    max_cost_usd: float = Field(default=25.0, ge=0.0)
    max_source_requests: int = Field(default=1_000, ge=0)
    max_source_bytes: int = Field(default=2_000_000_000, ge=0)
    max_hypotheses: int = Field(default=100, ge=0)
    max_retries: int = Field(default=30, ge=0)
    max_wall_seconds: int = Field(default=43_200, ge=1)


class BudgetUsage(BaseModel):
    """Observed or proposed resource usage in the same units as the policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_calls: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    source_requests: int = Field(default=0, ge=0)
    source_bytes: int = Field(default=0, ge=0)
    hypotheses: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    wall_seconds: int = Field(default=0, ge=0)

    def __add__(self, other: "BudgetUsage") -> "BudgetUsage":
        return BudgetUsage(
            model_calls=self.model_calls + other.model_calls,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            source_requests=self.source_requests + other.source_requests,
            source_bytes=self.source_bytes + other.source_bytes,
            hypotheses=self.hypotheses + other.hypotheses,
            retries=self.retries + other.retries,
            wall_seconds=self.wall_seconds + other.wall_seconds,
        )


_BUDGET_FIELDS: tuple[tuple[str, str], ...] = (
    ("model_calls", "max_model_calls"),
    ("input_tokens", "max_input_tokens"),
    ("output_tokens", "max_output_tokens"),
    ("cost_usd", "max_cost_usd"),
    ("source_requests", "max_source_requests"),
    ("source_bytes", "max_source_bytes"),
    ("hypotheses", "max_hypotheses"),
    ("retries", "max_retries"),
    ("wall_seconds", "max_wall_seconds"),
)


class BudgetGuard:
    """In-memory reservation guard backed by a caller-persisted usage snapshot.

    The caller must append the returned usage to the run ledger before starting
    work.  Keeping persistence outside this class avoids inventing a second
    source of truth next to the hash-chained event ledger.
    """

    def __init__(self, policy: BudgetPolicy, usage: BudgetUsage | None = None) -> None:
        self.policy = policy
        self.usage = usage or BudgetUsage()
        self._assert_within(self.usage)

    def violations(self, proposed: BudgetUsage | None = None) -> tuple[str, ...]:
        """Return every ceiling the current plus proposed usage would cross."""
        total = self.usage + (proposed or BudgetUsage())
        violations: list[str] = []
        for used_field, limit_field in _BUDGET_FIELDS:
            used = getattr(total, used_field)
            limit = getattr(self.policy, limit_field)
            if used > limit:
                violations.append(f"{used_field} {used} exceeds {limit_field} {limit}")
        return tuple(violations)

    def reserve(self, proposed: BudgetUsage) -> BudgetUsage:
        """Reserve capacity or fail before any external request is made."""
        violations = self.violations(proposed)
        if violations:
            raise BudgetExceededError("; ".join(violations))
        self.usage = self.usage + proposed
        return self.usage

    def _assert_within(self, usage: BudgetUsage) -> None:
        violations = self.violations(usage) if usage != self.usage else self.violations()
        if violations:
            raise BudgetExceededError("; ".join(violations))


class ExternalAction(StrEnum):
    """Operations with materially different authority requirements."""

    READ_PUBLIC_SOURCE = "read_public_source"
    MODEL_CALL = "model_call"
    WRITE_LOCAL_ARTIFACT = "write_local_artifact"
    REQUEST_INTERVIEW = "request_interview"
    SEND_OUTREACH = "send_outreach"
    BUY_ADS = "buy_ads"
    ACCEPT_DEPOSIT = "accept_deposit"
    CONTACT_INVESTOR = "contact_investor"
    BUY_DATA = "buy_data"


_AUTONOMOUS_ACTIONS = frozenset(
    {
        ExternalAction.READ_PUBLIC_SOURCE,
        ExternalAction.MODEL_CALL,
        ExternalAction.WRITE_LOCAL_ARTIFACT,
    }
)


class ActionDecision(BaseModel):
    """Auditable policy result for one proposed external action."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    action: ExternalAction
    allowed: bool
    requires_human_approval: bool
    reason: str
    approval_event_id: str | None = None


def authorize_action(
    action: ExternalAction,
    *,
    approval_event_id: str | None = None,
) -> ActionDecision:
    """Permit research/local work and require an explicit event for market actions."""
    if action in _AUTONOMOUS_ACTIONS:
        return ActionDecision(
            action=action,
            allowed=True,
            requires_human_approval=False,
            reason="allowed by the autonomous research policy",
        )
    if approval_event_id:
        return ActionDecision(
            action=action,
            allowed=True,
            requires_human_approval=True,
            reason="authorized by a recorded human approval event",
            approval_event_id=approval_event_id,
        )
    return ActionDecision(
        action=action,
        allowed=False,
        requires_human_approval=True,
        reason="external commercial action requires a recorded human approval event",
    )


class KillSwitch:
    """A fixed stop file inside the run root, checked between every operation."""

    filename = "STOP"

    def __init__(self, run_root: Path) -> None:
        self.run_root = run_root.expanduser().resolve()

    @property
    def path(self) -> Path:
        return self.run_root / self.filename

    def engaged(self) -> bool:
        return self.path.is_file()

    def assert_clear(self) -> None:
        if self.engaged():
            raise BudgetExceededError(f"kill switch engaged at {self.path}")


class PacketRole(StrEnum):
    """The independent job receiving a deliberately limited view."""

    RESEARCHER = "researcher"
    FALSIFIER = "falsifier"
    VERIFIER = "verifier"
    OUTCOME_ADJUDICATOR = "outcome_adjudicator"


class BlindPacket(BaseModel):
    """Information a role may see, with forbidden context absent by construction."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    packet_id: str = Field(min_length=1)
    role: PacketRole
    information_cutoff: str = Field(min_length=1)
    payload: dict[str, Any]
    visible_fields: tuple[str, ...]

    @model_validator(mode="after")
    def _payload_matches_allowlist(self) -> "BlindPacket":
        extra = set(self.payload) - set(self.visible_fields)
        if extra:
            raise ValueError(f"payload contains fields outside allowlist: {sorted(extra)}")
        return self


_ROLE_FIELDS: dict[PacketRole, tuple[str, ...]] = {
    PacketRole.RESEARCHER: (
        "question",
        "geography",
        "industry_universe",
        "source_policy",
        "output_schema",
    ),
    PacketRole.FALSIFIER: (
        "anonymized_thesis",
        "claims",
        "evidence_refs",
        "geography",
        "source_policy",
        "falsification_checklist",
    ),
    PacketRole.VERIFIER: (
        "claims",
        "evidence_refs",
        "raw_capture_refs",
        "transform_refs",
        "verification_checklist",
    ),
    PacketRole.OUTCOME_ADJUDICATOR: (
        "target_definition",
        "outcome_source",
        "eligible_population",
        "matures_at",
    ),
}

_ALWAYS_FORBIDDEN = frozenset(
    {
        "current_rank",
        "preferred_outcome",
        "proponent_identity",
        "researcher_identity",
        "falsifier_identity",
        "investor_preference",
        "predicted_probability",
        "prediction_interval",
        "proponent_rationale",
    }
)


def make_blind_packet(
    *,
    packet_id: str,
    role: PacketRole,
    information_cutoff: str,
    available: dict[str, Any],
) -> BlindPacket:
    """Project available context through the role's explicit field allowlist."""
    visible = _ROLE_FIELDS[role]
    payload = {key: available[key] for key in visible if key in available}
    leaked = _ALWAYS_FORBIDDEN & set(payload)
    if leaked:  # defensive if an allowlist is loosened in a future edit
        raise ValueError(f"blind packet would leak forbidden fields: {sorted(leaked)}")
    return BlindPacket(
        packet_id=packet_id,
        role=role,
        information_cutoff=information_cutoff,
        payload=payload,
        visible_fields=visible,
    )


__all__ = [
    "ActionDecision",
    "BlindPacket",
    "BudgetExceededError",
    "BudgetGuard",
    "BudgetPolicy",
    "BudgetUsage",
    "ExternalAction",
    "KillSwitch",
    "PacketRole",
    "authorize_action",
    "make_blind_packet",
]
