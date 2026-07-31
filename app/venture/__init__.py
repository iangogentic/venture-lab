"""Evidence-governed venture discovery.

This package is deliberately separate from the upstream software-opportunity
pipeline.  It treats an idea as a falsifiable thesis, keeps measurements in
natural units, and makes stage progression a deterministic policy decision.
"""

from app.venture.evaluation import (
    BinaryForecast,
    CalibrationReport,
    NumericIntervalForecast,
    calibration_report,
    interval_score,
)
from app.venture.operations import (
    ActionDecision,
    BlindPacket,
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

__all__ = [
    "ActionDecision",
    "BinaryForecast",
    "BlindPacket",
    "BudgetExceededError",
    "BudgetGuard",
    "BudgetPolicy",
    "BudgetUsage",
    "CalibrationReport",
    "ExternalAction",
    "KillSwitch",
    "NumericIntervalForecast",
    "PacketRole",
    "authorize_action",
    "calibration_report",
    "interval_score",
    "make_blind_packet",
]
