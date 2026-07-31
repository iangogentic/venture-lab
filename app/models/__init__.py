"""SQLModel tables — the run ledger.

The ledger records *what happened*: runs, the attempts inside them, and an index
over what they produced. The artifacts themselves live in `workspace/` as JSON.
Keeping the two apart means a workspace stays readable (and diffable) without the
database, and the database stays rebuildable by rescanning the workspace —
`op runs sync` does exactly that, which is what keeps the index from becoming a
second, quietly disagreeing source of truth.

Every table module must be imported here so importing this package registers the
full schema on `SQLModel.metadata`.
"""

from app.models.base import TimestampMixin, UTCDateTime, utcnow
from app.models.evidence import EvidenceRecord
from app.models.opportunity import OpportunityRecord
from app.models.run import Run, RunStatus
from app.models.source import Source
from app.models.stage_run import StageRun, StageState

__all__ = [
    "EvidenceRecord",
    "OpportunityRecord",
    "Run",
    "RunStatus",
    "Source",
    "StageRun",
    "StageState",
    "TimestampMixin",
    "UTCDateTime",
    "utcnow",
]
