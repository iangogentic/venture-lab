"""Data access. Callers talk to repositories, never to sessions directly."""

from app.storage.repositories.base import BaseRepository
from app.storage.repositories.evidence import EvidenceRepository
from app.storage.repositories.opportunity import OpportunityRepository
from app.storage.repositories.run import RunRepository
from app.storage.repositories.source import SourceRepository
from app.storage.repositories.stage_run import StageRunRepository

__all__ = [
    "BaseRepository",
    "EvidenceRepository",
    "OpportunityRepository",
    "RunRepository",
    "SourceRepository",
    "StageRunRepository",
]
