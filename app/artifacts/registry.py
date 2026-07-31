"""The artifact registry: a JSON-file store rooted at `workspace/`.

There is deliberately no database here. An artifact is one pretty-printed JSON
file at `workspace/<directory>/<id>.json`, which means the entire state of a run
is greppable, diffable, and reviewable in a pull request. The cost is that
"search" is a directory scan; at the volumes this pipeline produces that is a
better trade than a schema to migrate.

Layout::

    workspace/
      evidence/
        ev_3f9c….json              <- current version
        .history/
          ev_3f9c….v1.json         <- superseded revisions, byte-for-byte

Two distinct notions of change, matching the envelope in `base.py`:

* `update()`   — corrects an artifact in place. Same `id`, same `version`.
* `version()`  — publishes a new revision. Same `id`, `version` incremented, and
  the previous file preserved under `.history/`.

`supersedes` is the third, different case: a *new* artifact (new id) replacing an
old one. The registry never sets it — that is a judgement the pipeline makes.
"""

import json
import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from pydantic import ValidationError

from app.artifacts.base import Artifact, ArtifactKind, ArtifactRef, ArtifactStatus, EvidenceLevel
from app.artifacts.competition_analysis import CompetitionAnalysis
from app.artifacts.contradiction_analysis import ContradictionAnalysis
from app.artifacts.decision import Decision
from app.artifacts.evidence import Evidence
from app.artifacts.interview_plan import InterviewPlan
from app.artifacts.lead import Lead
from app.artifacts.market_analysis import MarketAnalysis
from app.artifacts.opportunity import Opportunity
from app.artifacts.pain_cluster import PainCluster
from app.artifacts.question import Question
from app.artifacts.report import Report
from app.artifacts.research_brief import ResearchBrief
from app.utils.errors import ArtifactError
from app.utils.paths import WorkspacePaths, get_workspace_paths
from app.utils.time import utcnow

HISTORY_DIRNAME: Final[str] = ".history"
"""Hidden, so a plain `*.json` listing of a stage directory shows only current artifacts."""

MODELS: Final[Mapping[ArtifactKind, type[Artifact]]] = MappingProxyType(
    {
        ArtifactKind.QUESTION: Question,
        ArtifactKind.EVIDENCE: Evidence,
        ArtifactKind.RESEARCH_BRIEF: ResearchBrief,
        ArtifactKind.PAIN_CLUSTER: PainCluster,
        ArtifactKind.OPPORTUNITY: Opportunity,
        ArtifactKind.MARKET_ANALYSIS: MarketAnalysis,
        ArtifactKind.COMPETITION_ANALYSIS: CompetitionAnalysis,
        ArtifactKind.CONTRADICTION_ANALYSIS: ContradictionAnalysis,
        ArtifactKind.DECISION: Decision,
        ArtifactKind.INTERVIEW_PLAN: InterviewPlan,
        ArtifactKind.LEAD: Lead,
        ArtifactKind.REPORT: Report,
    },
)
"""Which concrete model implements each kind. Needed because the JSON is loaded by kind."""

_PREFIX_TO_KIND: Final[Mapping[str, ArtifactKind]] = MappingProxyType(
    {model.id_prefix: kind for kind, model in MODELS.items()}
)


def model_for(kind: ArtifactKind) -> type[Artifact]:
    """Return the concrete model class implementing `kind`."""
    return MODELS[kind]


def kind_for_id(artifact_id: str) -> ArtifactKind | None:
    """Infer a kind from an id's prefix, e.g. `ev_3f9c…` -> EVIDENCE.

    Only a hint — ids are opaque and a caller may have minted one by hand, so
    every lookup that uses this must fall back to scanning.
    """
    prefix, separator, _ = artifact_id.partition("_")
    if not separator:
        return None
    return _PREFIX_TO_KIND.get(prefix)


class ArtifactRegistry:
    """Reads and writes artifacts as JSON files under `workspace/`."""

    def __init__(self, paths: WorkspacePaths | None = None) -> None:
        self.paths = paths if paths is not None else get_workspace_paths()

    # ------------------------------------------------------------------ paths

    def directory_for(self, kind: ArtifactKind) -> Path:
        """The stage directory holding artifacts of this kind."""
        return self.paths.for_directory(kind.directory)

    def path_for(self, kind: ArtifactKind, artifact_id: str) -> Path:
        """Where the current version of this artifact lives."""
        return self.directory_for(kind) / f"{artifact_id}.json"

    def history_dir_for(self, kind: ArtifactKind) -> Path:
        """Where superseded revisions of this kind are kept."""
        return self.directory_for(kind) / HISTORY_DIRNAME

    def history_path_for(self, kind: ArtifactKind, artifact_id: str, version: int) -> Path:
        """Where revision `version` of this artifact is archived."""
        return self.history_dir_for(kind) / f"{artifact_id}.v{version}.json"

    def exists(self, kind: ArtifactKind, artifact_id: str) -> bool:
        """Whether a current version of this artifact is on disk."""
        return self.path_for(kind, artifact_id).is_file()

    # ------------------------------------------------------------------- save

    def save(self, artifact: Artifact, *, overwrite: bool = True) -> Path:
        """Write `artifact` as JSON and return the path written.

        The write is atomic — a temp file plus `os.replace` — so a crash mid-write
        cannot leave a truncated artifact behind for the next stage to read.
        """
        kind = type(artifact).kind
        path = self.path_for(kind, artifact.id)

        if path.exists() and not overwrite:
            raise ArtifactError(f"{kind.value} {artifact.id!r} already exists at {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        self._atomic_write(path, self._serialise(artifact))
        return path

    # ------------------------------------------------------------------- load

    def load(self, kind: ArtifactKind, artifact_id: str) -> Artifact:
        """Load an artifact, choosing the model class from `kind`."""
        return self.load_as(kind, artifact_id, model_for(kind))

    def load_as[ArtifactT: Artifact](
        self,
        kind: ArtifactKind,
        artifact_id: str,
        model: type[ArtifactT],
    ) -> ArtifactT:
        """Load an artifact and validate it against a specific model."""
        path = self.path_for(kind, artifact_id)
        if not path.is_file():
            raise ArtifactError(f"No {kind.value} artifact {artifact_id!r} at {path}")
        return self._deserialise(path, kind, model)

    def delete(self, kind: ArtifactKind, artifact_id: str, *, missing_ok: bool = False) -> None:
        """Remove the current version. History is left intact."""
        path = self.path_for(kind, artifact_id)
        if not path.is_file():
            if missing_ok:
                return
            raise ArtifactError(f"No {kind.value} artifact {artifact_id!r} at {path}")
        path.unlink()

    # --------------------------------------------------------- update/version

    def update[ArtifactT: Artifact](self, artifact: ArtifactT, **changes: Any) -> ArtifactT:
        """Apply `changes` in place and save. Same id, same version.

        Round-tripped through validation rather than `model_copy`, so a bad value
        is rejected here instead of surfacing as a corrupt file later.
        """
        updated = self._apply(artifact, changes, version=artifact.version)
        self.save(updated)
        return updated

    def version[ArtifactT: Artifact](self, artifact: ArtifactT, **changes: Any) -> ArtifactT:
        """Publish a new revision: archive what is on disk, then bump `version`.

        The archived copy is the exact bytes previously stored, not a re-render of
        the in-memory object, so the history is a faithful record even if the
        caller's copy has drifted.
        """
        kind = type(artifact).kind
        current = self.path_for(kind, artifact.id)

        if current.is_file():
            raw = current.read_text(encoding="utf-8")
            archive = self.history_path_for(kind, artifact.id, self._version_of(raw, current))
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_text(raw, encoding="utf-8")

        revised = self._apply(artifact, changes, version=artifact.version + 1)
        self.save(revised)
        return revised

    def versions(self, kind: ArtifactKind, artifact_id: str) -> list[int]:
        """Archived revision numbers for this artifact, oldest first."""
        history = self.history_dir_for(kind)
        if not history.is_dir():
            return []

        found: list[int] = []
        for path in history.glob(f"{artifact_id}.v*.json"):
            _, _, tail = path.name.partition(".v")
            number = tail.removesuffix(".json")
            if number.isdigit():
                found.append(int(number))
        return sorted(found)

    def load_version(self, kind: ArtifactKind, artifact_id: str, version: int) -> Artifact:
        """Load an archived revision."""
        path = self.history_path_for(kind, artifact_id, version)
        if not path.is_file():
            raise ArtifactError(f"No version {version} of {kind.value} {artifact_id!r} at {path}")
        return self._deserialise(path, kind, model_for(kind))

    # ----------------------------------------------------------------- search

    def list_ids(self, kind: ArtifactKind) -> list[str]:
        """Ids of every current artifact of this kind, sorted."""
        directory = self.directory_for(kind)
        if not directory.is_dir():
            return []
        return sorted(path.stem for path in directory.glob("*.json"))

    def iter_by_type(self, kind: ArtifactKind) -> Iterator[Artifact]:
        """Stream every current artifact of one kind, so a large stage need not fit in memory."""
        model = model_for(kind)
        directory = self.directory_for(kind)
        if not directory.is_dir():
            return
        for path in sorted(directory.glob("*.json")):
            yield self._deserialise(path, kind, model)

    def find_by_type(
        self,
        kind: ArtifactKind,
        *,
        status: ArtifactStatus | None = None,
        run_id: str | None = None,
        min_evidence_level: EvidenceLevel | None = None,
        min_confidence: float | None = None,
        limit: int | None = None,
    ) -> list[Artifact]:
        """Search one kind, narrowing on the envelope fields every artifact shares.

        Filters are conjunctive; `None` means "do not filter on this". Artifacts
        whose `confidence` is unassessed are excluded by `min_confidence` — an
        unknown value cannot satisfy a threshold.
        """
        results: list[Artifact] = []
        for artifact in self.iter_by_type(kind):
            if status is not None and artifact.status is not status:
                continue
            if run_id is not None and artifact.run_id != run_id:
                continue
            if (
                min_evidence_level is not None
                and artifact.evidence_level.rank < min_evidence_level.rank
            ):
                continue
            if min_confidence is not None and (
                artifact.confidence is None or artifact.confidence < min_confidence
            ):
                continue

            results.append(artifact)
            if limit is not None and len(results) >= limit:
                break
        return results

    def find_by_fingerprint(self, fingerprint: str, kind: ArtifactKind) -> list[Artifact]:
        """Artifacts of one kind already produced by an identical computation.

        The whole idempotency story: if this returns anything, the same skill with
        the same prompt, model and inputs has already run, and running it again
        would spend money to produce what is already on disk.
        """
        return [
            artifact for artifact in self.iter_by_type(kind) if artifact.fingerprint == fingerprint
        ]

    def find_by_id(self, artifact_id: str) -> Artifact | None:
        """Find an artifact by id alone, across every kind. `None` if absent.

        Tries the kind implied by the id prefix first, then falls back to scanning
        the remaining kinds so hand-minted ids still resolve.
        """
        located = self.locate(artifact_id)
        return None if located is None else self.load(located.kind, located.id)

    def locate(self, artifact_id: str) -> ArtifactRef | None:
        """Resolve an id to a `(kind, id)` reference without reading the file body."""
        hinted = kind_for_id(artifact_id)
        if hinted is not None and self.exists(hinted, artifact_id):
            return ArtifactRef(kind=hinted, id=artifact_id)

        for kind in ArtifactKind:
            if kind is not hinted and self.exists(kind, artifact_id):
                return ArtifactRef(kind=kind, id=artifact_id)
        return None

    def resolve(self, ref: ArtifactRef) -> Artifact:
        """Load the artifact a reference points at."""
        return self.load(ref.kind, ref.id)

    def count(self, kind: ArtifactKind) -> int:
        """How many current artifacts of this kind are stored."""
        return len(self.list_ids(kind))

    # --------------------------------------------------------------- internals

    @staticmethod
    def _serialise(artifact: Artifact) -> str:
        """Render an artifact as JSON, tagged with its kind.

        `kind` is a ClassVar, so it is not one of the model's fields; it is injected
        here to keep files self-describing. A file copied out of its directory can
        still be identified, and `_deserialise` uses it as an integrity check.
        """
        payload: dict[str, Any] = {"kind": type(artifact).kind.value}
        payload.update(artifact.model_dump(mode="json"))
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    @staticmethod
    def _deserialise[ArtifactT: Artifact](
        path: Path,
        kind: ArtifactKind,
        model: type[ArtifactT],
    ) -> ArtifactT:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"{path} is not valid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ArtifactError(f"{path} does not contain a JSON object")

        declared = data.pop("kind", None)
        if declared is not None and declared != kind.value:
            raise ArtifactError(
                f"{path} declares kind {declared!r} but sits in the {kind.value!r} directory"
            )

        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise ArtifactError(f"{path} does not match {model.__name__}: {exc}") from exc

    @staticmethod
    def _version_of(raw: str, path: Path) -> int:
        """Read the `version` out of stored JSON, for naming its archive file."""
        try:
            stored = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ArtifactError(f"{path} is not valid JSON: {exc}") from exc

        version = stored.get("version", 1) if isinstance(stored, dict) else 1
        return version if isinstance(version, int) and version >= 1 else 1

    @staticmethod
    def _apply[ArtifactT: Artifact](
        artifact: ArtifactT,
        changes: dict[str, Any],
        *,
        version: int,
    ) -> ArtifactT:
        data = artifact.model_dump()
        data.update(changes)
        data["version"] = version
        data["updated_at"] = utcnow()

        model = type(artifact)
        try:
            return model.model_validate(data)
        except ValidationError as exc:
            raise ArtifactError(
                f"Invalid change to {model.__name__} {artifact.id!r}: {exc}"
            ) from exc

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        temp = path.with_name(f"{path.name}.tmp")
        temp.write_text(text, encoding="utf-8")
        os.replace(temp, path)


__all__ = [
    "HISTORY_DIRNAME",
    "MODELS",
    "ArtifactRegistry",
    "kind_for_id",
    "model_for",
]
