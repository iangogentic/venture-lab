"""Collector that searches a local corpus of files.

The escape hatch for evidence that has no API behind it: interview notes, an
exported support thread, a saved article, a transcript someone pasted into a
file. It is what keeps the pipeline honest about private signal — a claim made
from a note in a folder is exactly as checkable as one made from a public post,
because `external_id` is the path relative to the search root and `url` is a
`file://` URI, so whoever holds the corpus can open the file and read the line.

Nothing on disk is trusted to be well-formed. A binary blob, a file in some
1990s encoding, a directory this process may not read, a symlink loop — each is
skipped rather than raised, because a corpus is a pile somebody accumulated, not
a curated dataset, and losing two hundred good notes to one bad byte would break
the collector precisely when it is carrying the evidence nothing else can.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import ClassVar, Final, NamedTuple

from pydantic import Field

from app.collectors.base import Collector, CollectorConfig, SourceItem, register
from app.utils.errors import CollectorError
from app.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_GLOB: Final = "**/*.md"

# Files above this are skipped whole rather than truncated. A note this long is a
# database export or a log, not something a person wrote, and the size cap doubles
# as the length bound on `text`: because we never cut a file short, an excerpt is
# always checkable against the entire document rather than against a prefix of it.
_MAX_FILE_BYTES: Final = 400_000
# Ceiling on how much of a tree one search will walk. A glob pointed at a home
# directory should cost a bounded amount of time, not all of it.
_MAX_CANDIDATES: Final = 5_000
# How far into a file to look for its heading before concluding it has none.
_TITLE_SCAN_LINES: Final = 5


class FilesystemConfig(CollectorConfig):
    """Settings for `FilesystemCollector`."""

    paths: list[str] = Field(
        default_factory=list,
        description="Directories to walk, or individual files to read. `~` is expanded.",
    )
    glob: str = Field(
        default=_DEFAULT_GLOB,
        description="Which files under each directory to read, e.g. `**/*.txt`.",
    )


class _Candidate(NamedTuple):
    """A file worth reading, with the root that will name it."""

    path: Path
    root: Path
    modified_at: datetime


@register
class FilesystemCollector(Collector):
    """Search a local corpus for files mentioning the query.

    With no `paths` configured there is no corpus, so `available()` is False and
    `search` returns `[]`: an unpointed filesystem collector is unconfigured, not
    broken, and should be skipped with a reason rather than fail a run.
    """

    name: ClassVar[str] = "filesystem"
    description: ClassVar[str] = "Search a local corpus of notes, transcripts and saved articles."
    requires_credentials: ClassVar[bool] = False

    def search(self, query: str, *, limit: int | None = None) -> list[SourceItem]:
        """Return the files under the configured paths that mention `query`.

        Raises:
            CollectorError: If no configured path exists at all. A path that has
                been typoed or never created is a configuration error the operator
                can fix, and returning `[]` for it would look exactly like a corpus
                with nothing to say. If at least one path exists the others are
                logged and skipped, so a laptop missing one mount still searches.
        """
        cap = limit or self.config.limit
        roots = _configured_paths(self.config)
        if not roots:
            return []

        pattern = _glob_pattern(self.config)
        candidates: list[_Candidate] = []
        missing: list[str] = []
        for raw in roots:
            root = Path(raw).expanduser()
            if not root.exists():
                missing.append(raw)
                continue
            candidates.extend(self._candidates(root, pattern))

        if len(missing) == len(roots):
            raise CollectorError(f"no configured path exists: {', '.join(missing)}")
        for raw in missing:
            logger.warning("filesystem: skipping missing path %s", raw)

        # Two stable sorts, not one compound key: path order gives a deterministic
        # result for files sharing a timestamp (a git checkout gives whole trees the
        # same mtime), and recency ordering then decides which survive the cap.
        candidates.sort(key=lambda candidate: candidate.path)
        candidates.sort(key=lambda candidate: candidate.modified_at, reverse=True)

        found: list[SourceItem] = []
        seen: set[Path] = set()
        for candidate in candidates:
            if len(found) >= cap:
                break
            # Two configured roots can reach the same file; it should be cited once.
            resolved = _resolved(candidate.path)
            if resolved in seen:
                continue
            seen.add(resolved)
            item = self._to_item(candidate, query)
            if item is not None:
                found.append(item)
        return found

    def available(self) -> bool:
        """False when no paths are configured — there would be no corpus to search."""
        return self.config.enabled and bool(_configured_paths(self.config))

    # --------------------------------------------------------------- internals

    def _candidates(self, root: Path, pattern: str) -> list[_Candidate]:
        """Every readable, plausibly-sized file under one configured path."""
        if root.is_file():
            # A path may name a single file. "Search this transcript" is a
            # reasonable thing to configure, and the glob has nothing to walk.
            # The root that names it is then its directory, so the id stays a
            # file name rather than becoming empty.
            single = self._describe(root, root.parent)
            return [single] if single is not None else []

        candidates: list[_Candidate] = []
        try:
            # `Path.glob` does not descend into symlinked directories, so a corpus
            # containing a link back to its own parent cannot spin here.
            for path in root.glob(pattern):
                if len(candidates) >= _MAX_CANDIDATES:
                    logger.warning(
                        "filesystem: stopped at %d files under %s; narrow the glob",
                        _MAX_CANDIDATES,
                        root,
                    )
                    break
                if not path.is_file():
                    continue  # a glob like `**/*` matches directories too
                candidate = self._describe(path, root)
                if candidate is not None:
                    candidates.append(candidate)
        except (OSError, ValueError) as exc:
            # OSError: a directory this process may not read. ValueError: a glob
            # pattern the caller mistyped. Neither should end the whole search.
            logger.warning("filesystem: cannot walk %s with %r: %s", root, pattern, exc)
        return candidates

    def _describe(self, path: Path, root: Path) -> _Candidate | None:
        """Stat one file, or None if it is unreadable or too large to be prose."""
        try:
            stat = path.stat()
        except OSError as exc:
            logger.debug("filesystem: cannot stat %s: %s", path, exc)
            return None
        if stat.st_size > _MAX_FILE_BYTES:
            logger.debug("filesystem: skipping %s, %d bytes", path, stat.st_size)
            return None
        # Aware UTC because the Evidence model downstream rejects naive datetimes,
        # and an mtime is the only date most notes will ever carry.
        modified_at = datetime.fromtimestamp(stat.st_mtime, tz=UTC)
        return _Candidate(path=path, root=root, modified_at=modified_at)

    def _to_item(self, candidate: _Candidate, query: str) -> SourceItem | None:
        """One file as a `SourceItem`, or None if it is unreadable or does not match."""
        try:
            text = candidate.path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            # A PDF, an image, a file written in cp1252: not evidence, not an error.
            logger.debug("filesystem: skipping unreadable %s: %s", candidate.path, exc)
            return None
        if not text.strip():
            return None

        external_id = _relative_id(candidate.path, candidate.root)
        if not _mentions(query, external_id, text):
            return None

        try:
            return SourceItem(
                collector=self.name,
                external_id=external_id,
                # Verbatim, whole and unreflowed: the file *is* the source, so the
                # excerpt check downstream runs against exactly what is on disk.
                text=text,
                title=_title(text, candidate.path),
                url=_file_uri(candidate.path),
                # A file records who wrote it nowhere a reader could trust, so the
                # author stays unset rather than guessing from ownership.
                author=None,
                published_at=candidate.modified_at,
            )
        except Exception as exc:
            # Deliberately broad, and the reason is the guarantee: one unrepresentable
            # file must never lose the other two hundred that read cleanly.
            logger.debug("filesystem: skipping %s: %s", candidate.path, exc)
            return None


def _configured_paths(config: CollectorConfig) -> list[str]:
    """The configured search paths, defaulting to none.

    Read defensively rather than off `FilesystemConfig`: `CollectorConfig` allows
    extra keys, so a config loaded from a file can carry `paths` without ever
    being a `FilesystemConfig`, and one stray non-string should cost that entry
    rather than the collector's construction.
    """
    raw: object = getattr(config, "paths", None)
    if isinstance(raw, str):
        raw = [raw]  # a single directory written as a bare string is a natural mistake
    if not isinstance(raw, list):
        return []
    return [entry.strip() for entry in raw if isinstance(entry, str) and entry.strip()]


def _glob_pattern(config: CollectorConfig) -> str:
    """The configured glob, defaulting to markdown anywhere beneath each path."""
    raw: object = getattr(config, "glob", None)
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return _DEFAULT_GLOB


def _mentions(query: str, external_id: str, text: str) -> bool:
    """Whether `query` appears in the file's body or its path, case-insensitively.

    The path counts because in a real corpus the topic is often only in the name —
    `interviews/2026-06-ci-flakiness.md` says what the transcript is about long
    before the word appears in the prose. An empty query matches the whole corpus.
    """
    needle = query.strip().lower()
    if not needle:
        return True
    return needle in text.lower() or needle in external_id.lower()


def _relative_id(path: Path, root: Path) -> str:
    """The file's path relative to the root it was found under, as a stable id."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        # A symlink can land the match outside the root it was reached through;
        # the absolute path is uglier but still names exactly one file.
        return path.as_posix()


def _file_uri(path: Path) -> str | None:
    """A `file://` URI a reader can open, or None if the path cannot be resolved."""
    try:
        return _resolved(path).as_uri()
    except (OSError, ValueError):
        return None


def _resolved(path: Path) -> Path:
    """The absolute, symlink-free path, falling back to the absolute path."""
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _title(text: str, path: Path) -> str:
    """The document's own heading if it opens with one, else the file's name.

    Notes almost never carry metadata, but they nearly always start with a title,
    and a citation reading "CI flakiness, June interviews" is worth more to a
    reader than one reading "note-14".
    """
    for line in text.splitlines()[:_TITLE_SCAN_LINES]:
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                return heading
        if stripped:
            break  # the file opens with prose, so it has no heading to find
    return path.stem


__all__ = ["FilesystemCollector", "FilesystemConfig"]
