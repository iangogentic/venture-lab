"""Logging setup, routed through Rich so log output matches console output."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from rich.logging import RichHandler

from app.utils.console import err_console


@contextmanager
def log_to_file(path: Path, level: int = logging.INFO) -> Iterator[Path]:
    """Also write the log to a file for the duration of the block.

    A long run's warnings are the most perishable thing it produces: which forum
    would not answer, which collector was rate-limited, why a stage was retried.
    On screen they scroll away, and under a full-screen view they cannot even be
    selected — so they go to a file as well, where they can be read, grepped and
    pasted into a bug report after the fact.

    Added rather than swapped in: whatever was already printing to the terminal
    keeps printing. A file that cannot be opened costs the file, never the run.
    """
    root = logging.getLogger()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
    except OSError as exc:
        root.debug("could not open a log file at %s: %s", path, exc)
        yield path
        return

    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    root.addHandler(handler)
    try:
        yield path
    finally:
        root.removeHandler(handler)
        handler.close()


def configure_logging(level: str = "INFO") -> None:
    """Install a Rich log handler on the root logger.

    Idempotent: repeated calls replace the existing handlers rather than
    stacking duplicates.
    """
    logging.basicConfig(
        level=level.upper(),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=err_console, rich_tracebacks=True, show_path=False)],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)


__all__ = ["configure_logging", "get_logger", "log_to_file"]
