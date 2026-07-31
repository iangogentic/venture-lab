"""Shared Rich consoles.

Import these rather than constructing new `Console` objects so styling, width
detection, and stderr routing stay consistent everywhere.
"""

from rich.console import Console
from rich.theme import Theme

THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "yellow",
        "danger": "bold red",
        "muted": "dim",
        "stage": "bold magenta",
    }
)

console = Console(theme=THEME)
"""Primary console for user-facing output (stdout)."""

err_console = Console(theme=THEME, stderr=True)
"""Console for errors, logs, and diagnostics (stderr)."""

__all__ = ["THEME", "console", "err_console"]
