"""Terminal UI for the long-running commands.

The rest of the CLI prints and exits, which is right for commands that finish in
a second. `op auto` does not: it can spend a quarter of an hour inside one stage,
and a scrolling wall of text is a poor way to answer the only two questions
anyone has while waiting — where is it now, and is it still alive.
"""

from app.cli.tui.auto import AutoApp, run_with_tui

__all__ = ["AutoApp", "run_with_tui"]
