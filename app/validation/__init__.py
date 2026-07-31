"""Deterministic scaffolding for validation experiments.

No model is involved anywhere in this package. A landing page is a public claim
made in the founder's name, so every word on it is either drawn verbatim from an
audited artifact or written here, where a reviewer can read it — nothing is
generated at scaffold time.

The templates live in this package rather than in `app/cli` because the
no-prompt-text guard scans `app/skills`, `app/pipeline` and `app/cli` for
prompt-sized string literals. These are page markup, not prompts, but keeping
them out of the scanned packages keeps that guard's rule absolute: nothing long
lives in those three.
"""

from app.validation.scaffold import attributed_quotes, build_scaffold
from app.validation.templates import (
    CALCOM_PLACEHOLDER,
    HONEST_BANNER,
    LISTMONK_PLACEHOLDER,
    POSTHOG_PLACEHOLDER,
)

__all__ = [
    "CALCOM_PLACEHOLDER",
    "HONEST_BANNER",
    "LISTMONK_PLACEHOLDER",
    "POSTHOG_PLACEHOLDER",
    "attributed_quotes",
    "build_scaffold",
]
