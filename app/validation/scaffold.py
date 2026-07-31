"""Build the three scaffold files from a decided opportunity's artifacts.

Pure functions over artifacts — no registry, no network, no model — so the whole
scaffold is reproducible from the workspace and testable without either. The
caller (`op validate scaffold`) loads the artifacts; this module only renders.

Two rules are enforced here rather than trusted to callers:

* **No usernames on a public page.** A person consented to their words appearing
  on the platform where they posted, not to being named on a stranger's landing
  page — quotes are attributed by platform only.
* **Quotes stay verbatim.** They are interpolated (HTML-escaped, nothing more)
  from the pain cluster's `quotes`, which upstream stages already verified
  against collected evidence.
"""

import html
import re
from collections.abc import Sequence
from typing import Final

from app.artifacts import Decision, Opportunity
from app.validation.templates import (
    CALCOM_PLACEHOLDER,
    DECISION_SECTION_TEMPLATE,
    HONEST_BANNER,
    LANDING_CSS,
    LANDING_TEMPLATE,
    LISTMONK_PLACEHOLDER,
    NO_DECISION_SECTION,
    POSTHOG_PLACEHOLDER,
    QUOTE_TEMPLATE,
    QUOTES_SECTION_TEMPLATE,
    RESEARCH_ATTRIBUTION,
    SCAFFOLD_README_TEMPLATE,
    VALIDATION_PLAN_TEMPLATE,
)

MAX_QUOTES: Final[int] = 3
"""Quotes shown on the page. Three is social proof; ten is a wall of complaints."""

_WHITESPACE: Final[re.Pattern[str]] = re.compile(r"\s+")


def attributed_quotes(
    quotes: Sequence[str],
    sources: Sequence[tuple[str, str]],
    *,
    limit: int = MAX_QUOTES,
) -> list[tuple[str, str]]:
    """Pair each quote with a platform-only attribution, e.g. "a developer on reddit".

    `sources` are `(text, platform)` pairs from the run's leads and evidence. A
    quote whose text appears in a source (whitespace-normalised, either direction
    — cluster quotes and lead quotes are excerpts of each other) is attributed to
    that source's platform. When no source matches, the attribution honestly says
    the quote came from our research rather than guessing a platform: a fabricated
    attribution on a validation page would be the exact sin the pipeline exists
    to prevent.
    """
    attributed: list[tuple[str, str]] = []
    for quote in quotes[:limit]:
        needle = _normalise(quote)
        platform = next(
            (
                platform
                for text, platform in sources
                if needle in _normalise(text) or _normalise(text) in needle
            ),
            None,
        )
        label = f"a developer on {platform}" if platform else f"a developer, {RESEARCH_ATTRIBUTION}"
        attributed.append((quote, label))
    return attributed


def build_scaffold(
    opportunity: Opportunity,
    *,
    quotes: Sequence[tuple[str, str]],
    decision: Decision | None,
) -> dict[str, str]:
    """Render the scaffold: filename → content, ready to be written to a directory."""
    return {
        "index.html": _landing_page(opportunity, quotes),
        "validation-plan.md": _validation_plan(opportunity, decision),
        "README.md": SCAFFOLD_README_TEMPLATE.format(
            opportunity_id=opportunity.id,
            listmonk=LISTMONK_PLACEHOLDER,
            posthog=POSTHOG_PLACEHOLDER,
            calcom=CALCOM_PLACEHOLDER,
        ),
    }


def _landing_page(opportunity: Opportunity, quotes: Sequence[tuple[str, str]]) -> str:
    """The public page. Headline from the problem, workflow as-is, quotes verbatim."""
    items = "".join(
        QUOTE_TEMPLATE.format(quote=html.escape(quote), attribution=html.escape(attribution))
        for quote, attribution in quotes[:MAX_QUOTES]
    )
    section = QUOTES_SECTION_TEMPLATE.format(items=items) if items else ""
    return LANDING_TEMPLATE.format(
        title=html.escape(opportunity.title),
        posthog=POSTHOG_PLACEHOLDER,
        css=LANDING_CSS,
        # Not escaped: the banner is this package's own constant, and escaping its
        # apostrophe would stop the exact agreed sentence appearing in the page.
        banner=HONEST_BANNER,
        headline=html.escape(opportunity.problem),
        workflow=html.escape(opportunity.workflow),
        quotes=section,
        listmonk=LISTMONK_PLACEHOLDER,
        calcom=CALCOM_PLACEHOLDER,
    )


def _validation_plan(opportunity: Opportunity, decision: Decision | None) -> str:
    """The pre-registered plan, carrying the decision's own doubts when it exists."""
    if decision is not None:
        section = DECISION_SECTION_TEMPLATE.format(
            decision_id=decision.id,
            biggest_unknown=decision.biggest_unknown,
            next_validation_step=decision.next_validation_step,
        )
    else:
        section = NO_DECISION_SECTION
    return VALIDATION_PLAN_TEMPLATE.format(
        opportunity_title=opportunity.title,
        opportunity_id=opportunity.id,
        problem=opportunity.problem,
        icp=opportunity.icp,
        decision_section=section,
    )


def _normalise(text: str) -> str:
    """Collapse whitespace so a reflowed quote still matches its source."""
    return _WHITESPACE.sub(" ", text).strip().casefold()


__all__ = ["MAX_QUOTES", "attributed_quotes", "build_scaffold"]
