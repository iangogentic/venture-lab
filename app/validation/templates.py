"""The text of the validation scaffold: landing page, plan, and deploy notes.

Everything here is `str.format`-substituted by `app/validation/scaffold.py`. The
service placeholders — `{{LISTMONK_URL}}` and friends — are deliberately *not*
format fields: they are passed in as literal values so they survive rendering
and reach the written file for the founder to replace by hand.

The honesty constraints are structural, not stylistic. The banner is part of the
template rather than an argument, so no caller can scaffold a page without it;
the page names no price, no availability, and no product, because the thing being
validated does not exist and the page saying otherwise would poison the very
signal the experiment is for.
"""

from typing import Final

LISTMONK_PLACEHOLDER: Final[str] = "{{LISTMONK_URL}}"
"""Where the waitlist form posts. Replaced by the founder's listmonk endpoint."""

POSTHOG_PLACEHOLDER: Final[str] = "{{POSTHOG_SNIPPET}}"
"""Marks where the PostHog analytics snippet goes, inside a head comment."""

CALCOM_PLACEHOLDER: Final[str] = "{{CALCOM_URL}}"
"""The interview booking link. Replaced by the founder's Cal.com event URL."""

HONEST_BANNER: Final[str] = (
    "We're exploring this idea — nothing is built yet. Join the waitlist to shape it."
)
"""The one sentence the page may never lose. A validation page that implies an
existing product measures willingness to be misled, not demand."""

RESEARCH_ATTRIBUTION: Final[str] = "from our research"
"""Fallback attribution when a quote's platform could not be established.
Honest about the limit of what is known — never a guessed platform."""

LANDING_CSS: Final[str] = """\
:root { color-scheme: light dark; }
* { box-sizing: border-box; margin: 0; }
body {
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  line-height: 1.6;
  color: #1a1a24;
  background: #fdfdfc;
}
@media (prefers-color-scheme: dark) {
  body { color: #e8e8ef; background: #16161d; }
  .banner { background: #3a2f10; color: #f2d9a0; }
  blockquote { border-left-color: #4a4a58; }
  footer, .muted, blockquote footer { color: #9a9aa8; }
  input[type="email"] { background: #21212b; color: #e8e8ef; border-color: #4a4a58; }
}
.banner {
  background: #fff3d6;
  color: #6b4e0f;
  text-align: center;
  padding: 0.75rem 1rem;
  font-weight: 600;
}
main { max-width: 40rem; margin: 0 auto; padding: 2rem 1.25rem 4rem; }
h1 { font-size: 1.9rem; line-height: 1.25; margin: 1.5rem 0 0.5rem; }
h2 { font-size: 1.2rem; margin: 2.5rem 0 0.75rem; }
p { margin: 0.6rem 0; }
.muted { color: #6a6a78; font-size: 0.95rem; }
blockquote {
  border-left: 3px solid #d8d8e0;
  margin: 1.25rem 0;
  padding: 0.25rem 0 0.25rem 1rem;
  font-style: italic;
}
blockquote footer { font-style: normal; color: #6a6a78; font-size: 0.9rem; margin-top: 0.35rem; }
form { display: flex; gap: 0.5rem; flex-wrap: wrap; margin: 1rem 0; }
input[type="email"] {
  flex: 1 1 14rem;
  padding: 0.65rem 0.85rem;
  font-size: 1rem;
  border: 1px solid #c8c8d2;
  border-radius: 6px;
}
button {
  padding: 0.65rem 1.2rem;
  font-size: 1rem;
  font-weight: 600;
  border: none;
  border-radius: 6px;
  background: #2b59c3;
  color: #fff;
  cursor: pointer;
}
footer.page { text-align: center; padding: 1.5rem 1rem 2.5rem; color: #6a6a78; font-size: 0.9rem; }
"""

LANDING_TEMPLATE: Final[str] = """\
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<!-- {posthog} -->
<style>
{css}</style>
</head>
<body>
<div class="banner">{banner}</div>
<main>
  <h1>{headline}</h1>
  <section>
    <h2>How it works</h2>
    <p class="muted">The workflow that is broken today:</p>
    <p>{workflow}</p>
    <p>We're exploring a fix. Nothing exists yet — no product, no pricing, no
    launch date. The waitlist below is how you tell us what it should become.</p>
  </section>
{quotes}  <section>
    <h2>Join the waitlist</h2>
    <form action="{listmonk}" method="post">
      <input type="email" name="email" placeholder="you@example.com" required
        aria-label="Email address">
      <button type="submit">Join the waitlist</button>
    </form>
    <p>Prefer to talk? <a href="{calcom}">Book a 20-minute conversation</a> —
    we're interviewing people who live this problem.</p>
  </section>
</main>
<footer class="page">
  <p>This page is a research experiment: nothing described here exists yet.</p>
</footer>
</body>
</html>
"""

QUOTES_SECTION_TEMPLATE: Final[str] = """\
  <section>
    <h2>You're not alone</h2>
{items}  </section>
"""

QUOTE_TEMPLATE: Final[str] = """\
    <blockquote>
      <p>{quote}</p>
      <footer>— {attribution}</footer>
    </blockquote>
"""

VALIDATION_PLAN_TEMPLATE: Final[str] = """\
# Validation plan — {opportunity_title}

> **Fill in every threshold below BEFORE the page goes live.** Post-hoc
> thresholds rationalize any result: once the numbers are in, any of them can be
> argued into "good enough". Committing to pass/fail lines first is what makes
> this an experiment instead of a story you tell yourself afterwards.

## The hypothesis under test

- Opportunity: {opportunity_title} (`{opportunity_id}`)
- Problem: {problem}
- ICP: {icp}

## Pre-registered thresholds — fill in before launch

| Threshold | Target | Actual (at review) |
| --- | --- | --- |
| Visitor → waitlist conversion | ____ % | |
| Minimum waitlist size | ____ | |
| Interviews booked | ____ | |
| Interviews with a concrete commitment (time, money, or a referral) | ____ | |

**Review date:** ____ (pick it now; moving it later is the same failure as
moving a threshold)

{decision_section}
## Reading the result

- All thresholds met by the review date → the demand signal is real; the next
  step is whatever the interviews say it is.
- Any threshold missed → write down why before proposing a fix. A missed
  threshold explained after the fact is the exact rationalization this plan
  exists to prevent.
- A waitlist signup is interest; only a commitment of time, money, or a referral
  is evidence of demand.
"""

DECISION_SECTION_TEMPLATE: Final[str] = """\
## What the pipeline already flagged

- Biggest unknown (from `{decision_id}`): {biggest_unknown}
- Next validation step (from `{decision_id}`): {next_validation_step}

"""

NO_DECISION_SECTION: Final[str] = """\
## What the pipeline already flagged

No decision has been recorded for this opportunity yet — run the pipeline
through the decision stage, then regenerate this scaffold so its biggest unknown
and next validation step are carried in here.

"""

SCAFFOLD_README_TEMPLATE: Final[str] = """\
# Validation scaffold — {opportunity_id}

`index.html` is fully self-contained (inline CSS, no external assets), so any
static host serves it as-is: GitHub Pages, Cloudflare Pages, Netlify, or an S3
bucket behind a CDN.

## Replace the placeholders before deploying

| Placeholder | Where | Replace with |
| --- | --- | --- |
| `{listmonk}` | the waitlist `<form action>` | your listmonk subscription endpoint |
| `{posthog}` | a comment in `<head>` | your PostHog JS snippet |
| `{calcom}` | the interview link | your Cal.com event URL |

All three have self-hostable options:

- [listmonk](https://listmonk.app) manages the waitlist — the form posts to it.
- [PostHog](https://posthog.com) gives you the funnel: visit → signup → booking.
- [Cal.com](https://cal.com) takes the interview bookings.

## Before launch

Fill in `validation-plan.md` — the thresholds are pre-registered on purpose and
must be written down before the first visitor arrives.

## Sharing the page

Community posting is done personally, by a human: post it yourself, in your own
words, in communities where you already participate, and say plainly that it is
an idea you are validating. Nothing in this project automates posting, replies,
or outreach — automating that would burn the very communities the evidence came
from.
"""

__all__ = [
    "CALCOM_PLACEHOLDER",
    "DECISION_SECTION_TEMPLATE",
    "HONEST_BANNER",
    "LANDING_CSS",
    "LANDING_TEMPLATE",
    "LISTMONK_PLACEHOLDER",
    "NO_DECISION_SECTION",
    "POSTHOG_PLACEHOLDER",
    "QUOTES_SECTION_TEMPLATE",
    "QUOTE_TEMPLATE",
    "RESEARCH_ATTRIBUTION",
    "SCAFFOLD_README_TEMPLATE",
    "VALIDATION_PLAN_TEMPLATE",
]
