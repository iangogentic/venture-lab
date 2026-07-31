---
name: compose-report
description: Render a run's findings as one evidence-first report.
---

# Compose the report

Write the report of this run for one reader: a founder deciding where to spend their
next month. They were not in the pipeline. They will read this document alone, act on
it, and be right or wrong with real time and money. Truth over optimism — a report
that flatters the findings costs them a month.

You are **narrating, not concluding**. Every claim below already exists in an
artifact; your job is the through-line, not new judgement. The verdicts were taken by
the decision stage and are not yours to soften or upgrade.

## The question this run set out to answer

```json
${question}
```

## The research briefs

```json
${briefs}
```

## The pain clusters (may be empty)

```json
${clusters}
```

## The opportunities

```json
${opportunities}
```

## The decisions

```json
${decisions}
```

## The market analyses (may be empty)

```json
${market}
```

## The competition analyses (may be empty)

```json
${competition}
```

## The contradiction analyses (may be empty)

```json
${contradictions}
```

## The interview plans (may be empty)

```json
${interviews}
```

## The harvested leads (may be empty)

```json
${leads}
```

## What to return

- `executive_summary` — the findings without the argument, for the reader who stops
  here. State what was decided and what it rests on, in a short paragraph.
- `highlights` — the few lines worth reading if nothing else is. Findings, not
  headings.
- `body` — the full report, in Markdown.

## The body: one evidence-first section per opportunity

Open with the question and how much evidence the run actually gathered — say plainly
when it is thin. Then, for each opportunity, in this order:

1. **The thesis** — what the opportunity claims, and the pain cluster and briefs it
   grew from.
2. **Market and competition, in brief** — only what the analyses established. An
   unsized market is missing information, not a small market; report the `basis`
   honestly rather than dressing the gap.
3. **What contradicts it** — the counter-evidence, given its own weight, never
   softened and never netted away. A `blocking` finding leads this subsection; an
   empty contradiction analysis is reported together with what was searched for, so
   the reader can judge whether the silence means anything.
4. **The decision and why** — the verdict, its stated reasons, and its confidence.
   Include what argued the other way.
5. **The next validation step** — the decision's cheapest next action, and the
   interview plan or leads that would serve it when the run has them.

## Separate observation, inference, and hypothesis

Keep the register visible in the prose: what a source said ("developers report…"),
what was inferred from it ("which suggests…"), and what is still only proposed
("the untested assumption is…"). A report that flattens these into one confident
voice is wrong even when every sentence is individually true.

## Cite ids for every claim

Every claim points at the artifact that carries it, citing ids inline in parentheses
exactly as they appear in the inputs — `(ev_…)`, `(rb_…)`, `(dec_…)`, or several at
once `(ev_…, cx_…)`. A claim you cannot cite does not go in the report.

Ids nested inside the artifacts above are citable too, and are often the better
citation: the evidence a brief quotes, the cluster an opportunity answers, the
opportunity a decision decided. Copy such an id exactly as it is written — never
adapt, shorten or reconstruct one, and never write an id no input contains. The
application checks every citation against the run's artifacts, and a single id that
names nothing fails the whole reply.

## What not to compute

Do not invent totals, percentages, or market numbers that no artifact carries. The
application computes the title, the covered period, and the verdict tallies from the
artifacts themselves; anything you return in `verdict_counts` is discarded.
