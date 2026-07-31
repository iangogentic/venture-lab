---
name: collect-evidence
description: Search real sources for material bearing on the question.
---

# Collect evidence

Real material has already been fetched from external sources. Your job is to decide
which of it bears on the question, and to quote the passage that carries the point.

You are **selecting, not writing**. Every excerpt you keep is checked against the
text it came from; anything not literally present there is rejected and the whole
reply fails. You cannot introduce a source, a URL, a statistic or a quotation that
is not below.

## The question

```json
${question}
```

## What the sources returned

```json
${candidates}
```

## What to keep

Keep an item when it is **first-hand about the problem space** — someone describing
their own workflow, their own frustration, what they tried, what it cost them. That
is the material the rest of the pipeline is built to reason over.

Drop:

- marketing copy, launch announcements, and anything written to sell something;
- generic commentary that could have been written without doing the work;
- items that mention the topic but say nothing about how anyone actually works;
- duplicates and cross-posts — keep the fullest telling, not both.

Relevance beats volume. **Four strong items is a better result than twenty weak
ones**, and an empty selection is legitimate when nothing fetched genuinely bears on
the question. Padding the list is the failure mode here, not missing something.

## Choosing the excerpt

Copy the passage **exactly** as it appears in that candidate's `text`. Reflowing
across lines is fine; changing words, tidying, cutting mid-sentence or joining two
separate passages is not.

Pick the sentence or short paragraph a reader could see on its own and understand
what the problem is. A quote that needs the surrounding thread to make sense is a
poor excerpt — prefer a different item over a quote that cannot stand up alone.

`candidate_id` must match an `external_id` above exactly.

## Grading each one

- `evidence_kind` — what shape of observation this is.
- `evidence_level` — how well grounded. One person's account is `anecdotal`; several
  independent people saying the same thing is `corroborated`; a figure someone
  measured is `measured`. Most of what you see here will be `anecdotal`, and saying
  so costs nothing.
- `confidence` — how sure you are it bears on the question, not how strongly the
  author felt.
- `relevance` — one line on what it shows. Not a summary of the item; a statement of
  why it matters to the question asked.

## What would make this stage worthless

Selecting an item because it is *about the right topic* rather than because it shows
someone struggling. Grading everything `corroborated` because several items mention
the same tool. Quoting the most quotable line rather than the most informative one.
Keeping something because the list would otherwise look thin.
