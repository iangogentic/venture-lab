---
name: harvest-leads
description: Turn clustered evidence into leads - people, quotes, permalinks.
---

# Harvest leads

Each pain cluster below comes with candidates: evidence items that carry an author and
a permalink. Your job is to decide which candidates show a person actually expressing
the cluster's pain, quote the passage that shows it, and read what their words express.

You are **selecting, not writing**. Every quote you keep is checked against the excerpt
it came from; anything not literally present there is rejected and the whole reply
fails. You cannot add a person, a URL, or a candidate that is not below.

## The pain clusters

```json
${clusters}
```

## The candidates, per cluster

```json
${candidates}
```

## Observation before inference

The quote is the observation: the person's exact words. The intent is your inference
from those words and nothing else — not from the cluster's label, not from what a
person *probably* meant. If the quote does not support the intent on its own, pick a
different quote or skip the candidate.

## Intent

- `seeking` — actively asking for a solution. *"What are people using instead of X?
  I can't keep doing this manually."*
- `complaining` — expressing the pain unprompted: venting, describing the cost,
  nobody asked. *"We lose half a day every release to this sync step."*
- `mentioning` — the topic is named but no pain is expressed. *"We use X for our
  deployments alongside Y."*

Give `intent_rationale` in one line: which words carried the reading.

## Quotes are copied, not composed

Copy the passage **exactly** as it appears in that candidate's `excerpt`. Reflowing
across lines is fine; changing words, tidying, cutting mid-sentence or stitching two
passages together is not. Pick the passage a reader could see alone and understand
what the pain is.

`evidence_id` and `cluster_id` must match a candidate entry above exactly.

## Skipping is correct

A candidate that names the topic without expressing pain, or whose excerpt is about
someone else's problem, is not a lead — leave it out. A cluster yielding zero leads is
a true result, not a failure. Padding the list is the failure mode here.

## Never invent contact details

The permalink is the only channel. Do not infer, guess, or construct emails, handles
on other platforms, employers, or real names — not even when the text hints at them.
Engaging with a lead happens later, as a public, human-written reply on the platform
where the person chose to speak; nothing you produce here automates or addresses that
contact.
