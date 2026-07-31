---
name: cluster-pains
description: Fold research briefs into recurring, evidenced pain clusters.
---

# Cluster pains

Group what these briefs describe into distinct recurring pains — by the workflow problem
people actually have, not by the words they used for it.

## The question

```json
${question}
```

## The research briefs

```json
${briefs}
```

## Cluster by the underlying workflow problem

Vocabulary is a trap in both directions. Two briefs both saying "sync" may describe
unrelated problems; two briefs sharing no words at all — one about pasting between tabs,
one about a nightly export script — are often the same pain.

The test to apply: **would one fix plausibly relieve both?** If yes, they belong together.
If a single fix could not address both, they are separate pains however similar the
language. A second test that catches the rest: describe the moment the pain occurs — who is
doing what, at which step, with which tools open. Two descriptions that land on the same
moment are one cluster.

Also keep separate:

- the pain from the workaround people built for it,
- the pain from the tool they blame for it,
- the symptom from the cause, when the briefs distinguish them. When they do not, cluster
  on the symptom — that is what was observed — and say the cause is not established.

## No solutions. None.

Do not name, sketch, or imply what should be built. Not as an aside, not in a label, not as
"which suggests a need for". The next stage cannot infer an honest opportunity from a
problem statement that already contains the answer, and a cluster written around a solution
quietly stops being a description of what people said.

## No market estimation either

No sizing, no revenue, no share of market, no "this affects thousands of teams", no
willingness to pay. Nothing about buyers or budgets. Prevalence is a measurement, not an
impression: leave it unset unless the briefs actually measured it.

## Name it properly

- **Label** it in the sufferers' words where they have one. Their phrase for the problem is
  more useful downstream than your category name for it.
- **Describe** who hits it, when in the workflow they hit it, and what it costs them. A
  description that would fit any company in any industry describes nothing.
- **Segments** narrow enough that someone could go and find these people: "ops leads at
  Series B companies running their own billing reconciliation", not "businesses" or
  "users".
- **Quotes** come verbatim from the briefs. Two sources landing on the same phrasing is
  itself the finding, so keep the repeat.

## Counting and severity

- `source_count` is *distinct sources*. One source restating itself across three briefs
  counts once. This number is the whole difference between a signal and an anecdote.
- Severity is what the pain costs whoever has it, and observed behaviour is the tell:
  people paying for a workaround, hiring for it, or actively shopping is high; people
  complaining while changing nothing is low. Do not read severity off the intensity of the
  language — angry posts are cheap. Leave it unset if the briefs do not support a reading.

## A cluster of one is allowed

If a single source describes a real, specific, costly problem, it is a cluster. Record it
honestly as one source rather than dropping it: a sharp anecdote is more useful than a
vague aggregate, and the next stage can see the count. What is not allowed is dressing it
up as a pattern.

## Failure modes

- Keyword clustering: grouping on a shared noun instead of a shared problem.
- One enormous cluster that covers everything and therefore means nothing.
- A leftover "miscellaneous" or "other issues" bucket. Either it is a pain or it is not.
- Overlapping clusters that are one pain sliced by phrasing.
- Clusters that restate the question rather than what the briefs found.
- Inventing a segment the briefs never mention, or inflating a count to justify a cluster
  you find interesting.
- Any solution, product idea, sizing or buyer speculation leaking in.
