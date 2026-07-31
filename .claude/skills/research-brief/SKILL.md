---
name: research-brief
description: Synthesise collected evidence into cited research briefs.
---

# Research brief

Say what this evidence adds up to — and, just as importantly, how much it is worth.

## The question

```json
${question}
```

## The evidence

```json
${evidence}
```

## The job

A brief is what every later stage reads *instead of* the raw evidence. Someone who never
sees these excerpts must be able to act on your brief and know how far to trust it. That
second half is not metadata; it is half the output.

Cover one coherent slice of the question per brief. If the evidence genuinely splits into
separate stories — different segments, different problems, different timeframes — write a
brief for each. Do not fuse unrelated material because it arrived together, and do not
split one story into fragments to look thorough.

## Four different things, kept apart

**Signals** are what the evidence shows: one pattern per signal, stated so it could be
checked on its own. "Teams re-key the same order into two systems every morning" is a
signal. "There are issues around data entry" is not. Tie each signal to the specific
evidence ids carrying *that* statement, not to the whole pile you read — which subset
supports which claim is exactly what a reader needs.

Tag every signal:

- **observation** — the supplied evidence says this directly. You could quote it.
- **inference** — you reasoned it from the evidence. The reasoning must be legible in the
  statement itself.
- **hypothesis** — you are proposing it. Nothing supplied establishes it yet.

Most signals in a first pass are inference. Labelling one as observation because it feels
solid is the failure this field exists to catch.

**Quotes** are verbatim excerpts, kept attributable. They are the one place a downstream
reader can audit the pipeline, so a quote that cannot be traced back to its evidence is
worse than no quote. Never paraphrase inside a quote, never tidy the grammar, never merge
two speakers.

**Contradictions** are places where sources disagree — a topic plus at least two positions,
each stated in its own terms so whoever holds it would recognise it. Do not resolve them by
majority, by recency, or by which reads better. Do not average them into a claim neither
source made. A single position with a caveat is not a contradiction. Note also which kind
of disagreement it is: about what is happening, about what causes it, or about whether it
matters — those are three different findings.

**Unknowns** are what this evidence does not settle. Be concrete about what would settle
each one — the observation, source or measurement that would close the gap. These drive the
next collection round, so "more research needed" is useless; name the research. An empty
unknowns list is a very strong claim about your own coverage, and is almost never true.

## Grade your own evidence honestly

- **Quality** is what the evidence *is*: nothing behind it, a single unverified report,
  several independent sources agreeing, quantitative data, or something confirmed against a
  primary source. Grade the base as a whole, at the level the weakest load-bearing part
  supports — not at the level of your best single source.
- **Density** is how much of it there is. Sparse means a handful of sources, or many voices
  tracing back to one origin. Moderate means enough to see a pattern but not to size it.
  Dense means many genuinely independent sources. On a normal run, **sparse is the truthful
  answer**, and reporting it costs nothing — reporting dense over eight forum posts costs
  the decision at the end of the pipeline.
- **Source count** is distinct sources. One source restated in three excerpts counts once.
  Do not inflate it, and do not let the number of evidence records stand in for it.

## Failure modes

- **Laundering**: a claim in the summary that appears in no signal and no evidence.
- **Smoothing**: presenting a contested point as settled because most excerpts agreed.
- **"Many users report"** when it was two, or one source quoted twice.
- **Recapitulation**: replaying the excerpts in order instead of saying what they mean
  together. If your summary would be unchanged with half the evidence removed, you
  summarised the pile rather than synthesised it.
- Confident prose over an anecdote. Match the register to the grounding.
- Tagging inference as observation, or grading density up because the brief felt thin.
- Importing a fact you happen to know to fill a gap the evidence left.
