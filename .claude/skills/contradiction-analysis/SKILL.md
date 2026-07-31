---
name: contradiction-analysis
description: Search for evidence against each opportunity.
---

# Contradiction analysis

Go hunting for evidence *against* each opportunity below. Return what you find. Do not
weigh it, and do not conclude anything.

## The question

```json
${question}
```

## The opportunities

```json
${opportunities}
```

## The market analyses

```json
${market}
```

## The competition analyses

```json
${competition}
```

## Why this stage exists

Every stage before this one built the case *for*. Left alone, a pipeline of summarisers
converges on a coherent story, and coherence is not truth. Your job is adversarial: read
all of this as someone who wants the opportunity to be wrong, and go looking for what would
show it.

One analysis per opportunity, linked by its real id. Argue against that specific
opportunity — not against the category, not against the market in general.

## What to search for

- **Failed startups** — someone tried this, or close enough to this, and did not make it.
  What did they build, for whom, and what is known about why it ended?
- **Negative reviews** — users of the existing answers saying they do not work, or that
  they work well enough that nothing better is needed. Both are counter-evidence.
- **Abandonment** — people who adopted something for this problem and stopped. Churn,
  deprecated internal tools, a workflow that reverted to the spreadsheet. Abandonment is
  stronger counter-evidence than absence, because someone already validated the idea and
  it still did not hold.
- **Incumbent strength** — an existing player well placed to absorb this the moment it
  matters: distribution, the data, the contract, the relationship, the roadmap adjacency.
- **Market risk** — structural conditions working against it: budgets that do not exist,
  procurement that outlasts a startup, regulation, a platform dependency that could close,
  a buyer with no incentive to act, a segment too small to reach affordably.

Mine the supplied market and competition analyses as well as the opportunities. The
strongest counter-evidence is often already sitting in them unflagged: an adequate
substitute, a switching cost nobody can pay, a buyer who is not the sufferer, a size
estimate that could not be established. Lift those out and state them as what they are.

Grade severity by what it would do to the decision, not by how uncomfortable it feels:
`minor` where it colours the picture, `material` where the decision should be framed or
conditioned differently, `blocking` where a decision cannot safely be taken until it is
settled. Use `blocking` sparingly and mean it — it stops real work.

## Evidence only

- **No verdict.** Do not say whether the opportunity should proceed.
- **No weighing.** Do not net counter-evidence against the case for, do not say which
  finding matters most, do not write "on balance".
- **No conclusion, no recommendation, no mitigation.** "This could be addressed by…" is
  the decision stage's work, and offering it here lets the search stop early — as soon as
  the story resolves, you stop looking, which is exactly the failure this stage prevents.

Each finding is one observation, stated as an observation, attributed where it can be
attributed. Where it came from a supplied artifact, cite that artifact by its real id.
Where you cannot attribute it, say so rather than dressing it up; an unattributable
objection is still a lead, but it must not look like a fact.

## Finding nothing is a real result

If the search turns up nothing, return nothing — and record what you searched for. That
list is what makes an empty result interpretable instead of indistinguishable from not
having looked. Record it either way, whatever you found.

**Do not manufacture objections.** Padding this stage is more damaging than padding any
other: a decision stage that learns to discount your findings stops reading them, and then
the one real objection arrives with the noise.

## Failure modes

- Restating an upstream caveat as though it were a discovery.
- A named failed company you are not certain existed, or a cause of death you inferred and
  reported as known.
- Generic risk boilerplate ("the market is competitive", "execution is hard") that would
  apply to any opportunity.
- Grading everything `material` to look balanced, or upgrading to `blocking` for effect.
- Softening a genuinely blocking finding into a hedge.
- Any sentence that adds up the evidence, recommends, or resolves.
