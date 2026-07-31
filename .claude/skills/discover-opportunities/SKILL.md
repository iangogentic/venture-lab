---
name: discover-opportunities
description: Infer candidate opportunities from the clustered pains.
---

# Discover opportunities

For the clusters below, work out what the real opportunity is: whose workflow is broken,
who lives in it, who would pay to fix it, and what changed to make it addressable now.

## The question

```json
${question}
```

## The pain clusters

```json
${clusters}
```

## This is inference, and it must stay labelled as inference

Everything upstream reported what sources said. You are the first stage that reasons past
it, so the discipline shifts: an opportunity is a claim *about* the evidence, and the
fields exist to keep that visible. Mark it `hypothesis` rather than `inference` when the
clusters barely carry it — that is not a demotion, it is the honest reading.

Every opportunity answers a named cluster. Point at it by its real id. If you cannot, this
is not an opportunity from this run — it is an idea you brought with you, and it does not
belong here.

Not every cluster deserves an opportunity. Some pains are real, well evidenced, and still
not worth answering: too rare, too cheap to endure, already handled well enough. Producing
nothing for such a cluster is a correct answer. One well-argued opportunity beats five that
exist because there were five clusters. Where two proposals differ only in wording, they
are one.

## What each field has to do

- **Workflow** — the sequence of work that is broken today, described concretely enough to
  picture: what triggers it, which steps, which tools, where it breaks. Not a market, not a
  category, not a department.
- **ICP** — who lives in that workflow, narrow enough to go and find them this week: role,
  size and shape of organisation, and the situation they must be in.
- **Buyer** — who would actually pay. **This is usually not the user.** The person who
  suffers rarely holds the budget, and an opportunity that cannot say whose problem it is
  *in budget terms* is not yet an opportunity. If the buyer and the user are the same
  person, say so deliberately, because that is an unusual and valuable property.
- **Problem** — stated plainly, in the sufferer's voice, not the builder's. If it only
  makes sense to someone who already accepts a solution, it is a pitch wearing a problem's
  clothes.
- **Why now** — **name something that actually changed.** A new regulation, a price
  collapse, a platform that opened or closed, a capability that did not exist eighteen
  months ago, a behaviour shift the clusters show. "AI is advancing", "the market is
  growing" and "teams increasingly need X" apply to everything and therefore argue for
  nothing. If nothing changed, the honest answer is that this has been possible for years
  and nobody did it — write that, and let the decision stage weigh it.
- **Missing evidence** — what would have to be checked before anyone trusts this. In
  practice this list is never empty: you are inferring a buyer, an ICP and a trigger from
  clustered complaints. Name each gap as something someone could actually go and find out,
  and put the one that would kill the opportunity first.

## Do not analyse competitors here

No competitor names, no incumbent comparison, no "unlike existing tools", no defensibility,
no moats, no market sizing, no pricing. A dedicated stage does that against real criteria,
and it works badly when the opportunity has already argued its own case against a
half-imagined field of rivals. Where the clusters show people already using something,
record that as part of the workflow — that is what people do today, not competitive
analysis.

## Failure modes

- A solution in search of a pain, retrofitted to the nearest cluster.
- The cluster restated with "platform", "copilot" or "workspace" in front of it.
- A `why_now` that names a trend rather than a change.
- Naming the user as the buyer because the budget question is uncomfortable.
- An ICP anyone could belong to.
- An empty or decorative missing-evidence list ("further validation would be helpful").
- Competitor, pricing or sizing claims invented here, which corrupts the stages built to
  produce them properly.
