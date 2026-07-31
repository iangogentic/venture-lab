---
name: analyze-market
description: Make the market case for each candidate opportunity.
---

# Analyse the market

For each opportunity below: who buys, whose budget it comes from, what it could be charged
for, and how big it could plausibly be — with the derivation of every figure exposed.

## The question

```json
${question}
```

## The opportunities

```json
${opportunities}
```

## One analysis per opportunity

Link it to the opportunity by its real id. Analyses are read side by side later, so keep
each one about its own opportunity; do not smuggle in a comparison or a ranking.

## Never fabricate a number

You have no market data in front of you. An unsupported figure is worse than no figure:
downstream it becomes a fact, and nobody re-derives it. **"Cannot be sized on this
evidence" is a first-class answer** — write it, say precisely what is missing, and move on.

Every estimate states its basis either way:

- With a number, the basis is the derivation — the countable base, the rate applied to it,
  the price assumed, and where each came from. A method with three named unknowns is worth
  more than a confident total.
- Without a number, the basis is why it could not be established: which factor is unknown,
  and what would have to be observed to supply it.

An amount always carries its currency and the period it covers. "Twelve million" alone is
noise.

## SAM and SOM only

- **SAM** — the portion of demand this offering could actually serve: the named ICP, in the
  geographies and segments reachable through channels that exist, buying something like
  this. Not everyone with the problem.
- **SOM** — what could realistically be won in a stated period, given one team, one
  go-to-market motion, and incumbents who do not stand still.

**Do not produce a TAM.** Not as context, not as a ceiling, not as "for reference". It is
the number most often invented, it is never the number that decides anything, and putting
it next to a careful SAM makes the careful work look small.

## Buyer and budget owner are two questions

- **Buyer** — who holds the problem and would choose to pay to fix it.
- **Budget owner** — whose line the money actually comes out of. Frequently a different
  person, sometimes a different function, occasionally someone who has never met the user.

Where they differ, the gap between them is the deal, and it belongs in assumptions: an
approval step, a security review, a procurement threshold, an annual planning cycle.

## Pricing

Name the model — per seat, usage-based, flat platform fee, percentage of the flow — and say
what it is anchored to: what this segment already pays for something adjacent, what the
workaround costs them today, what the budget line already contains. If nothing in the
input anchors a price, say the model and state that the level is unknown. Do not produce a
number because a pricing field exists.

Complaining is not demand. Loud, frequent, articulate complaint is evidence a problem
exists; it is not evidence anyone will pay to remove it.

## Assumptions and unknowns

- **Assumptions** are what must hold for these figures to mean anything: an adoption rate,
  a reachable channel, a willingness to switch, a budget that exists. Write each so someone
  could go and check it.
- **Unknowns** are what you could not establish at all. An empty unknowns list here is
  almost never honest — you are sizing a market from clustered complaints.

## Failure modes

- Bottom-up arithmetic resting on invented multipliers, which is fabrication with a method
  attached.
- Any figure with no basis, or a basis that restates the figure.
- A TAM, a growth rate, a headcount or a spend figure that did not come from the input.
- Treating everyone with the problem as the addressable market.
- Confusing the sufferer with the budget holder.
- Optimism about willingness to pay that no artifact supports.
- Sizing something confidently rather than saying it cannot be sized.
