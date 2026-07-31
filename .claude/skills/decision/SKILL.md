---
name: decision
description: Choose build, reject or wait for each opportunity.
---

# Decide

Rule on each opportunity below — build, reject or wait — with reasoning explicit enough
that someone can re-judge the call later and see exactly where they disagree with you.

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

## The contradiction analyses

```json
${contradictions}
```

## One verdict per opportunity

Exactly one, linked to the opportunity by its real id. There is no "build with
reservations": that is `build`, plus a next validation step that says what you are watching.

- **build** — pursue it now. The case holds and the remaining risks are ones you are
  knowingly accepting.
- **reject** — do not pursue, and do not expect that to change. A judgement about the
  idea: the pain is not worth solving, or this is not the way to solve it.
- **wait** — plausible, but something has to change or be learned first. Name what, and
  name what would tell you it has happened. A `wait` with no trigger is avoidance wearing a
  verdict.

The distinction most often collapsed is `reject` versus `wait`. Rejecting because you do
not know enough closes a question nobody actually examined; waiting on an idea that is
plainly bad burns a research cycle. Be clear which one you mean. And do not use `wait` as
the safe middle: if every opportunity comes back `wait`, you have decided nothing.

## This is the only stage that weighs

Every stage before this one was forbidden from netting its findings against anything. The
market analysis made the case for and did not discount it; the competition analysis
described the field and did not conclude from it; the contradiction analysis returned
findings and no verdict. The trade-off is unresolved on purpose, and it is yours. Say
plainly what you traded against what — the weighing has to be visible in your reasons, not
just implied by the verdict you landed on.

## How to weigh the four readings

**The dimensions are not interchangeable.** They do not add up to a score, and a strong
reading on one does not buy down a weak reading on another.

- **A strong market case does not rescue an opportunity with a blocking contradiction.** A
  large, well-derived market makes a blocked opportunity a bigger miss, not a smaller
  risk. If a finding is graded `blocking`, the size of the prize is not the answer to it —
  either a reason states plainly why that finding does not bind this decision, or the
  verdict is `wait` with the finding as the trigger.
- **A crowded field is not by itself a reject.** Read `competitors` together with
  `switching_costs` and `substitutes`. Many incumbents with low switching costs means
  customers can move — the field is contested but not closed, and the honest verdict is
  usually `wait` on evidence that a wedge exists, not `reject`. High switching costs, or a
  `moats` entry that a new entrant has no route around, is what turns a crowded field into
  a genuine reject. Note also that `substitutes` are usually the real competition: a
  spreadsheet that is good enough is a harder opponent than a funded company.
- **`differentiation` is a claim, not a finding.** Treat an entry there as something still
  to be validated unless something in the evidence actually supports it.

## An unsized market is missing information, not a negative signal

A `SizeEstimate` can legitimately carry no number. Its `basis` is mandatory either way: with
an amount it is the derivation, without one it is why the market could not be sized. The
previous stage was built to make that refusal expressible so that nobody would have to
invent a figure.

**So do not read an absent SAM or SOM as a small market.** Treating "we could not size
this" as evidence against would punish exactly the honesty the stage was designed for, and
it would teach the pipeline that a fabricated number scores better than an admitted gap.
Read the `basis` and decide what the gap actually is:

- Unsized because the buyer segment is not yet identifiable → that is your `biggest
  unknown` and probably a `wait` with sizing as the next validation step.
- Unsized because the figure exists but was not to hand → it is a cheap next validation
  step and should barely move the verdict.
- Sized with a `basis` that is a chain of guesses → weaker than it looks. A number with a
  thin derivation deserves less weight than an honest refusal.

The same reading applies to `pricing`, to `assumptions`, and to `unknowns` — an empty
`unknowns` list is rarely honest, and is itself worth noticing. And check the `buyer`
against the `budget_owner`: when they differ, someone has to be persuaded who does not feel
the pain, and that is a real cost the size figure does not capture.

## Reasons must be traceable

Each reason points at something in the artifacts you were given — an opportunity's `why
now`, a size estimate that could not be established, a switching cost, a specific piece of
counter-evidence — and cites it by real id where it has one. A reason resting on a fact
that appears in none of these artifacts is not a reason; it is a preference.

Give the two or three things that actually drove the call, strongest first, not everything
true about the opportunity. **Include what argued the other way and why it lost.** A list
of only supporting points is advocacy, and it gives a later reader nothing to check.

## The counter-evidence has to show up

The contradiction analyses were produced deliberately, without weighing, so that the
weighing happens here and is visible.

- A `blocking` finding pushes hard away from `build`. Building over one is permitted, but
  only if a reason states plainly why that finding does not bind this decision. Otherwise
  the honest verdict is `wait`.
- `material` findings do not block, but should appear somewhere: in a reason, in the next
  validation step, or in an explicit statement that the call holds either way.
- If a contradiction analysis found nothing, note that you saw it — and note whether its
  `searched_for` list was thorough enough for the emptiness to mean anything. An empty
  result from a shallow search is not reassurance.

## Confidence, unknown, next step

Required for every verdict, including `reject`, and including a `build` you are sure of.

- **Confidence** grades this judgement, not the idea's attractiveness. Anchor it to how
  much the artifacts actually support the call. Thin, sparse evidence with a clear
  direction is a low-confidence `build`, and saying so is what makes the call reversible.
  Four readings that agree do not raise confidence if all four rest on the same one piece
  of evidence — check whether they are independent before you let them accumulate.
- **Biggest unknown** is required even when you are confident — *especially* then. A
  confident verdict that cannot name what it is still most likely to be wrong about has
  stopped reasoning. Name the specific thing, not "market dynamics".
- **Next validation step** is the single cheapest action that would move the confidence:
  who to talk to, what to look up, what to try. Days, not quarters. It applies to `reject`
  as much as `build` — what would make you revisit this, and what do you tell whoever asked
  the question?

## Failure modes

- A verdict that does not follow from its own stated reasons.
- Netting the four readings into an implied score, or letting market size answer an
  objection that is not about size.
- Reading an unsized market, or an honest `basis`, as evidence against.
- Rejecting on the mere presence of competitors, without reference to switching costs,
  substitutes or moats.
- `wait` used to avoid deciding, or with no trigger attached.
- Building on everything, or rejecting everything, which is the same failure twice.
- Reasoning that appeals to facts none of the supplied artifacts contain.
- Silence about the strongest argument against the verdict you chose.
- Confidence set by how good the idea sounds rather than by how much stands behind it.
- A biggest unknown so vague that no evidence could ever resolve it.
