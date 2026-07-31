# Venture Lab

Venture Lab is a standalone, evidence-governed business-opportunity research
workflow. It is designed to answer a narrower and more defensible question than
“What is the best business?”:

> Which business theses deserve the next validation dollar, what evidence
> supports them, and what observation would make us stop?

It does not contact customers, investors, or vendors; buy ads or data; accept
deposits; or perform regulated work. Those actions require a separately recorded
human approval.

## What one scan does

1. Loads one frozen `EvidencePacket` with an explicit information cutoff.
2. Lets a tool-free generator propose a bounded number of hypotheses.
3. Validates typed geography, evidence-reference, scenario, market-topic,
   provider-activity, and customer-industry boundary fields.
4. Performs a provisional comparative review of the sold activity against every
   eligible provider-side NAICS scope in the packet.
5. Gives a separate, blind falsifier only the evidence relevant to that candidate.
6. Applies deterministic G0-G7 policy gates. Unknown values remain unknown.
7. Writes candidates, exact comparative-review inputs and validated responses,
   simplified reviews, critiques, gates, and a report once.
8. Snapshots every run artifact, appends its hash to a chained ledger, and
   verifies the completed run end to end.

There is deliberately no weighted “best business” score. Scenarios have
different economics, and missing values cannot safely be converted into zeros.
Once every natural-unit metric exists, the included Pareto comparator can expose
the nondominated candidates within a capital scenario.

## Run it

Requirements are Python 3.13, `uv`, and a configured provider key for
model-backed mode. The example below uses OpenAI directly; the inherited
OpenRouter route remains available. The included offline fixtures and tests need
no credential.

```bash
uv sync
export OPENAI_API_KEY='set this outside the repository'
export LLM_TRANSPORT=openai
export LLM_USE_CATALOGUE=false

uv run op venture seed /tmp/venture-evidence.json
uv run op venture scan /tmp/venture-evidence.json \
  --run-id my-first-venture-scan \
  --mode llm \
  --model openai/gpt-5.4-mini \
  --max-hypotheses 6 \
  --max-cost-usd 20

uv run op venture show my-first-venture-scan
uv run op venture verify my-first-venture-scan
```

`seed` and completed run artifacts are write-once. Reusing a run ID with
different inputs fails; a partial run is preserved and requires a new ID.

To stop an active run between operations, create `STOP` either at the output
root or inside that run directory.

## Important interpretation rules

- `PASS` is local to one gate. In particular, G3 passes only when each required
  falsification dimension reports no substantive contradiction; it still does
  not mean the business thesis is validated.
- `HOLD` is the expected result when direct willingness to pay, capacity,
  competition, or stressed unit economics are still missing.
- `KILL` is reserved for a verified, explicit disqualifier or a preregistered
  field-test failure. Model allegations cannot execute a kill.
- BLS cohort survival is establishment survival, not founder, firm, investment,
  or candidate-specific survival.
- Economic Census receipts and payroll ratios, and IRS aggregate income
  statistics, are screening proxies. They are not product margin.
- CMS certified beds and average residents are not live vacancies or available
  admissions.
- A legal mandate proves required work, not that a customer outsources it or
  will buy the proposed offer.

The bundled pilot packet is a reviewed, normalized official-source snapshot.
The run proves exactly which normalized rows the models saw and preserves those
rows immutably. It does **not** yet prove complete raw-download and transform
lineage for every row. Source adapters and parser tests exist, but raw-capture
provenance remains a separate production-hardening milestone.

The current `budget_usage` object records conservative preflight reservations
made by the venture layer. It is not a provider invoice or a complete audit of
transport-level retries. Use provider project limits as a second ceiling and do
not describe that field as observed spend.

## Why this is not a Wayfinder map

Wayfinder is useful one level above this system: it can hold unresolved
decisions, blockers, and human-in-the-loop research tickets across sessions.
It explicitly plans; it does not execute the evidence collection, typed
verification, immutable run, or field experiment.

The recommended split is:

- **Venture Lab:** evidence ingestion, hypothesis generation, classification,
  falsification, gates, experiment preregistration, and outcome calibration.
- **Wayfinder:** optional shared issue-tracker map of the remaining decisions.
- **Codex:** the execution and review environment.
- **GPU Cats:** optional operator interface later, not the source of truth.

Keeping the research engine standalone makes it testable, reproducible, and
portable. It can be surfaced inside GPU Cats later without coupling the evidence
ledger to a chat UI.

The generator, classifier, and falsifier are role-separated and receive
different scoped prompts, but may use the same configured model and transport.
This is blind role separation, not independent-model adjudication.

## Validate the implementation

```bash
uv run pytest -q
uv run ruff check app tests
uv run ruff format --check app tests
uv run mypy --strict app tests
```

The venture-specific tests live under `tests/venture/`.
