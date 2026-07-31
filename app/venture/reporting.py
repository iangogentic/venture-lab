"""Plain Markdown rendering for immutable venture-pilot results."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from app.venture.core import GateDecision

if TYPE_CHECKING:
    from app.venture.pilot import PilotCandidateResult, PilotRunResult


def render_pilot_report(result: PilotRunResult) -> str:
    """Render evidence, falsification, and every G0-G7 predicate without ranking."""
    lines = [
        "# Venture evidence pilot",
        "",
        f"- Run: `{result.run_id}`",
        f"- Evidence packet: `{result.packet_id}`",
        f"- Information cutoff: {result.information_cutoff.isoformat()}",
        f"- Analyst mode: `{result.mode.value}`",
        f"- Candidates: {len(result.candidates)}",
        "",
        "> Research-only boundary: this run did not contact customers or investors, "
        "send outreach, buy ads or data, accept deposits, or spend outside its bounded "
        "model calls. No master score or single-winner ranking is computed.",
        "",
        "## Decision register",
        "",
        "| Candidate | Provider NAICS | Offer market | Context markets | "
        "Classification gate | Scenario | Policy decision | Gate path |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for item in result.candidates:
        path = " · ".join(
            f"{gate.gate.value}:{gate.decision.value.upper()}" for gate in item.gates.results
        )
        lines.append(
            "| "
            + " | ".join(
                (
                    _cell(item.hypothesis.title),
                    item.hypothesis.naics_codes[0],
                    item.hypothesis.offer_market_topic.value,
                    _cell(
                        ", ".join(topic.value for topic in item.hypothesis.context_market_topics)
                        or "none"
                    ),
                    _classification_gate_label(item),
                    _cell(item.hypothesis.scenario.value),
                    item.gates.decision.value.upper(),
                    _cell(path),
                )
            )
            + " |"
        )

    for index, item in enumerate(result.candidates, start=1):
        lines.extend(_candidate_section(index, item))

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `PASS` means every visible predicate at that gate passed.",
            "- `HOLD` means at least one prerequisite is unmet or unknown; unknowns are "
            "never converted to zero.",
            "- `KILL` is reserved for a verified explicit disqualifier or a "
            "preregistered field-test kill result.",
            "- Model-authored disqualifier allegations are listed for verification and "
            "cannot execute a kill by themselves.",
            "- A G3 predicate is satisfied only by `NO_CONTRADICTION_FOUND`; "
            "`WEAKENS`, `CONTRADICTS`, and `UNRESOLVED` remain `HOLD` even when "
            "coverage was completed.",
            "",
        ]
    )
    return "\n".join(lines)


def _candidate_section(index: int, item: PilotCandidateResult) -> list[str]:
    hypothesis = item.hypothesis
    lines = [
        "",
        f"## {index}. {_text(hypothesis.title)}",
        "",
        f"- Policy decision: **{item.gates.decision.value.upper()}**",
        f"- Thesis fingerprint: `{hypothesis.thesis_id}`",
        f"- Customer: {_text(hypothesis.customer)}",
        f"- Payer: {_text(hypothesis.payer)}",
        f"- Entity scope: {_text(hypothesis.entity_scope)}",
        f"- Geography: {_join(hypothesis.geography)}",
        f"- Provider/business NAICS: `{hypothesis.naics_codes[0]}`",
        f"- Offer market topic: `{hypothesis.offer_market_topic.value}`",
        "- Context market topics: "
        + (
            _join(topic.value for topic in hypothesis.context_market_topics)
            if hypothesis.context_market_topics
            else "none"
        ),
        "- Customer NAICS: "
        + (
            _join(hypothesis.customer_naics_codes)
            if hypothesis.customer_naics_codes
            else "none (consumer or government buyers may have no business NAICS)"
        ),
        f"- NAICS classification status: `{hypothesis.classification_status.value}`",
        f"- NAICS basis: {_text(hypothesis.naics_basis)}",
        f"- Scenario: `{hypothesis.scenario.value}`",
        f"- Archetype: `{hypothesis.archetype.value}`",
        f"- Problem: {_text(hypothesis.problem)}",
        f"- Mechanism: {_text(hypothesis.mechanism)}",
        f"- Business model: {_text(hypothesis.business_model)}",
        f"- Contestable-spend basis: {_text(hypothesis.contestable_spend_basis)}",
        f"- Adjacent markets excluded: {_join(hypothesis.adjacent_market_exclusions)}",
        f"- Why now: {_text(hypothesis.reason_for_now)}",
        f"- Critical assumptions: {_join(hypothesis.critical_assumptions)}",
        "- Disconfirming observations: " + _join(hypothesis.disconfirming_observations),
        "",
        "### Independent provider-classification gate",
    ]
    if item.classification_review is not None:
        review = item.classification_review
        lines.extend(
            [
                "",
                f"- Gate outcome: **{review.outcome.value.upper()}**",
                f"- Assignment: `{review.assignment_id}`; provider code "
                f"`{review.naics_code}`; official scope ref "
                f"`{review.scope_measurement_ref}`.",
                f"- Basis: {_text(review.analysis)}",
            ]
        )
        if review.mismatches:
            lines.append("- Scope mismatches: " + _join(review.mismatches) + ".")
        if review.missing_evidence:
            lines.append(
                "- Classification evidence missing: " + _join(review.missing_evidence) + "."
            )
        if review.outcome.value != "fit":
            lines.append(
                "- Policy effect: classification failed closed; market falsification "
                "was skipped and the cohort continued."
            )
    else:
        classification_failure = item.classification_review_failure
        assert classification_failure is not None
        lines.extend(
            [
                "",
                f"- Gate outcome: **FAIL-CLOSED ({_text(classification_failure.kind.value)})**",
                f"- Assignment `{classification_failure.assignment_id}` returned "
                f"`{_text(classification_failure.error_type)}`: "
                f"{_text(classification_failure.message)}",
                "- Policy effect: classification unresolved; market falsification "
                "was skipped and the cohort continued.",
            ]
        )

    lines.extend(["", "### Evidence cited", ""])
    for measurement in item.evidence:
        source = _safe_source_links(measurement.source_url)
        value = "unknown" if measurement.value is None else str(measurement.value)
        lines.append(
            f"- `{measurement.measurement_id}` — {_text(measurement.metric)}: "
            f"{_text(value)} {_text(measurement.unit)}; {_text(measurement.geography)}, "
            f"{_text(measurement.observed_period)}; {source}. "
            f"Caveat: {_text(measurement.caveat)}"
        )

    lines.extend(["", "### Independent falsification", ""])
    if item.falsification is None:
        classification_fit = (
            item.classification_review is not None
            and item.classification_review.outcome.value == "fit"
        )
        if not classification_fit:
            lines.append(
                "- Skipped: provider classification did not receive FIT; every G3 "
                "predicate remains unknown."
            )
        elif item.falsification_failure is None:
            lines.append("- Not run in this fixture; every G3 check remains unknown.")
        else:
            falsification_failure = item.falsification_failure
            lines.extend(
                [
                    f"- **Quarantined failure "
                    f"({_text(falsification_failure.kind.value)}).** "
                    f"Assignment `{falsification_failure.assignment_id}` returned "
                    f"`{_text(falsification_failure.error_type)}`: "
                    f"{_text(falsification_failure.message)}",
                    "- Policy effect: no falsification report was accepted; every G3 "
                    "predicate remains unknown and the cohort continued.",
                ]
            )
    else:
        for finding in item.falsification.findings:
            cited = _join(finding.evidence_refs) if finding.evidence_refs else "none"
            lines.append(
                f"- **{_text(finding.dimension.value)} — "
                f"{_text(finding.outcome.value)}.** {_text(finding.analysis)} "
                f"Evidence refs: {cited}."
            )
            if finding.missing_evidence:
                lines.append(f"  Missing: {_join(finding.missing_evidence)}.")
        if item.falsification.critical_unknowns:
            lines.append(
                "- Critical unknowns: " + _join(item.falsification.critical_unknowns) + "."
            )
    if item.unverified_disqualifier_allegations:
        lines.append(
            "- Unverified disqualifier allegations: "
            + _join(item.unverified_disqualifier_allegations)
            + ". These require independent verification."
        )

    lines.extend(
        [
            "",
            "### G0-G7 policy",
            "",
            "| Gate | Decision | Unknown or unmet predicates |",
            "|---|---|---|",
        ]
    )
    for gate in item.gates.results:
        pending = tuple(
            predicate.name for predicate in gate.predicates if predicate.satisfied is not True
        )
        lines.append(
            f"| {gate.gate.value} | {gate.decision.value.upper()} | "
            f"{_cell(', '.join(pending) if pending else 'none')} |"
        )

    next_checks = _pending_requirements(item)
    lines.extend(["", "### Next evidence needed", ""])
    lines.extend(f"- {_text(requirement)}" for requirement in next_checks)
    return lines


def _pending_requirements(item: PilotCandidateResult) -> tuple[str, ...]:
    requirements: list[str] = []
    for gate in item.gates.results:
        if gate.decision is GateDecision.PASS:
            continue
        requirements.extend(
            predicate.requirement
            for predicate in gate.predicates
            if predicate.satisfied is not True
        )
    return tuple(dict.fromkeys(requirements))


def _classification_gate_label(item: PilotCandidateResult) -> str:
    if item.classification_review is not None:
        return item.classification_review.outcome.value.upper()
    return "FAIL-CLOSED"


def _safe_source_links(value: str) -> str:
    """Render the catalog's explicit multi-source delimiter as separate links."""
    parts = tuple(part.strip() for part in value.split(" | ") if part.strip())
    if not parts:
        return ""
    return ", ".join(
        _safe_source_link(part, index=index, total=len(parts))
        for index, part in enumerate(parts, start=1)
    )


def _safe_source_link(value: str, *, index: int = 1, total: int = 1) -> str:
    cleaned = value.strip()
    parsed = urlsplit(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return _text(cleaned)
    safe = cleaned.replace("<", "%3C").replace(">", "%3E")
    label = "source" if total == 1 else f"source {index}"
    return f"[{label}](<{safe}>)"


def _join(values: Iterable[str]) -> str:
    return ", ".join(_text(value) for value in values)


def _cell(value: str) -> str:
    return _text(value).replace("|", "\\|")


def _text(value: str) -> str:
    escaped_html = html.escape(" ".join(value.split()), quote=False)
    return re.sub(r"([\\`*_\[\]()!#])", r"\\\1", escaped_html)


__all__ = ["render_pilot_report"]
