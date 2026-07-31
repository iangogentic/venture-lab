"""The validation scaffold: `op validate scaffold` and its deterministic renderer.

The guarantees under test are the feature's ethics made executable: the page
always carries the honest banner, never names a person, quotes only verbatim
cluster quotes attributed by platform alone, and the plan's thresholds are
pre-registered blanks — plus the plain mechanics of where files land.
"""

from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from app.artifacts import ArtifactKind, ArtifactRegistry, Opportunity
from app.cli.main import app
from app.utils.paths import WorkspacePaths
from app.validation import (
    CALCOM_PLACEHOLDER,
    HONEST_BANNER,
    LISTMONK_PLACEHOLDER,
    POSTHOG_PLACEHOLDER,
)
from tests.factories import make

runner = CliRunner()

RUN = "r1"

QUOTE = "Our CI takes 40 minutes and the team hates it."
AUTHOR = "jane_doe"


def _seed(
    registry: ArtifactRegistry,
    *,
    quotes: list[str] | None = None,
    with_cluster: bool = True,
    with_decision: bool = True,
) -> Opportunity:
    """A cluster with quotes, its opportunity, backing evidence, and a decision."""
    cluster = None
    if with_cluster:
        cluster = make(ArtifactKind.PAIN_CLUSTER, run_id=RUN, quotes=quotes or [QUOTE])
        registry.save(cluster)

    fields: dict[str, Any] = {"run_id": RUN}
    if cluster is not None:
        fields["pain_cluster"] = cluster.ref
    opportunity = make(ArtifactKind.OPPORTUNITY, **fields)
    assert isinstance(opportunity, Opportunity)
    registry.save(opportunity)

    evidence = make(
        ArtifactKind.EVIDENCE,
        run_id=RUN,
        collector="hacker-news",
        author=AUTHOR,
        excerpt=QUOTE,
    )
    registry.save(evidence)

    if with_decision:
        registry.save(
            make(
                ArtifactKind.DECISION,
                run_id=RUN,
                opportunity=opportunity.ref,
                biggest_unknown="Whether teams will pay rather than self-host.",
                next_validation_step="Interview five platform leads about build waits.",
            )
        )
    return opportunity


def _scaffold_dir(workspace: WorkspacePaths, opportunity: Opportunity) -> Path:
    return workspace.root / "validation" / opportunity.id


# ------------------------------------------------------------------ the files


def test_scaffold_writes_three_files_in_the_default_workspace_dir(
    workspace: WorkspacePaths,
) -> None:
    opportunity = _seed(ArtifactRegistry(workspace))

    result = runner.invoke(app, ["validate", "scaffold", opportunity.id])

    assert result.exit_code == 0
    target = _scaffold_dir(workspace, opportunity)
    for name in ("index.html", "validation-plan.md", "README.md"):
        assert (target / name).is_file(), f"missing {name}"


def test_scaffold_out_option_writes_elsewhere(workspace: WorkspacePaths) -> None:
    opportunity = _seed(ArtifactRegistry(workspace))
    target = workspace.root / "elsewhere"

    result = runner.invoke(app, ["validate", "scaffold", opportunity.id, "--out", str(target)])

    assert result.exit_code == 0
    assert (target / "index.html").is_file()
    assert not _scaffold_dir(workspace, opportunity).exists()


# ----------------------------------------------------------------- the page


def test_scaffold_page_carries_the_honest_banner_and_placeholders(
    workspace: WorkspacePaths,
) -> None:
    """The page may never imply an existing product, and the founder's stack is
    left as named placeholders, not guessed URLs."""
    opportunity = _seed(ArtifactRegistry(workspace))

    runner.invoke(app, ["validate", "scaffold", opportunity.id])

    page = (_scaffold_dir(workspace, opportunity) / "index.html").read_text(encoding="utf-8")
    assert HONEST_BANNER in page
    assert LISTMONK_PLACEHOLDER in page
    assert POSTHOG_PLACEHOLDER in page
    assert CALCOM_PLACEHOLDER in page
    assert opportunity.problem in page
    assert opportunity.workflow in page


def test_scaffold_quotes_are_verbatim_platform_only_no_usernames(
    workspace: WorkspacePaths,
) -> None:
    """The person consented to their words on the platform, not to being named here."""
    opportunity = _seed(ArtifactRegistry(workspace))

    runner.invoke(app, ["validate", "scaffold", opportunity.id])

    page = (_scaffold_dir(workspace, opportunity) / "index.html").read_text(encoding="utf-8")
    assert QUOTE in page
    assert "a developer on hacker-news" in page
    assert AUTHOR not in page


def test_scaffold_caps_the_quotes_at_three(workspace: WorkspacePaths) -> None:
    quotes = [QUOTE, "Second complaint here.", "Third complaint here.", "Fourth complaint here."]
    opportunity = _seed(ArtifactRegistry(workspace), quotes=quotes)

    runner.invoke(app, ["validate", "scaffold", opportunity.id])

    page = (_scaffold_dir(workspace, opportunity) / "index.html").read_text(encoding="utf-8")
    assert "Third complaint here." in page
    assert "Fourth complaint here." not in page


def test_scaffold_without_cluster_still_scaffolds(workspace: WorkspacePaths) -> None:
    """No quotes is a thinner page, never a refusal."""
    opportunity = _seed(ArtifactRegistry(workspace), with_cluster=False)

    result = runner.invoke(app, ["validate", "scaffold", opportunity.id])

    assert result.exit_code == 0
    page = (_scaffold_dir(workspace, opportunity) / "index.html").read_text(encoding="utf-8")
    assert HONEST_BANNER in page
    assert QUOTE not in page


# ----------------------------------------------------------------- the plan


def test_scaffold_plan_preregisters_thresholds_and_carries_the_decision(
    workspace: WorkspacePaths,
) -> None:
    opportunity = _seed(ArtifactRegistry(workspace))

    runner.invoke(app, ["validate", "scaffold", opportunity.id])

    plan = (_scaffold_dir(workspace, opportunity) / "validation-plan.md").read_text(
        encoding="utf-8"
    )
    assert "BEFORE the page goes live" in plan
    assert "waitlist conversion" in plan
    assert "waitlist size" in plan
    assert "Interviews booked" in plan
    assert "concrete commitment" in plan
    assert "Review date" in plan
    assert "Whether teams will pay rather than self-host." in plan
    assert "Interview five platform leads about build waits." in plan


def test_scaffold_plan_without_a_decision_says_so(workspace: WorkspacePaths) -> None:
    opportunity = _seed(ArtifactRegistry(workspace), with_decision=False)

    result = runner.invoke(app, ["validate", "scaffold", opportunity.id])

    assert result.exit_code == 0
    plan = (_scaffold_dir(workspace, opportunity) / "validation-plan.md").read_text(
        encoding="utf-8"
    )
    assert "No decision has been recorded" in plan


def test_scaffold_readme_names_the_self_host_stack(workspace: WorkspacePaths) -> None:
    opportunity = _seed(ArtifactRegistry(workspace))

    runner.invoke(app, ["validate", "scaffold", opportunity.id])

    notes = (_scaffold_dir(workspace, opportunity) / "README.md").read_text(encoding="utf-8")
    assert "listmonk" in notes
    assert "PostHog" in notes
    assert "Cal.com" in notes
    assert "personally, by a human" in notes


# ----------------------------------------------------------------- refusals


def test_scaffold_unknown_id_exits_non_zero(workspace: WorkspacePaths) -> None:
    result = runner.invoke(app, ["validate", "scaffold", "op_nope"])
    assert result.exit_code == 1
    assert "op_nope" in result.output


def test_scaffold_non_opportunity_id_exits_non_zero(workspace: WorkspacePaths) -> None:
    registry = ArtifactRegistry(workspace)
    cluster = make(ArtifactKind.PAIN_CLUSTER, run_id=RUN)
    registry.save(cluster)

    result = runner.invoke(app, ["validate", "scaffold", cluster.id])

    assert result.exit_code == 1
    assert "not an opportunity" in result.output
