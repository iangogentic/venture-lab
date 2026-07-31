"""`op venture` — run and inspect the evidence-governed venture pilot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer

from app.cli.render import print_json
from app.utils.console import console, err_console
from app.utils.errors import OpportunityEngineError
from app.utils.paths import get_workspace_paths
from app.venture.core import sha256_bytes
from app.venture.operations import BudgetPolicy
from app.venture.pilot import (
    PilotError,
    PilotMode,
    load_evidence_packet,
    load_offline_fixture,
    load_pilot_execution,
    report_path,
    run_pilot,
    verify_pilot_run,
)
from app.venture.pilot_evidence import (
    build_pilot_evidence_packet,
    pilot_evidence_packet_json,
)

app = typer.Typer(
    name="venture",
    help="Run evidence packets through bounded hypothesis, falsification, and G0-G7 gates.",
    no_args_is_help=True,
)

JsonOption = Annotated[bool, typer.Option("--json", help="Emit JSON and nothing else.")]
OutputRootOption = Annotated[
    Path | None,
    typer.Option(
        "--output-root",
        help="Immutable pilot store. Defaults to <WORKSPACE_DIR>/venture.",
        file_okay=False,
        resolve_path=True,
    ),
]


@app.command("seed")
def seed(
    packet_path: Annotated[
        Path,
        typer.Argument(
            help="Exact path for the frozen official EvidencePacket JSON.",
            dir_okay=False,
            resolve_path=True,
        ),
    ],
    as_json: JsonOption = False,
) -> None:
    """Write the reviewed official pilot packet once, without shell redirection."""
    packet = build_pilot_evidence_packet()
    content = pilot_evidence_packet_json(packet).encode("utf-8")
    try:
        _write_once(packet_path, content)
    except (OSError, PilotError) as exc:
        err_console.print(f"[danger]{exc}[/danger]")
        raise typer.Exit(code=1) from exc
    payload = {
        "packet_id": packet.packet_id,
        "measurements": len(packet.measurements),
        "sha256": sha256_bytes(content),
        "path": str(packet_path),
    }
    if as_json:
        print_json(payload)
        return
    console.print(
        f"[success]seeded[/success] [stage]{packet.packet_id}[/stage] · "
        f"{len(packet.measurements)} measurements · {packet_path}"
    )


@app.command("scan")
def scan(
    packet_path: Annotated[
        Path,
        typer.Argument(
            help="Normalized EvidencePacket JSON.",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="Safe immutable run identifier."),
    ],
    mode: Annotated[
        PilotMode,
        typer.Option(
            "--mode",
            help="'offline' uses a fixture; 'llm' makes bounded paid model calls.",
        ),
    ] = PilotMode.OFFLINE,
    fixture_path: Annotated[
        Path | None,
        typer.Option(
            "--fixture",
            help="OfflinePilotFixture JSON. Required in offline mode.",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Optional model slug for explicit LLM mode."),
    ] = None,
    max_hypotheses: Annotated[
        int,
        typer.Option("--max-hypotheses", min=1, max=25, help="Hard candidate ceiling."),
    ] = 12,
    max_cost_usd: Annotated[
        float,
        typer.Option("--max-cost-usd", min=0.0, help="Preflight model-cost reservation ceiling."),
    ] = 25.0,
    output_root: OutputRootOption = None,
    as_json: JsonOption = False,
) -> None:
    """Run one packet without performing any external commercial action."""
    root = _output_root(output_root)
    try:
        packet = load_evidence_packet(packet_path)
        fixture = load_offline_fixture(fixture_path) if fixture_path is not None else None
        execution = run_pilot(
            packet=packet,
            output_root=root,
            run_id=run_id,
            mode=mode,
            fixture=fixture,
            model=model,
            max_hypotheses=max_hypotheses,
            budget_policy=BudgetPolicy(max_cost_usd=max_cost_usd),
        )
    except (PilotError, OpportunityEngineError, OSError, ValueError) as exc:
        err_console.print(f"[danger]{exc}[/danger]")
        raise typer.Exit(code=1) from exc

    if as_json:
        print_json(execution.model_dump(mode="json"))
        return
    console.print(
        f"[success]completed[/success] [stage]{execution.result.run_id}[/stage] · "
        f"{len(execution.result.candidates)} candidates · no master score"
    )
    for item in execution.result.candidates:
        console.print(
            f"  [{_decision_style(item.gates.decision.value)}]"
            f"{item.gates.decision.value.upper()}[/] {item.hypothesis.title} "
            f"[muted]({item.hypothesis.scenario.value})[/muted]"
        )
    console.print(f"[muted]report[/muted] {root / execution.report_relative_path}")


@app.command("show")
def show(
    run_id: Annotated[str, typer.Argument(help="Completed pilot run identifier.")],
    output_root: OutputRootOption = None,
    as_json: JsonOption = False,
) -> None:
    """Show a completed result or its human-readable Markdown report."""
    root = _output_root(output_root)
    try:
        execution = load_pilot_execution(output_root=root, run_id=run_id)
        path = report_path(output_root=root, run_id=run_id)
    except (PilotError, OSError, ValueError) as exc:
        err_console.print(f"[danger]{exc}[/danger]")
        raise typer.Exit(code=1) from exc
    if as_json:
        print_json(execution.model_dump(mode="json"))
        return
    typer.echo(path.read_text(encoding="utf-8"))


@app.command("verify")
def verify(
    run_id: Annotated[str, typer.Argument(help="Completed pilot run identifier.")],
    output_root: OutputRootOption = None,
    as_json: JsonOption = False,
) -> None:
    """Verify file hashes, snapshots, ledger events, and the hash chain."""
    try:
        verification = verify_pilot_run(
            output_root=_output_root(output_root),
            run_id=run_id,
        )
    except (PilotError, OSError, ValueError) as exc:
        err_console.print(f"[danger]{exc}[/danger]")
        raise typer.Exit(code=1) from exc
    if as_json:
        print_json(verification.model_dump(mode="json"))
        return
    console.print(
        f"[success]valid[/success] [stage]{verification.run_id}[/stage] · "
        f"{verification.artifact_count} immutable artifacts · "
        f"{verification.ledger_event_count} ledger events"
    )


def _output_root(value: Path | None) -> Path:
    return (
        value.expanduser().resolve()
        if value is not None
        else (get_workspace_paths().root / "venture").resolve()
    )


def _decision_style(value: str) -> str:
    return {"pass": "success", "hold": "warning", "kill": "danger"}[value]


def _write_once(path: Path, content: bytes) -> None:
    """Create one exact file atomically; identical retries are idempotent."""
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(resolved, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        try:
            existing = resolved.read_bytes()
        except OSError as exc:
            raise PilotError(f"existing packet {resolved} is unreadable") from exc
        if existing != content:
            raise PilotError(f"refusing to overwrite different content at {resolved}") from None
        return
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(resolved, 0o444)
    except BaseException:
        try:
            resolved.unlink(missing_ok=True)
        finally:
            raise


__all__ = ["app"]
