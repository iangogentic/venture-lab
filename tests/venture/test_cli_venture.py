"""The `op venture` command surface."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from app.cli.main import app
from tests.venture.test_pilot import _fixture, _packet

runner = CliRunner()


def test_venture_help_exposes_scan_show_and_verify() -> None:
    result = runner.invoke(app, ["venture", "--help"])
    assert result.exit_code == 0
    assert "seed" in result.output
    assert "scan" in result.output
    assert "show" in result.output
    assert "verify" in result.output


def test_seed_writes_official_packet_atomically_and_is_idempotent(tmp_path: Path) -> None:
    packet_path = tmp_path / "nested" / "official-packet.json"
    arguments = ["venture", "seed", str(packet_path), "--json"]
    first = runner.invoke(app, arguments)
    second = runner.invoke(app, arguments)

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    payload = json.loads(first.stdout)
    assert payload["packet_id"] == "pilot-official-2026-07-31-v2"
    assert payload["measurements"] >= 200
    assert len(payload["sha256"]) == 64
    assert packet_path.is_file()
    assert packet_path.read_bytes().endswith(b"\n")
    assert json.loads(second.stdout)["sha256"] == payload["sha256"]


def test_seed_refuses_to_overwrite_different_content(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text('{"different":true}', encoding="utf-8")
    result = runner.invoke(app, ["venture", "seed", str(packet_path)])
    assert result.exit_code == 1
    assert "refusing to overwrite different content" in result.output


def test_offline_scan_and_verify_emit_pipeable_json(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    fixture_path = tmp_path / "fixture.json"
    output_root = tmp_path / "pilot"
    packet_path.write_text(_packet().model_dump_json(), encoding="utf-8")
    fixture_path.write_text(_fixture().model_dump_json(), encoding="utf-8")

    scan = runner.invoke(
        app,
        [
            "venture",
            "scan",
            str(packet_path),
            "--run-id",
            "cli-run",
            "--fixture",
            str(fixture_path),
            "--output-root",
            str(output_root),
            "--json",
        ],
    )
    assert scan.exit_code == 0, scan.output
    payload = json.loads(scan.stdout)
    assert payload["result"]["run_id"] == "cli-run"
    assert payload["result"]["candidates"][0]["gates"]["decision"] == "hold"

    verify = runner.invoke(
        app,
        [
            "venture",
            "verify",
            "cli-run",
            "--output-root",
            str(output_root),
            "--json",
        ],
    )
    assert verify.exit_code == 0, verify.output
    assert json.loads(verify.stdout)["valid"] is True


def test_offline_scan_without_fixture_fails_without_traceback(tmp_path: Path) -> None:
    packet_path = tmp_path / "packet.json"
    packet_path.write_text(_packet().model_dump_json(), encoding="utf-8")
    result = runner.invoke(
        app,
        ["venture", "scan", str(packet_path), "--run-id", "missing-fixture"],
    )
    assert result.exit_code == 1
    assert "offline mode requires an explicit fixture" in result.output
    assert "Traceback" not in result.output
