"""
Smoke tests for the ``econ-clean`` Typer console script.

We don't re-test the science here — the underlying helpers already have
dedicated unit tests. Instead we pin the public surface: subcommand
registration, ``--help`` text, and the error path when a required
input is missing. These are the cheapest tests that catch regressions
like an import typo, a dropped subcommand, or a typer.Exit that silently
turns into a zero exit code.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_help_lists_all_subcommands(runner: CliRunner):
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("verify-data", "clean", "analyze", "report", "lock"):
        assert cmd in result.stdout, f"{cmd!r} missing from --help output"


def test_analyze_help_mentions_master_option(runner: CliRunner):
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "--master" in result.stdout


def test_analyze_without_master_dataset_exits_1(runner: CliRunner, tmp_path: Path, monkeypatch):
    """Missing master parquet → red error + exit code 1, not a traceback."""
    missing = tmp_path / "does_not_exist.parquet"
    result = runner.invoke(app, ["analyze", "--master", str(missing)])
    assert result.exit_code == 1
    # stderr message may land in stdout under typer's default runner;
    # accept either channel.
    combined = result.stdout + (result.stderr or "")
    assert "not found" in combined.lower()


def test_clean_help_mentions_year_range(runner: CliRunner):
    result = runner.invoke(app, ["clean", "--help"])
    assert result.exit_code == 0
    assert "--year-min" in result.stdout
    assert "--year-max" in result.stdout
