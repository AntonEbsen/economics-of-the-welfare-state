"""
``econ-clean`` — unified console script for the welfare-state pipeline.

Registered as a ``[project.scripts]`` entry point in ``pyproject.toml``.
Each subcommand is a thin wrapper over an existing Python function or
shell tool so that ``make`` targets, ``run_analysis.sh``/``.bat``, and
CI can all share one surface instead of each learning a different
``python -m …`` incantation.

Subcommands
-----------

``verify-data``
    SHA-256 check on ``data/raw/*.xlsx`` via
    ``scripts/download_raw_data.py``.

``clean``
    Full ``process_all_datasets`` + ``merge_all_datasets`` run.
    Options: ``--year-min``, ``--year-max``, ``--repo-root``,
    ``--no-save``.

``analyze``
    Load ``data/final/master_dataset.parquet`` and export stepwise,
    subperiod, and regime-heterogeneity tables/figures to
    ``outputs/``.

``report``
    Render the Quarto HTML report to ``_site/``.

``lock``
    Regenerate ``requirements-lock.txt`` via ``pip-compile``.

Examples
--------

::

    econ-clean verify-data
    econ-clean clean --year-min 2015 --year-max 2022
    econ-clean analyze
    econ-clean --help
"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

import typer

from clean.constants import DEFAULT_YEAR_MAX, DEFAULT_YEAR_MIN

app = typer.Typer(
    name="econ-clean",
    add_completion=False,
    no_args_is_help=True,
    help="Economics of the Welfare State — pipeline console script.",
)

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    # src/cli.py → src/ → repo_root/
    return Path(__file__).resolve().parent.parent


@app.command("verify-data")
def verify_data(
    fetch: bool = typer.Option(
        False, "--fetch", help="Download any MISSING files from the pinned URLs."
    ),
    show: bool = typer.Option(False, "--show", help="Print the manifest and exit."),
    strict: bool = typer.Option(
        True, "--strict/--no-strict", help="Fail on checksum mismatch (default) vs. warn only."
    ),
) -> None:
    """Verify ``data/raw/*.xlsx`` against the pinned SHA-256 manifest."""
    script = _repo_root() / "scripts" / "download_raw_data.py"
    cmd: list[str] = [sys.executable, str(script)]
    if fetch:
        cmd.append("--fetch")
    if show:
        cmd.append("--show")
    if not strict:
        cmd.append("--no-strict")
    raise typer.Exit(code=subprocess.call(cmd))


@app.command("clean")
def clean(
    year_min: int = typer.Option(DEFAULT_YEAR_MIN, "--year-min", help="Minimum year (inclusive)."),
    year_max: int = typer.Option(DEFAULT_YEAR_MAX, "--year-max", help="Maximum year (inclusive)."),
    repo_root: Path = typer.Option(
        None, "--repo-root", help="Repository root (defaults to the installed package's parent)."
    ),
    save: bool = typer.Option(
        True, "--save/--no-save", help="Write processed + merged outputs to disk."
    ),
) -> None:
    """Run the full data-cleaning + merge pipeline."""
    from clean.pipeline import process_all_datasets
    from clean.utils import setup_logging

    root = Path(repo_root) if repo_root is not None else _repo_root()
    setup_logging(root / "pipeline.log")
    logger.info("Starting clean pipeline: %s..%s (save=%s)", year_min, year_max, save)
    process_all_datasets(
        repo_root=root,
        year_min=year_min,
        year_max=year_max,
        save_outputs=save,
        validate=True,
    )


@app.command("analyze")
def analyze(
    master: Path = typer.Option(
        None,
        "--master",
        help="Path to master_dataset.parquet (defaults to data/final/master_dataset.parquet).",
    ),
) -> None:
    """Export stepwise, subperiod, and heterogeneity regression tables."""
    import pandas as pd

    from analysis.correlations import export_correlation_matrix
    from analysis.robustness import (
        export_baseline_regression_table,
        export_feedback_regression_table,
        export_interaction_excl_postcommunist_table,
        export_interaction_regression_table,
        export_marginal_effects_tables,
        export_stepwise_robustness_tables,
        export_subcomponent_regression_table,
        export_subperiod_heterogeneity_regressions,
        export_subperiod_regressions,
    )
    from analysis.trend_plots import plot_kof_trend, plot_sstran_trend
    from clean.panel_utils import add_welfare_regimes
    from clean.utils import load_config, setup_logging

    root = _repo_root()
    setup_logging(root / "pipeline.log")
    master_path = (
        Path(master) if master is not None else root / "data" / "final" / "master_dataset.parquet"
    )
    if not master_path.exists():
        typer.secho(
            f"Master dataset not found at {master_path}. Run `econ-clean clean` first.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1)

    config = load_config()
    panel = add_welfare_regimes(pd.read_parquet(master_path))
    tables_dir = root / "outputs" / "tables"
    figures_dir = root / "outputs" / "figures"
    export_baseline_regression_table(panel, config, out_dir=tables_dir)
    export_interaction_regression_table(panel, config, out_dir=tables_dir)
    export_interaction_excl_postcommunist_table(panel, config, out_dir=tables_dir)
    export_marginal_effects_tables(panel, config, out_dir=tables_dir)
    export_stepwise_robustness_tables(panel, config)
    export_subperiod_regressions(panel, config)
    export_subperiod_heterogeneity_regressions(panel, config)
    export_feedback_regression_table(panel, config, out_dir=tables_dir)
    export_correlation_matrix(panel, tables_dir)
    try:
        export_subcomponent_regression_table(panel, config, out_dir=tables_dir)
    except ValueError as exc:
        # Master panel may not include the finer-grained KOF sub-components
        # (KOFTrGI, KOFFiGI, etc.). Log and continue rather than failing
        # the whole analyze step.
        typer.secho(f"Skipping sub-component table: {exc}", fg=typer.colors.YELLOW, err=True)
    plot_sstran_trend(panel, figures_dir)
    plot_kof_trend(panel, figures_dir)
    typer.secho("Tables and figures saved to outputs/", fg=typer.colors.GREEN)


@app.command("report")
def report() -> None:
    """Render the Quarto HTML report to ``_site/``."""
    root = _repo_root()
    try:
        rc = subprocess.call(["quarto", "render"], cwd=root)
    except FileNotFoundError:
        typer.secho(
            "quarto not found on PATH. See https://quarto.org/docs/get-started/.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(code=1) from None
    raise typer.Exit(code=rc)


@app.command("lock")
def lock() -> None:
    """Regenerate ``requirements-lock.txt`` via ``pip-compile``."""
    root = _repo_root()
    # Ensure pip-tools is available; install quietly if missing.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "pip-tools"])
    rc = subprocess.call(
        [
            sys.executable,
            "-m",
            "piptools",
            "compile",
            "--extra=dev",
            "--output-file=requirements-lock.txt",
            "pyproject.toml",
        ],
        cwd=root,
    )
    raise typer.Exit(code=rc)


if __name__ == "__main__":
    app()
