"""
Cross-country trend plots for publication.

Extracted from ``notebooks/02_modern_pipeline.ipynb`` (cells 32 & 34) so
the paper's headline time-series figures can be regenerated from the
pipeline, scripts, or CI without re-running the whole notebook. The
notebook still hosts the narrative; canonical implementations live
here.

Two public helpers:

``plot_sstran_trend``
    Cross-country mean of ``sstran`` (social security transfers, % GDP)
    plotted against year. Produces ``sstran_average.{png,pdf}``.

``plot_kof_trend``
    Cross-country mean of the four KOF globalization indices
    (``KOFGI``, ``KOFEcGI``, ``KOFSoGI``, ``KOFPoGI``) on a shared axis.
    Produces ``kof_indices_average.{png,pdf}``.

Both helpers return the aggregated :class:`pandas.DataFrame` used for
plotting so tests can assert on the numeric content without digging
into Matplotlib internals.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

DEFAULT_KOF_INDICES: list[str] = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]

DEFAULT_KOF_LABELS: dict[str, str] = {
    "KOFGI": "Total Globalization",
    "KOFEcGI": "Economic Globalization",
    "KOFSoGI": "Social Globalization",
    "KOFPoGI": "Political Globalization",
}

DEFAULT_KOF_COLORS: dict[str, str] = {
    "KOFGI": "#E11D48",
    "KOFEcGI": "#2563EB",
    "KOFSoGI": "#10B981",
    "KOFPoGI": "#F59E0B",
}


def _yearly_mean(df: pd.DataFrame, value_col: str, year_col: str = "year") -> pd.DataFrame:
    """Return a DataFrame with ``year``, ``mean``, and ``n`` columns.

    Non-numeric years and values are coerced and dropped. ``n`` is the
    count of non-null observations per year.
    """
    working = df[[year_col, value_col]].copy()
    working[year_col] = pd.to_numeric(working[year_col], errors="coerce")
    working[value_col] = pd.to_numeric(working[value_col], errors="coerce")
    working = working.dropna(subset=[year_col, value_col])
    return (
        working.groupby(year_col)[value_col]
        .agg(mean="mean", n="count")
        .reset_index()
        .sort_values(year_col)
        .reset_index(drop=True)
    )


def plot_sstran_trend(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    value_col: str = "sstran",
    year_col: str = "year",
    filename_stem: str = "sstran_average",
    figsize: tuple[float, float] = (11, 5),
) -> tuple[Path, Path, pd.DataFrame]:
    """Plot cross-country mean of ``sstran`` and save PNG + PDF.

    Parameters
    ----------
    df
        Panel DataFrame containing at least ``year_col`` and
        ``value_col``.
    out_dir
        Directory for output figures; created if missing.
    value_col, year_col
        Column names. Defaults match the master panel.
    filename_stem
        Basename (no extension) for the saved figures.
    figsize
        Matplotlib figure size.

    Returns
    -------
    tuple
        ``(png_path, pdf_path, aggregated_df)``. The aggregated frame
        has columns ``[year, mean, n]``.
    """
    agg = _yearly_mean(df, value_col=value_col, year_col=year_col)
    if agg.empty:
        raise ValueError(f"No non-null observations of {value_col!r} found after coercion")

    x_year = np.asarray(agg[year_col], dtype=float)
    y_mean = np.asarray(agg["mean"], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        x_year,
        y_mean,
        color="#2563EB",
        linewidth=2.5,
        marker="o",
        markersize=3.5,
        markeredgewidth=0,
        label="Cross-country mean",
    )
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Social Security Transfers (% GDP)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(framealpha=0.9, loc="upper left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_xlim(x_year.min(), x_year.max())

    n_min, n_max = int(agg["n"].min()), int(agg["n"].max())
    ax.annotate(
        f"N = {n_min}-{n_max} countries per year",
        xy=(0.02, 0.04),
        xycoords="axes fraction",
        fontsize=9,
        color="grey",
    )
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{filename_stem}.png"
    pdf_path = out_dir / f"{filename_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, agg


def plot_kof_trend(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    indices: list[str] | None = None,
    labels: dict[str, str] | None = None,
    colors: dict[str, str] | None = None,
    year_col: str = "year",
    filename_stem: str = "kof_indices_average",
    figsize: tuple[float, float] = (11, 6),
    y_limits: tuple[float, float] | None = (50, 90),
) -> tuple[Path, Path, pd.DataFrame]:
    """Plot cross-country mean of KOF indices and save PNG + PDF.

    Indices absent from ``df.columns`` are silently skipped so the
    helper also works on reduced master panels. Raises
    :class:`ValueError` if no requested index is present.

    Returns ``(png_path, pdf_path, aggregated_df)``, where the frame
    has columns ``[year] + available_indices``.
    """
    if indices is None:
        indices = list(DEFAULT_KOF_INDICES)
    if labels is None:
        labels = DEFAULT_KOF_LABELS
    if colors is None:
        colors = DEFAULT_KOF_COLORS

    available = [idx for idx in indices if idx in df.columns]
    if not available:
        raise ValueError(f"None of the requested KOF indices {indices!r} are columns of df")

    working = df[[year_col, *available]].copy()
    for col in [year_col, *available]:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    agg = (
        working.dropna(subset=available, how="all")
        .groupby(year_col)[available]
        .mean()
        .reset_index()
        .sort_values(year_col)
        .reset_index(drop=True)
    )
    if agg.empty:
        raise ValueError(f"No non-null observations across {available!r} after coercion")

    x_year = np.asarray(agg[year_col], dtype=float)

    fig, ax = plt.subplots(figsize=figsize)
    for idx in available:
        y_values = np.asarray(agg[idx], dtype=float)
        ax.plot(
            x_year,
            y_values,
            label=labels.get(idx, idx),
            color=colors.get(idx),
            linewidth=2.5,
            marker="o",
            markersize=3,
            markeredgewidth=0,
        )
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("KOF Indices of Globalization", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.legend(framealpha=0.9, loc="upper left", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.set_xlim(x_year.min(), x_year.max())
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    fig.tight_layout()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{filename_stem}.png"
    pdf_path = out_dir / f"{filename_stem}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path, agg
