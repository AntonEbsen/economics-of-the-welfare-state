"""
Tests for ``analysis.trend_plots`` — the publication-ready cross-country
trend figures. Lifted from ``notebooks/02_modern_pipeline.ipynb`` cells
32 (sstran trend) and 34 (KOF indices trend).

The tests use a non-interactive Matplotlib backend via ``matplotlib.use``
so they stay CI-safe; the helpers themselves are expected to close their
figures to avoid the ``RuntimeWarning: More than 20 figures have been
opened`` during the full test run.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # noqa: E402 — must precede pyplot-importing modules

import numpy as np
import pandas as pd
import pytest

from analysis.trend_plots import (
    DEFAULT_KOF_COLORS,
    DEFAULT_KOF_INDICES,
    DEFAULT_KOF_LABELS,
    plot_kof_trend,
    plot_sstran_trend,
)


def _synthetic_panel(seed: int = 7) -> pd.DataFrame:
    """6 countries × 20 years carrying sstran + the four KOF indices."""
    rng = np.random.default_rng(seed)
    countries = [f"C{i:02d}" for i in range(6)]
    years = list(range(2000, 2020))
    rows = []
    for c in countries:
        for t in years:
            rows.append(
                {
                    "iso3": c,
                    "year": t,
                    "sstran": 15.0 + rng.normal(0, 2),
                    "KOFGI": 60.0 + (t - 2000) * 0.5 + rng.normal(0, 1),
                    "KOFEcGI": 65.0 + rng.normal(0, 1),
                    "KOFSoGI": 55.0 + rng.normal(0, 1),
                    "KOFPoGI": 70.0 + rng.normal(0, 1),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# sstran trend plot
# ---------------------------------------------------------------------------


def test_plot_sstran_trend_writes_png_and_pdf(tmp_path):
    df = _synthetic_panel()
    png_path, pdf_path, agg = plot_sstran_trend(df, tmp_path)
    assert png_path.exists() and png_path.stat().st_size > 0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    assert png_path.name == "sstran_average.png"
    assert pdf_path.name == "sstran_average.pdf"


def test_plot_sstran_trend_aggregates_to_one_row_per_year(tmp_path):
    df = _synthetic_panel()
    _, _, agg = plot_sstran_trend(df, tmp_path)
    # 20 distinct years in the synthetic panel.
    assert len(agg) == 20
    assert list(agg.columns) == ["year", "mean", "n"]
    # 6 countries every year.
    assert (agg["n"] == 6).all()
    # Mean of a ~15-centred draw should be close to 15.
    assert 10 < agg["mean"].mean() < 20


def test_plot_sstran_trend_raises_on_empty_input(tmp_path):
    df = pd.DataFrame({"year": [], "sstran": []})
    with pytest.raises(ValueError, match="No non-null observations"):
        plot_sstran_trend(df, tmp_path)


# ---------------------------------------------------------------------------
# KOF trend plot
# ---------------------------------------------------------------------------


def test_default_kof_indices_are_the_four_headline_indices():
    assert DEFAULT_KOF_INDICES == ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
    assert set(DEFAULT_KOF_LABELS) == set(DEFAULT_KOF_INDICES)
    assert set(DEFAULT_KOF_COLORS) == set(DEFAULT_KOF_INDICES)


def test_plot_kof_trend_writes_png_and_pdf(tmp_path):
    df = _synthetic_panel()
    png_path, pdf_path, agg = plot_kof_trend(df, tmp_path)
    assert png_path.name == "kof_indices_average.png"
    assert pdf_path.name == "kof_indices_average.pdf"
    assert png_path.exists() and png_path.stat().st_size > 0
    assert pdf_path.exists() and pdf_path.stat().st_size > 0
    # All four indices survive the aggregation.
    for idx in DEFAULT_KOF_INDICES:
        assert idx in agg.columns


def test_plot_kof_trend_skips_missing_indices(tmp_path):
    df = _synthetic_panel().drop(columns=["KOFPoGI"])
    _, _, agg = plot_kof_trend(df, tmp_path)
    assert "KOFPoGI" not in agg.columns
    assert "KOFGI" in agg.columns


def test_plot_kof_trend_raises_when_no_indices_present(tmp_path):
    df = _synthetic_panel().drop(columns=list(DEFAULT_KOF_INDICES))
    with pytest.raises(ValueError, match="None of the requested KOF indices"):
        plot_kof_trend(df, tmp_path)


def test_plot_kof_trend_accepts_subset_via_indices_arg(tmp_path):
    df = _synthetic_panel()
    _, _, agg = plot_kof_trend(df, tmp_path, indices=["KOFGI"])
    assert list(agg.columns) == ["year", "KOFGI"]
