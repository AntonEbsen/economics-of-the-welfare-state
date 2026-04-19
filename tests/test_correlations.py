"""
Smoke-level tests for ``analysis.correlations`` — the lifted-from-notebook
correlation matrix helpers. These lock the shape and significance-stars
convention so changes in the notebook can't silently drift from the
module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.correlations import (
    DEFAULT_LAG_LABELS,
    build_correlation_matrix,
    export_correlation_matrix,
)


def _synthetic_panel(seed: int = 0) -> pd.DataFrame:
    """Two countries × 25 years with engineered correlations to sstran."""
    rng = np.random.default_rng(seed)
    rows = []
    for iso3 in ("DNK", "DEU"):
        for year in range(1995, 2020):
            x = rng.normal(0, 1)
            sstran = 0.5 * x + rng.normal(0, 0.3)
            rows.append(
                {
                    "iso3": iso3,
                    "year": year,
                    "sstran": sstran,
                    "KOFGI": x + rng.normal(0, 0.1),
                    "KOFEcGI": x * 0.8 + rng.normal(0, 0.2),
                    "ln_gdppc": rng.normal(10, 0.1),
                    "inflation_cpi": rng.normal(2, 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_build_correlation_matrix_is_lower_triangular_with_stars():
    df = _synthetic_panel()
    labels = {
        "KOFGI": "OG (t-1)",
        "KOFEcGI": "EG (t-1)",
        "ln_gdppc": "GDPpc (t-1)",
    }
    tbl = build_correlation_matrix(df, lag_labels=labels)

    # Shape: dependent + 3 lagged labels
    assert tbl.shape == (4, 4)
    # Diagonal is "1.00"
    for i in range(4):
        assert tbl.iloc[i, i] == "1.00"
    # Upper triangle is empty
    for i in range(4):
        for j in range(i + 1, 4):
            assert tbl.iloc[i, j] == ""
    # At least one significant star somewhere in the lower triangle
    flat = [tbl.iloc[i, j] for i in range(4) for j in range(i)]
    assert any("*" in cell for cell in flat)


def test_build_correlation_matrix_skips_absent_columns():
    df = _synthetic_panel()
    # 'KOFSoGI' is not in the df; helper should silently drop it.
    labels = {"KOFGI": "OG (t-1)", "KOFSoGI": "SG (t-1)"}
    tbl = build_correlation_matrix(df, lag_labels=labels)
    # Dependent + 1 actually-present lagged var = 2×2
    assert tbl.shape == (2, 2)
    assert "SG (t-1)" not in tbl.columns


def test_default_lag_labels_stable():
    # The labels feed into the paper's table — lock the public contract.
    assert DEFAULT_LAG_LABELS["KOFGI"] == "OG (t-1)"
    assert DEFAULT_LAG_LABELS["ln_gdppc"] == "GDPpc (t-1)"


def test_export_correlation_matrix_writes_both_formats(tmp_path):
    df = _synthetic_panel()
    labels = {"KOFGI": "OG (t-1)", "ln_gdppc": "GDPpc (t-1)"}
    csv_path, tex_path = export_correlation_matrix(df, tmp_path, lag_labels=labels)
    assert csv_path.exists() and csv_path.stat().st_size > 0
    assert tex_path.exists() and tex_path.stat().st_size > 0
    tex = tex_path.read_text(encoding="utf-8")
    assert "tab:correlation_matrix" in tex
    assert "OG (t-1)" in tex
