"""
Correlation matrix helpers with significance testing.

Extracted from ``notebooks/02_modern_pipeline.ipynb`` (cell 51) so that
the same logic is re-usable from the pipeline, from scripts, and from
tests. The notebook version is kept in place for exploratory narrative;
the canonical implementation lives here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Default display labels matching the notebook's output table.
DEFAULT_LAG_LABELS: dict[str, str] = {
    "KOFGI": "OG (t-1)",
    "KOFEcGI": "EG (t-1)",
    "KOFPoGI": "PG (t-1)",
    "KOFSoGI": "SG (t-1)",
    "ln_gdppc": "GDPpc (t-1)",
    "inflation_cpi": "Inf. (t-1)",
    "deficit": "Deficit (t-1)",
    "debt": "Gov. debt (t-1)",
    "ln_population": "Log pop. (t-1)",
    "dependency_ratio": "Dep. (t-1)",
}


def _pairwise_pvalues(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson-correlation p-value for every pair of columns.

    NaNs are dropped pairwise. Pairs with fewer than three non-NaN
    observations receive ``NaN`` instead of a p-value.
    """
    pvals = pd.DataFrame(index=df.columns, columns=df.columns, dtype=float)
    for r in df.columns:
        for c in df.columns:
            if r == c:
                pvals.loc[r, c] = 0.0
                continue
            mask = df[r].notna() & df[c].notna()
            if mask.sum() > 2:
                pvals.loc[r, c] = stats.pearsonr(df[r][mask], df[c][mask])[1]
            else:
                pvals.loc[r, c] = np.nan
    return pvals


def _format_with_stars(value: float, pvalue: float) -> str:
    if pd.isna(value) or pd.isna(pvalue):
        return ""
    base = f"{value:.2f}"
    if pvalue < 0.01:
        return base + "***"
    if pvalue < 0.05:
        return base + "**"
    if pvalue < 0.10:
        return base + "*"
    return base


def build_correlation_matrix(
    df: pd.DataFrame,
    dependent: str = "sstran",
    dependent_label: str = "WS",
    lag_labels: dict[str, str] | None = None,
    entity_col: str = "iso3",
    year_col: str = "year",
) -> pd.DataFrame:
    """Build a lower-triangle correlation table with significance stars.

    Parameters
    ----------
    df
        Panel in long format.
    dependent
        Name of the dependent-variable column (not lagged).
    dependent_label
        Display label for the dependent variable.
    lag_labels
        Mapping ``source_col -> display_label`` for columns to include
        at a 1-year lag. Defaults to :data:`DEFAULT_LAG_LABELS`.
    entity_col, year_col
        Panel key columns used to sort before the per-entity shift.

    Returns
    -------
    pandas.DataFrame
        Lower-triangle formatted strings (e.g. ``"0.37***"``). The
        upper triangle is empty and the diagonal is ``"1.00"``.
    """
    if lag_labels is None:
        lag_labels = DEFAULT_LAG_LABELS

    working = df.sort_values([entity_col, year_col]).copy()
    for col, label in lag_labels.items():
        if col in working.columns:
            working[label] = working.groupby(entity_col)[col].shift(1)
    working[dependent_label] = working[dependent]

    available = [label for col, label in lag_labels.items() if label in working.columns]
    subset = working[[dependent_label] + available]

    corr = subset.corr(method="pearson")
    pvals = _pairwise_pvalues(subset)

    formatted = pd.DataFrame(index=corr.index, columns=corr.columns, dtype=object)
    for i, r in enumerate(corr.index):
        for j, c in enumerate(corr.columns):
            if i == j:
                formatted.loc[r, c] = "1.00"
            elif j > i:
                formatted.loc[r, c] = ""
            else:
                formatted.loc[r, c] = _format_with_stars(corr.loc[r, c], pvals.loc[r, c])
    return formatted


def export_correlation_matrix(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    dependent: str = "sstran",
    dependent_label: str = "WS",
    lag_labels: dict[str, str] | None = None,
    caption: str = "Correlation Matrix",
    label: str = "tab:correlation_matrix",
) -> tuple[Path, Path]:
    """Build and write the correlation matrix to CSV and LaTeX.

    Returns the pair ``(csv_path, tex_path)``.
    """
    table = build_correlation_matrix(
        df,
        dependent=dependent,
        dependent_label=dependent_label,
        lag_labels=lag_labels,
    )
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "correlation_matrix.csv"
    tex_path = out_dir / "correlation_matrix.tex"
    table.to_csv(csv_path)
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(
            table.to_latex(
                caption=caption,
                label=label,
                column_format="l" + "c" * len(table.columns),
                position="htbp",
            )
        )
    return csv_path, tex_path
