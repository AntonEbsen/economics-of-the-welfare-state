"""
End-to-end smoke tests for the data-merging pipeline.

These tests don't read the real Excel inputs (that would require 7 MB of
source data and a working network for download); instead they feed a
synthetic ``results`` dict — the same shape that ``process_all_datasets()``
returns — through ``merge_all_datasets``, ``get_merge_summary``, and
``save_master_dataset``. This is enough to catch regressions in the merge
contract, the save formats, and the country/year indexing.
"""

from __future__ import annotations

import pandas as pd
import pytest

from clean.merge import get_merge_summary, merge_all_datasets, save_master_dataset
from clean.panel_utils import check_panel_balance, create_lags
from clean.validation import master_schema, validate_output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _panel_frame(cols: dict[str, list[float]], n_years: int = 4) -> pd.DataFrame:
    """Build a (iso3, year) panel for three countries for the last n_years."""
    iso = ["USA", "GBR", "DEU"]
    base_year = 2018
    rows = []
    for i, c in enumerate(iso):
        for t in range(n_years):
            row = {"iso3": c, "year": base_year + t}
            for col, values in cols.items():
                row[col] = values[i * n_years + t]
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_results_dict() -> dict:
    """Mimic the output of ``process_all_datasets`` with a tiny 3×4 panel."""
    cpds = _panel_frame(
        {
            "sstran": [
                10.0, 10.5, 11.0, 11.5,   # USA
                12.0, 12.5, 13.0, 13.5,   # GBR
                15.0, 15.5, 16.0, 16.5,   # DEU
            ],
            "deficit": [
                -3.0, -3.2, -3.1, -2.8,
                -2.0, -2.5, -2.1, -1.9,
                -1.0, -1.2, -0.8, -0.5,
            ],
            "debt": [
                80.0, 82.0, 85.0, 87.0,
                75.0, 78.0, 80.0, 82.0,
                60.0, 62.0, 64.0, 66.0,
            ],
        }
    )

    population = _panel_frame(
        {"ln_population": [
            19.50, 19.51, 19.52, 19.53,
            18.00, 18.01, 18.02, 18.03,
            18.20, 18.21, 18.22, 18.23,
        ]}
    )

    gdppc = _panel_frame(
        {"ln_gdppc": [
            10.80, 10.82, 10.84, 10.86,
            10.50, 10.52, 10.54, 10.56,
            10.60, 10.62, 10.64, 10.66,
        ]}
    )

    inflation = _panel_frame(
        {"inflation_cpi": [
            2.0, 1.8, 3.5, 4.0,
            1.5, 2.0, 4.0, 5.0,
            1.2, 1.4, 3.0, 4.2,
        ]}
    )

    dependency = _panel_frame(
        {"dependency_ratio": [
            20.0, 20.2, 20.4, 20.6,
            22.0, 22.1, 22.2, 22.3,
            30.0, 30.5, 31.0, 31.5,
        ]}
    )

    kof = _panel_frame(
        {
            "KOFGI": [80.0, 81.0, 82.0, 83.0, 85.0, 85.5, 86.0, 86.5, 87.0, 87.5, 88.0, 88.5],
            "KOFEcGI": [75.0, 75.5, 76.0, 76.5, 80.0, 80.5, 81.0, 81.5, 82.0, 82.5, 83.0, 83.5],
            "KOFSoGI": [82.0, 82.5, 83.0, 83.5, 85.0, 85.5, 86.0, 86.5, 88.0, 88.5, 89.0, 89.5],
            "KOFPoGI": [90.0, 90.5, 91.0, 91.5, 92.0, 92.5, 93.0, 93.5, 94.0, 94.5, 95.0, 95.5],
        }
    )

    return {
        "cpds": cpds,
        "population": population,
        "gdppc": gdppc,
        "inflation": inflation,
        "dependency": dependency,
        "kof": kof,
    }


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


def test_merge_produces_expected_columns(synthetic_results_dict):
    """The merged frame carries all source columns keyed on (iso3, year)."""
    master = merge_all_datasets(synthetic_results_dict, how="outer")

    expected = {
        "iso3",
        "year",
        "sstran",
        "deficit",
        "debt",
        "ln_population",
        "ln_gdppc",
        "inflation_cpi",
        "dependency_ratio",
        "KOFGI",
        "KOFEcGI",
        "KOFSoGI",
        "KOFPoGI",
    }
    assert expected.issubset(set(master.columns))


def test_merge_row_count_equals_panel_size(synthetic_results_dict):
    """3 countries × 4 years = 12 rows in the balanced output."""
    master = merge_all_datasets(synthetic_results_dict, how="outer")
    assert len(master) == 12
    assert master["iso3"].nunique() == 3
    assert master["year"].min() == 2018
    assert master["year"].max() == 2021


def test_merge_preserves_point_values(synthetic_results_dict):
    """Spot-check that per-country values land correctly after the join."""
    master = merge_all_datasets(synthetic_results_dict, how="outer")
    deu_2020 = master[(master["iso3"] == "DEU") & (master["year"] == 2020)].iloc[0]
    assert deu_2020["sstran"] == 16.0
    assert deu_2020["ln_gdppc"] == 10.64
    assert deu_2020["KOFGI"] == 88.0


def test_merged_panel_is_balanced(synthetic_results_dict):
    master = merge_all_datasets(synthetic_results_dict, how="outer")
    balance = check_panel_balance(master)
    assert balance["balanced"] is True
    assert balance["n_units"] == 3
    assert balance["n_periods"] == 4


# ---------------------------------------------------------------------------
# Post-merge validation
# ---------------------------------------------------------------------------


def test_merged_frame_passes_master_schema(synthetic_results_dict):
    """The merged output should satisfy the master Pandera schema."""
    master = merge_all_datasets(synthetic_results_dict, how="outer")
    # Schema has strict=False, so extra columns (KOF*) pass through.
    master_schema.validate(master)


def test_validate_output_accepts_merged_frame(synthetic_results_dict):
    master = merge_all_datasets(synthetic_results_dict, how="outer")
    validate_output(
        master,
        required_cols=["iso3", "year", "sstran", "KOFGI"],
        dataset_name="smoke-master",
        year_min=2018,
        year_max=2021,
    )


# ---------------------------------------------------------------------------
# Downstream utilities
# ---------------------------------------------------------------------------


def test_create_lags_on_merged_frame(synthetic_results_dict):
    """Lagging on the merged frame produces NaNs in year 0 per country."""
    master = merge_all_datasets(synthetic_results_dict, how="outer")
    lagged = create_lags(master, ["KOFGI", "sstran"], lags=[1])
    assert "KOFGI_lag1" in lagged.columns
    assert "sstran_lag1" in lagged.columns
    # First year per country must be NaN.
    usa_first = lagged[(lagged["iso3"] == "USA") & (lagged["year"] == 2018)].iloc[0]
    assert pd.isna(usa_first["KOFGI_lag1"])


def test_get_merge_summary_shape(synthetic_results_dict):
    master = merge_all_datasets(synthetic_results_dict, how="outer")
    summary = get_merge_summary(master)
    assert len(summary) == 3
    assert {"year_min", "year_max", "n_years"}.issubset(summary.columns)
    assert (summary["n_years"] == 4).all()


# ---------------------------------------------------------------------------
# Save pipeline
# ---------------------------------------------------------------------------


def test_save_master_dataset_writes_requested_formats(synthetic_results_dict, tmp_path):
    from pathlib import Path

    master = merge_all_datasets(synthetic_results_dict, how="outer")
    base = tmp_path / "master"
    written = save_master_dataset(master, str(base), formats=["parquet", "csv"])
    assert set(written.keys()) == {"parquet", "csv"}
    for path in written.values():
        assert Path(path).exists()
    # Round-trip the CSV to confirm it's readable and identical in shape.
    roundtripped = pd.read_csv(written["csv"])
    assert len(roundtripped) == len(master)
    assert set(master.columns) == set(roundtripped.columns)
