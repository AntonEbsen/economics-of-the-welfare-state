"""
Unit tests for src/clean/validation.py — Pandera schemas and validate_output.
"""

import logging

import pandas as pd
import pandera.pandas as pa
import pytest

from clean.validation import (
    master_schema,
    merged_panel_schema,
    validate_master_data,
    validate_merged_panel,
    validate_output,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_master_df() -> pd.DataFrame:
    """A minimal DataFrame satisfying master_schema."""
    return pd.DataFrame(
        {
            "iso3": ["USA", "GBR", "DEU"],
            "year": [2000, 2000, 2000],
            "sstran": [10.0, 12.0, 15.0],
            "ln_gdppc": [10.5, 10.3, 10.4],
            "inflation_cpi": [2.0, 1.8, 1.2],
            "deficit": [-3.0, -2.5, -1.0],
            "debt": [60.0, 80.0, 70.0],
            "ln_population": [19.5, 18.0, 18.2],
            "dependency_ratio": [20.0, 22.0, 30.0],
        }
    )


@pytest.fixture
def valid_merged_panel_df() -> pd.DataFrame:
    """A minimal DataFrame satisfying merged_panel_schema."""
    return pd.DataFrame(
        {
            "iso3": ["USA", "GBR"],
            "year": [2010, 2010],
            "sstran": [11.0, 13.0],
            "KOFGI": [82.0, 84.0],
            "KOFEcGI": [75.0, 78.0],
            "KOFSoGI": [85.0, 88.0],
            "KOFPoGI": [90.0, 92.0],
            "regime_conservative": [0, 0],
            "regime_mediterranean": [0, 0],
            "regime_liberal": [1, 1],
            "regime_post_communist": [0, 0],
        }
    )


# ---------------------------------------------------------------------------
# Pandera schemas
# ---------------------------------------------------------------------------


def test_validate_master_data_accepts_valid_frame(valid_master_df):
    """A well-formed frame passes master_schema validation."""
    validated = validate_master_data(valid_master_df)
    assert len(validated) == len(valid_master_df)
    assert set(validated.columns) == set(valid_master_df.columns)


def test_master_schema_rejects_bad_iso3():
    """iso3 codes must be exactly 3 characters."""
    bad = pd.DataFrame({"iso3": ["US"], "year": [2000]})
    with pytest.raises(pa.errors.SchemaError):
        master_schema.validate(bad)


def test_master_schema_rejects_pre_1960_year():
    """The master schema rejects rows before 1960."""
    bad = pd.DataFrame({"iso3": ["USA"], "year": [1950]})
    with pytest.raises(pa.errors.SchemaError):
        master_schema.validate(bad)


def test_master_schema_allows_extra_columns(valid_master_df):
    """strict=False means extra columns (e.g. regime dummies) pass through."""
    valid_master_df = valid_master_df.copy()
    valid_master_df["welfare_regime"] = "liberal"
    validated = master_schema.validate(valid_master_df)
    assert "welfare_regime" in validated.columns


def test_master_schema_allows_nullable_metrics():
    """Metric columns are nullable (real data has gaps)."""
    df = pd.DataFrame(
        {
            "iso3": ["USA"],
            "year": [2000],
            "sstran": [None],
            "ln_gdppc": [None],
            "inflation_cpi": [None],
            "deficit": [None],
            "debt": [None],
            "ln_population": [None],
            "dependency_ratio": [None],
        }
    )
    master_schema.validate(df)  # should not raise


def test_validate_merged_panel_accepts_valid_frame(valid_merged_panel_df):
    validated = validate_merged_panel(valid_merged_panel_df)
    assert len(validated) == len(valid_merged_panel_df)


def test_merged_panel_schema_rejects_bad_regime_dummy(valid_merged_panel_df):
    """regime_* dummies must be in {0, 1}."""
    bad = valid_merged_panel_df.copy()
    bad.loc[0, "regime_liberal"] = 2
    with pytest.raises(pa.errors.SchemaError):
        merged_panel_schema.validate(bad)


def test_merged_panel_schema_rejects_out_of_range_year(valid_merged_panel_df):
    """year must be within [1960, 2025]."""
    bad = valid_merged_panel_df.copy()
    bad.loc[0, "year"] = 1800
    with pytest.raises(pa.errors.SchemaError):
        merged_panel_schema.validate(bad)


# ---------------------------------------------------------------------------
# validate_output
# ---------------------------------------------------------------------------


def test_validate_output_passes_on_well_formed_frame(valid_master_df):
    """Happy path: required columns present, year range respected."""
    out = validate_output(
        valid_master_df,
        required_cols=["iso3", "year", "sstran"],
        dataset_name="TEST",
        year_min=1960,
        year_max=2025,
    )
    # Returns the input for chaining.
    assert out is valid_master_df


def test_validate_output_raises_on_missing_column(valid_master_df):
    """A missing required column is a hard error — merge cannot continue."""
    with pytest.raises(ValueError, match="missing required columns"):
        validate_output(
            valid_master_df,
            required_cols=["iso3", "year", "does_not_exist"],
            dataset_name="TEST",
        )


def test_validate_output_warns_on_year_out_of_range(valid_master_df, caplog):
    """Year values outside [year_min, year_max] log a warning (non-fatal)."""
    caplog.set_level(logging.WARNING, logger="clean.validation")
    validate_output(
        valid_master_df,
        required_cols=["iso3", "year"],
        dataset_name="TEST",
        year_min=2010,
        year_max=2020,
    )
    messages = " ".join(r.message for r in caplog.records)
    assert "below year_min" in messages


def test_validate_output_warns_when_country_count_wrong(valid_master_df, caplog):
    """expect_32_countries=True with only 3 iso3 codes should warn."""
    caplog.set_level(logging.WARNING, logger="clean.validation")
    validate_output(
        valid_master_df,
        required_cols=["iso3", "year"],
        dataset_name="TEST",
        expect_32_countries=True,
    )
    messages = " ".join(r.message for r in caplog.records)
    assert "expected 32 countries" in messages


def test_validate_output_is_a_noop_when_year_column_absent():
    """No 'year' column means year-range checks are silently skipped."""
    df = pd.DataFrame({"iso3": ["USA"], "value": [1.0]})
    # Should not raise and should not warn about year ranges.
    validate_output(
        df,
        required_cols=["iso3", "value"],
        dataset_name="TEST",
        year_min=2000,
        year_max=2020,
    )
