"""
Unit tests for merge utilities.
"""

import pandas as pd
import pytest

from clean.merge import get_merge_summary, merge_all_datasets


@pytest.fixture
def sample_datasets():
    """Create sample datasets for testing."""
    # CPDS data
    cpds = pd.DataFrame(
        {
            "iso3": ["USA", "USA", "GBR", "GBR"],
            "year": [2020, 2021, 2020, 2021],
            "sstran": [10.5, 11.0, 12.5, 13.0],
            "deficit": [-3.5, -4.0, -2.5, -3.0],
        }
    )

    # Population data
    population = pd.DataFrame(
        {
            "iso3": ["USA", "USA", "GBR", "GBR"],
            "year": [2020, 2021, 2020, 2021],
            "ln_population": [19.5, 19.51, 18.0, 18.01],
        }
    )

    # GDP data
    gdppc = pd.DataFrame(
        {
            "iso3": ["USA", "USA", "GBR", "GBR"],
            "year": [2020, 2021, 2020, 2021],
            "ln_gdppc": [10.8, 10.85, 10.5, 10.55],
        }
    )

    return {
        "cpds": cpds,
        "population": population,
        "gdppc": gdppc,
        "inflation": None,  # Simulate missing dataset
        "dependency": None,
    }


def test_merge_basic(sample_datasets):
    """Test basic merging functionality."""
    result = merge_all_datasets(sample_datasets, how="outer")

    # Check shape
    assert len(result) == 4, "Should have 4 rows after merging"

    # Check columns
    expected_cols = {"iso3", "year", "sstran", "deficit", "ln_population", "ln_gdppc"}
    assert set(result.columns) == expected_cols, "Missing or extra columns after merge"


def test_merge_preserves_data(sample_datasets):
    """Test that merge preserves original data."""
    result = merge_all_datasets(sample_datasets, how="outer")

    # Check a specific value
    usa_2020 = result[(result["iso3"] == "USA") & (result["year"] == 2020)]
    assert len(usa_2020) == 1
    assert usa_2020["sstran"].values[0] == 10.5
    assert usa_2020["ln_population"].values[0] == 19.5


def test_merge_with_none_datasets(sample_datasets):
    """Test that merge handles None datasets correctly."""
    # Should not fail even with None datasets
    result = merge_all_datasets(sample_datasets, how="outer")
    assert result is not None
    assert len(result) > 0


def test_get_merge_summary(sample_datasets):
    """Test merge summary function."""
    merged = merge_all_datasets(sample_datasets, how="outer")
    summary = get_merge_summary(merged)

    assert len(summary) == 2, "Should have summary for 2 countries"
    assert "year_min" in summary.columns
    assert "year_max" in summary.columns
    assert "n_years" in summary.columns
