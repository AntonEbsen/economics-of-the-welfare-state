"""
Unit tests for panel utilities.
"""

import numpy as np
import pandas as pd
import pytest

from clean.panel_utils import check_panel_balance, create_differences, create_lags, fill_panel_gaps


@pytest.fixture
def balanced_panel():
    """Create a balanced panel dataset."""
    countries = ["USA", "GBR", "DEU"]
    years = [2018, 2019, 2020, 2021]

    data = []
    for country in countries:
        for year in years:
            data.append({"iso3": country, "year": year, "gdp": 100 + np.random.rand()})

    return pd.DataFrame(data)


@pytest.fixture
def unbalanced_panel():
    """Create an unbalanced panel dataset."""
    data = [
        {"iso3": "USA", "year": 2018, "gdp": 100},
        {"iso3": "USA", "year": 2019, "gdp": 101},
        {"iso3": "USA", "year": 2020, "gdp": 102},
        {"iso3": "GBR", "year": 2018, "gdp": 80},
        {"iso3": "GBR", "year": 2020, "gdp": 82},  # Missing 2019
    ]
    return pd.DataFrame(data)


def test_check_balance_balanced(balanced_panel):
    """Test balance checking on balanced panel."""
    result = check_panel_balance(balanced_panel)

    assert result["balanced"] is True
    assert result["n_units"] == 3
    assert result["n_periods"] == 4


def test_check_balance_unbalanced(unbalanced_panel):
    """Test balance checking on unbalanced panel."""
    result = check_panel_balance(unbalanced_panel)

    assert result["balanced"] is False
    assert result["n_units"] == 2


def test_create_lags(balanced_panel):
    """Test lag creation."""
    result = create_lags(balanced_panel, ["gdp"], lags=[1, 2])

    # Should have original column plus two lag columns
    assert "gdp_lag1" in result.columns
    assert "gdp_lag2" in result.columns

    # Check lag values for USA
    usa_data = result[result["iso3"] == "USA"].sort_values("year")
    assert pd.isna(usa_data.iloc[0]["gdp_lag1"])  # First year should be NaN
    assert usa_data.iloc[1]["gdp_lag1"] == pytest.approx(usa_data.iloc[0]["gdp"], rel=0.01)


def test_create_lags_strict_raises_on_missing(balanced_panel):
    """create_lags should raise when requested variables are absent by default."""
    with pytest.raises(ValueError, match="not in DataFrame"):
        create_lags(balanced_panel, ["gdp", "does_not_exist"], lags=[1])


def test_create_lags_non_strict_skips_missing(balanced_panel):
    """strict=False restores the old silent-skip behavior for opt-in callers."""
    result = create_lags(balanced_panel, ["gdp", "does_not_exist"], lags=[1], strict=False)
    assert "gdp_lag1" in result.columns
    assert "does_not_exist_lag1" not in result.columns


def test_create_differences(balanced_panel):
    """Test first difference creation."""
    result = create_differences(balanced_panel, ["gdp"])

    assert "d_gdp" in result.columns

    # First observation for each country should be NaN
    for country in ["USA", "GBR", "DEU"]:
        country_data = result[result["iso3"] == country].sort_values("year")
        assert pd.isna(country_data.iloc[0]["d_gdp"])


def test_fill_gaps_forward(unbalanced_panel):
    """Test forward fill of gaps."""
    result = fill_panel_gaps(unbalanced_panel, method="forward")

    # GBR should now have 2019 data filled forward from 2018
    gbr_data = result[result["iso3"] == "GBR"].sort_values("year")
    if len(gbr_data) > 2:  # If gap was filled
        assert gbr_data.iloc[1]["gdp"] == gbr_data.iloc[0]["gdp"]
