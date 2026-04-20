"""
Panel data utilities for time series analysis.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def check_panel_balance(df: pd.DataFrame, id_var: str = "iso3", time_var: str = "year") -> dict:
    """
    Check if panel is balanced (all units have all time periods).

    Args:
        df: Panel DataFrame
        id_var: Panel identifier (default: 'iso3')
        time_var: Time identifier (default: 'year')

    Returns:
        Dictionary with balance information

    Example:
        >>> balance_info = check_panel_balance(master)
        >>> logger.info(f"Balanced: {balance_info['balanced']}")
    """
    # Get unique values
    units = df[id_var].unique()
    times = df[time_var].unique()

    # Count observations per unit
    obs_per_unit = df.groupby(id_var)[time_var].count()

    # Expected observations if balanced
    expected_obs = len(times)

    # Check if balanced
    balanced = all(obs_per_unit == expected_obs)

    result = {
        "balanced": balanced,
        "n_units": len(units),
        "n_periods": len(times),
        "expected_obs_per_unit": expected_obs,
        "min_obs": obs_per_unit.min(),
        "max_obs": obs_per_unit.max(),
        "mean_obs": obs_per_unit.mean(),
        "units_with_all_periods": sum(obs_per_unit == expected_obs),
    }

    if balanced:
        logger.info(
            f"✅ Panel is BALANCED: {result['n_units']} units × {result['n_periods']} periods"
        )
    else:
        logger.info("⚠️  Panel is UNBALANCED:")
        logger.info(f"   {result['n_units']} units, {result['n_periods']} periods")
        logger.info(
            f"   Observations per unit: {result['min_obs']}-{result['max_obs']} (mean: {result['mean_obs']:.1f})"
        )
        logger.info(f"   {result['units_with_all_periods']} units have all periods")

    return result


def create_lags(
    df: pd.DataFrame,
    variables: list[str],
    lags: list[int] = [1],
    id_var: str = "iso3",
    time_var: str = "year",
    strict: bool = True,
) -> pd.DataFrame:
    """
    Create lagged variables for panel data.

    Args:
        df: Panel DataFrame
        variables: List of variables to lag
        lags: List of lag periods (default: [1])
        id_var: Panel identifier
        time_var: Time identifier
        strict: When True (default), raise ValueError if any requested variable
            is missing from ``df``. Silent-skip in prior versions masked config
            typos and upstream schema drift — both produce regressions that run
            without the intended regressor.

    Returns:
        DataFrame with lagged variables added

    Example:
        >>> # Create 1-year and 2-year lags of GDP
        >>> df_with_lags = create_lags(master, ['ln_gdppc'], lags=[1, 2])
        >>> # Creates: ln_gdppc_lag1, ln_gdppc_lag2
    """
    missing = [v for v in variables if v not in df.columns]
    if missing:
        if strict:
            raise ValueError(
                f"create_lags: variables not in DataFrame: {missing}. "
                f"Pass strict=False to silently skip."
            )
        logger.warning("create_lags: variables not found, skipping: %s", missing)

    result = df.copy()

    # Sort by id and time
    result = result.sort_values([id_var, time_var])

    for var in variables:
        if var in missing:
            continue
        for lag in lags:
            lag_col = f"{var}_lag{lag}"
            result[lag_col] = result.groupby(id_var)[var].shift(lag)
            logger.info(f"✅ Created {lag_col}")

    return result


def create_leads(
    df: pd.DataFrame,
    variables: list[str],
    leads: list[int] = [1],
    id_var: str = "iso3",
    time_var: str = "year",
) -> pd.DataFrame:
    """
    Create lead variables for panel data (future values).

    Args:
        df: Panel DataFrame
        variables: List of variables to lead
        leads: List of lead periods (default: [1])
        id_var: Panel identifier
        time_var: Time identifier

    Returns:
        DataFrame with lead variables added
    """
    result = df.copy()
    result = result.sort_values([id_var, time_var])

    for var in variables:
        if var not in df.columns:
            logger.info(f"⚠️  Warning: Variable '{var}' not found, skipping")
            continue

        for lead in leads:
            lead_col = f"{var}_lead{lead}"
            result[lead_col] = result.groupby(id_var)[var].shift(-lead)
            logger.info(f"✅ Created {lead_col}")

    return result


def create_differences(
    df: pd.DataFrame, variables: list[str], id_var: str = "iso3", time_var: str = "year"
) -> pd.DataFrame:
    """
    Create first differences for panel data.

    Args:
        df: Panel DataFrame
        variables: List of variables to difference
        id_var: Panel identifier
        time_var: Time identifier

    Returns:
        DataFrame with differenced variables added

    Example:
        >>> df_with_diffs = create_differences(master, ['ln_gdppc'])
        >>> # Creates: d_ln_gdppc (first difference)
    """
    result = df.copy()
    result = result.sort_values([id_var, time_var])

    for var in variables:
        if var not in df.columns:
            logger.info(f"⚠️  Warning: Variable '{var}' not found, skipping")
            continue

        diff_col = f"d_{var}"
        result[diff_col] = result.groupby(id_var)[var].diff()
        logger.info(f"✅ Created {diff_col}")

    return result


def fill_panel_gaps(
    df: pd.DataFrame,
    method: str = "linear",
    id_var: str = "iso3",
    time_var: str = "year",
    limit: int = None,
) -> pd.DataFrame:
    """
    Fill missing years in panel data with interpolation.

    Args:
        df: Panel DataFrame
        method: Interpolation method ('linear', 'forward', 'backward')
        id_var: Panel identifier
        time_var: Time identifier
        limit: Maximum number of consecutive NaNs to fill

    Returns:
        DataFrame with filled gaps
    """
    # 1. Create a complete MultiIndex (all units x all periods)
    units = df[id_var].unique()
    times = range(df[time_var].min(), df[time_var].max() + 1)
    full_index = pd.MultiIndex.from_product([units, times], names=[id_var, time_var])

    # 2. Reindex the data
    result = df.set_index([id_var, time_var]).reindex(full_index).reset_index()
    result = result.sort_values([id_var, time_var])

    # 3. Fill gaps within each ID group
    if method == "forward":
        # ffill within groups, staying within each unit
        cols_to_fill = [c for c in result.columns if c not in [id_var, time_var]]
        result[cols_to_fill] = result.groupby(id_var)[cols_to_fill].ffill(limit=limit)
    elif method == "backward":
        cols_to_fill = [c for c in result.columns if c not in [id_var, time_var]]
        result[cols_to_fill] = result.groupby(id_var)[cols_to_fill].bfill(limit=limit)
    else:  # linear or other pandas methods
        # interpolate requires a numeric index or being applied per group
        cols_to_fill = [c for c in result.columns if c not in [id_var, time_var]]
        result[cols_to_fill] = (
            result.groupby(id_var)[cols_to_fill]
            .apply(lambda x: x.interpolate(method=method, limit=limit))
            .reset_index(drop=True)
        )

    logger.info(f"✅ Filled gaps using {method} interpolation (reindexed to full panel)")
    return result


def add_welfare_regimes(df: pd.DataFrame, id_var: str = "iso3") -> pd.DataFrame:
    """
    Categorize countries into welfare regimes and create indicator dummies.

    Welfare regimes included:
    - Liberal
    - Conservative (Corporatist)
    - Social Democrat
    - Mediterranean
    - Post-Communist

    Args:
        df: Panel DataFrame
        id_var: Country identifier (ISO3)

    Returns:
        DataFrame with indicator columns and 'welfare_regime' categorical column.
    """
    from .constants import WELFARE_REGIME_MAP

    result = df.copy()

    # 1. Create individual indicator columns (dummies)
    # Note: A country can belong to multiple regimes (e.g., Mediterranean countries
    # are often also in the Conservative list)
    for regime, list_iso3 in WELFARE_REGIME_MAP.items():
        col_name = f"regime_{regime.lower().replace(' ', '_').replace('-', '_')}"
        result[col_name] = result[id_var].isin(list_iso3).astype(int)
        logger.info(f"✅ Created indicator: {col_name}")

    # 2. Create a single 'welfare_regime' categorical column
    # We apply prioritizing logic for overlaps
    def get_regime(iso3):
        # Priority: Mediterranean > Post-Communist > Social Democrat > Conservative > Liberal
        # This highlights the specific Mediterranean sub-type over General Conservative
        if iso3 in WELFARE_REGIME_MAP.get("Mediterranean", []):
            return "Mediterranean"
        if iso3 in WELFARE_REGIME_MAP.get("Post-Communist", []):
            return "Post-Communist"
        if iso3 in WELFARE_REGIME_MAP.get("Social Democrat", []):
            return "Social Democrat"
        if iso3 in WELFARE_REGIME_MAP.get("Conservative", []):
            return "Conservative"
        if iso3 in WELFARE_REGIME_MAP.get("Liberal", []):
            return "Liberal"
        return "Other"

    result["welfare_regime"] = result[id_var].apply(get_regime)
    logger.info("✅ Created categorical column: welfare_regime")

    return result
