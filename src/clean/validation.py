"""
Data validation utilities for cleaned datasets.
"""
from __future__ import annotations

import pandas as pd
from .constants import TARGET_ISO3_32, DEFAULT_YEAR_MIN, DEFAULT_YEAR_MAX


def validate_output(
    df: pd.DataFrame,
    required_cols: list[str],
    dataset_name: str = "dataset",
    year_min: int = DEFAULT_YEAR_MIN,
    year_max: int = DEFAULT_YEAR_MAX,
    expect_32_countries: bool = True,
) -> None:
    """
    Validate that a cleaned dataset meets quality standards.
    
    Args:
        df: The dataframe to validate
        required_cols: List of required column names
        dataset_name: Name of dataset for error messages
        year_min: Expected minimum year
        year_max: Expected maximum year
        expect_32_countries: If True, expect exactly 32 countries
        
    Raises:
        AssertionError: If validation fails
    """
    # Check columns exist
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise AssertionError(
            f"{dataset_name}: Missing required columns: {sorted(missing_cols)}. "
            f"Available columns: {sorted(df.columns)}"
        )
    
    # Check not empty
    if len(df) == 0:
        raise AssertionError(f"{dataset_name}: Output is empty (0 rows)")
    
    # Check iso3 column
    if "iso3" in required_cols:
        # Check all iso3 values are in TARGET_ISO3_32
        invalid_iso3 = set(df["iso3"].unique()) - TARGET_ISO3_32
        if invalid_iso3:
            raise AssertionError(
                f"{dataset_name}: Contains ISO3 codes not in TARGET_ISO3_32: {sorted(invalid_iso3)}"
            )
        
        # Check country count
        n_countries = df["iso3"].nunique()
        if expect_32_countries and n_countries != 32:
            raise AssertionError(
                f"{dataset_name}: Expected 32 countries, got {n_countries}. "
                f"Missing: {sorted(TARGET_ISO3_32 - set(df['iso3'].unique()))}"
            )
    
    # Check year column
    if "year" in required_cols:
        actual_min = int(df["year"].min())
        actual_max = int(df["year"].max())
        
        if actual_min < year_min:
            raise AssertionError(
                f"{dataset_name}: Minimum year {actual_min} is before expected {year_min}"
            )
        
        if actual_max > year_max:
            raise AssertionError(
                f"{dataset_name}: Maximum year {actual_max} is after expected {year_max}"
            )
    
    # Check for duplicates on (iso3, year)
    if "iso3" in required_cols and "year" in required_cols:
        n_dupes = df.duplicated(subset=["iso3", "year"]).sum()
        if n_dupes > 0:
            raise AssertionError(
                f"{dataset_name}: Found {n_dupes} duplicate (iso3, year) rows"
            )
    
    # Check for null values
    # ID cols must be strictly non-null
    id_cols = [c for c in ["iso3", "year"] if c in required_cols]
    if id_cols:
        null_ids = df[id_cols].isnull().sum()
        if null_ids.any():
            raise AssertionError(
                f"{dataset_name}: Found null values in ID columns (iso3/year): {null_ids[null_ids > 0].to_dict()}"
            )
    
    # Data cols can have nulls but we warn
    data_cols = [c for c in required_cols if c not in ["iso3", "year"]]
    if data_cols:
        null_data = df[data_cols].isnull().sum()
        if null_data.any():
            cols_with_nulls = null_data[null_data > 0].to_dict()
            print(f"⚠️  {dataset_name}: Missing data detected in: {cols_with_nulls}")
    
    print(f"✅ {dataset_name} validation passed: {len(df)} rows, {df['iso3'].nunique() if 'iso3' in df.columns else 'N/A'} countries")
