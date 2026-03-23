"""
Dependency Ratio cleaning utilities
Processes dependency ratio data for the same 32 countries used in the main analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Import from centralized constants
from .constants import TARGET_ISO3_32
from .utils import (
    filter_to_target_countries,
    filter_to_year_range,
    read_excel_robust,
    save_dataframe,
)
from .worldbank import WorldBankProcessor


def read_dependency_excel(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read Dependency Ratio Excel file."""
    return read_excel_robust(path, sheet_name)


def standardize_dependency_to_long(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert dependency ratio data from wide to long format.
    Handles World Bank-style format with years as columns.
    Returns: country, year, dependency_ratio
    """
    return WorldBankProcessor.wide_to_long(df_raw, value_name="dependency_ratio")


# Removed: Now using shared map_country_to_iso3 from utils.py


def filter_32_countries(
    df_mapped: pd.DataFrame,
    year_min: int | None = 1980,
    year_max: int | None = 2023,
    target_iso3: set[str] = TARGET_ISO3_32,
) -> pd.DataFrame:
    """
    Filter dependency ratio data to:
    - The same 32 countries used in the analysis
    - Year range (default 1980-2023)
    - Keep only: iso3, year, dependency_ratio
    """
    # Use shared utility functions
    out = filter_to_year_range(df_mapped, year_min, year_max)
    out = filter_to_target_countries(out, target_iso3)

    # Keep only essential columns
    out = out[["iso3", "year", "dependency_ratio"]].copy()

    # Sort by iso3 and year
    out = out.sort_values(["iso3", "year"]).reset_index(drop=True)

    return out


def save_dependency(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Save processed dependency ratio data to parquet or CSV."""
    return save_dataframe(df, out_path)
