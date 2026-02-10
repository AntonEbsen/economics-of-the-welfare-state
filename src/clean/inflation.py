"""
Inflation CPI cleaning utilities
Processes inflation CPI data for the same 32 countries used in the main analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import re

import pandas as pd

# Import from centralized constants
from .constants import TARGET_ISO3_32, COUNTRY_TO_ISO3


def read_inflation_excel(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read Inflation CPI Excel file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_excel(path, sheet_name=sheet_name)


def standardize_inflation_to_long(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert inflation data from wide to long format.
    Handles World Bank-style format with years as columns.
    Returns: country, year, inflation_cpi
    """
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    # Find country column
    country_candidates = [
        c for c in df.columns 
        if c.strip().lower() in {
            "country", "location", "country name", "country_name",
            "reference area", "reference_area", "ref_area"
        }
    ]
    
    if not country_candidates:
        raise ValueError(
            f"Could not find country column. Available: {list(df.columns)}"
        )
    
    country_col = country_candidates[0]
    
    # Find year columns (4-digit numbers, possibly with [YR...] suffix)
    year_pat = re.compile(r"^(\d{4})(?:\s*\[YR\d{4}\])?\s*$")
    year_cols = []
    for c in df.columns:
        match = year_pat.match(str(c))
        if match:
            year_cols.append(c)
    
    if not year_cols:
        # Fallback: just 4-digit columns
        year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]
    
    if not year_cols:
        raise ValueError("No year columns found (expected format: '1980' or '1980 [YR1980]')")
    
    # Melt to long format
    df_long = df.melt(
        id_vars=[country_col],
        value_vars=year_cols,
        var_name="year_raw",
        value_name="inflation_cpi"
    )
    
    df_long = df_long.rename(columns={country_col: "country"})
    
    # Extract year from year column
    df_long["year"] = (
        df_long["year_raw"]
        .astype(str)
        .str.extract(r"^(\d{4})", expand=False)
    )
    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
    
    # Clean inflation values
    df_long["inflation_cpi"] = pd.to_numeric(df_long["inflation_cpi"], errors="coerce")
    
    # Clean country names
    df_long["country"] = df_long["country"].astype(str).str.strip()
    
    # Drop rows with missing keys
    df_long = df_long.dropna(subset=["country", "year"]).reset_index(drop=True)
    
    return df_long[["country", "year", "inflation_cpi"]]


def map_country_to_iso3(df_long: pd.DataFrame) -> pd.DataFrame:
    """Map country names to ISO3 codes."""
    out = df_long.copy()
    out["iso3"] = out["country"].map(COUNTRY_TO_ISO3)
    return out


def filter_32_countries(
    df_mapped: pd.DataFrame,
    year_min: int | None = 1980,
    year_max: int | None = 2023,
    target_iso3: set[str] = TARGET_ISO3_32,
) -> pd.DataFrame:
    """
    Filter inflation data to:
    - The same 32 countries used in the analysis
    - Year range (default 1980-2023)
    - Keep only: iso3, year, inflation_cpi
    """
    out = df_mapped.copy()
    
    # Filter years
    if year_min is not None:
        out = out[out["year"] >= year_min]
    if year_max is not None:
        out = out[out["year"] <= year_max]
    
    # Drop unmapped countries
    out = out[~out["iso3"].isna()].copy()
    
    # Filter to 32 countries
    out["iso3"] = out["iso3"].astype(str).str.strip().str.upper()
    out = out[out["iso3"].isin(target_iso3)].copy()
    
    # Keep only essential columns
    out = out[["iso3", "year", "inflation_cpi"]].copy()
    
    # Sort by iso3 and year
    out = out.sort_values(["iso3", "year"]).reset_index(drop=True)
    
    return out


def save_inflation(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Save processed inflation data to parquet or CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError("Output must end with .parquet or .csv")
    
    return out_path
