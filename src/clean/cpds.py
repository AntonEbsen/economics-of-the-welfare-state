"""
CPDS (Comparative Political Data Set) cleaning utilities
Extracts social spending transfers (sstran), deficit, and debt data
for the same 32 countries used in the main analysis.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

# Import the same 32 countries from kofgi
from clean.kofgi import TARGET_ISO3_32


def read_cpds_excel(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """Read CPDS Excel file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_excel(path, sheet_name=sheet_name)


def standardize_cpds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize CPDS data:
    - Clean country and year columns
    - Keep only required variables: sstran, deficit, debt
    """
    out = df.copy()
    
    # Ensure year is integer type
    if "year" in out.columns:
        out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")
    
    # Clean country name
    if "country" in out.columns:
        out["country"] = out["country"].astype(str).str.strip()
    
    # Clean iso3 code
    if "iso" in out.columns:
        out["iso3"] = out["iso"].astype(str).str.strip().str.upper()
    
    return out


def filter_cpds_32countries(
    df: pd.DataFrame,
    year_min: int | None = 1980,
    year_max: int | None = 2023,
    target_iso3: set[str] = TARGET_ISO3_32,
) -> pd.DataFrame:
    """
    Filter CPDS data to:
    - The same 32 countries used in the analysis
    - Year range (default 1980-2023)
    - Keep only: country, year, sstran, deficit, debt
    """
    out = df.copy()
    
    # Filter countries (assuming 'iso3' or 'iso' column exists)
    if "iso3" in out.columns:
        out = out[out["iso3"].isin(target_iso3)].copy()
    elif "iso" in out.columns:
        # If only 'iso' exists, create iso3 and filter
        out["iso3"] = out["iso"].astype(str).str.strip().str.upper()
        out = out[out["iso3"].isin(target_iso3)].copy()
    
    # Filter years
    if "year" in out.columns:
        if year_min is not None:
            out = out[out["year"] >= year_min]
        if year_max is not None:
            out = out[out["year"] <= year_max]
    
    # Keep only required columns
    required_cols = ["country", "year"]
    data_cols = ["sstran", "deficit", "debt"]
    
    # Check which columns exist
    existing_cols = required_cols.copy()
    for col in data_cols:
        if col in out.columns:
            existing_cols.append(col)
    
    out = out[existing_cols].copy()
    
    # Sort by country and year
    out = out.sort_values(["country", "year"]).reset_index(drop=True)
    
    return out


def save_cpds(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Save processed CPDS data to parquet or CSV."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    if out_path.suffix.lower() == ".parquet":
        df.to_parquet(out_path, index=False)
    elif out_path.suffix.lower() == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError("Output must end with .parquet or .csv")
    
    return out_path
