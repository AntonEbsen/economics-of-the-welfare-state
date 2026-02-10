"""
Common utility functions for data processing.
Shared across all dataset cleaning modules.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .constants import TARGET_ISO3_32, COUNTRY_TO_ISO3


def map_country_to_iso3(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map country names to ISO3 codes.
    
    Args:
        df: DataFrame with 'country' column
        
    Returns:
        DataFrame with added 'iso3' column
    """
    out = df.copy()
    out["iso3"] = out["country"].map(COUNTRY_TO_ISO3)
    return out


def save_dataframe(df: pd.DataFrame, out_path: str | Path) -> Path:
    """
    Save DataFrame to parquet or CSV based on file extension.
    
    Args:
        df: DataFrame to save
        out_path: Output path with .parquet or .csv extension
        
    Returns:
        Path to saved file
        
    Raises:
        ValueError: If file extension is not .parquet or .csv
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    suffix = out_path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    elif suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        raise ValueError(f"Output must end with .parquet or .csv, got: {suffix}")
    
    return out_path


def filter_to_target_countries(
    df: pd.DataFrame,
    target_iso3: set[str] = TARGET_ISO3_32,
) -> pd.DataFrame:
    """
    Filter DataFrame to target ISO3 countries.
    
    Args:
        df: DataFrame with 'iso3' column
        target_iso3: Set of target ISO3 codes
        
    Returns:
        Filtered DataFrame containing only target countries
    """
    out = df.copy()
    
    # Drop unmapped countries
    out = out[~out["iso3"].isna()].copy()
    
    # Normalize ISO3 codes
    out["iso3"] = out["iso3"].astype(str).str.strip().str.upper()
    
    # Filter to target countries
    out = out[out["iso3"].isin(target_iso3)].copy()
    
    return out


def filter_to_year_range(
    df: pd.DataFrame,
    year_min: int | None = None,
    year_max: int | None = None,
) -> pd.DataFrame:
    """
    Filter DataFrame to year range.
    
    Args:
        df: DataFrame with 'year' column
        year_min: Minimum year (inclusive)
        year_max: Maximum year (inclusive)
        
    Returns:
        Filtered DataFrame within year range
    """
    out = df.copy()
    
    if year_min is not None:
        out = out[out["year"] >= year_min]
    
    if year_max is not None:
        out = out[out["year"] <= year_max]
    
    return out


def read_excel_robust(path: str | Path, sheet_name: str | int = 0) -> pd.DataFrame:
    """
    Read Excel file with basic error handling.
    
    Args:
        path: Path to Excel file
        sheet_name: Sheet name or index
        
    Returns:
        DataFrame with Excel data
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_excel(path, sheet_name=sheet_name)
