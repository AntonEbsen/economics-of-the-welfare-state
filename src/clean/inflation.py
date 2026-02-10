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
    """
    Read Inflation CPI Excel file with extreme robustness.
    Includes smart header detection to skip metadata rows.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    
    # helper for smart header detection
    def find_header_row(data_rows):
        keywords = {"country", "reference area", "ref_area", "iso3", "location"}
        for i, row in enumerate(data_rows[:20]):
            row_str = [str(c).lower().strip() for c in row if c is not None]
            if any(k in row_str for k in keywords):
                return i
        return 0

    # 1. Try Calamine first (most robust against style corruption)
    try:
        try:
            from python_calamine import CalamineWorkbook
        except ImportError:
            from calamine import CalamineWorkbook
            
        workbook = CalamineWorkbook.from_path(str(path))
        sheet_names = workbook.sheet_names
        target = sheet_names[sheet_name] if isinstance(sheet_name, int) else sheet_name
        data = workbook.get_sheet_by_name(target).to_python()
        
        if data:
            header_idx = find_header_row(data)
            df = pd.DataFrame(data[header_idx + 1:], columns=data[header_idx])
            return df
    except Exception as e:
        print(f"⚠️  Calamine read failed: {e}")

    # 2. Try standard pandas read with manual header detection
    try:
        # Load a few rows to find header
        preview = pd.read_excel(path, sheet_name=sheet_name, nrows=20, header=None)
        data_preview = preview.values.tolist()
        header_idx = find_header_row(data_preview)
        
        # Reload with correct header
        return pd.read_excel(path, sheet_name=sheet_name, header=header_idx)
    except Exception as e:
        error_msg = str(e)
        print(f"⚠️  Pandas read failed: {error_msg}")
        
    # 3. CRITICAL FALLBACK: Style Stripping (if it's a style error)
    if "expected" in error_msg and "Fill" in error_msg:
        import zipfile
        import tempfile
        print("🛠️  Corrupted Excel styles detected. Cleaning file metadata...")
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir) / "cleaned_data.xlsx"
                with zipfile.ZipFile(path, 'r') as zin:
                    with zipfile.ZipFile(tmp_path, 'w') as zout:
                        for item in zin.infolist():
                            if item.filename != 'xl/styles.xml':
                                zout.writestr(item, zin.read(item.filename))
                
                # Try reading the cleaned version with header detection
                preview = pd.read_excel(tmp_path, sheet_name=sheet_name, nrows=20, header=None, engine='openpyxl')
                header_idx = find_header_row(preview.values.tolist())
                return pd.read_excel(tmp_path, sheet_name=sheet_name, header=header_idx, engine='openpyxl')
        except Exception as clean_e:
            print(f"❌ Style cleaning failed: {clean_e}")

    # 4. Final attempt: Manual openpyxl
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        ws = wb.worksheets[sheet_name] if isinstance(sheet_name, int) else wb[sheet_name]
        data = [row for row in ws.iter_rows(values_only=True)]
        if data:
            header_idx = find_header_row(data)
            return pd.DataFrame(data[header_idx + 1:], columns=data[header_idx])
    except Exception as final_e:
        print(f"❌ All extraction methods failed for {path.name}")
        raise final_e

    # 4. Final attempt: Manual openpyxl with values_only=True
    try:
        print("🔍 Attempting manual extraction (ignoring metadata)...")
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
        ws = wb.worksheets[sheet_name] if isinstance(sheet_name, int) else wb[sheet_name]
        data = [row for row in ws.iter_rows(values_only=True)]
        if data:
            return pd.DataFrame(data[1:], columns=data[0])
    except Exception as final_e:
        print(f"❌ All extraction methods failed for {path.name}")
        raise final_e


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
