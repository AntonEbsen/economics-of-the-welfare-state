# src/clean/population.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import re

# Import from centralized constants
from .constants import TARGET_ISO3_32
from .utils import save_dataframe


@dataclass(frozen=True)
class PopulationConfig:
    year_min: Optional[int] = 1980
    year_max: Optional[int] = 2023
    strict_32: bool = False
    sheet_name: Union[str, int, None] = None
    header: Union[int, None] = 0  # World Bank exports usually have header row


def read_population_excel(
    path: Union[str, Path],
    sheet_name: Union[str, int, None] = None,
    header: Union[int, None] = 0,
) -> pd.DataFrame:
    """
    Read population excel file.
    - sheet_name=None reads the first sheet
    - header=0 for normal tables; header=None for raw grid sheets
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    xls = pd.ExcelFile(path)

    if sheet_name is None:
        sheet_name = xls.sheet_names[0]

    if isinstance(sheet_name, str) and sheet_name not in xls.sheet_names:
        raise ValueError(f"Worksheet named '{sheet_name}' not found. Available sheets: {xls.sheet_names}")

    if isinstance(sheet_name, int):
        if sheet_name < 0 or sheet_name >= len(xls.sheet_names):
            raise ValueError(f"Worksheet index {sheet_name} out of range. Available sheets: {xls.sheet_names}")

    return pd.read_excel(xls, sheet_name=sheet_name, header=header)


def standardize_worldbank_population_to_long(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parses World Bank wide format:
      Series Name | Series Code | Country Name | Country Code | 1980 [YR1980] | ... | 2023 [YR2023]

    Returns long format:
      iso3, year, population
    """
    df = df_raw.copy()
    df.columns = [str(c).strip() for c in df.columns]

    required = {"Country Code"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Found columns: {list(df.columns)}"
        )

    # Year columns look like "1980 [YR1980]"
    year_pat = re.compile(r"^(\d{4})\s*\[YR\d{4}\]\s*$")
    year_cols = [c for c in df.columns if year_pat.match(c)]

    if not year_cols:
        # fallback: some WB exports use just "1960", "1961", ...
        year_pat2 = re.compile(r"^(\d{4})$")
        year_cols = [c for c in df.columns if year_pat2.match(c)]
        if not year_cols:
            raise ValueError("No year columns found (expected '1980 [YR1980]' style).")

    id_cols = [c for c in ["Country Code"] if c in df.columns]

    long = df.melt(
        id_vars=id_cols,
        value_vars=year_cols,
        var_name="year_col",
        value_name="population"
    )

    # Extract year
    long["year"] = (
        long["year_col"]
        .astype(str)
        .str.extract(r"^(\d{4})", expand=False)
    )
    long["year"] = pd.to_numeric(long["year"], errors="coerce").astype("Int64")

    # Clean population values (WB sometimes uses "..")
    long["population"] = pd.to_numeric(long["population"], errors="coerce")

    long = long.rename(columns={"Country Code": "iso3"})

    long["iso3"] = long["iso3"].astype(str).str.strip().str.upper()

    # Keep only core columns (drop year_col too)
    long = long[["iso3", "year", "population"]].dropna(subset=["iso3", "year"]).copy()

    return long


def filter_32_and_log(long_pop: pd.DataFrame, cfg: PopulationConfig = PopulationConfig()) -> pd.DataFrame:
    """
    Filter to 32 countries, year range, compute ln(population).
    Returns ONLY: iso3, year, ln_population
    (drops country name + raw population)
    """
    out = long_pop.copy()

    # Filter countries
    out = out[out["iso3"].isin(TARGET_ISO3_32)].copy()

    # Filter years
    if cfg.year_min is not None:
        out = out[out["year"] >= cfg.year_min]
    if cfg.year_max is not None:
        out = out[out["year"] <= cfg.year_max]

    # ln(population): only for strictly positive values
    out["ln_population"] = np.where(out["population"] > 0, np.log(out["population"]), np.nan)

    out = out.sort_values(["iso3", "year"]).reset_index(drop=True)

    if cfg.strict_32:
        got = set(out["iso3"].unique())
        missing = TARGET_ISO3_32 - got
        if missing:
            raise AssertionError(
                f"Population dataset missing some of the 32 ISO3 codes: {sorted(missing)}"
            )

    # ✅ final cleaned output: no country name, no raw population
    return out[["iso3", "year", "ln_population"]]


def save_processed(df: pd.DataFrame, out_path: Union[str, Path]) -> Path:
    """Save processed population dataset to .parquet or .csv."""
    return save_dataframe(df, out_path)
