# src/clean/gdppc.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

# Import from centralized constants
from .constants import TARGET_ISO3_32
from .utils import map_country_to_iso3, save_dataframe


@dataclass(frozen=True)
class GDPPCConfig:
    year_min: Optional[int] = 1980
    year_max: Optional[int] = 2023

    # If True: raise error if some of the 32 ISO3 codes are missing (at least once)
    strict_32: bool = False

    # Keep rows where country->iso3 mapping failed (useful for debugging)
    keep_unmapped: bool = False

    # If you know the exact country column name, set it (e.g. "Reference area")
    country_col: Optional[str] = None

    # If you know the exact year column name (long format), set it
    year_col: Optional[str] = None

    # If you know the exact value column name (long format), set it
    value_col: Optional[str] = None


def _clean_colnames(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _normalize_country(x) -> str:
    if pd.isna(x):
        return x
    s = str(x).strip()
    # light normalization for common issues
    s = s.replace("&", "and")
    s = " ".join(s.split())  # collapse multiple spaces
    return s


def read_gdppc_excel(path, sheet_name=None, header=0):
    """
    Read GDP per capita excel file.
    - sheet_name=None reads the first sheet
    - header can be 0 (normal) or None (raw grid)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")

    xls = pd.ExcelFile(path)

    if sheet_name is None:
        sheet_name = xls.sheet_names[0]

    if isinstance(sheet_name, str) and sheet_name not in xls.sheet_names:
        raise ValueError(
            f"Worksheet named '{sheet_name}' not found. Available sheets: {xls.sheet_names}"
        )

    if isinstance(sheet_name, int):
        if sheet_name < 0 or sheet_name >= len(xls.sheet_names):
            raise ValueError(
                f"Worksheet index {sheet_name} out of range. Available sheets: {xls.sheet_names}"
            )

    return pd.read_excel(xls, sheet_name=sheet_name, header=header)


def standardize_gdppc_to_long(
    df_raw: pd.DataFrame, cfg: GDPPCConfig = GDPPCConfig()
) -> pd.DataFrame:
    """
    Standardize GDP per capita dataset to long format with columns:
        country, year, gdppc

    Handles:
      - Wide format: years as columns (e.g., 1980, 1981, ...)
      - Long format: columns like (Reference area / Country), year/time, value

    NOTE: Your file uses a country column called "Reference area".
    """
    df_raw = _clean_colnames(df_raw)

    # ---- 1) Determine country column ----
    if cfg.country_col and cfg.country_col in df_raw.columns:
        country_col = cfg.country_col
    else:
        country_candidates = [
            c
            for c in df_raw.columns
            if c.strip().lower()
            in {
                "country",
                "location",
                "name",
                "country_name",
                "reference area",
                "reference_area",
                "ref_area",
                "referencearea",
            }
        ]
        if not country_candidates:
            # Show candidates to help debugging
            raise ValueError(
                "Could not find a country column. "
                "Expected e.g. 'Reference area' or 'country'. "
                f"Available columns: {list(df_raw.columns)}. "
                "Pass GDPPCConfig(country_col='Reference area') if needed."
            )
        country_col = country_candidates[0]

    # ---- 2) Detect wide format (4-digit year columns) ----
    year_cols = [c for c in df_raw.columns if str(c).isdigit() and len(str(c)) == 4]

    if year_cols:
        # WIDE -> LONG
        df_long = (
            df_raw.rename(columns={country_col: "country"})
            .melt(id_vars=["country"], value_vars=year_cols, var_name="year", value_name="gdppc")
            .copy()
        )
    else:
        # LONG format: find year column
        if cfg.year_col and cfg.year_col in df_raw.columns:
            year_col = cfg.year_col
        else:
            year_candidates = [
                c
                for c in df_raw.columns
                if c.strip().lower()
                in {"year", "time", "time_period", "time period", "periode", "år"}
            ]
            if not year_candidates:
                raise ValueError(
                    "Long format detected but no year column found. "
                    f"Available columns: {list(df_raw.columns)}. "
                    "Pass GDPPCConfig(year_col='<your year column>') if needed."
                )
            year_col = year_candidates[0]

        # Find value column
        if cfg.value_col and cfg.value_col in df_raw.columns:
            val_col = cfg.value_col
        else:
            val_candidates = [
                c
                for c in df_raw.columns
                if c.strip().lower()
                in {"gdppc", "gdp per capita", "gdp_per_capita", "gdp_pc", "value", "val"}
            ]
            if val_candidates:
                val_col = val_candidates[0]
            else:
                # fallback: last numeric column
                num_cols = [c for c in df_raw.columns if pd.api.types.is_numeric_dtype(df_raw[c])]
                if not num_cols:
                    raise ValueError(
                        "No obvious value column and no numeric columns found. "
                        f"Available columns: {list(df_raw.columns)}. "
                        "Pass GDPPCConfig(value_col='<your value column>') if needed."
                    )
                val_col = num_cols[-1]

        df_long = df_raw.rename(
            columns={country_col: "country", year_col: "year", val_col: "gdppc"}
        )[["country", "year", "gdppc"]].copy()

    # ---- 3) Type cleaning ----
    df_long["country"] = df_long["country"].map(_normalize_country)
    df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")
    df_long["gdppc"] = pd.to_numeric(df_long["gdppc"], errors="coerce")

    # Drop rows with missing keys
    df_long = df_long.dropna(subset=["country", "year"]).reset_index(drop=True)

    return df_long


# Removed: Now using shared map_country_to_iso3 from utils.py


def report_unmapped_countries(df_long: pd.DataFrame) -> pd.Series:
    mapped = map_country_to_iso3(df_long)
    return mapped.loc[mapped["iso3"].isna(), "country"].value_counts()


def filter_32_and_log(df_mapped: pd.DataFrame, cfg: GDPPCConfig = GDPPCConfig()) -> pd.DataFrame:
    """
    Filter to your 32 countries and compute ln(GDP per capita).
    Returns columns: iso3, country, year, gdppc, ln_gdppc
    """
    out = df_mapped.copy()

    # year filter
    if cfg.year_min is not None:
        out = out[out["year"] >= cfg.year_min]
    if cfg.year_max is not None:
        out = out[out["year"] <= cfg.year_max]

    # keep or drop unmapped
    if not cfg.keep_unmapped:
        out = out[~out["iso3"].isna()].copy()

    # filter to your 32
    out["iso3"] = out["iso3"].astype(str).str.strip().str.upper()
    out = out[out["iso3"].isin(TARGET_ISO3_32)].copy()

    # compute ln(gdppc) safely
    out["ln_gdppc"] = np.where(out["gdppc"] > 0, np.log(out["gdppc"]), np.nan)

    # tidy output
    out = (
        out[["iso3", "country", "year", "gdppc", "ln_gdppc"]]
        .sort_values(["iso3", "year"])
        .reset_index(drop=True)
    )

    # strict 32 check (at least once each)
    if cfg.strict_32:
        got = set(out["iso3"].unique())
        missing = TARGET_ISO3_32 - got
        if missing:
            raise AssertionError(
                f"GDP dataset is missing some of the 32 ISO3 codes: {sorted(missing)}. "
                "Likely a country-name mapping issue in COUNTRY_TO_ISO3."
            )

    return out


def get_final_gdppc(df_mapped: pd.DataFrame, cfg: GDPPCConfig = GDPPCConfig()) -> pd.DataFrame:
    """
    Filter to 32 countries and return minimal columns: iso3, year, ln_gdppc
    This is the final clean output for merging with other datasets.
    """
    df_full = filter_32_and_log(df_mapped, cfg=cfg)

    # Keep only essential columns
    return df_full[["iso3", "year", "ln_gdppc"]].copy()


def save_processed(df: pd.DataFrame, out_path: Union[str, Path]) -> Path:
    """
    Save processed dataset to .parquet or .csv
    """
    return save_dataframe(df, out_path)
