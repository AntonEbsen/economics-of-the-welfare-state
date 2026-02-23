from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

# Import centralized constants
from .constants import TARGET_ISO3_32
from .utils import save_dataframe

NAME_MAP = {
    # common variants you mentioned
    "Czech Rep. (Czechia)": "Czech Republic",
    "Czech Rep.": "Czech Republic",
    "Czechia": "Czech Republic",
    "Slovakia/Slovak republic": "Slovak Republic",
    "Slovak republic": "Slovak Republic",
    "U.K.": "United Kingdom",
    "UK": "United Kingdom",
    "United States of America": "United States",
}


@dataclass(frozen=True)
class KOFConfig:
    year_min: int | None = 1980
    year_max: int | None = 2023
    target_iso3: set[str] = None  # filled in __post_init__ style below
    drop_all_missing_index: bool = True

    def __post_init__(self):
        # dataclass(frozen=True) workaround
        object.__setattr__(self, "target_iso3", TARGET_ISO3_32 if self.target_iso3 is None else self.target_iso3)


def read_kof_excel(path: str | Path, sheet_name: str = "Sheet1") -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_excel(path, sheet_name=sheet_name)


def standardize_kof(df: pd.DataFrame) -> pd.DataFrame:
    required = {"code", "country", "year"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    out = df.copy()

    out["code"] = out["code"].astype(str).str.strip().str.upper()
    out["country"] = out["country"].astype(str).str.strip()
    out["year"] = pd.to_numeric(out["year"], errors="coerce").astype("Int64")

    out["country_clean"] = out["country"].replace(NAME_MAP)

    return out


def filter_kof_32countries(df: pd.DataFrame, cfg: KOFConfig = KOFConfig()) -> pd.DataFrame:
    out = df.copy()

    # filter countries
    out = out[out["code"].isin(cfg.target_iso3)].copy()

    # filter years
    if cfg.year_min is not None:
        out = out[out["year"] >= cfg.year_min]
    if cfg.year_max is not None:
        out = out[out["year"] <= cfg.year_max]

    # select core indices
    core_indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
    
    # check which ones exist in the dataframe to avoid errors
    available_indices = [c for c in core_indices if c in out.columns]
    
    # identifiers first
    id_cols = ["code", "country_clean", "year"]
    
    # drop rows with all index columns missing (optional but usually sensible)
    if cfg.drop_all_missing_index and available_indices:
        out = out.dropna(subset=available_indices, how="all")

    # final selection
    out = out[id_cols + available_indices].sort_values(["country_clean", "year"]).reset_index(drop=True)

    # rename for downstream merges
    out = out.rename(columns={"code": "iso3", "country_clean": "country"})

    return out


def assert_exact_32(df: pd.DataFrame, expected_iso3: Iterable[str] = TARGET_ISO3_32) -> None:
    expected_iso3 = set(expected_iso3)
    got = set(df["iso3"].dropna().astype(str).str.upper().unique())

    missing = expected_iso3 - got
    extra = got - expected_iso3

    if missing or extra:
        msg = []
        if missing:
            msg.append(f"Missing ISO3: {sorted(missing)}")
        if extra:
            msg.append(f"Unexpected ISO3: {sorted(extra)}")
        raise AssertionError(" | ".join(msg))

    if df["iso3"].nunique() != 32:
        raise AssertionError(f"Expected 32 unique ISO3 codes, got {df['iso3'].nunique()}.")


def save_processed(df: pd.DataFrame, out_path: str | Path) -> Path:
    """Save processed KOF data to .parquet or .csv."""
    return save_dataframe(df, out_path)
