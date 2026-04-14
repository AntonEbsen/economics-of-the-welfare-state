import logging
from typing import Iterable

import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema, Index

logger = logging.getLogger(__name__)

# Master Dataset Schema
# This ensures that our master dataset has the expected columns and types
master_schema = DataFrameSchema(
    columns={
        "iso3": Column(str, Check.str_length(3)),
        "year": Column(int, Check(lambda x: x >= 1960)),
        "sstran": Column(float, nullable=True),
        "ln_gdppc": Column(float, nullable=True),
        "inflation_cpi": Column(float, nullable=True),
        "deficit": Column(float, nullable=True),
        "debt": Column(float, nullable=True),
        "ln_population": Column(float, nullable=True),
        "dependency_ratio": Column(float, nullable=True),
    },
    index=Index(int),
    strict=False,  # Allow extra columns like region/regime indicators
    coerce=True,
)


def validate_master_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the master dataset using Pandera.
    """
    try:
        validated_df = master_schema.validate(df)
        print("✅ Data validation passed successfully.")
        return validated_df
    except pa.errors.SchemaErrors as err:
        print("❌ Data validation failed!")
        print(err)
        raise err


def validate_output(
    df: pd.DataFrame,
    required_cols: Iterable[str],
    dataset_name: str = "dataset",
    year_min: int | None = None,
    year_max: int | None = None,
    expect_32_countries: bool = False,
) -> pd.DataFrame:
    """
    Lightweight post-processing validation for individual cleaned datasets.

    Enforces the contract that downstream merge steps rely on: the required
    columns exist, and optionally checks that the year range and country
    coverage match expectations. Missing columns raise ``ValueError`` since
    the merge pipeline cannot proceed without them. Year-range and country-
    count mismatches log warnings rather than raising because they can be
    legitimate (e.g. KOF Index stops earlier than CPDS).

    Args:
        df: Cleaned DataFrame to validate.
        required_cols: Column names that must be present.
        dataset_name: Human-readable dataset label used in log messages.
        year_min: Optional lower bound for the ``year`` column.
        year_max: Optional upper bound for the ``year`` column.
        expect_32_countries: If True, warn when ``iso3`` does not contain
            exactly 32 unique codes (the target OECD sample).

    Returns:
        The input DataFrame (unchanged) for chaining convenience.

    Raises:
        ValueError: If any of ``required_cols`` is missing from ``df``.
    """
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{dataset_name}] missing required columns: {missing}. " f"Present: {list(df.columns)}"
        )

    if "year" in df.columns:
        if year_min is not None and df["year"].min() < year_min:
            logger.warning(
                "[%s] year column contains values below year_min=%s (min=%s)",
                dataset_name,
                year_min,
                df["year"].min(),
            )
        if year_max is not None and df["year"].max() > year_max:
            logger.warning(
                "[%s] year column contains values above year_max=%s (max=%s)",
                dataset_name,
                year_max,
                df["year"].max(),
            )

    if expect_32_countries and "iso3" in df.columns:
        n_countries = df["iso3"].nunique()
        if n_countries != 32:
            logger.warning("[%s] expected 32 countries, found %s", dataset_name, n_countries)

    return df


# Merged Analysis Panel Schema
merged_panel_schema = DataFrameSchema(
    columns={
        "iso3": Column(str, Check.str_length(3)),
        "year": Column(int, Check.in_range(1960, 2025)),
        "sstran": Column(float, nullable=True),
        "KOFGI": Column(float, nullable=True),
        "KOFEcGI": Column(float, nullable=True),
        "KOFSoGI": Column(float, nullable=True),
        "KOFPoGI": Column(float, nullable=True),
        "regime_conservative": Column(int, Check.isin([0, 1])),
        "regime_mediterranean": Column(int, Check.isin([0, 1])),
        "regime_liberal": Column(int, Check.isin([0, 1])),
        "regime_post_communist": Column(int, Check.isin([0, 1])),
    },
    strict=False,
    coerce=True,
)


def validate_merged_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate the final merged panel dataset used for regression modeling.
    Ensures core index variables and regime dummies are present and correctly bounded.
    """
    try:
        validated_df = merged_panel_schema.validate(df)
        print("✅ Merged Panel Data validation passed successfully.")
        return validated_df
    except pa.errors.SchemaErrors as err:
        print("❌ Merged Panel Data validation failed!")
        print(err)
        raise err
