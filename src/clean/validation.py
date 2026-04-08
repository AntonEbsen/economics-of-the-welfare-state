import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema, Index

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


def validate_output(*args, **kwargs):
    pass


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
