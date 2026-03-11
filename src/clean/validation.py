import pandas as pd
import pandera as pa
from pandera import Column, Check, Index, DataFrameSchema

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
    strict=False, # Allow extra columns like region/regime indicators
    coerce=True
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
