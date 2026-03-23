"""
World Bank data processing utilities.
Handles the common wide-to-long conversion pattern used in World Bank datasets.
"""

from __future__ import annotations

import re

import pandas as pd


class WorldBankProcessor:
    """Process World Bank-style Excel files (wide format with year columns)."""

    @staticmethod
    def find_country_column(df: pd.DataFrame) -> str:
        """
        Find country column using common naming patterns.

        Args:
            df: DataFrame to search

        Returns:
            Name of country column

        Raises:
            ValueError: If no country column found
        """
        candidates = [
            c
            for c in df.columns
            if c.strip().lower()
            in {
                "country",
                "location",
                "country name",
                "country_name",
                "reference area",
                "reference_area",
                "ref_area",
                "referencearea",
            }
        ]

        if not candidates:
            raise ValueError(f"Could not find country column. Available: {list(df.columns)}")

        return candidates[0]

    @staticmethod
    def find_year_columns(df: pd.DataFrame) -> list[str]:
        """
        Find year columns (4-digit numbers, possibly with [YR...] suffix).

        Args:
            df: DataFrame to search

        Returns:
            List of year column names

        Raises:
            ValueError: If no year columns found
        """
        # Pattern: "1980" or "1980 [YR1980]"
        year_pat = re.compile(r"^(\d{4})(?:\s*\[YR\d{4}\])?\s*$")
        year_cols = [c for c in df.columns if year_pat.match(str(c))]

        if not year_cols:
            # Fallback: just 4-digit columns
            year_cols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4]

        if not year_cols:
            raise ValueError("No year columns found (expected format: '1980' or '1980 [YR1980]')")

        return year_cols

    @classmethod
    def wide_to_long(
        cls,
        df_raw: pd.DataFrame,
        value_name: str,
    ) -> pd.DataFrame:
        """
        Convert World Bank wide format to long format.

        Args:
            df_raw: Raw DataFrame in wide format
            value_name: Name for the value column in long format

        Returns:
            DataFrame in long format with columns: country, year, {value_name}
        """
        df = df_raw.copy()
        df.columns = [str(c).strip() for c in df.columns]

        # Find columns
        country_col = cls.find_country_column(df)
        year_cols = cls.find_year_columns(df)

        # Melt to long format
        df_long = df.melt(
            id_vars=[country_col], value_vars=year_cols, var_name="year_raw", value_name=value_name
        )

        df_long = df_long.rename(columns={country_col: "country"})

        # Extract year from year column
        df_long["year"] = df_long["year_raw"].astype(str).str.extract(r"^(\d{4})", expand=False)
        df_long["year"] = pd.to_numeric(df_long["year"], errors="coerce").astype("Int64")

        # Clean values
        df_long[value_name] = pd.to_numeric(df_long[value_name], errors="coerce")
        df_long["country"] = df_long["country"].astype(str).str.strip()

        # Drop rows with missing keys
        df_long = df_long.dropna(subset=["country", "year"]).reset_index(drop=True)

        return df_long[["country", "year", value_name]]
