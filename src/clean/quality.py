"""
Data quality reporting and diagnostics.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_quality_report(df: pd.DataFrame, output_path: str = None) -> dict:
    """
    Generate comprehensive data quality report.

    Args:
        df: DataFrame to analyze
        output_path: Optional path to save report as HTML

    Returns:
        Dictionary with quality metrics
    """
    report = {}

    # Basic info
    report["n_rows"] = len(df)
    report["n_columns"] = len(df.columns)
    report["memory_usage_mb"] = df.memory_usage(deep=True).sum() / 1024 / 1024

    # Missing values
    missing = df.isnull().sum()
    report["missing_values"] = missing[missing > 0].to_dict()
    report["missing_pct"] = (missing[missing > 0] / len(df) * 100).to_dict()

    # Duplicates
    if "iso3" in df.columns and "year" in df.columns:
        dupes = df.duplicated(subset=["iso3", "year"]).sum()
        report["duplicate_rows"] = dupes

    # Numeric variables
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    report["numeric_variables"] = len(numeric_cols)

    # Outliers (IQR method)
    outliers = {}
    for col in numeric_cols:
        if col not in ["iso3", "year"]:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            if outlier_count > 0:
                outliers[col] = outlier_count
    report["outliers"] = outliers

    # Panel balance (if panel data)
    if "iso3" in df.columns and "year" in df.columns:
        n_countries = df["iso3"].nunique()
        n_years = df["year"].nunique()
        expected_obs = n_countries * n_years
        actual_obs = len(df)
        report["panel_balance"] = {
            "expected_obs": expected_obs,
            "actual_obs": actual_obs,
            "balanced": expected_obs == actual_obs,
        }

    # Print summary
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"\nDataset size: {report['n_rows']:,} rows × {report['n_columns']} columns")
    print(f"Memory usage: {report['memory_usage_mb']:.2f} MB")

    if report["missing_values"]:
        print(f"\n⚠️  Missing values found in {len(report['missing_values'])} variables:")
        for var, count in list(report["missing_values"].items())[:5]:
            pct = report["missing_pct"][var]
            print(f"   {var}: {count:,} ({pct:.1f}%)")
    else:
        print("\n✅ No missing values")

    if report["duplicate_rows"] > 0:
        print(f"\n⚠️  {report['duplicate_rows']} duplicate rows found")
    else:
        print("\n✅ No duplicate rows")

    if outliers:
        print(f"\n⚠️  Outliers detected in {len(outliers)} variables:")
        for var, count in list(outliers.items())[:5]:
            print(f"   {var}: {count:,} outliers")

    if "panel_balance" in report:
        if report["panel_balance"]["balanced"]:
            print("\n✅ Panel is balanced")
        else:
            print(
                f"\n⚠️  Panel is unbalanced: {report['panel_balance']['actual_obs']:,} / {report['panel_balance']['expected_obs']:,} observations"
            )

    print("\n" + "=" * 60)

    # Save HTML report if requested
    if output_path:
        save_html_report(report, df, output_path)

    return report


def save_html_report(report: dict, df: pd.DataFrame, output_path: str):
    """Save quality report as HTML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = f"""
    <html>
    <head>
        <title>Data Quality Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #4CAF50; color: white; }}
            .warning {{ color: #ff9800; }}
            .good {{ color: #4CAF50; }}
        </style>
    </head>
    <body>
        <h1>Data Quality Report</h1>
        <p>Generated: {pd.Timestamp.now()}</p>

        <h2>Dataset Overview</h2>
        <p>Rows: {report["n_rows"]:,} | Columns: {report["n_columns"]} | Memory: {
        report["memory_usage_mb"]:.2f} MB</p>

        <h2>Missing Values</h2>
        {
        pd.DataFrame(
            {
                "Variable": list(report["missing_values"].keys()),
                "Count": list(report["missing_values"].values()),
                "Percentage": [
                    f"{report['missing_pct'][v]:.1f}%" for v in report["missing_values"].keys()
                ],
            }
        ).to_html(index=False)
        if report["missing_values"]
        else '<p class="good">✅ No missing values</p>'
    }

        <h2>Summary Statistics</h2>
        {df.describe().to_html()}
    </body>
    </html>
    """

    with open(output_path, "w") as f:
        f.write(html)

    print(f"✅ HTML report saved to: {output_path}")


def check_time_series_breaks(
    df: pd.DataFrame, variables: list[str], threshold: float = 3.0
) -> dict:
    """
    Detect potential breaks or anomalies in time series.

    Args:
        df: Panel DataFrame
        variables: Variables to check
        threshold: Number of standard deviations to flag as anomaly

    Returns:
        Dictionary with detected breaks by variable
    """
    breaks = {}

    for var in variables:
        if var not in df.columns:
            continue

        # Calculate year-over-year changes by country
        df_sorted = df.sort_values(["iso3", "year"])
        df_sorted[f"{var}_change"] = df_sorted.groupby("iso3")[var].diff()

        # Detect outliers in changes
        mean_change = df_sorted[f"{var}_change"].mean()
        std_change = df_sorted[f"{var}_change"].std()

        anomalies = df_sorted[
            abs(df_sorted[f"{var}_change"] - mean_change) > threshold * std_change
        ][["iso3", "year", var, f"{var}_change"]]

        if len(anomalies) > 0:
            breaks[var] = anomalies
            print(f"\n⚠️  {var}: {len(anomalies)} potential breaks detected")

    return breaks
