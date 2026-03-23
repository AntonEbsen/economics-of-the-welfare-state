"""
Automatic documentation generation for research papers.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


def generate_methods_section(
    df: pd.DataFrame, output_path: str = None, include_summary_stats: bool = True
) -> str:
    """
    Auto-generate data and methods section for research paper.

    Args:
        df: Master dataset
        output_path: Optional path to save as markdown/text file
        include_summary_stats: Include summary statistics table

    Returns:
        Formatted methods section text

    Example:
        >>> methods = generate_methods_section(master)
        >>> print(methods)  # Copy into your paper!
    """
    from .metadata import get_variable_info

    sections = []

    # Title
    sections.append("# Data and Methods\n")

    # Data sources section
    sections.append("## Data Sources\n")
    sections.append("This study utilizes panel data from multiple sources:\n")

    # Group variables by source
    sources = {}
    for var in df.columns:
        if var not in ["iso3", "year"]:
            info = get_variable_info(var)
            if "error" not in info:
                source = info["source"]
                if source not in sources:
                    sources[source] = []
                sources[source].append((var, info["label"]))

    for source, vars_list in sources.items():
        sections.append(f"\n**{source}:**")
        for var, label in vars_list:
            sections.append(f"- *{label}* (`{var}`)")

    # Sample section
    sections.append("\n## Sample\n")
    sections.append(
        f"The analysis covers **{df['iso3'].nunique()} countries** over the period "
        f"**{int(df['year'].min())}–{int(df['year'].max())}**, yielding "
        f"**{len(df):,} country-year observations**.\n"
    )

    # List countries
    countries = sorted(df["iso3"].unique())
    sections.append("\n**Countries included:**")
    # Format in rows of 8
    for i in range(0, len(countries), 8):
        row = countries[i : i + 8]
        sections.append(" ".join(row))

    # Data structure
    sections.append("\n## Panel Structure\n")

    # Check balance
    n_countries = df["iso3"].nunique()
    n_years = df["year"].nunique()
    expected = n_countries * n_years
    actual = len(df)
    balance_pct = (actual / expected) * 100

    if balance_pct == 100:
        sections.append("The panel is **balanced**, with all countries observed in all years.")
    else:
        sections.append(
            f"The panel is **unbalanced**, with {balance_pct:.1f}% of potential "
            f"observations present ({actual:,} of {expected:,} possible country-years)."
        )

    # Variable construction
    sections.append("\n## Variable Construction\n")
    sections.append("\n**Dependent and Independent Variables:**\n")

    for var in df.columns:
        if var not in ["iso3", "year"]:
            info = get_variable_info(var)
            if "error" not in info:
                sections.append(f"- **{info['label']}** ({var}): {info['description']}")

    # Missing data
    sections.append("\n## Missing Data\n")
    missing = df.isnull().sum()
    missing_vars = missing[missing > 0]

    if len(missing_vars) == 0:
        sections.append("The dataset contains no missing values.")
    else:
        sections.append("Missing data patterns:\n")
        for var, count in missing_vars.items():
            pct = (count / len(df)) * 100
            sections.append(f"- {var}: {count:,} observations ({pct:.1f}%)")

    # Summary statistics
    if include_summary_stats:
        sections.append("\n## Summary Statistics\n")
        sections.append(
            "\nTable 1 presents summary statistics for all variables in the analysis.\n"
        )
        sections.append(
            "*(Summary statistics table would be inserted here using `generate_summary_stats()`)*\n"
        )

    # Estimation strategy (placeholder)
    sections.append("\n## Estimation Strategy\n")
    sections.append(
        "The empirical analysis employs panel data methods with fixed effects "
        "to control for unobserved country-specific and time-specific heterogeneity. "
        "Standard errors are clustered at the country level to account for "
        "within-country correlation over time.\n"
    )

    # Footer
    sections.append(f"\n---\n*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

    # Combine
    text = "\n".join(sections)

    # Save if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(text)
        print(f"✅ Methods section saved to: {output_path}")

    return text


def generate_data_appendix(df: pd.DataFrame, output_path: str = None) -> str:
    """
    Generate detailed data appendix with all variable definitions.

    Args:
        df: Master dataset
        output_path: Optional path to save

    Returns:
        Formatted data appendix
    """
    from .metadata import get_variable_info

    sections = []

    sections.append("# Data Appendix\n")
    sections.append("## Variable Definitions\n")

    # Table of variables
    rows = []
    for var in sorted(df.columns):
        if var not in ["iso3", "year"]:
            info = get_variable_info(var)
            if "error" not in info:
                rows.append(
                    f"| {var} | {info['label']} | {info['source']} | {info['unit']} | {info['description']} |"
                )

    if rows:
        sections.append("\n| Variable | Label | Source | Unit | Description |")
        sections.append("|----------|-------|--------|------|-------------|")
        sections.extend(rows)

    # Coverage table
    sections.append("\n## Data Coverage by Country\n")
    sections.append("\n*(Coverage table showing years available for each country)*\n")

    coverage = df.groupby("iso3")["year"].agg(["min", "max", "count"]).reset_index()
    coverage.columns = ["Country", "First Year", "Last Year", "N Years"]
    sections.append("\n" + coverage.to_markdown(index=False))

    text = "\n".join(sections)

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(text)
        print(f"✅ Data appendix saved to: {output_path}")

    return text
