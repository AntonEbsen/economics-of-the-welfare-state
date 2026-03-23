"""
Export utilities for R and other statistical software.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def export_to_r(
    df: pd.DataFrame,
    output_path: str,
    dataset_name: str = "master",
    include_packages: list[str] = None,
    include_sample_models: bool = True,
) -> None:
    """
    Generate R script for panel data analysis.

    Args:
        df: Panel DataFrame
        output_path: Path to save R script
        dataset_name: Name for dataset in R
        include_packages: R packages to load
        include_sample_models: Include example model specifications

    Example:
        >>> export_to_r(
        ...     master,
        ...     'analysis/analysis.R',
        ...     include_packages=['fixest', 'plm', 'did']
        ... )
    """
    from .metadata import get_variable_info

    if include_packages is None:
        include_packages = ["fixest", "plm", "data.table", "ggplot2"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Also save data as CSV
    data_path = output_path.parent / f"{dataset_name}.csv"
    df.to_csv(data_path, index=False)

    script = []

    # Header
    script.append("# ============================================================================")
    script.append("# Panel Data Analysis in R")
    script.append("# Auto-generated script")
    script.append(
        "# ============================================================================\n"
    )

    # Package installation
    script.append("# Install required packages (run once)")
    script.append("# install.packages(c('" + "', '".join(include_packages) + "'))\n")

    # Load packages
    script.append("# Load packages")
    for pkg in include_packages:
        script.append(f"library({pkg})")
    script.append("")

    # Load data
    script.append("# Load data")
    script.append(f"{dataset_name} <- read.csv('{data_path.name}')")
    script.append(f"head({dataset_name})")
    script.append("")

    # Variable labels (as comments)
    script.append("# Variable descriptions:")
    for var in df.columns:
        info = get_variable_info(var)
        if "error" not in info:
            script.append(f"# {var}: {info['label']} - {info['description']}")
    script.append("")

    # Convert to panel data format
    script.append("# Convert to panel data format (if using plm)")
    script.append(f"{dataset_name}_panel <- pdata.frame({dataset_name}, index = c('iso3', 'year'))")
    script.append("")

    # Sample models
    if include_sample_models:
        script.append(
            "# ============================================================================"
        )
        script.append("# Sample Model Specifications")
        script.append(
            "# ============================================================================\n"
        )

        # Identify potential variables
        numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_vars = [v for v in numeric_vars if v not in ["iso3", "year"]]

        if len(numeric_vars) >= 2:
            dep_var = numeric_vars[0]
            ind_vars = numeric_vars[1 : min(4, len(numeric_vars))]

            script.append(f"# Example: {dep_var} as dependent variable\n")

            # OLS
            script.append("# 1. Pooled OLS")
            script.append(
                f"ols_model <- lm({dep_var} ~ {' + '.join(ind_vars)}, data = {dataset_name})"
            )
            script.append("summary(ols_model)")
            script.append("")

            # Fixed effects (fixest)
            script.append("# 2. Two-way fixed effects (fixest)")
            script.append(f"fe_model <- feols({dep_var} ~ {' + '.join(ind_vars)} | iso3 + year, ")
            script.append(f"                   data = {dataset_name}, cluster = ~iso3)")
            script.append("summary(fe_model)")
            script.append("")

            # Fixed effects (plm)
            script.append("# 3. Two-way fixed effects (plm)")
            script.append(f"plm_model <- plm({dep_var} ~ {' + '.join(ind_vars)}, ")
            script.append(f"                 data = {dataset_name}_panel, ")
            script.append("                 model = 'within', effect = 'twoways')")
            script.append("summary(plm_model)")
            script.append("")

            # Random effects
            script.append("# 4. Random effects")
            script.append(f"re_model <- plm({dep_var} ~ {' + '.join(ind_vars)}, ")
            script.append(f"                data = {dataset_name}_panel, ")
            script.append("                model = 'random')")
            script.append("summary(re_model)")
            script.append("")

            # Hausman test
            script.append("# 5. Hausman test (FE vs RE)")
            script.append("phtest(fe_model, re_model)")
            script.append("")

    # Diagnostics
    script.append("# ============================================================================")
    script.append("# Diagnostics")
    script.append(
        "# ============================================================================\n"
    )
    script.append("# Panel characteristics")
    script.append(f"pdim({dataset_name}_panel)  # Panel dimensions")
    script.append("")
    script.append("# Summary statistics")
    script.append(f"summary({dataset_name})")
    script.append("")

    # Write script
    with open(output_path, "w") as f:
        f.write("\n".join(script))

    print(f"✅ R script saved to: {output_path}")
    print(f"✅ Data saved to: {data_path}")
    print("\nTo use:")
    print("  1. Open R or RStudio")
    print(f"  2. Run: source('{output_path.name}')")


def export_to_stata_script(
    df: pd.DataFrame,
    output_path: str,
    dataset_name: str = "master",
    include_sample_models: bool = True,
) -> None:
    """
    Generate Stata .do file for panel analysis.

    Args:
        df: Panel DataFrame
        output_path: Path to save .do file
        dataset_name: Name for dataset
        include_sample_models: Include example specifications
    """
    from .metadata import get_variable_info

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as Stata format
    data_path = output_path.parent / f"{dataset_name}.dta"
    df.to_stata(data_path, write_index=False)

    script = []

    # Header
    script.append("* ===========================================================================")
    script.append("* Panel Data Analysis")
    script.append("* Auto-generated Stata script")
    script.append("* ===========================================================================")
    script.append("")

    # Load data
    script.append("* Load data")
    script.append(f'use "{data_path.name}", clear')
    script.append("")

    # Variable labels
    script.append("* Variable labels")
    for var in df.columns:
        info = get_variable_info(var)
        if "error" not in info:
            script.append(f"label variable {var} \"{info['label']}\"")
    script.append("")

    # Declare panel
    script.append("* Declare panel structure")
    script.append("encode iso3, gen(country_id)")
    script.append("xtset country_id year")
    script.append("xtdescribe")
    script.append("")

    # Sample models
    if include_sample_models:
        numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_vars = [v for v in numeric_vars if v not in ["iso3", "year"]]

        if len(numeric_vars) >= 2:
            dep_var = numeric_vars[0]
            ind_vars = " ".join(numeric_vars[1 : min(4, len(numeric_vars))])

            script.append("* Sample regressions")
            script.append("")
            script.append("* 1. Pooled OLS")
            script.append(f"reg {dep_var} {ind_vars}")
            script.append("")
            script.append("* 2. Fixed effects")
            script.append(f"xtreg {dep_var} {ind_vars}, fe vce(cluster country_id)")
            script.append("")
            script.append("* 3. Random effects")
            script.append(f"xtreg {dep_var} {ind_vars}, re")
            script.append("")
            script.append("* 4. Hausman test")
            script.append("hausman ., sigmamore")
            script.append("")

    # Write script
    with open(output_path, "w") as f:
        f.write("\n".join(script))

    print(f"✅ Stata script saved to: {output_path}")
    print(f"✅ Data saved to: {data_path}")
