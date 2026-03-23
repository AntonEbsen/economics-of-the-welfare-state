"""
Summary statistics and table generation for research papers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_summary_stats(
    df: pd.DataFrame, variables: list[str] = None, output_format: str = "pandas"
) -> pd.DataFrame | str:
    """
    Generate Table 1 style summary statistics.

    Args:
        df: DataFrame with data
        variables: List of variables to summarize. If None, use all numeric.
        output_format: 'pandas', 'latex', 'markdown', or 'csv'

    Returns:
        Summary statistics table

    Example:
        >>> stats = generate_summary_stats(
        ...     master,
        ...     variables=['sstran', 'deficit', 'debt', 'ln_gdppc'],
        ...     output_format='latex'
        ... )
    """
    if variables is None:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude id/time variables
        variables = [v for v in variables if v not in ["iso3", "year"]]

    # Calculate statistics
    stats = pd.DataFrame()

    for var in variables:
        if var in df.columns:
            stats[var] = {
                "N": df[var].count(),
                "Mean": df[var].mean(),
                "Std": df[var].std(),
                "Min": df[var].min(),
                "p25": df[var].quantile(0.25),
                "Median": df[var].median(),
                "p75": df[var].quantile(0.75),
                "Max": df[var].max(),
            }

    stats = stats.T

    # Format based on output type
    if output_format == "latex":
        return stats.to_latex(float_format="%.2f")
    elif output_format == "markdown":
        return stats.to_markdown(floatfmt=".2f")
    elif output_format == "csv":
        return stats.to_csv(float_format="%.2f")
    else:
        return stats


def compare_groups(df: pd.DataFrame, variable: str, group_var: str, test: str = "t-test") -> dict:
    """
    Compare variable across groups.

    Args:
        df: DataFrame
        variable: Variable to compare
        group_var: Grouping variable
        test: Statistical test ('t-test', 'anova', 'median')

    Returns:
        Dictionary with test results
    """
    from scipy import stats as sp_stats

    groups = df.groupby(group_var)[variable].apply(list).to_dict()

    result = {
        "variable": variable,
        "group_var": group_var,
        "n_groups": len(groups),
    }

    if test == "t-test" and len(groups) == 2:
        g1, g2 = list(groups.values())
        t_stat, p_value = sp_stats.ttest_ind(g1, g2, nan_policy="omit")
        result["t_statistic"] = t_stat
        result["p_value"] = p_value
        result["significant"] = p_value < 0.05
    elif test == "anova":
        f_stat, p_value = sp_stats.f_oneway(*groups.values())
        result["f_statistic"] = f_stat
        result["p_value"] = p_value
        result["significant"] = p_value < 0.05

    return result


def correlation_matrix(
    df: pd.DataFrame,
    variables: list[str] = None,
    method: str = "pearson",
    output_format: str = "pandas",
) -> pd.DataFrame | str:
    """
    Generate correlation matrix.

    Args:
        df: DataFrame
        variables: Variables to include. If None, use all numeric.
        method: 'pearson', 'spearman', or 'kendall'
        output_format: 'pandas', 'latex', or 'markdown'

    Returns:
        Correlation matrix
    """
    if variables is None:
        variables = df.select_dtypes(include=[np.number]).columns.tolist()
        variables = [v for v in variables if v not in ["iso3", "year"]]

    corr = df[variables].corr(method=method)

    if output_format == "latex":
        return corr.to_latex(float_format="%.3f")
    elif output_format == "markdown":
        return corr.to_markdown(floatfmt=".3f")
    else:
        return corr


def export_stata_labels(df: pd.DataFrame, output_path: str):
    """
    Export Stata .do file with variable labels.

    Args:
        df: DataFrame
        output_path: Path to output .do file

    Example:
        >>> export_stata_labels(master, 'analysis/label_variables.do')
    """
    from pathlib import Path

    from .metadata import get_variable_info

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("* Variable labels for master dataset\n")
        f.write("* Auto-generated\n\n")

        for var in df.columns:
            info = get_variable_info(var)
            if "error" not in info:
                label = info["label"]
                f.write(f'label variable {var} "{label}"\n')

        f.write("\n* Variable notes\n")
        for var in df.columns:
            info = get_variable_info(var)
            if "error" not in info:
                desc = info["description"]
                f.write(f"notes {var}: {desc}\n")

    print(f"✅ Stata labels exported to: {output_path}")


def create_publication_table(
    results_list: list,
    model_names: list[str] = None,
    output_format: str = "latex",
    stars: bool = True,
    decimals: int = 3,
    add_stats: list[str] = ["nobs", "rsquared"],
) -> str:
    """
    Create publication-ready regression table from statsmodels results.

    Args:
        results_list: List of fitted regression models (statsmodels results)
        model_names: Names for each model column
        output_format: 'latex', 'html', or 'text'
        stars: Add significance stars (* p<0.1, ** p<0.05, *** p<0.01)
        decimals: Number of decimal places
        add_stats: Additional statistics to include

    Returns:
        Formatted table string

    Example:
        >>> from statsmodels.formula.api import ols
        >>> model1 = ols('sstran ~ ln_gdppc', data=df).fit()
        >>> model2 = ols('sstran ~ ln_gdppc + deficit', data=df).fit()
        >>> table = create_publication_table(
        ...     [model1, model2],
        ...     model_names=['Model 1', 'Model 2'],
        ...     output_format='latex'
        ... )
        >>> # Copy-paste into LaTeX paper!
    """
    try:
        from stargazer.stargazer import Stargazer
    except ImportError:
        print("⚠️  stargazer package recommended: pip install stargazer")
        # Fallback to manual formatting
        return _manual_regression_table(results_list, model_names, stars, decimals)

    # Use stargazer for beautiful tables
    sg = Stargazer(results_list)

    if model_names:
        sg.custom_columns(model_names, [1] * len(results_list))

    if output_format == "latex":
        return sg.render_latex()
    elif output_format == "html":
        return sg.render_html()
    else:
        return str(sg)


def _manual_regression_table(
    results_list: list, model_names: list[str] = None, stars: bool = True, decimals: int = 3
) -> str:
    """Fallback manual table formatting."""

    def format_coef(coef, pval, stars_enabled):
        """Format coefficient with stars."""
        if stars_enabled:
            if pval < 0.01:
                return f"{coef:.{decimals}f}***"
            elif pval < 0.05:
                return f"{coef:.{decimals}f}**"
            elif pval < 0.1:
                return f"{coef:.{decimals}f}*"
        return f"{coef:.{decimals}f}"

    if model_names is None:
        model_names = [f"({i+1})" for i in range(len(results_list))]

    # Collect all variables
    all_vars = set()
    for result in results_list:
        all_vars.update(result.params.index)
    all_vars = sorted(all_vars)

    # Build table
    lines = []
    lines.append("\n" + "=" * 60)
    lines.append("REGRESSION RESULTS")
    lines.append("=" * 60)

    # Header
    header = f"{'Variable':<20} " + " ".join([f"{name:>12}" for name in model_names])
    lines.append(header)
    lines.append("-" * 60)

    # Coefficients
    for var in all_vars:
        row = f"{var:<20}"
        for result in results_list:
            if var in result.params:
                coef = result.params[var]
                pval = result.pvalues[var]
                se = result.bse[var]
                row += f" {format_coef(coef, pval, stars):>12}"
            else:
                row += f" {'':>12}"
        lines.append(row)

        # Standard errors
        se_row = f"{'':20}"
        for result in results_list:
            if var in result.params:
                se = result.bse[var]
                se_row += f" ({se:.{decimals}f})".rjust(13)
            else:
                se_row += f" {'':>12}"
        lines.append(se_row)

    # Stats
    lines.append("-" * 60)
    lines.append(f"{'N':<20} " + " ".join([f"{int(r.nobs):>12}" for r in results_list]))
    lines.append(f"{'R-squared':<20} " + " ".join([f"{r.rsquared:>12.3f}" for r in results_list]))
    lines.append("=" * 60)

    if stars:
        lines.append("\nSignificance: * p<0.1, ** p<0.05, *** p<0.01")

    return "\n".join(lines)
