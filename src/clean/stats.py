import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS, RandomEffects
from scipy import stats
from scipy import stats as sp_stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

from .metadata import get_variable_info


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
        model_names = [f"({i + 1})" for i in range(len(results_list))]

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


def export_vif_latex(
    df: pd.DataFrame, indices: list[str], controls: list[str], out_dir: str
) -> None:
    """
    Calculate Variance Inflation Factor (VIF) for each model specification
    (each KOF index + controls) and export an aggregated LaTeX table.
    """
    vif_summary = []

    # We want a row for every possible regressor
    all_regressors = indices + controls

    for var in all_regressors:
        vif_summary.append({"Variable": var})

    summary_df = pd.DataFrame(vif_summary).set_index("Variable")

    print("\n" + "=" * 60)
    print("📈 CALCULATING VARIANCE INFLATION FACTORS (VIF)")
    print("=" * 60)

    for idx_name in indices:
        if idx_name not in df.columns:
            continue

        # The specific model specification
        model_vars = [idx_name] + controls

        # Drop NaNs across exactly these variables so VIF matches regression sample
        data_model = df[model_vars].dropna()

        # Add constant for VIF calculation (crucial, otherwise VIF is non-centered)
        X = sm.add_constant(data_model)

        vif_data = {}
        # VIF is calculated for each independent variable
        for i, col in enumerate(X.columns):
            if col == "const":
                continue
            try:
                vif_val = variance_inflation_factor(X.values, i)
                vif_data[col] = f"{vif_val:.2f}"
            except Exception:
                vif_data[col] = "-"

        # Fill the summary dataframe for this specific model
        summary_df[idx_name] = summary_df.index.map(lambda x: vif_data.get(x, "-"))

    summary_df = summary_df.reset_index()

    # Print to console
    print(summary_df.to_string(index=False))
    print("-" * 60)
    print("Rule of Thumb: VIF > 10 indicates high multicollinearity.")
    print("=" * 60 + "\n")

    # Format the LaTeX
    col_fmt = "l" + "c" * len(indices)
    latex_str = summary_df.to_latex(
        index=False,
        escape=False,
        column_format=col_fmt,
        caption="Variance Inflation Factor (VIF) by Model Specification",
        label="tab:vif",
    )

    # Import map for nice labels if present
    try:
        from analysis.regression_utils import LATEX_LABEL_MAP

        for old, new in LATEX_LABEL_MAP.items():
            latex_str = latex_str.replace(old, new)
    except ImportError:
        pass

    # Beautify LaTeX table
    beautified_header = (
        "\\toprule\n& \\multicolumn{"
        + str(len(indices))
        + "}{c}{Model Specification} \\\\\n\\cmidrule(lr){2-"
        + str(len(indices) + 1)
        + "}"
    )
    latex_str = latex_str.replace("\\toprule", beautified_header)

    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir / "vif_table.tex"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"✅ Exported VIF LaTeX table to: {out_file}")


def _modified_wald_groupwise_hetero(resids: pd.Series) -> tuple[float, float]:
    """Modified Wald test for groupwise heteroskedasticity in FE panels.

    Tests H₀: σ²ᵢ = σ² for all entities i (Greene 2000, §11.4.3).

    Under H₀ the test statistic W = Σᵢ nᵢ (σ̂²ᵢ − σ̂²)² / (2 σ̂⁴) is
    χ²(N−1) where N is the number of entities. The entity-specific
    variance estimator uses the FE residuals from the within-transformed
    regression.

    Returns ``(W_statistic, p_value)``.
    """
    if isinstance(resids.index, pd.MultiIndex):
        entity_level = resids.index.names[0]
        grouped = resids.groupby(level=entity_level)
    else:
        raise ValueError("Residuals must have a (entity, time) MultiIndex.")

    entity_vars = grouped.var(ddof=0)
    entity_counts = grouped.count()
    sigma2_pool = float((resids**2).mean())

    if sigma2_pool == 0:
        return np.nan, np.nan

    N = len(entity_vars)
    W = float((entity_counts * (entity_vars - sigma2_pool) ** 2).sum() / (2.0 * sigma2_pool**2))
    p_value = 1.0 - stats.chi2.cdf(W, N - 1)
    return W, p_value


def export_model_diagnostics_latex(final_models: dict, out_dir: str) -> pd.DataFrame:
    """
    Export post-estimation residual diagnostics for a dictionary of fitted models.

    Returns the diagnostics DataFrame so notebooks can display it inline
    alongside writing the LaTeX file.

    Heteroskedasticity battery:
        1. Breusch-Pagan (1979) — assumes errors are linear in the regressors.
        2. White (1980) — general test, no functional-form assumption.
        3. Modified Wald (Greene 2000) — panel-specific test for groupwise
           heteroskedasticity across entities (H₀: σ²ᵢ = σ² for all i).
    Serial correlation:
        4. Ljung-Box at lag 1.
    """
    print("\n" + "=" * 60)
    print("🔬 CALCULATING POST-ESTIMATION MODEL DIAGNOSTICS")
    print("=" * 60)

    bp_stat_row = {"Test": "Breusch-Pagan Stat"}
    bp_pval_row = {"Test": "Breusch-Pagan p-val"}
    wh_stat_row = {"Test": "White Stat"}
    wh_pval_row = {"Test": "White p-val"}
    mw_stat_row = {"Test": "Mod. Wald Stat"}
    mw_pval_row = {"Test": "Mod. Wald p-val"}
    lb_stat_row = {"Test": "Ljung-Box (Lag 1) Stat"}
    lb_pval_row = {"Test": "Ljung-Box (Lag 1) p-val"}

    for idx_name, res in final_models.items():
        resids = res.resids

        # 1. Breusch-Pagan
        try:
            exog = res.model.exog.dataframe
            if "const" not in exog.columns:
                exog = sm.add_constant(exog)

            bp_stat, bp_p, _, _ = het_breuschpagan(resids, exog)
            bp_stat_row[idx_name] = f"{bp_stat:.2f}"
            bp_pval_row[idx_name] = f"{bp_p:.3f}"
        except Exception:
            bp_stat_row[idx_name] = "-"
            bp_pval_row[idx_name] = "-"

        # 2. White's test
        try:
            exog = res.model.exog.dataframe
            if "const" not in exog.columns:
                exog = sm.add_constant(exog)

            wh_stat, wh_p, _, _ = het_white(resids, exog)
            wh_stat_row[idx_name] = f"{wh_stat:.2f}"
            wh_pval_row[idx_name] = f"{wh_p:.3f}"
        except Exception:
            wh_stat_row[idx_name] = "-"
            wh_pval_row[idx_name] = "-"

        # 3. Modified Wald for groupwise heteroskedasticity
        try:
            mw_stat, mw_p = _modified_wald_groupwise_hetero(resids)
            mw_stat_row[idx_name] = f"{mw_stat:.2f}" if not np.isnan(mw_stat) else "-"
            mw_pval_row[idx_name] = f"{mw_p:.3f}" if not np.isnan(mw_p) else "-"
        except Exception:
            mw_stat_row[idx_name] = "-"
            mw_pval_row[idx_name] = "-"

        # 4. Serial Correlation (Ljung-Box)
        try:
            if isinstance(resids, pd.DataFrame):
                resids_series = resids.iloc[:, 0]
            else:
                resids_series = resids

            resids_sorted = (
                resids_series.reset_index()
                .sort_values(by=list(resids_series.index.names))
                .set_index(resids_series.index.names)
                .iloc[:, 0]
            )
            lb_res = acorr_ljungbox(resids_sorted.dropna(), lags=[1], return_df=True)
            lb_stat = lb_res["lb_stat"].iloc[0]
            lb_p = lb_res["lb_pvalue"].iloc[0]

            lb_stat_row[idx_name] = f"{lb_stat:.2f}"
            lb_pval_row[idx_name] = f"{lb_p:.3f}"
        except Exception:
            lb_stat_row[idx_name] = "-"
            lb_pval_row[idx_name] = "-"

    records = [
        bp_stat_row,
        bp_pval_row,
        wh_stat_row,
        wh_pval_row,
        mw_stat_row,
        mw_pval_row,
        lb_stat_row,
        lb_pval_row,
    ]
    summary_df = pd.DataFrame(records)

    print(summary_df.to_string(index=False))
    print("-" * 60)
    print("Interpretation:")
    print("BP Test  (H0): Homoskedasticity (linear form)")
    print("White    (H0): Homoskedasticity (general, with cross-terms)")
    print("Mod.Wald (H0): Equal variance across entities (groupwise)")
    print("LB Test  (H0): No Serial Corr. at lag 1")
    print("=" * 60 + "\n")

    col_fmt = "l" + "c" * len(final_models)
    latex_str = summary_df.to_latex(
        index=False,
        escape=False,
        column_format=col_fmt,
        caption="Post-Estimation Residual Diagnostics (Fully Specified Models)",
        label="tab:model_diagnostics",
    )

    beautified_header = (
        "\\toprule\n& \\multicolumn{"
        + str(len(final_models))
        + "}{c}{Model Specification (Full Controls)} \\\\\n\\cmidrule(lr){2-"
        + str(len(final_models) + 1)
        + "}"
    )
    latex_str = latex_str.replace("\\toprule", beautified_header)

    try:
        from analysis.regression_utils import LATEX_LABEL_MAP

        for old, new in LATEX_LABEL_MAP.items():
            latex_str = latex_str.replace(old, new)
    except ImportError:
        pass

    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir / "model_diagnostics.tex"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"✅ Exported Model Diagnostics LaTeX table to: {out_file}")
    return summary_df


def export_hausman_latex(
    final_model_data: dict,
    dep_var: str,
    out_dir: str,
) -> None:
    """
    Export Hausman test results (FE vs RE) for each model specification
    into a single LaTeX table.

    Args:
        final_model_data: dict mapping index name -> (ols_data, exog_vars)
        dep_var: Dependent variable name.
        out_dir: Output directory for LaTeX table.
    """
    print("\n" + "=" * 60)
    print("⚖️  RUNNING HAUSMAN SPECIFICATION TESTS (FE vs RE)")
    print("=" * 60)

    records = []

    for idx_name, (ols_data, exog_vars) in final_model_data.items():
        try:
            exog = sm.add_constant(ols_data[exog_vars])
            exog = exog.loc[:, ~exog.columns.duplicated()]

            fe_res = PanelOLS(ols_data[dep_var], exog, entity_effects=True, time_effects=False).fit(
                cov_type="unadjusted"
            )
            re_res = RandomEffects(ols_data[dep_var], exog).fit(cov_type="unadjusted")

            shared = [v for v in fe_res.params.index if v in re_res.params.index and v != "const"]

            b_fe = fe_res.params[shared].values
            b_re = re_res.params[shared].values
            diff = b_fe - b_re

            cov_diff = fe_res.cov.loc[shared, shared].values - re_res.cov.loc[shared, shared].values
            cov_inv = np.linalg.pinv(cov_diff)
            h_stat = float(diff @ cov_inv @ diff)
            df = len(shared)
            p_value = 1 - stats.chi2.cdf(h_stat, df)

            verdict = "FE" if p_value < 0.05 else "RE"

            records.append(
                {
                    "Model": idx_name,
                    "$\\chi^2$ Stat": f"{h_stat:.2f}",
                    "d.f.": str(df),
                    "p-value": f"{p_value:.3f}",
                    "Preferred": verdict,
                }
            )

            status = "✅ FE preferred" if p_value < 0.05 else "⚠️  RE consistent"
            print(f"  {idx_name}: χ²={h_stat:.2f}, p={p_value:.3f} → {status}")

        except Exception as e:
            records.append(
                {
                    "Model": idx_name,
                    "$\\chi^2$ Stat": "-",
                    "d.f.": "-",
                    "p-value": "-",
                    "Preferred": "-",
                }
            )
            print(f"  {idx_name}: Error — {e}")

    print("=" * 60 + "\n")

    summary_df = pd.DataFrame(records)

    latex_str = summary_df.to_latex(
        index=False,
        escape=False,
        column_format="lcccc",
        caption="Hausman Specification Test: Fixed Effects vs.\\ Random Effects",
        label="tab:hausman",
    )

    # Import map for nice labels
    try:
        from analysis.regression_utils import LATEX_LABEL_MAP

        for old, new in LATEX_LABEL_MAP.items():
            latex_str = latex_str.replace(old, new)
    except ImportError:
        pass

    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir / "hausman_table.tex"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"✅ Exported Hausman Test LaTeX table to: {out_file}")


def export_granger_causality_latex(
    df: pd.DataFrame,
    indices: list[str],
    dep_var: str = "sstran",
    id_var: str = "iso3",
    time_var: str = "year",
    max_lag: int = 2,
    out_dir: str = "outputs/tables",
) -> None:
    """
    Panel Granger causality test for each globalization index.

    Tests two directions:
      (A) Does lagged globalization Granger-cause welfare spending?
      (B) Does lagged welfare spending Granger-cause globalization?

    The test adds lagged values of the "cause" variable to a regression
    of the "effect" variable on its own lags, then runs an F-test on the
    joint significance of the added lags.
    """
    print("\n" + "=" * 60)
    print("🔄 RUNNING PANEL GRANGER CAUSALITY TESTS")
    print("=" * 60)

    records = []

    for idx_name in indices:
        if idx_name not in df.columns:
            continue

        # Build panel with lags
        panel = df[[id_var, time_var, dep_var, idx_name]].dropna().copy()
        panel = panel.sort_values([id_var, time_var])

        for lag in range(1, max_lag + 1):
            panel[f"{dep_var}_L{lag}"] = panel.groupby(id_var)[dep_var].shift(lag)
            panel[f"{idx_name}_L{lag}"] = panel.groupby(id_var)[idx_name].shift(lag)

        panel = panel.dropna()

        own_lags_y = [f"{dep_var}_L{lag_idx}" for lag_idx in range(1, max_lag + 1)]
        own_lags_x = [f"{idx_name}_L{lag_idx}" for lag_idx in range(1, max_lag + 1)]

        # ── Direction A: X → Y (does globalization Granger-cause welfare?) ──
        try:
            # Restricted: y ~ own lags of y + entity dummies
            entities = pd.get_dummies(panel[id_var], drop_first=True, dtype=float)
            X_r = sm.add_constant(pd.concat([panel[own_lags_y], entities], axis=1))
            # Unrestricted: adds lagged X
            X_u = sm.add_constant(pd.concat([panel[own_lags_y + own_lags_x], entities], axis=1))

            res_r = sm.OLS(panel[dep_var], X_r).fit()
            res_u = sm.OLS(panel[dep_var], X_u).fit()

            n = len(panel)
            q = len(own_lags_x)  # number of restrictions
            k_u = X_u.shape[1]

            f_stat = ((res_r.ssr - res_u.ssr) / q) / (res_u.ssr / (n - k_u))
            f_p = 1 - sp_stats.f.cdf(f_stat, q, n - k_u)

            direction_a = "Yes" if f_p < 0.05 else "No"
            records.append(
                {
                    "Direction": f"{idx_name} $\\rightarrow$ {dep_var}",
                    "F-Stat": f"{f_stat:.2f}",
                    "p-value": f"{f_p:.3f}",
                    "Granger-Causes?": direction_a,
                }
            )
            print(f"  {idx_name} → {dep_var}: F={f_stat:.2f}, p={f_p:.3f} → {direction_a}")
        except Exception as e:
            records.append(
                {
                    "Direction": f"{idx_name} $\\rightarrow$ {dep_var}",
                    "F-Stat": "-",
                    "p-value": "-",
                    "Granger-Causes?": "-",
                }
            )
            print(f"  {idx_name} → {dep_var}: Error — {e}")

        # ── Direction B: Y → X (reverse causality check) ──
        try:
            X_r2 = sm.add_constant(pd.concat([panel[own_lags_x], entities], axis=1))
            X_u2 = sm.add_constant(pd.concat([panel[own_lags_x + own_lags_y], entities], axis=1))

            res_r2 = sm.OLS(panel[idx_name], X_r2).fit()
            res_u2 = sm.OLS(panel[idx_name], X_u2).fit()

            q2 = len(own_lags_y)
            k_u2 = X_u2.shape[1]

            f_stat2 = ((res_r2.ssr - res_u2.ssr) / q2) / (res_u2.ssr / (n - k_u2))
            f_p2 = 1 - sp_stats.f.cdf(f_stat2, q2, n - k_u2)

            direction_b = "Yes" if f_p2 < 0.05 else "No"
            records.append(
                {
                    "Direction": f"{dep_var} $\\rightarrow$ {idx_name}",
                    "F-Stat": f"{f_stat2:.2f}",
                    "p-value": f"{f_p2:.3f}",
                    "Granger-Causes?": direction_b,
                }
            )
            print(f"  {dep_var} → {idx_name}: F={f_stat2:.2f}, p={f_p2:.3f} → {direction_b}")
        except Exception as e:
            records.append(
                {
                    "Direction": f"{dep_var} $\\rightarrow$ {idx_name}",
                    "F-Stat": "-",
                    "p-value": "-",
                    "Granger-Causes?": "-",
                }
            )
            print(f"  {dep_var} → {idx_name}: Error — {e}")

    print("=" * 60 + "\n")

    summary_df = pd.DataFrame(records)

    latex_str = summary_df.to_latex(
        index=False,
        escape=False,
        column_format="lccc",
        caption=f"Panel Granger Causality Tests (Lags = {max_lag})",
        label="tab:granger",
    )

    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir / "granger_causality_table.tex"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"✅ Exported Granger Causality LaTeX table to: {out_file}")


def export_reset_test_latex(final_models: dict, out_dir: str) -> None:
    """
    Run the Ramsey RESET test for functional form misspecification
    on each fully specified model and export results to LaTeX.

    The test adds powers of fitted values (ŷ², ŷ³) to the regression
    and tests their joint significance. Rejecting H0 (p < 0.05) suggests
    the linear functional form is misspecified.
    """
    print("\n" + "=" * 60)
    print("📐 RUNNING RAMSEY RESET TESTS (Functional Form)")
    print("=" * 60)

    records = []

    for idx_name, res in final_models.items():
        try:
            fitted = res.fitted_values

            # Get exog from the model
            exog = res.model.exog.dataframe
            if "const" not in exog.columns:
                exog = sm.add_constant(exog)

            # Dependent variable
            y = res.model.dependent.dataframe.iloc[:, 0]

            # Align indices
            common_idx = y.index.intersection(exog.index).intersection(fitted.index)
            y = y.loc[common_idx]
            exog = exog.loc[common_idx]
            fitted_vals = fitted.loc[common_idx]

            if isinstance(fitted_vals, pd.DataFrame):
                fitted_vals = fitted_vals.iloc[:, 0]

            # Restricted model (original specification)
            res_r = sm.OLS(y.values, exog.values).fit()

            # Unrestricted model: add ŷ² and ŷ³
            fv = fitted_vals.values
            powers = np.column_stack([fv**2, fv**3])
            X_u = np.column_stack([exog.values, powers])
            res_u = sm.OLS(y.values, X_u).fit()

            n = len(y)
            q = 2  # two added terms (ŷ², ŷ³)
            k_u = X_u.shape[1]

            f_stat = ((res_r.ssr - res_u.ssr) / q) / (res_u.ssr / (n - k_u))
            f_p = 1 - sp_stats.f.cdf(f_stat, q, n - k_u)

            verdict = "Misspecified" if f_p < 0.05 else "Adequate"
            records.append(
                {
                    "Model": idx_name,
                    "RESET F-Stat": f"{f_stat:.2f}",
                    "p-value": f"{f_p:.3f}",
                    "Functional Form": verdict,
                }
            )

            status = "⚠️  Misspecified" if f_p < 0.05 else "✅ Adequate"
            print(f"  {idx_name}: F={f_stat:.2f}, p={f_p:.3f} → {status}")

        except Exception as e:
            records.append(
                {
                    "Model": idx_name,
                    "RESET F-Stat": "-",
                    "p-value": "-",
                    "Functional Form": "-",
                }
            )
            print(f"  {idx_name}: Error — {e}")

    print("=" * 60 + "\n")

    summary_df = pd.DataFrame(records)

    latex_str = summary_df.to_latex(
        index=False,
        escape=False,
        column_format="lccc",
        caption="Ramsey RESET Test for Functional Form Misspecification",
        label="tab:reset",
    )

    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir / "reset_test_table.tex"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"✅ Exported RESET Test LaTeX table to: {out_file}")


def export_cointegration_latex(
    df: pd.DataFrame,
    indices: list[str],
    dep_var: str = "sstran",
    controls: list[str] = None,
    id_var: str = "iso3",
    time_var: str = "year",
    out_dir: str = "outputs/tables",
) -> None:
    """
    Kao (1999) Panel Cointegration Test.

    Tests whether non-stationary I(1) variables share a genuine long-run
    equilibrium (are cointegrated). The procedure:
      1. Estimate fixed-effects regression: y_it = α_i + β·x_it + e_it
      2. Obtain residuals ê_it
      3. Run ADF test on the pooled residuals
      4. If residuals are I(0), the variables are cointegrated

    H0: No cointegration (residuals have a unit root)
    Rejecting H0 (p < 0.05) means the variables ARE cointegrated,
    and the regression captures a real long-run relationship.
    """
    print("\n" + "=" * 60)
    print("🔗 RUNNING KAO PANEL COINTEGRATION TESTS")
    print("=" * 60)

    records = []

    for idx_name in indices:
        if idx_name not in df.columns:
            continue

        try:
            # Build the cointegrating regression variables
            reg_vars = [idx_name]
            if controls:
                reg_vars += [c for c in controls if c in df.columns]

            # Subset and drop NaNs
            panel = df[[id_var, time_var, dep_var] + reg_vars].dropna().copy()

            # Create entity dummies (fixed effects)
            entities = pd.get_dummies(panel[id_var], drop_first=True, dtype=float)
            X = sm.add_constant(pd.concat([panel[reg_vars], entities], axis=1))

            # Step 1: Estimate the cointegrating regression
            ols_res = sm.OLS(panel[dep_var], X).fit()

            # Step 2: Get residuals
            residuals = ols_res.resid

            # Step 3: ADF test on pooled residuals
            adf_stat, adf_p, used_lag, nobs, crit_vals, _ = adfuller(residuals, autolag="AIC")

            # Kao adjustment: the test statistic needs adjustment for
            # the number of regressors. Under Kao (1999), the ADF t-stat
            # on the residuals is compared against critical values that
            # account for the panel structure.
            cointegrated = "Yes" if adf_p < 0.05 else "No"

            records.append(
                {
                    "Specification": f"{dep_var} ~ {idx_name}"
                    + (" + controls" if controls else ""),
                    "ADF Stat": f"{adf_stat:.3f}",
                    "p-value": f"{adf_p:.3f}",
                    "Lags Used": str(used_lag),
                    "Cointegrated?": cointegrated,
                }
            )

            status = "✅ Cointegrated" if adf_p < 0.05 else "⚠️  No cointegration"
            print(f"  {dep_var} ~ {idx_name}: ADF={adf_stat:.3f}, p={adf_p:.3f} → {status}")

        except Exception as e:
            records.append(
                {
                    "Specification": f"{dep_var} ~ {idx_name}",
                    "ADF Stat": "-",
                    "p-value": "-",
                    "Lags Used": "-",
                    "Cointegrated?": "-",
                }
            )
            print(f"  {dep_var} ~ {idx_name}: Error — {e}")

    print("-" * 60)
    print("H0: No cointegration (residuals are non-stationary)")
    print("Reject H0 (p < 0.05) → Variables share a long-run equilibrium")
    print("=" * 60 + "\n")

    summary_df = pd.DataFrame(records)

    latex_str = summary_df.to_latex(
        index=False,
        escape=False,
        column_format="lcccc",
        caption="Kao Panel Cointegration Test (ADF on Residuals)",
        label="tab:cointegration",
    )

    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir / "cointegration_table.tex"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"✅ Exported Cointegration Test LaTeX table to: {out_file}")


def build_latex_appendix(
    tables_dir: str = "outputs/tables",
    out_file: str | None = None,
) -> None:
    """
    Compile all diagnostic LaTeX tables into a single appendix file
    with section headers, ready to paste into Overleaf.

    When ``out_file`` is None the appendix is written next to ``tables_dir``
    as ``<tables_dir>/appendix_diagnostics.tex``. Previously the default
    pointed at a hardcoded repo-relative path, which caused test pollution
    — export_stepwise_robustness_tables(out_dir=tmp_path) still wrote the
    appendix into the real ``outputs/tables/`` directory.
    """
    tables_dir = Path(tables_dir)
    out_file = Path(out_file) if out_file is not None else tables_dir / "appendix_diagnostics.tex"

    # Define the table ordering and section titles
    sections = [
        (
            "Pre-Estimation Variable Diagnostics",
            [
                (
                    "diagnostics_table.tex",
                    "Stationarity, Normality, Serial Correlation, and Cross-Sectional Dependence",
                ),
                ("vif_table.tex", "Variance Inflation Factor (Multicollinearity)"),
                ("cointegration_table.tex", "Panel Cointegration (Kao Test)"),
            ],
        ),
        (
            "Model Specification Tests",
            [
                ("hausman_table.tex", "Hausman Test: Fixed Effects vs.\\ Random Effects"),
                ("reset_test_table.tex", "Ramsey RESET Test for Functional Form"),
            ],
        ),
        (
            "Post-Estimation Residual Diagnostics",
            [
                (
                    "model_diagnostics.tex",
                    "Heteroskedasticity (Breusch-Pagan, White, Modified Wald)"
                    " and Ljung-Box Serial Correlation",
                ),
            ],
        ),
        (
            "Causality and Endogeneity",
            [
                ("granger_causality_table.tex", "Panel Granger Causality Tests"),
            ],
        ),
        (
            "Structural Stability",
            [
                (
                    "chow_test_table.tex",
                    "Chow Structural Break Test (known break at China WTO accession, 2000)",
                ),
                (
                    "qlr_test_table.tex",
                    "QLR / Sup-Wald Test for Unknown Structural Break (Andrews 1993)",
                ),
            ],
        ),
    ]

    print("\n" + "=" * 60)
    print("📚 BUILDING LATEX DIAGNOSTICS APPENDIX")
    print("=" * 60)

    lines = []
    lines.append("% ============================================================")
    lines.append("% ECONOMETRIC DIAGNOSTICS APPENDIX")
    lines.append("% Auto-generated — do not edit manually")
    lines.append("% ============================================================")
    lines.append("")

    tables_included = 0

    for section_title, tables in sections:
        # Check if any tables in this section exist
        existing = [(f, desc) for f, desc in tables if (tables_dir / f).exists()]
        if not existing:
            continue

        lines.append(f"\\subsection{{{section_title}}}")
        lines.append("")

        for filename, description in existing:
            filepath = tables_dir / filename

            # Read the table content
            with open(filepath, "r", encoding="utf-8") as f:
                table_content = f.read().strip()

            lines.append(f"% --- {description} ---")
            lines.append(table_content)
            lines.append("")
            tables_included += 1
            print(f"  ✅ Included: {filename}")

    # Write the combined appendix
    os.makedirs(out_file.parent, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n📚 Appendix compiled: {tables_included} tables → {out_file}")
    print("=" * 60 + "\n")
