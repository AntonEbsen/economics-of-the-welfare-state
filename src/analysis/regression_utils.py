"""
Regression utilities for panel data analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# Centralized LaTeX variable to label mapping
LATEX_LABEL_MAP = {
    "KOFGI\\_lag1": "Globalization (Overall)$_{t-1}$",
    "KOFEcGI\\_lag1": "Globalization (Economic)$_{t-1}$",
    "KOFTrGI\\_lag1": "Globalization (Trade)$_{t-1}$",
    "KOFFiGI\\_lag1": "Globalization (Financial)$_{t-1}$",
    "KOFSoGI\\_lag1": "Globalization (Social)$_{t-1}$",
    "KOFIpGI\\_lag1": "Globalization (Interpersonal)$_{t-1}$",
    "KOFInGI\\_lag1": "Globalization (Informational)$_{t-1}$",
    "KOFCuGI\\_lag1": "Globalization (Cultural)$_{t-1}$",
    "KOFPoGI\\_lag1": "Globalization (Political)$_{t-1}$",
    "sstran\\_lag1": "Social Security Transfers$_{t-1}$",
    "int\\_conservative": "$\\times$ Conservative",
    "int\\_mediterranean": "$\\times$ Mediterranean",
    "int\\_liberal": "$\\times$ Liberal",
    "int\\_post_communist": "$\\times$ Post-Communist",
    "ln\\_gdppc\\_lag1": "log GDP per capita$_{t-1}$",
    "inflation\\_cpi\\_lag1": "Inflation$_{t-1}$",
    "deficit\\_lag1": "Deficit$_{t-1}$",
    "debt\\_lag1": "Debt$_{t-1}$",
    "ln\\_population\\_lag1": "log Population$_{t-1}$",
    "dependency\\_ratio\\_lag1": "Dependency Ratio$_{t-1}$",
    "const": "Constant",
}


def prepare_regression_data(
    df: pd.DataFrame,
    dep_var: str,
    indep_var: str,
    ctrls_lagged: list[str],
    interactions: bool = False,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Prepare DataFrame and exog_vars for PanelOLS regression.

    Args:
        df: The DataFrame with ALL variables including lags created.
        dep_var: The name of the dependent variable column.
        indep_var: The name of the independent variable column.
        ctrls_lagged: List of control variables that have already been created/lagged.
        interactions: If True, creates interaction terms with welfare_regime indicators.

    Returns:
        tuple containing:
        - ols_data: DataFrame with NaN rows dropped and MultiIndex set.
        - exog_vars: List of independent variables (including interactions if any) + controls.
    """
    # Create interactions if requested
    reg_data = df.copy()
    if interactions:
        reg_data["int_conservative"] = reg_data[indep_var] * reg_data["regime_conservative"]
        reg_data["int_mediterranean"] = reg_data[indep_var] * reg_data["regime_mediterranean"]
        reg_data["int_liberal"] = reg_data[indep_var] * reg_data["regime_liberal"]
        reg_data["int_post_communist"] = reg_data[indep_var] * reg_data["regime_post_communist"]

        exog_vars = [
            indep_var,
            "int_conservative",
            "int_mediterranean",
            "int_liberal",
            "int_post_communist",
        ] + ctrls_lagged
    else:
        exog_vars = [indep_var] + ctrls_lagged

    # CRITICAL FIX: Ensure no duplicate column names before dict.fromkeys execution
    reg_data = reg_data.loc[:, ~reg_data.columns.duplicated()]

    # Make exog_vars unique incase indep_var == dep_var or similar (shouldn't happen but safe)
    exog_vars = list(dict.fromkeys(exog_vars))

    # Filter out NaNs for the variables to be used in model
    vars_to_check = [dep_var] + exog_vars
    ols_data = reg_data.dropna(subset=vars_to_check).copy()

    # Set MultiIndex for panel regression
    if "iso3" in ols_data.columns and "year" in ols_data.columns:
        ols_data = ols_data.set_index(["iso3", "year"])

    return ols_data, exog_vars


def run_panel_ols(
    ols_data: pd.DataFrame,
    dep_var: str,
    exog_vars: list[str],
    entity_effects: bool = True,
    time_effects: bool = True,
    cov_type: str = "clustered",
    cluster_entity: bool = True,
    cluster_time: bool = True,
):
    """
    Execute standard PanelOLS with robust standard errors.
    By default, uses Two-Way Clustering (entity + time).
    """
    exog = sm.add_constant(ols_data[exog_vars])
    exog = exog.loc[:, ~exog.columns.duplicated()]

    model = PanelOLS(
        ols_data[dep_var], exog, entity_effects=entity_effects, time_effects=time_effects
    )
    if cov_type == "clustered":
        results = model.fit(
            cov_type=cov_type, cluster_entity=cluster_entity, cluster_time=cluster_time
        )
    else:
        results = model.fit(cov_type=cov_type)
    return results


def run_event_study(
    ols_data: pd.DataFrame,
    dep_var: str,
    treat_var: str,
    event_year: int = 2000,
    window: int = 5,
    exog_vars: list[str] = None,
):
    """
    Compute an event study design around a specific year.
    Creates interactions between the treatment variable and year relative to the event.
    """
    df_event = ols_data.copy()

    # Extract 'year' from the index if needed
    if "year" not in df_event.columns and "year" in df_event.index.names:
        df_event = df_event.reset_index()

    df_event["rel_time"] = df_event["year"] - event_year
    df_event = df_event[(df_event["rel_time"] >= -window) & (df_event["rel_time"] <= window)].copy()

    interaction_cols = []
    for t in range(-window, window + 1):
        if t == -1:
            continue  # Baseline
        col_name = f"event_{t}"
        df_event[col_name] = (df_event["rel_time"] == t).astype(int) * df_event[treat_var]
        interaction_cols.append(col_name)

    final_exogs = interaction_cols
    if exog_vars:
        # Filter out the main treatment var if it was provided, since we interact it
        final_exogs += [v for v in exog_vars if v != treat_var]

    df_event = df_event.set_index(["iso3", "year"])
    res = run_panel_ols(df_event, dep_var, final_exogs)

    plot_data = []
    for t in range(-window, window + 1):
        if t == -1:
            plot_data.append({"rel_time": t, "coef": 0.0, "lower": 0.0, "upper": 0.0})
        else:
            col_name = f"event_{t}"
            coef = res.params[col_name]
            se = res.std_errors[col_name]
            plot_data.append(
                {"rel_time": t, "coef": coef, "lower": coef - 1.96 * se, "upper": coef + 1.96 * se}
            )

    return pd.DataFrame(plot_data), res


def run_hausman_test(
    ols_data: pd.DataFrame,
    dep_var: str,
    exog_vars: list[str],
) -> pd.DataFrame:
    """
    Perform the Hausman specification test (Fixed Effects vs. Random Effects).

    The null hypothesis is that Random Effects (RE) is consistent and efficient.
    Rejecting H0 (low p-value) means FE is preferred.

    Args:
        ols_data: Cleaned DataFrame with MultiIndex (entity, time).
        dep_var: Dependent variable name.
        exog_vars: List of independent variable names.

    Returns:
        DataFrame summarising the Hausman test statistic, p-value, and verdict.
    """
    import numpy as np
    from linearmodels.panel import RandomEffects
    from scipy import stats

    exog = sm.add_constant(ols_data[exog_vars])
    exog = exog.loc[:, ~exog.columns.duplicated()]

    # Estimate FE (within) and RE models
    fe_res = PanelOLS(ols_data[dep_var], exog, entity_effects=True, time_effects=False).fit(
        cov_type="unadjusted"
    )
    re_res = RandomEffects(ols_data[dep_var], exog).fit(cov_type="unadjusted")

    # Shared coefficients (exclude constant, which is absent in FE)
    shared = [v for v in fe_res.params.index if v in re_res.params.index and v != "const"]

    b_fe = fe_res.params[shared].values
    b_re = re_res.params[shared].values

    diff = b_fe - b_re
    cov_fe = fe_res.cov.loc[shared, shared].values
    cov_re = re_res.cov.loc[shared, shared].values

    cov_diff = cov_fe - cov_re

    # Use pseudo-inverse for numerical stability
    cov_inv = np.linalg.pinv(cov_diff)
    h_stat = float(diff @ cov_inv @ diff)
    df = len(shared)
    p_value = 1 - stats.chi2.cdf(h_stat, df)

    verdict = (
        "Reject H₀ → Fixed Effects preferred"
        if p_value < 0.05
        else "Fail to reject H₀ → Random Effects consistent"
    )

    return pd.DataFrame(
        {
            "Hausman Statistic": [round(h_stat, 4)],
            "Degrees of Freedom": [df],
            "P-Value": [round(p_value, 4)],
            "Verdict (α=0.05)": [verdict],
        }
    )


def generate_marginal_effects(results, g_var: str) -> pd.DataFrame:
    """
    Given regression results fitted with interactions, generate a
    marginal-effects table with standard errors, t-statistics, p-values,
    and significance stars.

    The marginal effect for the reference group (Social Democrat) is
    simply β₁.  For each non-reference regime *k* it is β₁ + β_k.
    Standard errors are computed from the variance–covariance matrix::

        SE(β₁ + β_k) = sqrt(Var(β₁) + Var(β_k) + 2 Cov(β₁, β_k))

    Only regimes whose interaction term actually appears in the model
    are included (so ``export_interaction_excl_postcommunist_table``
    produces a four-row table without Post-Communist).

    Args:
        results: Fitted PanelResults from linearmodels.
        g_var: Name of the base globalisation variable (e.g. ``KOFGI_lag1``).

    Returns:
        DataFrame with columns ``Welfare Regime``, ``Marginal Effect``,
        ``Std. Error``, ``t-stat``, ``p-value``, ``Sig.``.
    """
    from scipy import stats as sp_stats

    params = results.params
    cov = results.cov

    b1 = params[g_var]
    var_b1 = cov.loc[g_var, g_var]

    interaction_map = [
        ("int_conservative", "Conservative"),
        ("int_mediterranean", "Mediterranean"),
        ("int_liberal", "Liberal"),
        ("int_post_communist", "Post-Communist"),
    ]

    rows = []

    # Reference group (Social Democrat): ME = β₁, SE = SE(β₁)
    se_ref = np.sqrt(var_b1)
    t_ref = b1 / se_ref if se_ref > 0 else np.nan
    p_ref = (
        2 * (1 - sp_stats.t.cdf(abs(t_ref), df=results.df_resid)) if not np.isnan(t_ref) else np.nan
    )
    rows.append(
        {
            "Welfare Regime": "Social Democrat (Ref)",
            "Marginal Effect": b1,
            "Std. Error": se_ref,
            "t-stat": t_ref,
            "p-value": p_ref,
            "Sig.": significance_stars(p_ref),
        }
    )

    # Non-reference regimes: ME = β₁ + β_k
    for int_term, regime_label in interaction_map:
        if int_term not in params.index:
            continue
        bk = params[int_term]
        me = b1 + bk
        var_bk = cov.loc[int_term, int_term]
        cov_b1_bk = cov.loc[g_var, int_term]
        se = np.sqrt(var_b1 + var_bk + 2 * cov_b1_bk)
        t_val = me / se if se > 0 else np.nan
        p_val = (
            2 * (1 - sp_stats.t.cdf(abs(t_val), df=results.df_resid))
            if not np.isnan(t_val)
            else np.nan
        )
        rows.append(
            {
                "Welfare Regime": regime_label,
                "Marginal Effect": me,
                "Std. Error": se,
                "t-stat": t_val,
                "p-value": p_val,
                "Sig.": significance_stars(p_val),
            }
        )

    return pd.DataFrame(rows)


def significance_stars(p: float) -> str:
    """Return conventional significance stars for a p-value.

    ``***`` (<0.01), ``**`` (<0.05), ``*`` (<0.10). NaN-safe.
    """
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def plot_coefficients(results, title: str = "Regression Coefficients"):
    """
    Plot coefficients with 95% confidence intervals.

    Args:
        results: Fitted results from linearmodels.
        title: Title of the forest plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    params = results.params
    std_errors = results.std_errors
    variables = params.index

    # Calculate confidence intervals (1.96 * SE for 95%)
    lower_ci = params - 1.96 * std_errors
    upper_ci = params + 1.96 * std_errors

    df_plot = pd.DataFrame(
        {"variable": variables, "coefficient": params, "lower": lower_ci, "upper": upper_ci}
    )

    # Don't plot the constant
    df_plot = df_plot[df_plot["variable"] != "const"]

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Plot point estimates
    plt.errorbar(
        x=df_plot["coefficient"],
        y=df_plot["variable"],
        xerr=[df_plot["coefficient"] - df_plot["lower"], df_plot["upper"] - df_plot["coefficient"]],
        fmt="o",
        color="royalblue",
        capsize=5,
        markersize=8,
    )

    plt.axvline(x=0, color="red", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def adjust_pvalues(pvalues: pd.Series, method: str = "fdr_bh") -> pd.DataFrame:
    """
    Adjust p-values for multiple hypothesis testing.

    Args:
        pvalues: Series of raw p-values.
        method: Correction method ('fdr_bh' for Benjamini-Hochberg, 'bonferroni', etc.)
    """
    from statsmodels.stats.multitest import multipletests

    rejected, corrected, _, _ = multipletests(pvalues, alpha=0.05, method=method)

    return pd.DataFrame(
        {
            "Variable": pvalues.index,
            "Raw P-Value": pvalues.values,
            "Corrected P-Value": corrected,
            "Significant (0.05)": rejected,
        }
    )


def run_placebo_test(
    ols_data: pd.DataFrame,
    dep_var: str,
    indep_var: str,
    exog_vars: list[str],
    n_sims: int = 100,
    seed: int = 42,
):
    """
    Run a placebo test by shuffling the independent variable.

    Args:
        ols_data: DataFrame for regression.
        dep_var: Dependent variable.
        indep_var: The variable to shuffle.
        exog_vars: All exogenous variables (including the one to be shuffled).
        n_sims: Number of simulations.
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    coefficients = []

    for _ in range(n_sims):
        mock_data = ols_data.copy()
        # Shuffle the independent variable within each entity (country) to preserve structure
        # but destroy the correlation with the dependent variable
        mock_data[indep_var] = mock_data.groupby(level=0)[indep_var].transform(rng.permutation)

        results = run_panel_ols(mock_data, dep_var, exog_vars)
        coefficients.append(results.params[indep_var])

    return np.array(coefficients)
