"""
Regression utilities for panel data analysis.
"""
from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

# Centralized LaTeX variable to label mapping
LATEX_LABEL_MAP = {
    "KOFGI\\_lag1": "Globalization (Overall)$_{t-1}$",
    "KOFEcGI\\_lag1": "Globalization (Economic)$_{t-1}$",
    "KOFSoGI\\_lag1": "Globalization (Social)$_{t-1}$",
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
    "const": "Constant"
}

def prepare_regression_data(
    df: pd.DataFrame, 
    dep_var: str, 
    indep_var: str, 
    ctrls_lagged: list[str], 
    interactions: bool = False
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
        
        exog_vars = [indep_var, "int_conservative", "int_mediterranean", "int_liberal", "int_post_communist"] + ctrls_lagged
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
    if 'iso3' in ols_data.columns and 'year' in ols_data.columns:
        ols_data = ols_data.set_index(["iso3", "year"])
        
    return ols_data, exog_vars

def run_panel_ols(ols_data: pd.DataFrame, dep_var: str, exog_vars: list[str], entity_effects: bool = True, time_effects: bool = True):
    """
    Execute standard PanelOLS with clustered standard errors.
    
    Args:
        ols_data: Cleaned DataFrame with a MultiIndex.
        dep_var: Dependent Variable.
        exog_vars: Independent Variables to feed into model.
        entity_effects: Whether to include fixed entity effects.
        time_effects: Whether to include fixed time effects.
    """
    exog = sm.add_constant(ols_data[exog_vars])
    exog = exog.loc[:, ~exog.columns.duplicated()]
    
    model = PanelOLS(ols_data[dep_var], exog, entity_effects=entity_effects, time_effects=time_effects)
    results = model.fit(cov_type="clustered", cluster_entity=True)
    return results

def generate_marginal_effects(results, g_var: str) -> pd.DataFrame:
    """
    Given regression results model fitted with interactions, generate Marginal Effects Table.
    Assumes Social Democrat is reference group.
    
    Args:
        results: Fitted results from linearmodels.
        g_var: String representing the main variable that interactions are derived from.
    """
    params = results.params
    b1 = params[g_var]
    b2 = params.get("int_conservative", 0)
    b3 = params.get("int_mediterranean", 0)
    b4 = params.get("int_liberal", 0)
    b5 = params.get("int_post_communist", 0)

    me_table = pd.DataFrame({
        "Welfare Regime": ["Social Democrat (Ref)", "Conservative", "Mediterranean", "Liberal", "Post-Communist"],
        "Formula": ["β1", "β1 + β2", "β1 + β3", "β1 + β4", "β1 + β5"],
        "Marginal Effect": [b1, b1 + b2, b1 + b3, b1 + b4, b1 + b5]
    })
    
    return me_table
