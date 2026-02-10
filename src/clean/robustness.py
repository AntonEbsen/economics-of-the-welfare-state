"""
Robustness checks for regression analysis.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Callable


def run_robustness_checks(
    df: pd.DataFrame,
    model_formula: str,
    model_func: Callable,
    checks: list[str] = None
) -> dict:
    """
    Run automated robustness checks on a regression model.
    
    Args:
        df: Panel DataFrame
        model_formula: Regression formula (e.g., 'y ~ x1 + x2')
        model_func: Regression function (e.g., from statsmodels)
        checks: List of robustness checks to run:
            - 'drop_outliers': Remove extreme observations
            - 'winsorize': Winsorize at 1st and 99th percentiles
            - 'subsample_high': Top 50% by GDP per capita
            - 'subsample_low': Bottom 50% by GDP per capita
            - 'pre_2008': Before financial crisis
            - 'post_2008': After financial crisis
            
    Returns:
        Dictionary with results for each specification
        
    Example:
        >>> from statsmodels.formula.api import ols
        >>> results = run_robustness_checks(
        ...     master,
        ...     'sstran ~ ln_gdppc + deficit',
        ...     ols,
        ...     checks=['drop_outliers', 'winsorize', 'pre_2008']
        ... )
    """
    if checks is None:
        checks = ['drop_outliers', 'winsorize', 'pre_2008', 'post_2008']
    
    results = {}
    
    # Baseline model
    print("Running baseline model...")
    results['baseline'] = model_func(model_formula, data=df).fit()
    
    # Run each robustness check
    for check in checks:
        print(f"Running robustness check: {check}")
        
        if check == 'drop_outliers':
            df_robust = _drop_outliers(df, model_formula)
        elif check == 'winsorize':
            df_robust = _winsorize_data(df, model_formula)
        elif check == 'subsample_high':
            df_robust = _subsample_by_gdp(df, top_half=True)
        elif check == 'subsample_low':
            df_robust = _subsample_by_gdp(df, top_half=False)
        elif check == 'pre_2008':
            df_robust = df[df['year'] < 2008].copy()
        elif check == 'post_2008':
            df_robust = df[df['year'] >= 2008].copy()
        else:
            print(f"⚠️  Unknown check: {check}, skipping")
            continue
        
        if len(df_robust) < 100:
            print(f"⚠️  {check}: Too few observations ({len(df_robust)}), skipping")
            continue
        
        try:
            results[check] = model_func(model_formula, data=df_robust).fit()
            print(f"✅ {check}: {len(df_robust)} observations")
        except Exception as e:
            print(f"⚠️  {check} failed: {e}")
    
    return results


def _drop_outliers(df: pd.DataFrame, formula: str, threshold: float = 3.0) -> pd.DataFrame:
    """Drop observations with extreme values (>3 std dev from mean)."""
    # Extract variable names from formula
    import re
    vars_in_formula = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)
    vars_in_formula = [v for v in vars_in_formula if v in df.columns]
    
    df_clean = df.copy()
    for var in vars_in_formula:
        if df[var].dtype in [np.float64, np.int64]:
            mean = df[var].mean()
            std = df[var].std()
            df_clean = df_clean[
                (df_clean[var] >= mean - threshold * std) &
                (df_clean[var] <= mean + threshold * std)
            ]
    
    return df_clean


def _winsorize_data(df: pd.DataFrame, formula: str, limits: tuple = (0.01, 0.01)) -> pd.DataFrame:
    """Winsorize variables at specified percentiles."""
    from scipy.stats import mstats
    import re
    
    vars_in_formula = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', formula)
    vars_in_formula = [v for v in vars_in_formula if v in df.columns]
    
    df_wins = df.copy()
    for var in vars_in_formula:
        if df[var].dtype in [np.float64, np.int64]:
            df_wins[var] = mstats.winsorize(df[var].dropna(), limits=limits)
    
    return df_wins


def _subsample_by_gdp(df: pd.DataFrame, top_half: bool = True) -> pd.DataFrame:
    """Subsample by GDP per capita."""
    if 'ln_gdppc' not in df.columns:
        return df
    
    median_gdp = df.groupby('iso3')['ln_gdppc'].mean().median()
    
    if top_half:
        high_gdp_countries = df.groupby('iso3')['ln_gdppc'].mean()[
            df.groupby('iso3')['ln_gdppc'].mean() >= median_gdp
        ].index
        return df[df['iso3'].isin(high_gdp_countries)].copy()
    else:
        low_gdp_countries = df.groupby('iso3')['ln_gdppc'].mean()[
            df.groupby('iso3')['ln_gdppc'].mean() < median_gdp
        ].index
        return df[df['iso3'].isin(low_gdp_countries)].copy()


def compare_robustness_results(results_dict: dict, variable: str = None) -> pd.DataFrame:
    """
    Compare coefficients across robustness checks.
    
    Args:
        results_dict: Dictionary from run_robustness_checks()
        variable: Specific variable to compare. If None, show all.
        
    Returns:
        DataFrame comparing coefficients across specifications
    """
    comparison = []
    
    for spec_name, result in results_dict.items():
        if variable:
            if variable in result.params:
                comparison.append({
                    'specification': spec_name,
                    'coefficient': result.params[variable],
                    'std_error': result.bse[variable],
                    'p_value': result.pvalues[variable],
                    'n_obs': int(result.nobs)
                })
        else:
            for var in result.params.index:
                comparison.append({
                    'specification': spec_name,
                    'variable': var,
                    'coefficient': result.params[var],
                    'std_error': result.bse[var],
                    'p_value': result.pvalues[var]
                })
    
    df_comparison = pd.DataFrame(comparison)
    
    print("\n" + "=" * 60)
    print("ROBUSTNESS CHECK COMPARISON")
    print("=" * 60)
    if variable:
        print(f"\nVariable: {variable}\n")
    print(df_comparison.to_string(index=False))
    print("=" * 60)
    
    return df_comparison
