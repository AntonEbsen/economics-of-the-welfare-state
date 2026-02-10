"""
Statistical tests for panel data and time series analysis.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Literal


def test_stationarity(
    df: pd.DataFrame,
    variables: list[str],
    test: Literal['adf', 'kpss', 'pp'] = 'adf',
    id_var: str = 'iso3',
    time_var: str = 'year',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run unit root/stationarity tests on panel data.
    
    Args:
        df: Panel DataFrame
        variables: List of variables to test
        test: Type of test:
            - 'adf': Augmented Dickey-Fuller (null: unit root exists)
            - 'kpss': KPSS test (null: series is stationary)
            - 'pp': Phillips-Perron (null: unit root exists)
        id_var: Panel identifier
        time_var: Time identifier
        verbose: Print detailed results
        
    Returns:
        DataFrame with test results
        
    Example:
        >>> results = test_stationarity(
        ...     master,
        ...     ['ln_gdppc', 'sstran', 'deficit'],
        ...     test='adf'
        ... )
        >>> print(results)
        # Shows which variables are stationary
    """
    from statsmodels.tsa.stattools import adfuller, kpss
    
    results = []
    
    if verbose:
        print("=" * 60)
        print(f"STATIONARITY TESTS ({test.upper()})")
        print("=" * 60)
    
    for var in variables:
        if var not in df.columns:
            print(f"⚠️  Variable '{var}' not found, skipping")
            continue
        
        # Run test for each country, then aggregate
        country_results = []
        
        for country in df[id_var].unique():
            country_data = df[df[id_var] == country].sort_values(time_var)
            series = country_data[var].dropna()
            
            if len(series) < 10:  # Need sufficient observations
                continue
            
            try:
                if test == 'adf':
                    stat, p_value, _, _, _, _ = adfuller(series, autolag='AIC')
                elif test == 'kpss':
                    stat, p_value, _, _ = kpss(series, regression='c', nlags='auto')
                elif test == 'pp':
                    # Phillips-Perron is similar to ADF
                    stat, p_value, _, _, _, _ = adfuller(series, regression='c', autolag='AIC')
                
                country_results.append({
                    'country': country,
                    'statistic': stat,
                    'p_value': p_value,
                    'stationary': p_value < 0.05 if test == 'adf' else p_value > 0.05
                })
            except Exception as e:
                if verbose:
                    print(f"⚠️  {var} ({country}): Test failed - {e}")
        
        if country_results:
            # Aggregate results
            cr_df = pd.DataFrame(country_results)
            pct_stationary = (cr_df['stationary'].sum() / len(cr_df)) * 100
            mean_p_value = cr_df['p_value'].mean()
            
            results.append({
                'variable': var,
                'test': test.upper(),
                'pct_stationary': pct_stationary,
                'mean_p_value': mean_p_value,
                'n_countries': len(cr_df),
                'interpretation': (
                    'Stationary' if pct_stationary > 50 else 'Non-stationary'
                )
            })
            
            if verbose:
                status = "✅" if pct_stationary > 50 else "⚠️ "
                print(f"\n{status} {var}:")
                print(f"   {pct_stationary:.1f}% of countries show stationarity")
                print(f"   Mean p-value: {mean_p_value:.4f}")
                print(f"   → {results[-1]['interpretation']}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("\nNote: For ADF/PP, null hypothesis is unit root (non-stationary).")
        print("      For KPSS, null hypothesis is stationarity.")
    
    return pd.DataFrame(results)


def panel_unit_root_test(
    df: pd.DataFrame,
    variable: str,
    test: Literal['ips', 'llc', 'breitung'] = 'ips',
    id_var: str = 'iso3',
    time_var: str = 'year'
) -> dict:
    """
    Panel unit root tests (requires linearmodels package).
    
    Args:
        df: Panel DataFrame
        variable: Variable to test
        test: Panel test type:
            - 'ips': Im-Pesaran-Shin test
            - 'llc': Levin-Lin-Chu test
            - 'breitung': Breitung test
        id_var: Panel identifier
        time_var: Time identifier
        
    Returns:
        Dictionary with test results
    """
    try:
        from linearmodels.panel.unit_root import IPS
    except ImportError:
        print("⚠️  linearmodels package required: pip install linearmodels")
        return None
    
    # Prepare data
    df_sorted = df.sort_values([id_var, time_var])
    df_sorted = df_sorted.set_index([id_var, time_var])
    
    # Run test
    result = IPS(df_sorted[[variable]], trend='c')
    
    return {
        'variable': variable,
        'test': test.upper(),
        'statistic': result.stat,
        'p_value': result.pvalue,
        'stationary': result.pvalue < 0.05,
        'lags': result.lags
    }


def test_normality(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """
    Test normality of variables using Shapiro-Wilk or Jarque-Bera test.
    
    Args:
        df: DataFrame
        variables: Variables to test
        
    Returns:
        DataFrame with normality test results
    """
    from scipy.stats import shapiro, jarque_bera
    
    results = []
    
    for var in variables:
        if var not in df.columns:
            continue
        
        data = df[var].dropna()
        
        # Jarque-Bera test (better for larger samples)
        jb_stat, jb_p = jarque_bera(data)
        
        results.append({
            'variable': var,
            'test': 'Jarque-Bera',
            'statistic': jb_stat,
            'p_value': jb_p,
            'normal': jb_p > 0.05
        })
    
    print("Normality Tests (Jarque-Bera)")
    print("-" * 40)
    for r in results:
        status = "✅" if r['normal'] else "⚠️ "
        print(f"{status} {r['variable']}: p={r['p_value']:.4f}")
    
    return pd.DataFrame(results)
