"""
Statistical tests for panel data and time series analysis.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Internal helper: Phillips-Perron test
# ---------------------------------------------------------------------------


def _phillips_perron(series: np.ndarray) -> tuple[float, float]:
    """
    Phillips-Perron (1988) unit root test – demeaned (constant) specification.

    PP modifies the ADF t-statistic with a non-parametric Newey-West HAC
    correction for serial correlation and heteroskedasticity, so no lag
    truncation choice is needed (unlike ADF where the lag order matters).

    Returns
    -------
    (stat, p_value) : the PP tau statistic and its MacKinnon (1994) p-value.
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.stats.sandwich_covariance import cov_hac
    from statsmodels.tsa.adfvalues import mackinnonp

    y = np.asarray(series, dtype=float)
    n = len(y)

    # --- OLS: Δy_t = α + β·y_{t-1} + ε_t  (no lags of Δy — that is the key
    #     difference from ADF; PP handles serial correlation non-parametrically)
    dy = np.diff(y)  # Δy_t
    y_lag = y[:-1]  # y_{t-1}
    X = np.column_stack([np.ones(len(dy)), y_lag])

    res = OLS(dy, X).fit()

    # Standard OLS quantities
    beta_hat = res.params[1]  # OLS estimate of β (unit-root coeff)
    se_ols = res.bse[1]  # OLS s.e. of β
    s2 = res.mse_resid  # OLS residual variance s²

    # Newey-West HAC long-run variance estimate (bandwidth by rule-of-thumb)
    bw = int(np.floor(4 * (n / 100) ** (2 / 9)))  # common bandwidth choice
    hac_cov = cov_hac(res, nlags=bw)
    lrv = hac_cov[1, 1] * (n - 2)  # scale HAC cov back to long-run variance

    # PP-adjusted t-statistic (Hamilton 1994, eq. 17.7)
    # Zt = (s²/lrv)^0.5 * t_OLS - 0.5*(lrv - s²) * n * se_ols / (lrv^0.5 * s²^0.5)
    t_ols = beta_hat / se_ols
    if lrv <= 0 or s2 <= 0:
        return float(t_ols), float(np.nan)

    pp_stat = ((s2 / lrv) ** 0.5) * t_ols - 0.5 * (lrv - s2) * (n * se_ols) / (lrv**0.5 * s2**0.5)

    # MacKinnon p-value table (same critical values as ADF, constant-only spec)
    p_value = mackinnonp(float(pp_stat), regression="c", N=1)
    p_value = float(np.clip(p_value, 0.0, 1.0))

    return float(pp_stat), p_value


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def test_stationarity(
    df: pd.DataFrame,
    variables: list[str],
    test: Literal["adf", "kpss", "pp", "all"] = "adf",
    id_var: str = "iso3",
    time_var: str = "year",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run unit root / stationarity tests on panel data.

    For each variable the chosen test is applied country-by-country;
    results are then aggregated (% stationary, mean p-value).

    Args:
        df: Panel DataFrame indexed by (id_var, time_var).
        variables: Column names to test.
        test: Which test(s) to run:
            - 'adf'  – Augmented Dickey-Fuller  (H0: unit root exists)
            - 'kpss' – KPSS                     (H0: series is stationary)
            - 'pp'   – Phillips-Perron           (H0: unit root exists)
            - 'all'  – All three; returns a pivoted side-by-side table.
        id_var: Column for cross-sectional identifier (default 'iso3').
        time_var: Column for time identifier (default 'year').
        verbose: Print results while running.

    Returns:
        For test in ('adf','kpss','pp'): long-form DataFrame with one row
        per variable, columns: variable, test, null_H0, pct_stationary,
        mean_p_value, n_countries, interpretation.

        For test='all': wide-form DataFrame with one row per variable,
        one column group per test, plus a 'consensus' column that shows
        'I(0) ✅' only when all three tests agree on stationarity.

    Notes:
        - ADF and PP share the same null (unit root); KPSS has the opposite
          null (stationarity).  The gold standard is: ADF rejects AND PP
          rejects AND KPSS does NOT reject → conclude I(0).
        - Phillips-Perron uses a Newey-West HAC correction, removing the
          need to choose a lag truncation like ADF.

    Example::

        # Single test
        results = test_stationarity(master, ['ln_gdppc', 'sstran'], test='adf')

        # All three in one call (recommended for publication)
        results = test_stationarity(
            master,
            ['ln_gdppc', 'sstran', 'deficit'],
            test='all'
        )
        print(results[['variable', 'consensus']])
    """
    from statsmodels.tsa.stattools import adfuller, kpss

    tests_to_run: list[str] = ["adf", "kpss", "pp"] if test == "all" else [test]

    if verbose:
        print("=" * 65)
        print(f"STATIONARITY TESTS  ({' + '.join(t.upper() for t in tests_to_run)})")
        print("=" * 65)
        print("H0: ADF / PP → unit root   (reject p<0.05 → stationary)")
        print("H0: KPSS     → stationary  (fail to reject p>0.05 → stationary)")
        print("-" * 65)

    records: list[dict] = []

    for var in variables:
        if var not in df.columns:
            print(f"⚠️  Variable '{var}' not found — skipping")
            continue

        per_test: dict[str, list[dict]] = {t: [] for t in tests_to_run}

        for country in df[id_var].unique():
            series = df[df[id_var] == country].sort_values(time_var)[var].dropna()
            if len(series) < 10:
                continue

            for t in tests_to_run:
                try:
                    if t == "adf":
                        stat, p_val, *_ = adfuller(series, autolag="AIC")
                        stationary = p_val < 0.05
                    elif t == "kpss":
                        # Note reversed interpretation: fail to reject → stationary
                        stat, p_val, *_ = kpss(series, regression="c", nlags="auto")
                        stationary = p_val > 0.05
                    elif t == "pp":
                        stat, p_val = _phillips_perron(series.values)
                        stationary = (not np.isnan(p_val)) and (p_val < 0.05)

                    per_test[t].append(
                        dict(country=country, stat=stat, p_val=p_val, stationary=stationary)
                    )
                except Exception as exc:
                    if verbose:
                        print(f"   ⚠️  {var} / {country} [{t.upper()}]: {exc}")

        for t in tests_to_run:
            rows = per_test[t]
            if not rows:
                continue

            cr = pd.DataFrame(rows)
            pct_stat = cr["stationary"].mean() * 100
            mean_p = cr["p_val"].mean()
            interp = "Stationary" if pct_stat > 50 else "Non-stationary"

            records.append(
                dict(
                    variable=var,
                    test=t.upper(),
                    null_H0="Unit root" if t in ("adf", "pp") else "Stationary",
                    pct_stationary=round(pct_stat, 1),
                    mean_p_value=round(mean_p, 4),
                    n_countries=len(cr),
                    interpretation=interp,
                )
            )

            if verbose:
                icon = "✅" if interp == "Stationary" else "⚠️ "
                print(f"\n{icon} {var}  [{t.upper()}]")
                print(f"   {pct_stat:.1f}% of countries stationary  |  mean p = {mean_p:.4f}")
                print(f"   → {interp}")

    if verbose:
        print("\n" + "=" * 65)
        if test != "all":
            print("Tip: use test='all' to run ADF + KPSS + PP in one call.")

    results_df = pd.DataFrame(records)
    if results_df.empty:
        return results_df

    # 'all' mode: pivot to a wide comparison table
    if test == "all":
        pivot = results_df.pivot(
            index="variable",
            columns="test",
            values=["pct_stationary", "mean_p_value", "interpretation"],
        )
        pivot.columns = [f"{metric}_{t}" for metric, t in pivot.columns]
        pivot = pivot.reset_index()

        expected = {"interpretation_ADF", "interpretation_KPSS", "interpretation_PP"}
        if expected.issubset(pivot.columns):
            pivot["consensus"] = pivot.apply(
                lambda r: (
                    "I(0) ✅"
                    if (
                        r["interpretation_ADF"] == "Stationary"
                        and r["interpretation_PP"] == "Stationary"
                        and r["interpretation_KPSS"] == "Stationary"
                    )
                    else "I(1) ⚠️ "
                ),
                axis=1,
            )

        if verbose:
            print("\nSide-by-side summary:")
            cols = ["variable"] + [
                c for c in pivot.columns if "interpretation" in c or c == "consensus"
            ]
            print(pivot[cols].to_string(index=False))

        return pivot

    return results_df


# ---------------------------------------------------------------------------


def panel_unit_root_test(
    df: pd.DataFrame,
    variable: str,
    test: Literal["ips", "llc", "breitung"] = "ips",
    id_var: str = "iso3",
    time_var: str = "year",
) -> dict:
    """
    Panel unit root tests (requires linearmodels package).

    Args:
        df: Panel DataFrame
        variable: Variable to test
        test: 'ips' (Im-Pesaran-Shin), 'llc' (Levin-Lin-Chu), or 'breitung'
        id_var: Panel identifier column
        time_var: Time identifier column

    Returns:
        Dictionary with test results
    """
    try:
        from linearmodels.panel.unit_root import IPS
    except ImportError:
        print("⚠️  linearmodels package required: pip install linearmodels")
        return None

    df_sorted = df.sort_values([id_var, time_var]).set_index([id_var, time_var])
    result = IPS(df_sorted[[variable]], trend="c")

    return {
        "variable": variable,
        "test": test.upper(),
        "statistic": result.stat,
        "p_value": result.pvalue,
        "stationary": result.pvalue < 0.05,
        "lags": result.lags,
    }


def test_normality(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """
    Test normality of variables using the Jarque-Bera test.

    Args:
        df: DataFrame
        variables: Variables to test

    Returns:
        DataFrame with normality test results
    """
    from scipy.stats import jarque_bera

    results = []
    for var in variables:
        if var not in df.columns:
            continue
        data = df[var].dropna()
        jb_stat, jb_p = jarque_bera(data)
        results.append(
            {
                "variable": var,
                "test": "Jarque-Bera",
                "statistic": jb_stat,
                "p_value": jb_p,
                "normal": jb_p > 0.05,
            }
        )

    print("Normality Tests (Jarque-Bera)")
    print("-" * 40)
    for r in results:
        status = "✅" if r["normal"] else "⚠️ "
        print(f"{status} {r['variable']}: p={r['p_value']:.4f}")

    return pd.DataFrame(results)
