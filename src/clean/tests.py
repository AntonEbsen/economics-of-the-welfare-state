import os
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, norm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.tsa.adfvalues import mackinnonp
from statsmodels.tsa.stattools import adfuller, kpss

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
        df: Panel DataFrame (levels, not diffed)
        variable: Variable to test
        test: 'ips' (Im-Pesaran-Shin), 'llc' (Levin-Lin-Chu), or 'breitung'
        id_var: Panel identifier column
        time_var: Time identifier column

    Returns:
        Dictionary with test results
    """
    try:
        from linearmodels.panel.unit_root import IPS, LLC, Breitung
    except ImportError:
        print("⚠️  linearmodels package required: pip install linearmodels")
        return None

    # Linearmodels requires a 2-level MultiIndex (entity, time)
    df_sorted = df.sort_values([id_var, time_var]).set_index([id_var, time_var])
    data = df_sorted[[variable]].dropna()

    if test == "ips":
        result = IPS(data, trend="c")
    elif test == "llc":
        result = LLC(data, trend="c")
    elif test == "breitung":
        result = Breitung(data, trend="c")
    else:
        raise ValueError(f"Unknown test type: {test}")

    return {
        "variable": variable,
        "test": test.upper(),
        "statistic": round(float(result.stat), 4),
        "p_value": round(float(result.pvalue), 4),
        "stationary": result.pvalue < 0.05,
    }


def generate_diagnostic_report(
    df: pd.DataFrame,
    variables: list[str],
    id_var: str = "iso3",
    time_var: str = "year",
) -> pd.DataFrame:
    """
    Generate a research-grade diagnostic report for key variables.

    Aggregates:
    - Stationarity (IPS Panel Unit Root)
    - Normality (Jarque-Bera)
    - Basic missingness information

    Args:
        df: Merged panel DataFrame
        variables: Key variables to analyze
        id_var: Identifier for countries
        time_var: Identifier for years

    Returns:
        Summary DataFrame
    """
    print("\n" + "=" * 60)
    print("📊 GENERATING RESEARCH DIAGNOSTIC REPORT")
    print("=" * 60)

    summary_records = []

    for var in variables:
        if var not in df.columns:
            continue

        # 1. Missingness
        n_missing = df[var].isnull().sum()
        pct_missing = (n_missing / len(df)) * 100

        # 2. Stationarity (ADF Panel Test, aggregated Fisher-style)
        try:
            # Returns a DataFrame with 'pct_stationary', 'mean_p_value', 'interpretation'
            st_df = test_stationarity(
                df, [var], test="adf", id_var=id_var, time_var=time_var, verbose=False
            )

            if not st_df.empty:
                is_stat = st_df.iloc[0]["interpretation"] == "Stationary"
                st_status = "I(0) ✅" if is_stat else "I(1) ⚠️ "
                st_p = st_df.iloc[0]["mean_p_value"]
            else:
                st_status = "Fail"
                st_p = None
        except Exception:
            st_status = "Fail"
            st_p = None

        # 3. Normality (Jarque-Bera)
        try:
            data = df[var].dropna()
            _, jb_p = jarque_bera(data)
            norm_status = "Normal ✅" if jb_p > 0.05 else "Skewed ⚠️ "
        except Exception:
            norm_status = "Fail"

        summary_records.append(
            {
                "Variable": var,
                "Missing (%)": round(pct_missing, 1),
                "Stationarity": st_status,
                "ADF mean p-value": st_p,
                "Normality": norm_status,
            }
        )

    summary_df = pd.DataFrame(summary_records)

    # Print results to console
    print(summary_df.to_string(index=False))
    print("-" * 60)
    print("Interpretation: I(0) = Stationary, I(1) = Non-stationary (Unit Root)")
    print("Normal: p > 0.05 (H0: Series is normal)")
    print("=" * 60 + "\n")

    return summary_df


def test_normality(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    """
    Test normality of variables using the Jarque-Bera test.

    Args:
        df: DataFrame
        variables: Variables to test

    Returns:
        DataFrame with normality test results
    """
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


def test_pesaran_cd(
    df: pd.DataFrame, var: str, id_var: str = "iso3", time_var: str = "year"
) -> tuple[float, float]:
    """
    Computes the Pesaran CD test for cross-sectional dependence on unbalanced panels.
    null: Cross-Sectional Independence (residuals or variable is uncorrelated across entities).
    p < 0.05 implies presence of cross-sectional dependence.
    """

    # Pivot to wide format: rows=years, cols=countries
    df_wide = df.pivot_table(index=time_var, columns=id_var, values=var)

    # Calculate pairwise correlation over overlapping non-null periods
    corr_matrix = df_wide.corr(method="pearson").values

    # Calculate pairwise number of non-null overlapping observations (T_ij)
    not_nulls = (~df_wide.isna()).astype(int)
    T_ij_matrix = not_nulls.T.dot(not_nulls).values

    N = df_wide.shape[1]

    cd_stat = 0.0
    valid_pairs = 0

    for i in range(N - 1):
        for j in range(i + 1, N):
            T_ij = T_ij_matrix[i, j]
            rho_ij = corr_matrix[i, j]

            # Only include if they have at least 3 overlapping periods to compute correlation reliably
            if T_ij >= 3 and not np.isnan(rho_ij):
                cd_stat += np.sqrt(T_ij) * rho_ij
                valid_pairs += 1

    if valid_pairs == 0:
        return np.nan, np.nan

    # Scale by the square root of the number of valid pairs to handle highly unbalanced panels
    cd_stat = cd_stat * np.sqrt(1.0 / valid_pairs)
    p_value = 2 * (1 - norm.cdf(abs(cd_stat)))

    return cd_stat, p_value


def export_diagnostics_latex(
    df: pd.DataFrame,
    variables: list[str],
    out_dir: str,
    id_var: str = "iso3",
    time_var: str = "year",
) -> None:
    """
    Generate and export a LaTeX table with Stationarity, Normality,
    Serial Correlation, and Cross-Sectional Dependence (Pesaran CD) test
    statistics for individual variables.
    """
    records = []

    for var in variables:
        if var not in df.columns:
            continue

        row = {"Variable": var}
        data_clean = df[var].dropna()

        # 1. Stationarity (ADF Panel mean test stat / mean p-value)
        try:
            st_df = test_stationarity(
                df, [var], test="adf", id_var=id_var, time_var=time_var, verbose=False
            )
            if not st_df.empty:
                adf_stat, adf_p, _, _, _, _ = adfuller(data_clean, autolag="AIC")

                row["ADF Stat"] = f"{adf_stat:.3f}"
                row["ADF p-val"] = f"{adf_p:.3f}"
            else:
                row["ADF Stat"] = "-"
                row["ADF p-val"] = "-"
        except Exception:
            row["ADF Stat"] = "-"
            row["ADF p-val"] = "-"

        # 2. Normality (Jarque-Bera)
        try:
            jb_stat, jb_p = jarque_bera(data_clean)
            row["JB Stat"] = f"{jb_stat:.3f}"
            row["JB p-val"] = f"{jb_p:.3f}"
        except Exception:
            row["JB Stat"] = "-"
            row["JB p-val"] = "-"

        # 3. Serial Correlation (Ljung-Box lag 1)
        try:
            # Sort by panel logic (country, then year) to avoid mixing boundaries too much
            data_sc = df.sort_values([id_var, time_var])[var].dropna()
            lb_res = acorr_ljungbox(data_sc, lags=[1], return_df=True)
            lb_stat = lb_res["lb_stat"].iloc[0]
            lb_p = lb_res["lb_pvalue"].iloc[0]
            row["LB Stat"] = f"{lb_stat:.3f}"
            row["LB p-val"] = f"{lb_p:.3f}"
        except Exception:
            row["LB Stat"] = "-"
            row["LB p-val"] = "-"

        # 4. Cross-Sectional Dependence (Pesaran CD)
        try:
            cd_stat, cd_p = test_pesaran_cd(df, var, id_var=id_var, time_var=time_var)
            if np.isnan(cd_stat):
                row["CD Stat"] = "-"
                row["CD p-val"] = "-"
            else:
                row["CD Stat"] = f"{cd_stat:.3f}"
                row["CD p-val"] = f"{cd_p:.3f}"
        except Exception:
            row["CD Stat"] = "-"
            row["CD p-val"] = "-"

        records.append(row)

    table_df = pd.DataFrame(records)

    # Format the LaTeX
    latex_str = table_df.to_latex(
        index=False,
        escape=False,
        column_format="lcccccccc",
        caption="Pre-Estimation Diagnostics: Stationarity, Normality, Serial Correlation, and Cross-Sectional Dependence",
        label="tab:diagnostics",
    )

    # Import map for nice labels if present
    try:
        from analysis.regression_utils import LATEX_LABEL_MAP

        for old, new in LATEX_LABEL_MAP.items():
            latex_str = latex_str.replace(old, new)
    except ImportError:
        pass

    # Beautify LaTeX table
    latex_str = latex_str.replace(
        "\\toprule",
        "\\toprule\n& \\multicolumn{2}{c}{Stationarity (ADF)} & \\multicolumn{2}{c}{Normality (JB)} & \\multicolumn{2}{c}{Serial Corr. (LB)} & \\multicolumn{2}{c}{Cross-Sectional Dep. (CD)} \\\\\n\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}",
    )

    out_dir = Path(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_file = out_dir / "diagnostics_table.tex"

    with open(out_file, "w", encoding="utf-8") as f:
        f.write(latex_str)

    print(f"✅ Exported diagnostics LaTeX table to: {out_file}")
