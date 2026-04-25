"""
Structural Break Tests for panel data.

Implements three complementary approaches for testing parameter instability
in the globalization → welfare spending relationship:

1. Chow Test (known break at China WTO accession, year 2000)
2. QLR / Sup-Wald Test (unknown break date via Andrews 1993)
3. Rolling OLS coefficient plots (visual instability analysis)
4. Bai-Perron (1998, 2003) sequential multiple structural break test

All tests pool the within-country (demeaned) variation so that results
are consistent with the PanelOLS + entity-FE estimator used elsewhere.

References:
    Chow (1960), Econometrica.
    Andrews (1993), Econometrica.
    Bai (1997), Econometrica.
    Bai & Perron (1998), Econometrica.
    Bai & Perron (2003), Journal of Applied Econometrics.
    Stock & Watson (2020), Introduction to Econometrics, ch. 14.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BREAK_YEAR = 2000  # China's WTO accession
TRIM_FRAC = 0.15  # Andrews (1993) 15 % trimming for QLR
ROLLING_WINDOW = 10  # years

# Labels that match the rest of the project
_INDEX_LABELS = {
    "KOFGI": "Globalization (Overall)",
    "KOFEcGI": "Globalization (Economic)",
    "KOFSoGI": "Globalization (Social)",
    "KOFPoGI": "Globalization (Political)",
}

# Andrews (1993) asymptotic 5 % and 1 % critical values for the sup-F (QLR)
# test with one structural break, for k=1 restriction.  These are the
# standard values tabulated in Andrews (1993, Table 1) for 15 % trimming.
_QLR_CV = {
    "k1_10pct": 7.12,
    "k1_5pct": 8.68,
    "k1_1pct": 12.16,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _demean_panel(
    df: pd.DataFrame,
    dep_var: str,
    indep_var: str,
    controls: list[str],
    id_var: str = "iso3",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Within-transform (demean) all variables by entity so the pooled OLS
    on the demeaned data is equivalent to fixed-effects estimation.

    Returns the demeaned DataFrame and the list of regressor column names.
    """
    all_vars = [dep_var, indep_var] + controls
    data = df[all_vars + [id_var]].dropna().copy()

    # Within demean
    for col in all_vars:
        data[col] = data[col] - data.groupby(id_var)[col].transform("mean")

    regressors = [indep_var] + controls
    return data, regressors


def _ols_on_demeaned(
    data: pd.DataFrame,
    dep_var: str,
    regressors: list[str],
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Run plain OLS on (already demeaned) data. No constant needed post-demeaning."""
    X = data[regressors].values
    y = data[dep_var].values
    return sm.OLS(y, X).fit()


# ---------------------------------------------------------------------------
# 1. Chow Test
# ---------------------------------------------------------------------------


def chow_test(
    df: pd.DataFrame,
    dep_var: str,
    indep_var: str,
    controls: list[str],
    break_year: int = BREAK_YEAR,
    time_var: str = "year",
    id_var: str = "iso3",
) -> dict:
    """
    Chow (1960) test for a structural break at a *known* date.

    The test partitions the demeaned panel into pre- and post-break
    sub-samples, runs OLS in each, and computes an F-statistic testing
    whether the slope coefficients are equal across the two periods.

    H0: No structural break (coefficients identical pre/post)
    Reject H0 (p < 0.05) → evidence of a structural break at break_year.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel (must contain id_var, time_var, dep_var, indep_var,
        and all elements of controls).
    dep_var : str
        Dependent variable (e.g. 'sstran').
    indep_var : str
        Main independent variable (e.g. 'KOFGI_lag1').
    controls : list[str]
        Control variable column names.
    break_year : int
        Candidate structural break year (default 2000).
    time_var : str
        Time column name (default 'year').
    id_var : str
        Entity column name (default 'iso3').

    Returns
    -------
    dict with keys: indep_var, break_year, f_stat, df_num, df_den,
                    p_value, verdict, n_pre, n_post.
    """
    data, regressors = _demean_panel(df, dep_var, indep_var, controls, id_var)
    k = len(regressors)

    pre = data[df[time_var] < break_year].copy()
    post = data[df[time_var] >= break_year].copy()

    # Need enough observations in each sub-period
    min_obs = k + 5
    if len(pre) < min_obs or len(post) < min_obs:
        logger.warning(
            f"Chow test for {indep_var}: insufficient observations "
            f"(pre={len(pre)}, post={len(post)}, need ≥{min_obs}). Skipping."
        )
        return {
            "indep_var": indep_var,
            "break_year": break_year,
            "f_stat": np.nan,
            "df_num": np.nan,
            "df_den": np.nan,
            "p_value": np.nan,
            "verdict": "Insufficient data",
            "n_pre": len(pre),
            "n_post": len(post),
        }

    res_full = _ols_on_demeaned(data, dep_var, regressors)
    res_pre = _ols_on_demeaned(pre, dep_var, regressors)
    res_post = _ols_on_demeaned(post, dep_var, regressors)

    # Chow F-statistic
    # F = [(RSS_full - (RSS_pre + RSS_post)) / k] / [(RSS_pre + RSS_post) / (n - 2k)]
    rss_full = res_full.ssr
    rss_restricted = res_pre.ssr + res_post.ssr
    n = len(data)
    df_num = k
    df_den = n - 2 * k

    if df_den <= 0 or rss_restricted <= 0:
        f_stat = np.nan
        p_value = np.nan
    else:
        f_stat = ((rss_full - rss_restricted) / df_num) / (rss_restricted / df_den)
        p_value = 1 - sp_stats.f.cdf(f_stat, df_num, df_den)

    verdict = "Break detected ✓" if (not np.isnan(p_value) and p_value < 0.05) else "No break"

    return {
        "indep_var": indep_var,
        "break_year": break_year,
        "f_stat": f_stat,
        "df_num": df_num,
        "df_den": df_den,
        "p_value": p_value,
        "verdict": verdict,
        "n_pre": len(pre),
        "n_post": len(post),
    }


# ---------------------------------------------------------------------------
# 2. QLR / Sup-Wald (Andrews 1993)
# ---------------------------------------------------------------------------


def qlr_test(
    df: pd.DataFrame,
    dep_var: str,
    indep_var: str,
    controls: list[str],
    time_var: str = "year",
    id_var: str = "iso3",
    trim: float = TRIM_FRAC,
) -> dict:
    """
    Quandt Likelihood Ratio (QLR) / Sup-Wald test for a structural break
    at an *unknown* date (Andrews 1993).

    The test computes Chow F-statistics at every candidate breakpoint
    within the trimmed interior [T*trim, T*(1-trim)] of the time span,
    and reports the supremum.  The estimated break date is the year that
    maximises the F-statistic.

    H0: No structural break at any point in the sample.
    Reject H0 if sup-F exceeds the Andrews (1993) asymptotic critical value.

    Parameters
    ----------
    trim : float
        Fraction to trim from each end (default 0.15, per Andrews 1993).

    Returns
    -------
    dict with keys: indep_var, sup_f, break_year_est, p_approx, cv_5pct,
                    cv_1pct, verdict, n_candidates.
    """
    data, regressors = _demean_panel(df, dep_var, indep_var, controls, id_var)
    k = len(regressors)

    years = sorted(df[time_var].unique())
    n_years = len(years)
    lo_idx = int(np.floor(trim * n_years))
    hi_idx = int(np.ceil((1 - trim) * n_years))
    candidate_years = years[lo_idx:hi_idx]

    if len(candidate_years) < 2:
        logger.warning(f"QLR test for {indep_var}: not enough candidate years.")
        return {
            "indep_var": indep_var,
            "sup_f": np.nan,
            "break_year_est": np.nan,
            "p_approx": np.nan,
            "cv_5pct": _QLR_CV["k1_5pct"],
            "cv_1pct": _QLR_CV["k1_1pct"],
            "verdict": "Insufficient data",
            "n_candidates": 0,
        }

    f_stats = {}
    res_full = _ols_on_demeaned(data, dep_var, regressors)
    rss_full = res_full.ssr
    n = len(data)

    for yr in candidate_years:
        mask_pre = df[time_var] < yr
        mask_post = df[time_var] >= yr

        pre = data[
            (
                mask_pre.values[: len(data)]
                if len(mask_pre) == len(data)
                else data.index.isin(df[mask_pre].index)
            )
        ].copy()
        post = data[
            (
                mask_post.values[: len(data)]
                if len(mask_post) == len(data)
                else data.index.isin(df[mask_post].index)
            )
        ].copy()

        min_obs = k + 3
        if len(pre) < min_obs or len(post) < min_obs:
            continue

        try:
            res_pre = _ols_on_demeaned(pre, dep_var, regressors)
            res_post = _ols_on_demeaned(post, dep_var, regressors)
            rss_restricted = res_pre.ssr + res_post.ssr
            df_den = n - 2 * k
            if df_den > 0 and rss_restricted > 0:
                f_val = ((rss_full - rss_restricted) / k) / (rss_restricted / df_den)
                f_stats[yr] = f_val
        except Exception as exc:
            logger.debug(f"QLR: skipping year {yr} for {indep_var}: {exc}")

    if not f_stats:
        return {
            "indep_var": indep_var,
            "sup_f": np.nan,
            "break_year_est": np.nan,
            "p_approx": np.nan,
            "cv_5pct": _QLR_CV["k1_5pct"],
            "cv_1pct": _QLR_CV["k1_1pct"],
            "verdict": "Could not compute",
            "n_candidates": 0,
        }

    sup_f = max(f_stats.values())
    break_year_est = max(f_stats, key=f_stats.get)

    # Approximate p-value: compare sup-F to an F(k, n-2k) distribution.
    # This is conservative (true critical values from Andrews tables are higher),
    # so we also report the Andrews 5% and 1% critical values for k=1.
    df_den = n - 2 * k
    p_approx = 1 - sp_stats.f.cdf(sup_f, k, max(df_den, 1))

    cv_5 = _QLR_CV["k1_5pct"]
    cv_1 = _QLR_CV["k1_1pct"]

    if sup_f >= cv_1:
        verdict = "Break detected ✓ (p<0.01)"
    elif sup_f >= cv_5:
        verdict = "Break detected ✓ (p<0.05)"
    else:
        verdict = "No break"

    return {
        "indep_var": indep_var,
        "sup_f": sup_f,
        "f_by_year": f_stats,
        "break_year_est": break_year_est,
        "p_approx": p_approx,
        "cv_5pct": cv_5,
        "cv_1pct": cv_1,
        "verdict": verdict,
        "n_candidates": len(f_stats),
    }


# ---------------------------------------------------------------------------
# 3. Rolling OLS
# ---------------------------------------------------------------------------


def rolling_ols_coefficients(
    df: pd.DataFrame,
    dep_var: str,
    indep_var: str,
    controls: list[str],
    window: int = ROLLING_WINDOW,
    time_var: str = "year",
    id_var: str = "iso3",
) -> pd.DataFrame:
    """
    Rolling (fixed-window) OLS on the demeaned panel data.

    For each window ending at year t, fits OLS and records the coefficient
    and 95 % CI for indep_var.  This reveals how the estimated effect of
    globalisation on welfare spending drifts over time.

    Parameters
    ----------
    window : int
        Number of years in each rolling window (default 10).

    Returns
    -------
    DataFrame with columns: year_end, coef, se, ci_lo, ci_hi, n_obs.
    """
    data_full, regressors = _demean_panel(df, dep_var, indep_var, controls, id_var)
    years = sorted(df[time_var].unique())

    records = []
    for i in range(len(years) - window + 1):
        window_years = years[i : i + window]
        year_end = window_years[-1]

        mask = df[time_var].isin(window_years)
        # Align mask to demeaned data (which may have fewer rows due to NaN drop)
        subset = data_full.loc[data_full.index.intersection(df[mask].index)]
        if len(subset) < len(regressors) + 5:
            continue

        try:
            res = _ols_on_demeaned(subset, dep_var, regressors)
            # indep_var is always the first regressor
            coef = res.params[0]
            se = res.bse[0]
            records.append(
                {
                    "year_end": year_end,
                    "coef": coef,
                    "se": se,
                    "ci_lo": coef - 1.96 * se,
                    "ci_hi": coef + 1.96 * se,
                    "n_obs": len(subset),
                }
            )
        except Exception as exc:
            logger.debug(f"Rolling OLS: skipping window ending {year_end}: {exc}")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Bai-Perron Sequential Multiple Break Test
# ---------------------------------------------------------------------------


def bai_perron_test(
    df: pd.DataFrame,
    dep_var: str,
    indep_var: str,
    controls: list[str],
    max_breaks: int = 5,
    trim: float = TRIM_FRAC,
    significance: float = 0.05,
    time_var: str = "year",
    id_var: str = "iso3",
) -> dict:
    """
    Bai-Perron (1998, 2003) sequential multiple structural break test.

    Uses the sequential procedure of Bai (1997): applies a sup-F test
    to the full sample to locate the first break.  If significant, the
    sample is split and each resulting segment is searched for further
    breaks.  Iteration continues until *max_breaks* is reached or no
    segment produces a significant sup-F.

    The number of breaks is also validated via BIC (Schwarz criterion):

        BIC(m) = n ln(RSS_m / n) + (m+1) k ln(n)

    where m is the number of breaks, k the number of regressors, and
    RSS_m the total residual sum of squares across the m+1 segments.

    Significance is assessed against Andrews (1993) asymptotic critical
    values for sup-F (15 % trimming, k=1 restriction).  This is a
    conservative approximation; exact Bai-Perron (2003, Table II)
    critical values are break-number-specific and not tabulated here.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format panel with id_var, time_var, dep_var, indep_var,
        and all controls.
    max_breaks : int
        Maximum number of breaks to search for (default 5).
    trim : float
        Fraction trimmed from each end of each segment (default 0.15).
    significance : float
        Significance level for sup-F rejection (0.05 or 0.01).

    Returns
    -------
    dict with keys: indep_var, break_years, n_breaks, sup_f_stats,
                    cv_used, significance, bic_by_k, k_star_bic, n_obs.
    """
    data, regressors = _demean_panel(df, dep_var, indep_var, controls, id_var)
    k = len(regressors)
    n_total = len(data)

    data = data.copy()
    data["_year"] = df.loc[data.index, time_var]
    all_years = sorted(data["_year"].unique())

    cv = _QLR_CV["k1_5pct"] if significance >= 0.05 else _QLR_CV["k1_1pct"]

    # ---- Sequential procedure (Bai 1997) --------------------------------
    segments: list[tuple[int, int]] = [(all_years[0], all_years[-1])]
    breaks_found: list[int] = []
    sup_f_stats: list[float] = []

    for _iteration in range(max_breaks):
        best_f = -np.inf
        best_break: int | None = None
        best_seg_idx: int | None = None

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            seg_years = [y for y in all_years if seg_start <= y <= seg_end]
            n_seg_years = len(seg_years)
            lo = max(1, int(np.floor(trim * n_seg_years)))
            hi = min(n_seg_years - 1, int(np.ceil((1 - trim) * n_seg_years)))
            candidates = seg_years[lo:hi]

            if not candidates:
                continue

            seg_data = data[(data["_year"] >= seg_start) & (data["_year"] <= seg_end)]
            if len(seg_data) < 2 * k + 10:
                continue

            try:
                res_seg = _ols_on_demeaned(seg_data, dep_var, regressors)
            except Exception:
                continue
            rss_seg = res_seg.ssr
            n_seg = len(seg_data)

            for yr in candidates:
                pre = seg_data[seg_data["_year"] < yr]
                post = seg_data[seg_data["_year"] >= yr]
                min_obs = k + 3
                if len(pre) < min_obs or len(post) < min_obs:
                    continue

                try:
                    res_pre = _ols_on_demeaned(pre, dep_var, regressors)
                    res_post = _ols_on_demeaned(post, dep_var, regressors)
                    rss_split = res_pre.ssr + res_post.ssr
                    df_den = n_seg - 2 * k
                    if df_den > 0 and rss_split > 0:
                        f_val = ((rss_seg - rss_split) / k) / (rss_split / df_den)
                        if f_val > best_f:
                            best_f = f_val
                            best_break = yr
                            best_seg_idx = seg_idx
                except Exception:
                    pass

        if best_break is None or best_f < cv:
            break

        breaks_found.append(best_break)
        sup_f_stats.append(round(float(best_f), 3))

        seg_start, seg_end = segments[best_seg_idx]
        segments.pop(best_seg_idx)
        segments.append((seg_start, best_break - 1))
        segments.append((best_break, seg_end))
        segments.sort()

    breaks_sorted = sorted(breaks_found)

    # ---- BIC for 0 .. len(breaks_sorted) --------------------------------
    bic_values: dict[int, float] = {}
    for m in range(len(breaks_sorted) + 1):
        bp = breaks_sorted[:m]
        boundaries = [all_years[0]] + bp + [all_years[-1] + 1]
        total_rss = 0.0
        total_n = 0
        valid = True

        for j in range(len(boundaries) - 1):
            b_lo, b_hi = boundaries[j], boundaries[j + 1]
            seg_data = data[(data["_year"] >= b_lo) & (data["_year"] < b_hi)]
            if len(seg_data) < k + 3:
                valid = False
                break
            try:
                res = _ols_on_demeaned(seg_data, dep_var, regressors)
                total_rss += res.ssr
                total_n += len(seg_data)
            except Exception:
                valid = False
                break

        if valid and total_rss > 0 and total_n > 0:
            bic_values[m] = total_n * np.log(total_rss / total_n) + (m + 1) * k * np.log(total_n)

    k_star = min(bic_values, key=bic_values.get) if bic_values else 0

    return {
        "indep_var": indep_var,
        "break_years": breaks_sorted,
        "n_breaks": len(breaks_sorted),
        "sup_f_stats": sup_f_stats,
        "cv_used": cv,
        "significance": significance,
        "bic_by_k": bic_values,
        "k_star_bic": k_star,
        "n_obs": n_total,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_rolling_ols(
    rolling_df: pd.DataFrame,
    indep_var: str,
    qlr_break_year: int | None = None,
    chow_break_year: int = BREAK_YEAR,
    out_dir: Path | str | None = None,
) -> None:
    """
    Publication-quality plot of rolling OLS coefficients with 95 % CI,
    annotated with the Chow and QLR break-year estimates.
    """
    if rolling_df.empty:
        logger.warning(f"plot_rolling_ols: empty DataFrame for {indep_var}, skipping.")
        return

    label = _INDEX_LABELS.get(indep_var.replace("_lag1", ""), indep_var)

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(11, 5))

    x = rolling_df["year_end"].values
    coef = rolling_df["coef"].values
    ci_lo = rolling_df["ci_lo"].values
    ci_hi = rolling_df["ci_hi"].values

    ax.fill_between(x, ci_lo, ci_hi, alpha=0.18, color="#3B82F6", label="95% CI")
    ax.plot(x, coef, color="#1D4ED8", linewidth=2.2, label="Rolling coefficient")
    ax.axhline(0, color="#EF4444", linewidth=1.4, linestyle="--", alpha=0.7)

    # Chow assumed break
    ax.axvline(
        chow_break_year,
        color="#F59E0B",
        linewidth=1.6,
        linestyle="--",
        label=f"Chow break ({chow_break_year})",
    )

    # QLR estimated break (if different)
    if qlr_break_year is not None and not np.isnan(qlr_break_year):
        qlr_yr = int(qlr_break_year)
        if qlr_yr != chow_break_year:
            ax.axvline(
                qlr_yr,
                color="#10B981",
                linewidth=1.6,
                linestyle=":",
                label=f"QLR break est. ({qlr_yr})",
            )

    ax.set_title(
        f"Rolling OLS Coefficient — {label}",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel(f"Window end year ({ROLLING_WINDOW}-year window)", fontsize=10)
    ax.set_ylabel("Coefficient on Globalization Index", fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.set_xlim(x.min() - 0.5, x.max() + 0.5)
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"rolling_ols_{indep_var.replace('_lag1', '')}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Saved rolling OLS plot: {fname}")

    plt.close(fig)


def plot_qlr_f_path(
    f_by_year: dict[int, float],
    indep_var: str,
    sup_f: float,
    break_year_est: int,
    cv_5pct: float,
    cv_1pct: float,
    out_dir: Path | str | None = None,
) -> None:
    """
    Plot the F-statistic path across candidate break years for the QLR test,
    with Andrews critical-value reference lines.
    """
    if not f_by_year:
        return

    label = _INDEX_LABELS.get(indep_var.replace("_lag1", ""), indep_var)
    years = sorted(f_by_year)
    f_vals = [f_by_year[y] for y in years]

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(years, f_vals, color="#1D4ED8", linewidth=2.0, label="Chow F-statistic")
    ax.axhline(cv_5pct, color="#F59E0B", linewidth=1.5, linestyle="--", label=f"5% CV ({cv_5pct})")
    ax.axhline(cv_1pct, color="#EF4444", linewidth=1.5, linestyle=":", label=f"1% CV ({cv_1pct})")
    ax.axvline(
        break_year_est,
        color="#10B981",
        linewidth=1.6,
        linestyle="--",
        label=f"Sup-F year ({break_year_est})",
    )

    ax.set_title(
        f"QLR Test — F-Statistic Path: {label}",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Candidate break year", fontsize=10)
    ax.set_ylabel("Chow F-statistic", fontsize=10)
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"qlr_path_{indep_var.replace('_lag1', '')}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        logger.info(f"✅ Saved QLR F-path plot: {fname}")

    plt.close(fig)


def plot_bai_perron_bic(
    bic_by_k: dict[int, float],
    indep_var: str,
    k_star: int,
    out_dir: Path | str | None = None,
) -> None:
    """BIC vs. number of breaks for Bai-Perron model selection."""
    if not bic_by_k:
        return

    label = _INDEX_LABELS.get(indep_var.replace("_lag1", ""), indep_var)
    ks = sorted(bic_by_k)
    bics = [bic_by_k[m] for m in ks]

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ks, bics, "o-", color="#1D4ED8", linewidth=2, markersize=8)
    ax.axvline(k_star, color="#10B981", linestyle="--", linewidth=1.5, label=f"k* = {k_star}")
    ax.set_xticks(ks)
    ax.set_xlabel("Number of breaks (m)", fontsize=10)
    ax.set_ylabel("BIC", fontsize=10)
    ax.set_title(f"Bai-Perron BIC — {label}", fontsize=13, fontweight="bold", pad=14)
    ax.legend(fontsize=9)
    sns.despine()
    plt.tight_layout()

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = out_dir / f"bai_perron_bic_{indep_var.replace('_lag1', '')}.png"
        fig.savefig(fname, dpi=300, bbox_inches="tight")
        logger.info(f"Saved Bai-Perron BIC plot: {fname}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# LaTeX export helpers
# ---------------------------------------------------------------------------


def _chow_results_to_latex(records: list[dict], out_dir: Path) -> None:
    """Export Chow test results table to LaTeX."""
    rows = []
    for r in records:
        label = _INDEX_LABELS.get(r["indep_var"].replace("_lag1", ""), r["indep_var"])
        f_str = f"{r['f_stat']:.3f}" if not np.isnan(r["f_stat"]) else "—"
        p_str = f"{r['p_value']:.3f}" if not np.isnan(r["p_value"]) else "—"
        rows.append(
            {
                "Index": label,
                "Obs (pre)": int(r["n_pre"]) if not np.isnan(r["n_pre"]) else "—",
                "Obs (post)": int(r["n_post"]) if not np.isnan(r["n_post"]) else "—",
                f"F-Stat (break={r['break_year']})": f_str,
                "p-value": p_str,
                "Verdict": r["verdict"].replace("✓", "\\checkmark"),
            }
        )

    df_out = pd.DataFrame(rows)
    latex_str = df_out.to_latex(
        index=False,
        escape=False,
        column_format="lccccl",
        caption=(
            f"Chow Structural Break Test (break year = {BREAK_YEAR}). "
            r"H$_0$: No structural break. Rejection at $p<0.05$ implies "
            "parameter instability at the candidate date."
        ),
        label="tab:chow_test",
    )
    _write_latex(latex_str, out_dir / "chow_test_table.tex")


def _qlr_results_to_latex(records: list[dict], out_dir: Path) -> None:
    """Export QLR test results table to LaTeX."""
    rows = []
    for r in records:
        label = _INDEX_LABELS.get(r["indep_var"].replace("_lag1", ""), r["indep_var"])
        sup_f_str = f"{r['sup_f']:.3f}" if not np.isnan(r["sup_f"]) else "—"
        byr_str = str(int(r["break_year_est"])) if not np.isnan(r["break_year_est"]) else "—"
        p_str = f"{r['p_approx']:.3f}" if not np.isnan(r["p_approx"]) else "—"
        rows.append(
            {
                "Index": label,
                "Sup-F": sup_f_str,
                "Est. Break Year": byr_str,
                "Approx. p-value": p_str,
                r"CV 5\% (Andrews)": f"{r['cv_5pct']:.2f}",
                r"CV 1\% (Andrews)": f"{r['cv_1pct']:.2f}",
                "Verdict": r["verdict"]
                .replace("✓", "\\checkmark")
                .replace("(p<0.01)", "($p<0.01$)")
                .replace("(p<0.05)", "($p<0.05$)"),
            }
        )

    df_out = pd.DataFrame(rows)
    latex_str = df_out.to_latex(
        index=False,
        escape=False,
        column_format="lcccccc",
        caption=(
            "QLR (Sup-Wald) Structural Break Test — Unknown Break Date "
            r"(Andrews 1993, 15\% trimming). "
            r"H$_0$: No structural break at any date. Critical values from "
            "Andrews (1993, Table 1, $k=1$)."
        ),
        label="tab:qlr_test",
    )
    _write_latex(latex_str, out_dir / "qlr_test_table.tex")


def _bai_perron_results_to_latex(records: list[dict], out_dir: Path) -> None:
    """Export Bai-Perron test results table to LaTeX."""
    rows = []
    for r in records:
        label = _INDEX_LABELS.get(r["indep_var"].replace("_lag1", ""), r["indep_var"])
        breaks_str = ", ".join(str(y) for y in r["break_years"]) if r["break_years"] else "None"
        f_str = "; ".join(f"{f:.2f}" for f in r["sup_f_stats"]) if r["sup_f_stats"] else "---"
        rows.append(
            {
                "Index": label,
                "Breaks (seq.)": breaks_str,
                "Sup-F values": f_str,
                r"$k^*$ (BIC)": r["k_star_bic"],
                "N": r["n_obs"],
            }
        )

    df_out = pd.DataFrame(rows)
    latex_str = df_out.to_latex(
        index=False,
        escape=False,
        column_format="llccc",
        caption=(
            "Bai-Perron Sequential Multiple Structural Break Test. "
            r"Breaks identified via sequential sup-F (Andrews 1993 CV at 5\%). "
            r"$k^*$ selected by BIC."
        ),
        label="tab:bai_perron_test",
    )
    _write_latex(latex_str, out_dir / "bai_perron_test_table.tex")


def _write_latex(latex_str: str, out_path: Path) -> None:
    os.makedirs(out_path.parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    logger.info(f"✅ Exported LaTeX table: {out_path}")
    print(f"  ✅ Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_structural_breaks_latex(
    master_regimes: pd.DataFrame,
    config: dict,
    out_dir: str | Path | None = None,
    fig_dir: str | Path | None = None,
    time_var: str = "year",
    id_var: str = "iso3",
) -> None:
    """
    Run all structural break tests for the four primary KOF indices and
    export results to LaTeX tables and publication-ready figures.

    Tests run:
        1. Chow Test at year 2000 (China WTO accession)
        2. QLR / Sup-Wald test (Andrews 1993) with unknown break date
        3. Rolling OLS coefficient plots (10-year window)

    LaTeX outputs  (in out_dir):
        - chow_test_table.tex
        - qlr_test_table.tex

    Figure outputs (in fig_dir):
        - rolling_ols_<index>.png
        - qlr_path_<index>.png

    Parameters
    ----------
    master_regimes : pd.DataFrame
        Panel DataFrame with all variables + regime dummies.
    config : dict
        Parsed config.yaml dict (uses 'controls' and 'dependent_var').
    out_dir : path-like, optional
        Directory for LaTeX tables.  Defaults to outputs/tables/.
    fig_dir : path-like, optional
        Directory for figures.  Defaults to outputs/figures/.
    """
    from .panel_utils import create_lags

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if fig_dir is None:
        fig_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "figures"
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
    macro_controls = config.get(
        "controls",
        ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"],
    )
    dep_var = config.get("dependent_var", "sstran")

    # Validate that indices and dep_var exist
    valid_indices = [idx for idx in indices if idx in master_regimes.columns]
    if not valid_indices:
        logger.error(
            "No valid KOF indices found in master_regimes. Aborting structural break tests."
        )
        return

    print("\n" + "=" * 65)
    print("🔍 STRUCTURAL BREAK TESTS")
    print(f"   Break year (Chow): {BREAK_YEAR}  |  Panel: 1980–2023")
    print(f"   Indices: {', '.join(valid_indices)}")
    print("=" * 65)

    # Create lagged variables once
    all_needed = valid_indices + macro_controls
    reg_data = create_lags(master_regimes, all_needed, lags=[1])

    lagged_controls = [f"{c}_lag1" for c in macro_controls if f"{c}_lag1" in reg_data.columns]

    chow_records = []
    qlr_records = []

    for idx_name in valid_indices:
        g_var = f"{idx_name}_lag1"
        if g_var not in reg_data.columns:
            logger.warning(f"Lagged variable {g_var} not found, skipping {idx_name}.")
            continue

        label = _INDEX_LABELS.get(idx_name, idx_name)
        print(f"\n  ── {label} ({idx_name}) ──")

        # ── Chow Test ────────────────────────────────────────────────────────
        print(f"     Chow test (break={BREAK_YEAR})...", end=" ", flush=True)
        chow = chow_test(
            reg_data,
            dep_var,
            g_var,
            lagged_controls,
            break_year=BREAK_YEAR,
            time_var=time_var,
            id_var=id_var,
        )
        chow_records.append(chow)
        _verdict_icon = "✓" if "Break detected" in chow["verdict"] else "✗"
        print(
            f"F={chow['f_stat']:.3f}, p={chow['p_value']:.3f} → {_verdict_icon} {chow['verdict']}"
            if not np.isnan(chow["f_stat"])
            else "skipped (insufficient data)"
        )

        # ── QLR Test ─────────────────────────────────────────────────────────
        print("     QLR test (unknown break)...", end=" ", flush=True)
        qlr = qlr_test(
            reg_data,
            dep_var,
            g_var,
            lagged_controls,
            time_var=time_var,
            id_var=id_var,
        )
        qlr_records.append(qlr)
        if not np.isnan(qlr["sup_f"]):
            print(
                f"Sup-F={qlr['sup_f']:.3f}, est. break={int(qlr['break_year_est'])} "
                f"→ {qlr['verdict']}"
            )
            # QLR F-path plot
            plot_qlr_f_path(
                qlr.get("f_by_year", {}),
                g_var,
                qlr["sup_f"],
                qlr["break_year_est"],
                qlr["cv_5pct"],
                qlr["cv_1pct"],
                out_dir=fig_dir,
            )
        else:
            print("skipped (insufficient data)")

        # ── Rolling OLS ───────────────────────────────────────────────────────
        print(f"     Rolling OLS ({ROLLING_WINDOW}-year window)...", end=" ", flush=True)
        rolling_df = rolling_ols_coefficients(
            reg_data,
            dep_var,
            g_var,
            lagged_controls,
            window=ROLLING_WINDOW,
            time_var=time_var,
            id_var=id_var,
        )
        if not rolling_df.empty:
            print(f"computed {len(rolling_df)} windows")
            qlr_break = qlr.get("break_year_est", np.nan)
            plot_rolling_ols(
                rolling_df,
                g_var,
                qlr_break_year=None if np.isnan(qlr_break) else int(qlr_break),
                chow_break_year=BREAK_YEAR,
                out_dir=fig_dir,
            )
        else:
            print("skipped (no valid windows)")

    # ── Export LaTeX tables ───────────────────────────────────────────────────
    print("\n  Exporting LaTeX tables...")
    if chow_records:
        _chow_results_to_latex(chow_records, out_dir)
    if qlr_records:
        _qlr_results_to_latex(qlr_records, out_dir)

    print("=" * 65 + "\n")
