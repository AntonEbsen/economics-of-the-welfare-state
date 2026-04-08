"""
Robustness checks execution pipelines.
"""

import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from clean.panel_utils import create_lags
from clean.stats import (
    build_latex_appendix,
    export_cointegration_latex,
    export_granger_causality_latex,
    export_hausman_latex,
    export_model_diagnostics_latex,
    export_reset_test_latex,
)
from linearmodels.panel import compare

from .regression_utils import LATEX_LABEL_MAP, prepare_regression_data, run_panel_ols

logger = logging.getLogger(__name__)


def export_stepwise_robustness_tables(
    master_regimes: pd.DataFrame, config: dict, out_dir: str | Path = None
) -> None:
    """
    Run stepwise robustness checks adding one macroeconomic control at a time,
    and export the model comparison matrices to LaTeX tables and spec curve plots.

    Args:
        master_regimes: DataFrame containing merged variables and regime indicators
        config: Dictionary from config.yaml containing dependencies
        out_dir: Output directory to save LaTeX tables
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    indices = config.get("indices", ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"])
    macro_controls = config.get(
        "controls",
        ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"],
    )
    dep_var = config.get("dependent_var", "sstran")

    final_models = {}
    final_model_data = {}  # for Hausman: idx_name -> (ols_data, exog_vars)

    for idx_name in indices:
        models = {}
        current_ctrls = []

        for step in range(len(macro_controls) + 1):
            if step > 0:
                current_ctrls.append(macro_controls[step - 1])

            # Current step variables
            all_needed_vars = [idx_name] + current_ctrls

            # Create lags
            reg_data = create_lags(master_regimes, all_needed_vars, lags=[1])

            g_var = f"{idx_name}_lag1"
            lagged_ctrls = [f"{v}_lag1" for v in current_ctrls]

            ols_data, exog_vars = prepare_regression_data(
                reg_data, dep_var, g_var, lagged_ctrls, interactions=False
            )

            if step == 0:
                model_name = "Baseline"
            else:
                ctrl = macro_controls[step - 1]
                model_name = f"+ {LATEX_LABEL_MAP.get(f'{ctrl}_lag1', ctrl)}"

            res = run_panel_ols(ols_data, dep_var, exog_vars)
            models[model_name] = res

            # If this is the last step (fully specified model), save it
            if step == len(macro_controls):
                final_models[idx_name] = res
                final_model_data[idx_name] = (ols_data, exog_vars)

        comparison = compare(models, stars=True)
        logger.info(f"{'=' * 20} Stepwise Robustness: {idx_name} {'=' * 20}")
        logger.info(comparison)

        output_file = out_dir / f"stepwise_robustness_{idx_name}.tex"

        latex_str = comparison.summary.as_latex()
        for old, new in LATEX_LABEL_MAP.items():
            latex_str = latex_str.replace(old, new)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_str)
        logger.info(f"Saved table to: {output_file}")

        # Generate specification curve for this index
        fig_dir = out_dir.parent / "figures"
        plot_specification_curve(models, idx_name, macro_controls, fig_dir)

        # Log robustness summary
        robustness_df = get_robustness_summary(models)
        logger.info(f"\n📈 Robustness Summary for {idx_name}:")
        logger.info("\n" + robustness_df.to_string())

    # ── Post-Estimation Diagnostics Suite ──
    if final_models:
        try:
            export_model_diagnostics_latex(final_models, out_dir=out_dir)
            export_hausman_latex(final_model_data, dep_var=dep_var, out_dir=out_dir)
            export_reset_test_latex(final_models, out_dir=out_dir)
            export_granger_causality_latex(
                master_regimes, indices=indices, dep_var=dep_var, out_dir=out_dir
            )
            # Panel Cointegration (Kao Test)
            export_cointegration_latex(
                master_regimes,
                indices=indices,
                dep_var=dep_var,
                controls=macro_controls,
                out_dir=out_dir,
            )
            # Build the final combined appendix
            build_latex_appendix(tables_dir=out_dir)
        except Exception as e:
            logger.error(f"Failed to export post-estimation diagnostics: {e}")


def plot_specification_curve(
    models: dict,
    idx_name: str,
    macro_controls: list[str],
    out_dir: str | Path = None,
) -> None:
    """
    Generate a publication-ready Specification Curve plot.

    Traces the stability of the main independent variable's coefficient
    (with 95% CI error bars) across the sequential stepwise models.

    Args:
        models: Ordered dict of fitted PanelOLS results from stepwise loop.
        idx_name: Name of the KOF globalization index (e.g. 'KOFGI').
        macro_controls: List of control variable names in the order they were added.
        out_dir: Output directory to save the figure.
    """

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "figures"
    else:
        out_dir = Path(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    g_var = f"{idx_name}_lag1"
    model_labels = list(models.keys())
    coefficients = []
    lower_ci = []
    upper_ci = []

    for name, result in models.items():
        coef = result.params[g_var]
        se = result.std_errors[g_var]
        coefficients.append(coef)
        lower_ci.append(coef - 1.96 * se)
        upper_ci.append(coef + 1.96 * se)

    x = np.arange(len(model_labels))
    coefficients = np.array(coefficients)
    lower_ci = np.array(lower_ci)
    upper_ci = np.array(upper_ci)
    err_lower = coefficients - lower_ci
    err_upper = upper_ci - coefficients

    # --- Plot ---

    sns.set_theme(style="whitegrid", palette="muted")
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot confidence intervals as a shaded area for a cleaner look
    ax.fill_between(x, lower_ci, upper_ci, alpha=0.15, color="#3B82F6", label="95% CI")

    # Plot point estimates with a distinct style
    ax.errorbar(
        x,
        coefficients,
        yerr=[err_lower, err_upper],
        fmt="o",
        color="#1D4ED8",
        ecolor="#93C5FD",
        elinewidth=1.5,
        capsize=4,
        markersize=7,
        markerfacecolor="white",
        markeredgewidth=2,
        zorder=3,
        label="Coefficient",
    )

    # Zero reference line (stronger)
    ax.axhline(0, color="#EF4444", linestyle="-", linewidth=1.5, alpha=0.6)

    # Aesthetics
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=35, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    idx_label = LATEX_LABEL_MAP.get(g_var, idx_name)
    ax.set_title(
        f"Specification Curve: {idx_label}",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylabel("Coefficient Estimate", fontsize=11, fontweight="semibold")
    ax.set_xlabel("Model Specification (Cumulative Controls)", fontsize=11, fontweight="semibold")

    # Remove top/right spines
    sns.despine()
    plt.tight_layout()

    out_path = out_dir / f"specification_curve_{idx_name}.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")  # High DPI for publication
    plt.close(fig)
    logger.info(f"✅ Saved High-DPI Specification Curve: {out_path}")


def get_robustness_summary(models: dict) -> pd.DataFrame:
    """
    Generate a summary table showing the robustness of each control variable.
    Calculates the percentage of models where the variable is significant.
    """
    summary = []

    # Flatten all parameters and p-values from all models
    for name, res in models.items():
        params = res.params
        pvalues = res.pvalues
        for var in params.index:
            if var == "const":
                continue
            summary.append(
                {
                    "Model": name,
                    "Variable": var,
                    "Coef": params[var],
                    "Significant": pvalues[var] < 0.05,
                }
            )

    df = pd.DataFrame(summary)
    if df.empty:
        return pd.DataFrame()

    # Aggregate by variable
    robustness = df.groupby("Variable").agg(
        {"Significant": ["sum", "count", "mean"], "Coef": "mean"}
    )

    robustness.columns = ["N Significant", "Total Models", "Robustness (%)", "Avg Coef"]
    robustness["Robustness (%)"] = (robustness["Robustness (%)"] * 100).round(1)

    return robustness.sort_values("Robustness (%)", ascending=False)


def export_subperiod_regressions(
    master_regimes: pd.DataFrame, config: dict, out_dir: str | Path = None
) -> None:
    """
    Run fully specified foundational models across different eras (Pre/Post China shock).
    Generates a table per era, with columns representing different indices.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    # Restrict to primary components: Total, Economic, Social, Political
    indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
    macro_controls = config.get(
        "controls",
        ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"],
    )
    dep_var = config.get("dependent_var", "sstran")

    subperiods = {
        "pre_china_shock": (1980, 1999),
        "post_china_shock": (2000, 2023),
        "pre_gfc": (1980, 2007),
        "post_gfc": (2008, 2023),
    }

    print("\n" + "=" * 60)
    print("🕰️ RUNNING SUBPERIOD REGRESSIONS (China Shock & GFC)")
    print("=" * 60)

    valid_indices = [idx for idx in indices if idx in master_regimes.columns]

    for period_name, (start_year, end_year) in subperiods.items():
        models = {}

        for idx_name in valid_indices:
            all_needed_vars = [idx_name] + macro_controls
            reg_data = create_lags(master_regimes, all_needed_vars, lags=[1])

            g_var = f"{idx_name}_lag1"
            lagged_ctrls = [f"{v}_lag1" for v in macro_controls]

            if "year" in reg_data.columns:
                period_data = reg_data[
                    (reg_data["year"] >= start_year) & (reg_data["year"] <= end_year)
                ].copy()
            else:
                period_data = reg_data.copy()

            ols_data, exog_vars = prepare_regression_data(
                period_data, dep_var, g_var, lagged_ctrls, interactions=False
            )

            if len(ols_data) < len(exog_vars) + 10:
                logger.warning(f"Not enough observations for {idx_name} in {period_name}")
                continue

            try:
                res = run_panel_ols(ols_data, dep_var, exog_vars)
                # Map to human readable index name for the column header
                header_name = (
                    LATEX_LABEL_MAP.get(g_var, idx_name).replace("_{t-1}", "").replace("$", "")
                )
                models[header_name] = res
            except Exception as e:
                logger.error(f"Error running {idx_name} for {period_name}: {e}")

        if not models:
            continue

        comparison = compare(models, stars=True)
        print(f"  Generated table for {period_name} with {len(models)} indices")

        output_file = out_dir / f"baseline_regressions_{period_name}.tex"
        latex_str = comparison.summary.as_latex()

        for old, new in LATEX_LABEL_MAP.items():
            latex_str = latex_str.replace(old, new)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_str)

    print("=" * 60 + "\n")


def export_subperiod_heterogeneity_regressions(
    master_regimes: pd.DataFrame, config: dict, out_dir: str | Path = None
) -> None:
    """
    Run heterogeneity (regime interaction) models across different eras (Pre/Post China shock).
    Generates a table per era, with columns representing different indices.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    # Restrict to primary components: Total, Economic, Social, Political
    indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
    macro_controls = config.get(
        "controls",
        ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"],
    )
    dep_var = config.get("dependent_var", "sstran")

    subperiods = {
        "pre_china_shock": (1980, 1999),
        "post_china_shock": (2000, 2023),
        "pre_gfc": (1980, 2007),
        "post_gfc": (2008, 2023),
    }

    print("\n" + "=" * 60)
    print("🕰️ RUNNING SUBPERIOD REGRESSIONS (Heterogeneity)")
    print("=" * 60)

    # Need regime dummy columns to create interactions
    regime_cols = [
        "regime_conservative",
        "regime_mediterranean",
        "regime_liberal",
    ]
    missing_regimes = [col for col in regime_cols if col not in master_regimes.columns]
    if missing_regimes:
        logger.error(
            f"Missing regime columns: {missing_regimes}. Cannot run heterogeneity analysis."
        )
        return

    valid_indices = [idx for idx in indices if idx in master_regimes.columns]

    for period_name, (start_year, end_year) in subperiods.items():
        models = {}

        for idx_name in valid_indices:
            all_needed_vars = [idx_name] + macro_controls + regime_cols
            reg_data = create_lags(master_regimes, all_needed_vars, lags=[1])

            g_var = f"{idx_name}_lag1"
            lagged_ctrls = [f"{v}_lag1" for v in macro_controls]

            if "year" in reg_data.columns:
                period_data = reg_data[
                    (reg_data["year"] >= start_year) & (reg_data["year"] <= end_year)
                ].copy()
            else:
                period_data = reg_data.copy()

            # Conservative, Mediterranean, Liberal interactions (Social Democrat = reference, Post-Communist excluded)
            period_data["int_conservative"] = (
                period_data[g_var] * period_data["regime_conservative"]
            )
            period_data["int_mediterranean"] = (
                period_data[g_var] * period_data["regime_mediterranean"]
            )
            period_data["int_liberal"] = period_data[g_var] * period_data["regime_liberal"]

            custom_exog_ctrls = [
                "int_conservative",
                "int_mediterranean",
                "int_liberal",
            ] + lagged_ctrls

            ols_data, exog_vars = prepare_regression_data(
                period_data, dep_var, g_var, custom_exog_ctrls, interactions=False
            )

            if len(ols_data) < len(exog_vars) + 10:
                logger.warning(f"Not enough observations for {idx_name} in {period_name}")
                continue

            try:
                res = run_panel_ols(ols_data, dep_var, exog_vars)
                # Map to human readable index name for the column header
                header_name = (
                    LATEX_LABEL_MAP.get(g_var, idx_name).replace("_{t-1}", "").replace("$", "")
                )
                models[header_name] = res
            except Exception as e:
                logger.error(f"Error running {idx_name} for {period_name}: {e}")

        if not models:
            continue

        comparison = compare(models, stars=True)
        print(f"  Generated heterogeneity table for {period_name} with {len(models)} indices")

        output_file = out_dir / f"heterogeneity_regressions_{period_name}.tex"
        latex_str = comparison.summary.as_latex()

        for old, new in LATEX_LABEL_MAP.items():
            latex_str = latex_str.replace(old, new)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_str)

    print("=" * 60 + "\n")


def export_event_study_plots(
    master_regimes: pd.DataFrame, config: dict, out_dir: str | Path = None
) -> None:
    """
    Run event study models around the year 2000 (China Shock) and export plots.
    """
    from .regression_utils import run_event_study

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "figures"
    else:
        out_dir = Path(out_dir)

    os.makedirs(out_dir, exist_ok=True)

    indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]
    macro_controls = config.get(
        "controls",
        ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"],
    )
    dep_var = config.get("dependent_var", "sstran")

    event_year = 2000
    window = 5

    logger.info("\n=======================================================")
    logger.info(f"📈 RUNNING EVENT STUDY ({event_year} +/- {window} years)")
    logger.info("=======================================================")

    for idx_name in indices:
        if idx_name not in master_regimes.columns:
            continue

        all_needed_vars = [idx_name] + macro_controls
        reg_data = create_lags(master_regimes, all_needed_vars, lags=[1])

        g_var = f"{idx_name}_lag1"
        lagged_ctrls = [f"{v}_lag1" for v in macro_controls]

        ols_data, exog_vars = prepare_regression_data(
            reg_data, dep_var, g_var, lagged_ctrls, interactions=False
        )

        try:
            plot_df, _ = run_event_study(
                ols_data=ols_data,
                dep_var=dep_var,
                treat_var=g_var,
                event_year=event_year,
                window=window,
                exog_vars=lagged_ctrls,
            )

            sns.set_theme(style="whitegrid")
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.errorbar(
                x=plot_df["rel_time"],
                y=plot_df["coef"],
                yerr=[plot_df["coef"] - plot_df["lower"], plot_df["upper"] - plot_df["coef"]],
                fmt="o-",
                color="#2563EB",
                capsize=5,
                markersize=8,
                linewidth=2,
            )

            ax.axhline(0, color="red", linestyle="--", alpha=0.7)
            ax.axvline(0, color="gray", linestyle=":", alpha=0.5)

            ax.set_title(
                f"Event Study: Impact of {idx_name} on {dep_var} around {event_year}",
                fontsize=14,
                pad=15,
            )
            ax.set_xlabel(f"Years relative to {event_year}", fontsize=12)
            ax.set_ylabel("Coefficient Estimate", fontsize=12)

            sns.despine()
            plt.tight_layout()

            out_file = out_dir / f"event_study_{idx_name}_{event_year}.png"
            fig.savefig(out_file, dpi=300)
            plt.close(fig)

            logger.info(f"  ✅ Saved Event Study Plot: {out_file.name}")
        except Exception as e:
            logger.error(f"  ❌ Error running event study for {idx_name}: {e}")
