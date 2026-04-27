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
from linearmodels.panel import compare

from clean.panel_utils import create_lags
from clean.stats import (
    build_latex_appendix,
    export_cointegration_latex,
    export_granger_causality_latex,
    export_hausman_latex,
    export_model_diagnostics_latex,
    export_reset_test_latex,
)

from .regression_utils import (
    LATEX_LABEL_MAP,
    generate_marginal_effects,
    prepare_regression_data,
    run_panel_ols,
    significance_stars,
)

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

    logger.info("🕰️ Running subperiod regressions (China Shock & GFC)")

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
        logger.info("Generated table for %s with %d indices", period_name, len(models))

        output_file = out_dir / f"baseline_regressions_{period_name}.tex"
        latex_str = comparison.summary.as_latex()

        for old, new in LATEX_LABEL_MAP.items():
            latex_str = latex_str.replace(old, new)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_str)


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

    logger.info("🕰️ Running subperiod heterogeneity regressions")

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
        logger.info(
            "Generated heterogeneity table for %s with %d indices", period_name, len(models)
        )

        output_file = out_dir / f"heterogeneity_regressions_{period_name}.tex"
        latex_str = comparison.summary.as_latex()

        for old, new in LATEX_LABEL_MAP.items():
            latex_str = latex_str.replace(old, new)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_str)


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


def run_feedback_regressions(
    master_regimes: pd.DataFrame,
    config: dict,
    indices: list[str] | None = None,
) -> dict:
    """Estimate "reverse direction" regressions: ``KOFxx = β·sstran_{t-1} + Xγ``.

    The main paper regresses the welfare-state proxy ``sstran`` on lagged
    globalisation. This helper flips the direction to address the worry
    that welfare generosity itself drives the measured globalisation
    indices (e.g. by changing trade intensity). If the coefficient on
    ``sstran_lag1`` is statistically indistinguishable from zero, the
    baseline's causal-direction interpretation is more credible.

    Lifted from ``notebooks/02_modern_pipeline.ipynb`` cell 67 so the
    notebook can become thin orchestration and this specification is
    unit-testable.

    Parameters
    ----------
    master_regimes
        Panel frame with ``iso3``/``year`` and the globalisation indices.
    config
        Loaded ``config.yaml``; ``indices`` and ``controls`` keys are
        consulted for defaults.
    indices
        Dependent variables (globalisation indices). Defaults to
        ``config["indices"]`` then the four KOF aggregates.

    Returns
    -------
    dict[str, PanelResults]
        Mapping ``index_name -> fitted linearmodels PanelOLS result``.
    """
    if indices is None:
        indices = config.get("indices", ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"])
    ctrl_vars = config.get(
        "controls",
        ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"],
    )
    iv_var = "sstran"

    models: dict = {}
    for dv_name in indices:
        all_needed_vars = [dv_name, iv_var] + ctrl_vars
        reg_data = create_lags(master_regimes, all_needed_vars, lags=[1])

        iv_lagged = f"{iv_var}_lag1"
        ctrls_lagged = [f"{v}_lag1" for v in ctrl_vars]
        ols_data, exog_vars = prepare_regression_data(
            reg_data, dv_name, iv_lagged, ctrls_lagged, interactions=False
        )
        models[dv_name] = run_panel_ols(ols_data, dv_name, exog_vars)

    return models


def export_feedback_regression_table(
    master_regimes: pd.DataFrame,
    config: dict,
    out_dir: str | Path | None = None,
) -> Path:
    """Run :func:`run_feedback_regressions` and write the LaTeX comparison.

    Writes ``feedback_regression_table.tex`` into ``out_dir`` (or
    ``outputs/tables/`` by default), with ``LATEX_LABEL_MAP`` applied so
    the table labels match the rest of the paper.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = run_feedback_regressions(master_regimes, config)
    comparison = compare(models, stars=True)

    latex_str = comparison.summary.as_latex()
    for old, new in LATEX_LABEL_MAP.items():
        latex_str = latex_str.replace(old, new)

    out_path = out_dir / "feedback_regression_table.tex"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(latex_str)
    logger.info(f"✅ Feedback regression table saved to: {out_path}")
    return out_path


# Default finer-grained KOF sub-components (one step below the four
# KOFEcGI / KOFSoGI / KOFPoGI aggregates).
#
#   KOFTrGI  Trade globalisation   (de-facto trade flows)
#   KOFFiGI  Financial globalisation
#   KOFIpGI  Interpersonal globalisation
#   KOFInGI  Informational globalisation
#   KOFCuGI  Cultural globalisation
DEFAULT_SUBCOMPONENTS: list[str] = ["KOFTrGI", "KOFFiGI", "KOFIpGI", "KOFInGI", "KOFCuGI"]


def run_subcomponent_regressions(
    master: pd.DataFrame,
    config: dict,
    subcomponents: list[str] | None = None,
) -> dict:
    """Baseline Driscoll-Kraay spec run against each KOF sub-component.

    Complements :func:`export_stepwise_robustness_tables`, which sweeps
    over the four aggregate indices. This helper drives the same
    specification against the finer ``KOFTrGI / KOFFiGI / KOFIpGI /
    KOFInGI / KOFCuGI`` decomposition — useful for decomposing which
    channel of globalisation (trade vs. finance vs. information, etc.)
    is driving the headline result.

    Lifted from ``notebooks/02_modern_pipeline.ipynb`` cell 38.
    """
    if subcomponents is None:
        subcomponents = DEFAULT_SUBCOMPONENTS

    ctrl_vars = config.get(
        "controls",
        ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"],
    )
    dep_var = config.get("dependent_var", "sstran")

    models: dict = {}
    for comp in subcomponents:
        if comp not in master.columns:
            logger.warning("Sub-component %s not in master panel; skipping.", comp)
            continue
        current_ctrl_vars = [comp] + ctrl_vars
        reg_data = create_lags(master, current_ctrl_vars, lags=[1])

        indep_var = f"{comp}_lag1"
        lagged_ctrls = [f"{v}_lag1" for v in ctrl_vars]
        ols_data, exog_vars = prepare_regression_data(
            reg_data, dep_var, indep_var, lagged_ctrls, interactions=False
        )
        models[comp] = run_panel_ols(ols_data, dep_var, exog_vars)

    return models


def export_subcomponent_regression_table(
    master: pd.DataFrame,
    config: dict,
    out_dir: str | Path | None = None,
    subcomponents: list[str] | None = None,
) -> Path:
    """Run :func:`run_subcomponent_regressions` and write the LaTeX comparison."""
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = run_subcomponent_regressions(master, config, subcomponents=subcomponents)
    if not models:
        raise ValueError("No sub-components produced a fitted model — check column names.")

    comparison = compare(models, stars=True)
    latex_str = comparison.summary.as_latex()
    for old, new in LATEX_LABEL_MAP.items():
        latex_str = latex_str.replace(old, new)

    out_path = out_dir / "component_regression_table.tex"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(latex_str)
    logger.info(f"✅ Sub-component comparison table saved to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Baseline and interaction regression tables (cells 54 and 60/61)
# ---------------------------------------------------------------------------


def _run_regressions_per_index(
    master_regimes: pd.DataFrame,
    config: dict,
    *,
    interactions: bool,
    indices: list[str] | None = None,
    cluster_time: bool = True,
) -> dict:
    """Shared loop: for each index run a single spec with all controls.

    Returns ``{idx_name: PanelResults}``. Indices absent from the panel
    are silently skipped so the helper works on reduced master frames.

    Parameters
    ----------
    cluster_time : bool
        If True (default), use two-way clustering (entity + time).
        If False, cluster by entity only — this matches the standard
        panel-data convention and should be used for marginal-effects
        tables where the interaction-term covariances matter.
    """
    if indices is None:
        indices = config.get("indices", ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"])
    ctrl_vars = config.get(
        "controls",
        ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"],
    )
    dep_var = config.get("dependent_var", "sstran")

    models: dict = {}
    for idx_name in indices:
        if idx_name not in master_regimes.columns:
            logger.warning("Index %s not in master panel; skipping.", idx_name)
            continue
        all_needed_vars = [idx_name] + ctrl_vars
        reg_data = create_lags(master_regimes, all_needed_vars, lags=[1])

        indep_var = f"{idx_name}_lag1"
        lagged_ctrls = [f"{v}_lag1" for v in ctrl_vars]
        ols_data, exog_vars = prepare_regression_data(
            reg_data, dep_var, indep_var, lagged_ctrls, interactions=interactions
        )
        models[idx_name] = run_panel_ols(ols_data, dep_var, exog_vars, cluster_time=cluster_time)
    return models


def run_baseline_regressions(
    master_regimes: pd.DataFrame,
    config: dict,
    indices: list[str] | None = None,
) -> dict:
    """Baseline PanelOLS (no regime interactions) per globalisation index.

    Each regression is ``sstran ~ idx_{t-1} + controls_{t-1}`` with
    entity + time fixed effects. Lifted from
    ``notebooks/02_modern_pipeline.ipynb`` cell 54.
    """
    return _run_regressions_per_index(master_regimes, config, interactions=False, indices=indices)


def export_baseline_regression_table(
    master_regimes: pd.DataFrame,
    config: dict,
    out_dir: str | Path | None = None,
    indices: list[str] | None = None,
) -> Path:
    """Run :func:`run_baseline_regressions` and write the LaTeX comparison."""
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = run_baseline_regressions(master_regimes, config, indices=indices)
    if not models:
        raise ValueError("No baseline models produced — check that index columns exist.")

    comparison = compare(models, stars=True)
    latex_str = comparison.summary.as_latex()
    for old, new in LATEX_LABEL_MAP.items():
        latex_str = latex_str.replace(old, new)

    out_path = out_dir / "baseline_regression_table.tex"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(latex_str)
    logger.info(f"✅ Baseline comparison table saved to: {out_path}")
    return out_path


def run_interaction_regressions(
    master_regimes: pd.DataFrame,
    config: dict,
    indices: list[str] | None = None,
    cluster_time: bool = True,
) -> dict:
    """Interaction PanelOLS: idx × welfare-regime dummies, per index.

    Uses ``prepare_regression_data(interactions=True)`` so the interaction
    terms follow the project-wide convention (social-democrat reference,
    conservative / mediterranean / liberal / post-communist interactions).
    Lifted from ``notebooks/02_modern_pipeline.ipynb`` cells 60/61.

    Requires the ``regime_*`` dummy columns added by
    :func:`clean.panel_utils.add_welfare_regimes`.
    """
    return _run_regressions_per_index(
        master_regimes, config, interactions=True, indices=indices, cluster_time=cluster_time
    )


def export_interaction_regression_table(
    master_regimes: pd.DataFrame,
    config: dict,
    out_dir: str | Path | None = None,
    indices: list[str] | None = None,
) -> Path:
    """Run :func:`run_interaction_regressions` and write the LaTeX comparison."""
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = run_interaction_regressions(master_regimes, config, indices=indices)
    if not models:
        raise ValueError("No interaction models produced — check that index columns exist.")

    comparison = compare(models, stars=True)
    latex_str = comparison.summary.as_latex()
    for old, new in LATEX_LABEL_MAP.items():
        latex_str = latex_str.replace(old, new)

    out_path = out_dir / "interaction_regression_table.tex"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(latex_str)
    logger.info(f"✅ Interaction comparison table saved to: {out_path}")
    return out_path


def export_marginal_effects_tables(
    master_regimes: pd.DataFrame,
    config: dict,
    out_dir: str | Path | None = None,
    indices: list[str] | None = None,
) -> dict[str, Path]:
    """Write per-index marginal-effects-by-regime LaTeX tables.

    For each KOF index, fits the regime-interaction PanelOLS (same spec
    as :func:`run_interaction_regressions`) and uses
    :func:`analysis.regression_utils.generate_marginal_effects` to turn
    the interaction coefficients into a marginal-effect-per-regime
    table. Each table is written to
    ``outputs/tables/marginal_effects_{idx}.tex``.

    Previously these tables were printed inside
    ``notebooks/02_modern_pipeline.ipynb`` cell 60 via ``display`` but
    never persisted — so swapping the notebook to a thin call lost the
    information. This helper restores it.

    Returns ``{idx_name: Path}`` for every index that produced a model.
    Raises :class:`ValueError` when no index in the panel yields a fit.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = run_interaction_regressions(master_regimes, config, indices=indices)
    if not models:
        raise ValueError("No interaction models produced — check that index columns exist.")

    out_paths: dict[str, Path] = {}
    for idx_name, result in models.items():
        g_var = f"{idx_name}_lag1"
        me_table = generate_marginal_effects(result, g_var)
        # Round numerics for presentation
        num_cols = ["Marginal Effect", "Std. Error", "t-stat", "p-value"]
        me_table[num_cols] = me_table[num_cols].round(4)
        caption = f"Marginal Effects by Welfare Regime — {idx_name}"
        label = f"tab:marginal_effects_{idx_name}"
        latex_str = me_table.to_latex(
            index=False,
            caption=caption,
            label=label,
            column_format="lccccc",
            position="htbp",
        )
        out_path = out_dir / f"marginal_effects_{idx_name}.tex"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(latex_str)
        logger.info(f"✅ Marginal-effects table saved to: {out_path}")
        out_paths[idx_name] = out_path
    return out_paths


def export_consolidated_marginal_effects_table(
    master_regimes: pd.DataFrame,
    config: dict,
    out_dir: str | Path | None = None,
    indices: list[str] | None = None,
) -> Path:
    """Consolidated marginal-effects table (indices as columns, regimes as rows).

    Produces a publication-ready LaTeX table using booktabs formatting with:

    - Point estimates and significance stars (en-dash ``$-$`` for negatives)
    - Standard errors in parentheses from the full variance--covariance matrix:
      ``SE(β₁ + β_k) = sqrt(Var(β₁) + Var(β_k) + 2 Cov(β₁, β_k))``
    - Model statistics footer (FE, controls, N, R²)
    - Explanatory notes

    Uses entity-clustered SEs (matching the original interaction regression
    table). Also prints numeric results to the console for verification.

    Returns the output file path.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if indices is None:
        indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]

    models = run_interaction_regressions(
        master_regimes, config, indices=indices, cluster_time=False
    )
    if not models:
        raise ValueError("No interaction models produced — check that index columns exist.")

    all_me, model_stats = _collect_marginal_effects(models)

    latex_str = _build_consolidated_latex(
        all_me,
        model_stats,
        indices,
        caption="Marginal Effects of Globalization on Social Security Transfers by Welfare Regime",
        label="tab:marginal_effects_consolidated",
    )
    out_path = out_dir / "marginal_effects_consolidated.tex"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(latex_str)
    logger.info(f"✅ Consolidated marginal effects table saved to: {out_path}")

    print("\n" + "=" * 70)
    print("MARGINAL EFFECTS BY WELFARE REGIME (entity-clustered SEs)")
    print("=" * 70)
    _print_me_summary(all_me, indices, "Full sample")
    print("=" * 70)

    return out_path


def wald_test_marginal_effect(result, g_var: str, int_term: str) -> dict:
    """Wald test sanity check for a single marginal effect β₁ + β_k = 0.

    Constructs the restriction matrix R = [0 ... 1 ... 1 ... 0] (ones at
    the positions of *g_var* and *int_term*) and tests Rβ = 0 via the
    result object's variance--covariance matrix.

    Returns a dict with ``me``, ``se_manual``, ``t_manual``, ``chi2_wald``,
    ``p_wald`` so the caller can verify that t² ≈ χ².
    """
    from scipy import stats as sp_stats

    params = result.params
    cov = result.cov
    param_names = list(params.index)

    b1 = float(params[g_var])
    bk = float(params[int_term])
    me = b1 + bk

    var_b1 = float(cov.loc[g_var, g_var])
    var_bk = float(cov.loc[int_term, int_term])
    cov_b1_bk = float(cov.loc[g_var, int_term])
    se = np.sqrt(var_b1 + var_bk + 2 * cov_b1_bk)
    t_manual = me / se if se > 0 else np.nan

    R = np.zeros((1, len(param_names)))
    R[0, param_names.index(g_var)] = 1.0
    R[0, param_names.index(int_term)] = 1.0
    q = np.array([0.0])

    beta = params.values
    cov_mat = cov.values
    diff = R @ beta - q
    var_diff = R @ cov_mat @ R.T
    chi2 = float(diff @ np.linalg.solve(var_diff, diff))
    p_wald = 1.0 - sp_stats.chi2.cdf(chi2, df=1)

    return {
        "me": me,
        "se_manual": se,
        "t_manual": t_manual,
        "t_squared": t_manual**2 if not np.isnan(t_manual) else np.nan,
        "chi2_wald": chi2,
        "p_manual": 2 * (1 - sp_stats.t.cdf(abs(t_manual), df=result.df_resid)),
        "p_wald": p_wald,
    }


# ---------------------------------------------------------------------------
# GFC subsample marginal effects
# ---------------------------------------------------------------------------


def _run_interaction_on_window(
    master_regimes: pd.DataFrame,
    config: dict,
    year_min: int,
    year_max: int,
    indices: list[str],
) -> dict:
    """Fit interaction PanelOLS on a year-restricted window, entity-clustered.

    Lags are created on the full panel so the first year in the window
    still has a valid lag, then the estimation sample is filtered.
    """
    ctrl_vars = config.get(
        "controls",
        ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"],
    )
    dep_var = config.get("dependent_var", "sstran")

    models: dict = {}
    for idx_name in indices:
        if idx_name not in master_regimes.columns:
            continue
        all_needed = [idx_name] + ctrl_vars
        reg_data = create_lags(master_regimes, all_needed, lags=[1])
        window = reg_data[(reg_data["year"] >= year_min) & (reg_data["year"] <= year_max)].copy()

        indep_var = f"{idx_name}_lag1"
        lagged_ctrls = [f"{v}_lag1" for v in ctrl_vars]
        ols_data, exog_vars = prepare_regression_data(
            window, dep_var, indep_var, lagged_ctrls, interactions=True
        )
        models[idx_name] = run_panel_ols(ols_data, dep_var, exog_vars, cluster_time=False)
    return models


def _build_consolidated_latex(
    all_me: dict[str, pd.DataFrame],
    model_stats: dict[str, dict],
    indices: list[str],
    caption: str,
    label: str,
    notes_extra: str = "",
) -> str:
    """Shared LaTeX builder for consolidated marginal-effects tables."""
    regime_order = [
        "Social Democrat (Ref)",
        "Conservative",
        "Mediterranean",
        "Liberal",
        "Post-Communist",
    ]
    idx_labels = {
        "KOFGI": "Overall",
        "KOFEcGI": "Economic",
        "KOFSoGI": "Social",
        "KOFPoGI": "Political",
    }

    def _fmt_coef(val, stars):
        if pd.isna(val):
            return ""
        sign = "$-$" if val < 0 else "\\phantom{$-$}"
        return f"{sign}{abs(val):.3f}{stars}"

    def _fmt_se(val):
        if pd.isna(val):
            return ""
        return f"({val:.3f})"

    n_cols = len(indices)
    col_fmt = "l" + "c" * n_cols

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        r"\toprule",
    ]

    header1 = " & ".join(idx_labels.get(idx, idx) for idx in indices)
    header2 = " & ".join(f"({idx})" for idx in indices)
    lines.append(f" & {header1} \\\\")
    lines.append(f" & {header2} \\\\")
    lines.append(r"\midrule")

    for i, regime in enumerate(regime_order):
        coef_cells, se_cells = [], []
        for idx_name in indices:
            me_df = all_me.get(idx_name)
            if me_df is not None:
                row = me_df[me_df["Welfare Regime"] == regime]
                if not row.empty:
                    coef_cells.append(
                        _fmt_coef(row["Marginal Effect"].iloc[0], row["Sig."].iloc[0])
                    )
                    se_cells.append(_fmt_se(row["Std. Error"].iloc[0]))
                else:
                    coef_cells.append("")
                    se_cells.append("")
            else:
                coef_cells.append("")
                se_cells.append("")

        lines.append(f"{regime} & " + " & ".join(coef_cells) + " \\\\")
        lines.append(" & " + " & ".join(se_cells) + " \\\\")
        if i < len(regime_order) - 1:
            lines.append(r"\addlinespace")

    lines.append(r"\midrule")
    yes_row = " & ".join(["Yes"] * n_cols)
    lines.append(f"Country FE & {yes_row} \\\\")
    lines.append(f"Year FE & {yes_row} \\\\")
    lines.append(f"Controls & {yes_row} \\\\")

    nobs_row = " & ".join(str(model_stats[idx]["nobs"]) for idx in indices)
    lines.append(f"Observations & {nobs_row} \\\\")

    r2_row = " & ".join(f"{model_stats[idx]['r2_within']:.3f}" for idx in indices)
    lines.append(f"$R^2$ (within) & {r2_row} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(r"\small")
    lines.append(
        r"\item \textit{Notes:} Each cell reports the marginal effect of the lagged "
        r"globalization index on social security transfers (\% of GDP) for the indicated "
        r"welfare regime. Standard errors (in parentheses) are computed from the full "
        r"variance--covariance matrix: "
        r"$\mathrm{SE}(\hat\beta_1 + \hat\beta_k) = "
        r"\sqrt{\mathrm{Var}(\hat\beta_1) + \mathrm{Var}(\hat\beta_k) + "
        r"2\,\mathrm{Cov}(\hat\beta_1, \hat\beta_k)}$. "
        r"All specifications include entity and time fixed effects with "
        r"standard errors clustered by country. Controls: "
        r"log GDP per capita, inflation, deficit/GDP, government debt/GDP, "
        r"log population, and dependency ratio (all lagged one period). "
        r"Social Democratic is the reference category. "
        + notes_extra
        + r"$^{***}\,p<0.01$, $^{**}\,p<0.05$, $^{*}\,p<0.10$."
    )
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def _collect_marginal_effects(
    models: dict,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Extract marginal-effects DataFrames and model stats from fitted models."""
    all_me: dict[str, pd.DataFrame] = {}
    model_stats: dict[str, dict] = {}
    for idx_name, result in models.items():
        g_var = f"{idx_name}_lag1"
        all_me[idx_name] = generate_marginal_effects(result, g_var)
        model_stats[idx_name] = {
            "nobs": int(result.nobs),
            "r2_within": float(result.rsquared_within),
        }
    return all_me, model_stats


def _print_me_summary(all_me: dict[str, pd.DataFrame], indices: list[str], label: str) -> None:
    """Print marginal effects to console."""
    print(f"\n  {label}:")
    for idx_name in indices:
        me_df = all_me.get(idx_name)
        if me_df is None:
            continue
        print(f"    {idx_name}:")
        for _, row in me_df.iterrows():
            print(
                f"      {row['Welfare Regime']:25s}  ME={row['Marginal Effect']:+.4f}  "
                f"SE={row['Std. Error']:.4f}  t={row['t-stat']:+.3f}  "
                f"p={row['p-value']:.4f} {row['Sig.']}"
            )


def export_gfc_marginal_effects_tables(
    master_regimes: pd.DataFrame,
    config: dict,
    out_dir: str | Path | None = None,
    indices: list[str] | None = None,
    break_year: int = 2008,
) -> dict[str, Path]:
    """Marginal-effects tables for pre- and post-GFC subsamples.

    Fits the interaction specification (globalisation × welfare regime)
    separately on 1980–2007 and 2008–2023 windows with entity-clustered
    SEs, then produces one consolidated booktabs LaTeX table per era.

    Returns ``{"pre_gfc": Path, "post_gfc": Path}``.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables" / "gfc"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if indices is None:
        indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]

    year_min = int(master_regimes["year"].min())
    year_max = int(master_regimes["year"].max())

    windows = {
        "pre_gfc": (year_min, break_year - 1),
        "post_gfc": (break_year, year_max),
    }

    print("\n" + "=" * 70)
    print(f"MARGINAL EFFECTS BY WELFARE REGIME — GFC SPLIT (break = {break_year})")
    print("=" * 70)

    out_paths: dict[str, Path] = {}
    for era_key, (w_min, w_max) in windows.items():
        era_label = (
            f"Pre-GFC ({w_min}–{w_max})" if "pre" in era_key else f"Post-GFC ({w_min}–{w_max})"
        )
        models = _run_interaction_on_window(master_regimes, config, w_min, w_max, indices)
        if not models:
            logger.warning("No models for %s; skipping.", era_key)
            continue

        all_me, model_stats = _collect_marginal_effects(models)
        _print_me_summary(all_me, indices, era_label)

        latex_str = _build_consolidated_latex(
            all_me,
            model_stats,
            indices,
            caption=f"Marginal Effects of Globalization by Welfare Regime — {era_label}",
            label=f"tab:marginal_effects_{era_key}",
            notes_extra=f"Sample restricted to {w_min}--{w_max}. ",
        )
        out_path = out_dir / f"marginal_effects_{era_key}.tex"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(latex_str)
        logger.info(f"✅ {era_label} marginal effects saved to: {out_path}")
        print(f"\n  Saved: {out_path}")
        out_paths[era_key] = out_path

    print("=" * 70)
    return out_paths


# ---------------------------------------------------------------------------
# Post-Communist exclusion robustness check (notebook cell 59)
# ---------------------------------------------------------------------------


def run_interaction_regressions_excl_postcommunist(
    master_regimes: pd.DataFrame,
    config: dict,
    indices: list[str] | None = None,
) -> dict:
    """Interaction PanelOLS excluding the post-communist regime.

    Same specification as :func:`run_interaction_regressions` but with
    only three interaction terms (conservative, mediterranean, liberal)
    — the post-communist regime is folded into the social-democratic
    reference category.  This is a common robustness check because the
    post-communist welfare-state model developed under very different
    conditions and may distort the main interaction pattern.

    Lifted from ``notebooks/02_modern_pipeline.ipynb`` cell 59.
    """
    if indices is None:
        indices = config.get("indices", ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"])
    ctrl_vars = config.get(
        "controls",
        ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"],
    )
    dep_var = config.get("dependent_var", "sstran")

    models: dict = {}
    for idx_name in indices:
        if idx_name not in master_regimes.columns:
            logger.warning("Index %s not in master panel; skipping.", idx_name)
            continue
        all_needed_vars = [idx_name] + ctrl_vars
        reg_data = create_lags(master_regimes, all_needed_vars, lags=[1])

        indep_var = f"{idx_name}_lag1"
        lagged_ctrls = [f"{v}_lag1" for v in ctrl_vars]

        # Manually build 3 interaction terms (no post_communist)
        reg_data["int_conservative"] = reg_data[indep_var] * reg_data["regime_conservative"]
        reg_data["int_mediterranean"] = reg_data[indep_var] * reg_data["regime_mediterranean"]
        reg_data["int_liberal"] = reg_data[indep_var] * reg_data["regime_liberal"]

        custom_ctrls = ["int_conservative", "int_mediterranean", "int_liberal"] + lagged_ctrls
        ols_data, exog_vars = prepare_regression_data(
            reg_data, dep_var, indep_var, custom_ctrls, interactions=False
        )
        header = LATEX_LABEL_MAP.get(indep_var, idx_name).replace("_{t-1}", "").replace("$", "")
        models[header] = run_panel_ols(ols_data, dep_var, exog_vars)
    return models


def export_interaction_excl_postcommunist_table(
    master_regimes: pd.DataFrame,
    config: dict,
    out_dir: str | Path | None = None,
    indices: list[str] | None = None,
) -> Path:
    """Run the post-communist exclusion robustness check and write LaTeX.

    Produces ``interaction_excl_postcommunist_table.tex`` alongside the
    standard interaction table for easy comparison.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = run_interaction_regressions_excl_postcommunist(master_regimes, config, indices=indices)
    if not models:
        raise ValueError(
            "No interaction (excl. post-communist) models produced — "
            "check that index columns exist."
        )

    comparison = compare(models, stars=True)
    latex_str = comparison.summary.as_latex()
    for old, new in LATEX_LABEL_MAP.items():
        latex_str = latex_str.replace(old, new)

    out_path = out_dir / "interaction_excl_postcommunist_table.tex"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(latex_str)
    logger.info(f"✅ Interaction (excl. post-communist) table saved to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Residual-based Pesaran CD test (post-estimation diagnostic)
# ---------------------------------------------------------------------------


def export_residual_cd_table(
    master_regimes: pd.DataFrame,
    config: dict,
    out_dir: str | Path | None = None,
    indices: list[str] | None = None,
) -> Path:
    """Run the Pesaran CD test on baseline residuals and export LaTeX.

    Two-way clustered SEs (the project default) handle *weak* cross-
    sectional dependence but become inconsistent under *strong* CSD —
    i.e. common shocks that hit many countries simultaneously (oil
    shocks, 2008 financial crisis, COVID-19). The Pesaran (2004) CD
    test on the regression residuals diagnoses whether this matters:

        H0: E[ε_{it} · ε_{jt}] = 0  for all i ≠ j

    Rejecting H0 indicates that Driscoll-Kraay standard errors (or a
    factor-augmented model) would be a more conservative choice. We
    run the test per globalisation index on the *baseline* (no
    interactions) specification, which is the cleanest residual series.

    Writes ``outputs/tables/residual_cd_test.tex``.
    """
    from clean.tests import test_pesaran_cd

    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = run_baseline_regressions(master_regimes, config, indices=indices)
    if not models:
        raise ValueError("No baseline models for CD test — check that index columns exist.")

    rows = []
    for idx_name, result in models.items():
        # linearmodels exposes resids with (iso3, year) MultiIndex
        residuals = result.resids.reset_index()
        residuals.columns = ["iso3", "year", "resid"]
        cd_stat, p_value = test_pesaran_cd(residuals, var="resid")
        rows.append(
            {
                "Index": idx_name,
                "CD Statistic": round(cd_stat, 4) if not np.isnan(cd_stat) else np.nan,
                "p-value": round(p_value, 4) if not np.isnan(p_value) else np.nan,
                "Verdict": (
                    "Reject H₀ → CSD present"
                    if not np.isnan(p_value) and p_value < 0.05
                    else "Fail to reject H₀"
                ),
            }
        )

    cd_df = pd.DataFrame(rows)
    latex_str = cd_df.to_latex(
        index=False,
        caption="Pesaran (2004) Cross-Sectional Dependence Test on Baseline Residuals",
        label="tab:residual_cd_test",
        column_format="lccl",
        position="htbp",
        escape=False,
    )

    out_path = out_dir / "residual_cd_test.tex"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(latex_str)
    logger.info(f"✅ Residual CD test table saved to: {out_path}")
    return out_path


def _se_comparison_row(
    data: pd.DataFrame,
    idx_name: str,
    ctrl_vars: list[str],
    dep_var: str,
) -> dict | None:
    """Refit the baseline spec three ways and return one row for the SE table.

    Expects ``data`` to already contain lagged columns (``{v}_lag1``) for
    the index and each control. Returns ``None`` when the PanelOLS fit
    drops to an empty sample — caller decides how to surface that.
    """
    indep_var = f"{idx_name}_lag1"
    lagged_ctrls = [f"{v}_lag1" for v in ctrl_vars]
    ols_data, exog_vars = prepare_regression_data(
        data, dep_var, indep_var, lagged_ctrls, interactions=False
    )
    if len(ols_data) < len(exog_vars) + 10:
        return None

    one_way = run_panel_ols(ols_data, dep_var, exog_vars, cluster_entity=True, cluster_time=False)
    two_way = run_panel_ols(ols_data, dep_var, exog_vars, cluster_entity=True, cluster_time=True)
    dk = run_panel_ols(ols_data, dep_var, exog_vars, cov_type="kernel")

    row = {"Index": idx_name, "N": int(one_way.nobs)}
    for label, res in [
        ("One-way entity", one_way),
        ("Two-way", two_way),
        ("Driscoll-Kraay", dk),
    ]:
        coef = float(res.params[indep_var])
        se = float(res.std_errors[indep_var])
        pval = float(res.pvalues[indep_var])
        row[f"{label} coef"] = f"{coef:.4f}{significance_stars(pval)}"
        row[f"{label} SE"] = f"({se:.4f})"
        row[f"{label} p"] = round(pval, 4)
    return row


def export_se_comparison_table(
    master_regimes: pd.DataFrame,
    config: dict,
    out_dir: str | Path | None = None,
    indices: list[str] | None = None,
) -> Path:
    """Side-by-side SE sensitivity: one-way entity clustering vs Driscoll-Kraay.

    Papers in the globalisation-vs-welfare-state literature typically cluster
    standard errors by country only (one-way entity clustering). That handles
    within-country serial correlation but assumes cross-sectional
    independence — which the Pesaran CD test on our residuals rejects
    (see :func:`export_residual_cd_table`). Driscoll-Kraay kernel SEs are
    robust to both serial correlation and cross-sectional dependence.

    This helper refits the baseline per-index spec three times — once with
    one-way entity clustering (literature convention), once with two-way
    clustering (project default), and once with Driscoll-Kraay — and writes
    a single LaTeX table comparing the coefficient on the lagged
    globalisation index under each covariance estimator. Only the
    coefficient on the KOF index is shown because that is the quantity of
    interest; controls are in the table footer.

    Writes ``outputs/tables/se_comparison.tex``.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if indices is None:
        indices = config.get("indices", ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"])
    ctrl_vars = config.get(
        "controls",
        ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"],
    )
    dep_var = config.get("dependent_var", "sstran")

    rows = []
    for idx_name in indices:
        if idx_name not in master_regimes.columns:
            logger.warning("Index %s not in master panel; skipping.", idx_name)
            continue
        reg_data = create_lags(master_regimes, [idx_name] + ctrl_vars, lags=[1])
        row = _se_comparison_row(reg_data, idx_name, ctrl_vars, dep_var)
        if row is not None:
            rows.append(row)

    if not rows:
        raise ValueError("No indices found in panel — cannot build SE comparison table.")

    se_df = pd.DataFrame(rows)
    latex_str = se_df.to_latex(
        index=False,
        caption=(
            "Sensitivity of the lagged globalisation coefficient to the covariance "
            "estimator: one-way entity clustering (literature convention) vs. two-way "
            "clustering (project default) vs. Driscoll-Kraay kernel SEs (robust to CSD). "
            "Stars follow conventional thresholds (*** p<0.01, ** p<0.05, * p<0.10)."
        ),
        label="tab:se_comparison",
        column_format="lc" + "ccc" * 3,
        position="htbp",
        escape=False,
    )

    out_path = out_dir / "se_comparison.tex"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(latex_str)
    logger.info(f"✅ SE comparison table saved to: {out_path}")
    return out_path


SE_COMPARISON_SUBPERIODS: dict[str, tuple[int, int]] = {
    "pre_china_shock": (1980, 1999),
    "post_china_shock": (2000, 2023),
    "pre_gfc": (1980, 2007),
    "post_gfc": (2008, 2023),
}


def export_subperiod_se_comparison_tables(
    master_regimes: pd.DataFrame,
    config: dict,
    out_dir: str | Path | None = None,
    indices: list[str] | None = None,
) -> dict[str, Path]:
    """Per-era SE sensitivity: one-way entity vs two-way vs Driscoll-Kraay.

    Same idea as :func:`export_se_comparison_table` but repeated for each of
    the four subperiods used elsewhere in the project
    (pre/post China shock, pre/post GFC). Writes one LaTeX table per era:

    ``outputs/tables/se_comparison_{period}.tex``.

    This matters because papers that split on the China shock or the GFC
    almost always keep one-way entity clustering even though the shorter
    post-break samples are exactly where residual cross-sectional
    dependence bites hardest (fewer years to average out common shocks).
    The table lets a reviewer see whether a subperiod result that looks
    significant under the literature's SE convention survives Driscoll-
    Kraay.

    Returns a mapping ``{period_name: output_path}`` for every era that
    produced at least one fit.
    """
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent.parent.parent / "outputs" / "tables"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if indices is None:
        indices = config.get("indices", ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"])
    ctrl_vars = config.get(
        "controls",
        ["ln_gdppc", "inflation_cpi", "deficit", "debt", "ln_population", "dependency_ratio"],
    )
    dep_var = config.get("dependent_var", "sstran")

    valid_indices = [i for i in indices if i in master_regimes.columns]
    if not valid_indices:
        raise ValueError("No indices found in panel — cannot build subperiod SE tables.")

    written: dict[str, Path] = {}
    for period_name, (start_year, end_year) in SE_COMPARISON_SUBPERIODS.items():
        rows = []
        for idx_name in valid_indices:
            # Lag on the full panel, *then* filter by year so the earliest
            # year in the subperiod still has a valid lag.
            reg_data = create_lags(master_regimes, [idx_name] + ctrl_vars, lags=[1])
            if "year" not in reg_data.columns:
                logger.warning("year column missing; cannot slice subperiod %s", period_name)
                continue
            period_data = reg_data[
                (reg_data["year"] >= start_year) & (reg_data["year"] <= end_year)
            ].copy()
            row = _se_comparison_row(period_data, idx_name, ctrl_vars, dep_var)
            if row is not None:
                rows.append(row)

        if not rows:
            logger.warning("Subperiod %s produced no fits; skipping.", period_name)
            continue

        se_df = pd.DataFrame(rows)
        pretty = period_name.replace("_", " ").title()
        latex_str = se_df.to_latex(
            index=False,
            caption=(
                f"SE sensitivity in subperiod {pretty} ({start_year}--{end_year}): "
                "one-way entity clustering vs. two-way clustering vs. Driscoll-Kraay. "
                "Stars: *** p<0.01, ** p<0.05, * p<0.10."
            ),
            label=f"tab:se_comparison_{period_name}",
            column_format="lc" + "ccc" * 3,
            position="htbp",
            escape=False,
        )
        out_path = out_dir / f"se_comparison_{period_name}.tex"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(latex_str)
        logger.info(f"✅ Subperiod SE comparison ({period_name}) saved to: {out_path}")
        written[period_name] = out_path

    return written
