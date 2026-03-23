"""
Robustness checks execution pipelines.
"""

import logging
import os
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
from linearmodels.panel import compare

from clean.panel_utils import create_lags

from .regression_utils import LATEX_LABEL_MAP, prepare_regression_data, run_panel_ols


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

            models[model_name] = run_panel_ols(ols_data, dep_var, exog_vars)

        comparison = compare(models, stars=True)
        logger.info(f"{'='*20} Stepwise Robustness: {idx_name} {'='*20}")
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
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np

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
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(
        x,
        coefficients,
        yerr=[err_lower, err_upper],
        fmt="o",
        color="#2563EB",
        ecolor="#93C5FD",
        elinewidth=2,
        capsize=6,
        markersize=8,
        markerfacecolor="white",
        markeredgewidth=2,
        zorder=3,
    )

    # Fill confidence band
    ax.fill_between(x, lower_ci, upper_ci, alpha=0.12, color="#2563EB")

    # Zero reference line
    ax.axhline(0, color="#EF4444", linestyle="--", linewidth=1.2, alpha=0.8)

    # Connect point estimates with a thin line
    ax.plot(x, coefficients, color="#2563EB", linewidth=1, alpha=0.4, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, rotation=30, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    idx_label = LATEX_LABEL_MAP.get(g_var, idx_name)
    ax.set_title(
        f"Specification Curve: {idx_label}",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Coefficient estimate (95% CI)", fontsize=10)
    ax.set_xlabel("Model specification", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()

    out_path = out_dir / f"specification_curve_{idx_name}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved specification curve to: {out_path}")
