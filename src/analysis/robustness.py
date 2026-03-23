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
    and export the model comparison matrices to LaTeX tables.

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
                model_name = f"+ {LATEX_LABEL_MAP.get(f'{macro_controls[step-1]}_lag1', macro_controls[step-1])}"

            models[model_name] = run_panel_ols(ols_data, dep_var, exog_vars)

        comparison = compare(models, stars=True)
        logger.info(f"\\n{'='*20} Stepwise Robustness: {idx_name} {'='*20}")
        logger.info(comparison)

        output_file = out_dir / f"stepwise_robustness_{idx_name}.tex"

        latex_str = comparison.summary.as_latex()
        for old, new in LATEX_LABEL_MAP.items():
            latex_str = latex_str.replace(old, new)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(latex_str)
        logger.info(f"✅ Saved table to: {output_file}")
