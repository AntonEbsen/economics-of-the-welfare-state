# Initialization for analysis module

from .latex_injector import format_latex_table, inject_tables_into_tex
from .regression_utils import (
    adjust_pvalues,
    generate_marginal_effects,
    plot_coefficients,
    prepare_regression_data,
    run_panel_ols,
    run_placebo_test,
)
from .robustness import export_stepwise_robustness_tables

__all__ = [
    "prepare_regression_data",
    "run_panel_ols",
    "generate_marginal_effects",
    "plot_coefficients",
    "adjust_pvalues",
    "run_placebo_test",
    "inject_tables_into_tex",
    "format_latex_table",
    "export_stepwise_robustness_tables",
]
