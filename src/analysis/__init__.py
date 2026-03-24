# Initialization for analysis module

from .latex_injector import inject_latex_results, update_paper_stats
from .regression_utils import (
    adjust_pvalues,
    generate_marginal_effects,
    plot_coefficients,
    prepare_regression_data,
    run_hausman_test,
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
    "run_hausman_test",
    "inject_latex_results",
    "update_paper_stats",
    "export_stepwise_robustness_tables",
]
