"""
Clean package - Data cleaning utilities for economics research.

This package provides utilities for processing and cleaning economic datasets.
"""

# Core processing
# Constants
from .constants import COUNTRY_TO_ISO3, DEFAULT_YEAR_MAX, DEFAULT_YEAR_MIN, TARGET_ISO3_32
from .documentation import generate_data_appendix, generate_methods_section
from .merge import get_merge_summary, merge_all_datasets, save_master_dataset

# Advanced utilities
from .metadata import export_codebook_to_csv, get_variable_info, print_codebook
from .panel_utils import (
    check_panel_balance,
    create_differences,
    create_lags,
    create_leads,
    fill_panel_gaps,
)
from .pipeline import process_all_datasets
from .quality import check_time_series_breaks, generate_quality_report
from .stats import (
    build_latex_appendix,
    correlation_matrix,
    create_publication_table,
    export_cointegration_latex,
    export_granger_causality_latex,
    export_hausman_latex,
    export_model_diagnostics_latex,
    export_reset_test_latex,
    export_stata_labels,
    export_vif_latex,
    generate_summary_stats,
)
from .subsets import COUNTRY_GROUPS, filter_by_region, list_regions
from .tests import (
    export_diagnostics_latex,
    generate_diagnostic_report,
    test_normality,
    test_stationarity,
)

# Shared utilities
from .utils import (
    filter_to_target_countries,
    filter_to_year_range,
    load_config,
    map_country_to_iso3,
    save_dataframe,
    setup_logging,
)

# Validation
from .validation import validate_master_data
from .viz import plot_correlation_matrix, plot_country_coverage, plot_distribution, plot_time_series
from .worldbank import WorldBankProcessor

__version__ = "0.3.0"

__all__ = [
    # Core pipeline
    "process_all_datasets",
    "merge_all_datasets",
    "get_merge_summary",
    "save_master_dataset",
    "validate_master_data",
    # Constants
    "TARGET_ISO3_32",
    "COUNTRY_TO_ISO3",
    "DEFAULT_YEAR_MIN",
    "DEFAULT_YEAR_MAX",
    "load_config",
    "setup_logging",
    # Metadata
    "get_variable_info",
    "print_codebook",
    "export_codebook_to_csv",
    # Subsets
    "COUNTRY_GROUPS",
    "filter_by_region",
    "list_regions",
    # Panel utilities
    "check_panel_balance",
    "create_lags",
    "create_leads",
    "create_differences",
    "fill_panel_gaps",
    # Statistics
    "generate_summary_stats",
    "correlation_matrix",
    "export_stata_labels",
    "create_publication_table",
    "export_vif_latex",
    "export_model_diagnostics_latex",
    "export_hausman_latex",
    "export_reset_test_latex",
    "export_granger_causality_latex",
    "export_cointegration_latex",
    "build_latex_appendix",
    # Shared utilities
    "filter_to_target_countries",
    "filter_to_year_range",
    "load_config",
    "map_country_to_iso3",
    "save_dataframe",
    "setup_logging",
    # Validation
    "validate_master_data",
    "WorldBankProcessor",
    # Visualization
    "plot_time_series",
    "plot_correlation_matrix",
    "plot_country_coverage",
    "plot_distribution",
    # Quality
    "generate_quality_report",
    "generate_diagnostic_report",
    "check_time_series_breaks",
    # Tests
    "test_stationarity",
    "test_normality",
    "export_diagnostics_latex",
    # Documentation
    "generate_methods_section",
    "generate_data_appendix",
]
