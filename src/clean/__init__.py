"""
Clean package - Data cleaning utilities for economics research.

This package provides utilities for processing and cleaning economic datasets.
"""

# Core processing
from .pipeline import process_all_datasets
from .merge import merge_all_datasets, get_merge_summary, save_master_dataset

# Shared utilities
from .utils import (
    map_country_to_iso3,
    filter_to_target_countries,
    filter_to_year_range,
    save_dataframe,
)
from .worldbank import WorldBankProcessor

# Validation
from .validation import validate_output

# Constants
from .constants import TARGET_ISO3_32, COUNTRY_TO_ISO3, DEFAULT_YEAR_MIN, DEFAULT_YEAR_MAX

# Advanced utilities
from .metadata import get_variable_info, print_codebook, export_codebook_to_csv
from .subsets import COUNTRY_GROUPS, filter_by_region, list_regions
from .panel_utils import (
    check_panel_balance,
    create_lags,
    create_leads,
    create_differences,
    fill_panel_gaps
)
from .stats import generate_summary_stats, correlation_matrix, export_stata_labels, create_publication_table
from .viz import (
    plot_time_series,
    plot_correlation_matrix,
    plot_country_coverage,
    plot_distribution
)
from .quality import generate_quality_report, check_time_series_breaks
from .tests import test_stationarity, test_normality
from .documentation import generate_methods_section, generate_data_appendix

__version__ = "0.3.0"

__all__ = [
    # Core pipeline
    'process_all_datasets',
    'merge_all_datasets',
    'get_merge_summary',
    'save_master_dataset',
    'validate_output',
    
    # Constants
    'TARGET_ISO3_32',
    'COUNTRY_TO_ISO3',
    'DEFAULT_YEAR_MIN',
    'DEFAULT_YEAR_MAX',
    
    # Metadata
    'get_variable_info',
    'print_codebook',
    'export_codebook_to_csv',
    
    # Subsets
    'COUNTRY_GROUPS',
    'filter_by_region',
    'list_regions',
    
    # Panel utilities
    'check_panel_balance',
    'create_lags',
    'create_leads',
    'create_differences',
    'fill_panel_gaps',
    
    # Statistics
    'generate_summary_stats',
    'correlation_matrix',
    'export_stata_labels',
    'create_publication_table',
    
    # Visualization
    'plot_time_series',
    'plot_correlation_matrix',
    'plot_country_coverage',
    'plot_distribution',
    
    # Quality
    'generate_quality_report',
    'check_time_series_breaks',
    
    # Tests
    'test_stationarity',
    'test_normality',
    
    # Documentation
    'generate_methods_section',
    'generate_data_appendix',
]
