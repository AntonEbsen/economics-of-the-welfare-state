# Economics Research Toolkit

A comprehensive Python package for economics panel data analysis, from raw data processing to publication-ready outputs.

## Features

### 🎯 Core Processing

- **Automated data pipeline** - Process 5 datasets with one command
- **Smart validation** - Catch errors before analysis
- **Consistent formatting** - All datasets use ISO3 codes and standard year ranges

### 📊 Analysis Tools

- **Panel utilities** - Lags, leads, differences, balance checking
- **Summary stats** - Publication-ready tables (LaTeX, markdown, HTML)
- **Visualization** - Time series, correlations, coverage plots
- **Statistical tests** - Unit root (ADF, KPSS), normality tests

### 🔬 Research Tools

- **Robustness checks** - Automated specification tests
- **Publication tables** - Format regression results for papers
- **Methods generator** - Auto-write data sections
- **R/Stata export** - Generate ready-to-run scripts

## Quick Start

```python
from pathlib import Path
from clean import process_all_datasets, merge_all_datasets

# Process all datasets
results = process_all_datasets(Path("."), validate=True)

# Merge into master dataset
master = merge_all_datasets(results)

# You're ready to analyze!
print(f"{len(master):,} observations ready for analysis")
```

## Installation

```bash
# Install dependencies
pip install pandas numpy scipy statsmodels matplotlib seaborn

# Optional (for advanced features)
pip install linearmodels stargazer
```

## Tutorials

See `notebooks/` for detailed tutorials:

1. **00_quick_start.ipynb** - 5-minute introduction
2. **01_basic_usage.ipynb** - Complete workflow
3. **02_panel_analysis.ipynb** - Advanced panel methods
4. **03_publication_workflow.ipynb** - Paper preparation

## Documentation

- [API Reference](docs/API_REFERENCE.md)
- [Getting Started](docs/GETTING_STARTED.md)

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src/clean tests/
```

## Example Workflow

```python
# 1. Load and merge data
from clean import process_all_datasets, merge_all_datasets, create_lags
results = process_all_datasets(Path("."))
master = merge_all_datasets(results)

# 2. Prepare for analysis
master = create_lags(master, ['ln_gdppc'], lags=[1, 2])

# 3. Run robustness checks
from clean.robustness import run_robustness_checks
from statsmodels.formula.api import ols

robust_results = run_robustness_checks(
    master,
    'sstran ~ ln_gdppc + deficit',
    ols,
    checks=['drop_outliers', 'winsorize', 'pre_2008']
)

# 4. Create publication table
from clean import create_publication_table
table = create_publication_table(
    list(robust_results.values()),
    output_format='latex'
)

# 5. Generate methods section
from clean import generate_methods_section
methods = generate_methods_section(master, 'paper/methods.md')

# 6. Export to R/Stata
from clean.export import export_to_r
export_to_r(master, 'analysis/analysis.R')
```

## Package Structure

```
src/clean/
├── constants.py         # Centralized constants
├── validation.py        # Data validation
├── pipeline.py          # Master processing
├── merge.py            # Dataset merging
├── metadata.py         # Variable documentation
├── subsets.py          # Country grouping
├── panel_utils.py      # Panel data utilities
├── stats.py            # Summary statistics
├── viz.py              # Visualizations
├── quality.py          # Data quality reports
├── tests.py            # Statistical tests
├── documentation.py    # Methods generation
├── robustness.py       # Robustness checks
└── export.py          # R/Stata export
```

## Contributing

Contributions welcome! Please add tests for new features.

## License

MIT License

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{economics_toolkit,
  title = {Economics Research Toolkit},
  year = {2026},
  author = {Your Name},
  url = {https://github.com/yourusername/economics-toolkit}
}
```
