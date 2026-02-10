# Getting Started Guide

## Installation

### 1. Install Python Dependencies

```bash
pip install pandas numpy scipy statsmodels matplotlib seaborn
```

### 2. Optional Packages

For full functionality:

```bash
# Panel unit root tests
pip install linearmodels

# Beautiful regression tables
pip install stargazer

# Testing
pip install pytest pytest-cov
```

## First Steps

### Process Your Data

```python
from pathlib import Path
from clean import process_all_datasets

# Process all raw datasets
results = process_all_datasets(
    repo_root=Path("."),
    year_min=1980,
    year_max=2023,
    validate=True
)
```

This processes:

- CPDS (sstran, deficit, debt)
- Population (ln_population)
- GDP per capita (ln_gdppc)
- Inflation CPI
- Dependency ratio

### Merge Datasets

```python
from clean import merge_all_datasets

master = merge_all_datasets(results, how='outer')
print(f"Master dataset: {len(master):,} observations")
```

### Quick Analysis

```python
from clean import generate_summary_stats

# Get summary statistics
stats = generate_summary_stats(master)
print(stats)

# Or export to LaTeX
latex_stats = generate_summary_stats(master, output_format='latex')
```

## Common Tasks

### Create Lags for Regression

```python
from clean import create_lags

master = create_lags(
    master, 
    variables=['ln_gdppc', 'deficit'],
    lags=[1, 2, 3]
)
```

### Test Stationarity

```python
from clean import test_stationarity

results = test_stationarity(
    master,
    variables=['ln_gdppc', 'sstran'],
    test='adf'
)
```

### Run Robustness Checks

```python
from clean.robustness import run_robustness_checks
from statsmodels.formula.api import ols

robust = run_robustness_checks(
    master,
    'sstran ~ ln_gdppc + deficit',
    ols,
    checks=['drop_outliers', 'winsorize']
)
```

### Export to R

```python
from clean.export import export_to_r

export_to_r(
    master,
    'analysis/analysis.R',
    include_packages=['fixest', 'plm']
)
```

## Next Steps

- See `notebooks/01_basic_usage.ipynb` for detailed tutorial
- Check `docs/API_REFERENCE.md` for all available functions
- Run `pytest tests/` to verify installation

## Getting Help

- Check documentation in `docs/`
- Review example notebooks in `notebooks/`
- Run `python -c "from clean import *; help(process_all_datasets)"`
