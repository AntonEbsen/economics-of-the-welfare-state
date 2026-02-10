# Installation Guide

## Quick Install (Editable Mode)

From the project root directory:

```bash
pip install -e .
```

This installs the package in "editable" mode, so changes to the code are immediately available.

## Install with Optional Dependencies

### Development Tools

```bash
pip install -e ".[dev]"
```

Includes: pytest, pytest-cov

### Advanced Features

```bash
pip install -e ".[advanced]"
```

Includes: linearmodels (panel unit root tests), stargazer (beautiful tables)

### Everything

```bash
pip install -e ".[all]"
```

## Verify Installation

```python
# Test import
from clean import process_all_datasets, merge_all_datasets
print("✅ Installation successful!")

# Check version
import clean
print(f"Version: {clean.__version__}")
```

## Run Tests

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=src/clean tests/

# Specific test file
pytest tests/test_merge.py -v
```

## Usage After Installation

Once installed, you can import from anywhere:

```python
from clean import (
    process_all_datasets,
    merge_all_datasets,
    create_lags,
    test_stationarity,
    generate_summary_stats
)
```

## Uninstall

```bash
pip uninstall econ-clean
```
