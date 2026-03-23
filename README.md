# Economics of the Welfare State

[![CI](https://github.com/AntonEbsen/economics-of-the-welfare-state/actions/workflows/ci.yml/badge.svg)](https://github.com/AntonEbsen/economics-of-the-welfare-state/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Empirical analysis of how globalization affects social security spending across OECD welfare states, using panel data from 1980–2022 with two-way fixed effects regressions.

---

## Research Question

> **Does globalization reduce social security transfers across OECD welfare states, and does this effect differ across welfare state regimes?**

The analysis uses the [KOF Globalization Index](https://kof.ethz.ch/en/forecasts-and-indicators/indicators/kof-globalisation-index.html) and the [Comparative Political Data Set (CPDS)](https://www.cpds-data.org/) across 30+ OECD countries from 1980 to 2022.

---

## Repository Structure

```
economics-of-the-welfare-state/
├── config.yaml                  # Centralized model configuration
├── Makefile                     # One-command pipeline (make all)
├── pyproject.toml               # Package metadata & dependencies
├── notebooks/
│   ├── 01_cleaning_pipeline.ipynb
│   └── 02_modern_pipeline.ipynb # Main analysis notebook
├── src/
│   ├── clean/                   # Data processing pipeline
│   │   ├── pipeline.py          # CLI + process_all_datasets()
│   │   ├── utils.py             # load_config(), setup_logging()
│   │   ├── validation.py        # pandera schema validation
│   │   └── panel_utils.py       # create_lags() and panel helpers
│   └── analysis/
│       ├── regression_utils.py  # run_panel_ols(), run_hausman_test()
│       └── robustness.py        # Stepwise checks + spec curve plots
├── data/
│   ├── raw/                     # Original source files
│   └── processed/               # Cleaned .parquet outputs
└── outputs/
    ├── tables/                  # LaTeX regression tables (.tex)
    └── figures/                 # Specification curve plots (.png)
```

---

## Variables

### Dependent Variable

| Variable | Description | Source |
|---|---|---|
| `sstran` | Social security transfers (% of GDP) | CPDS |

### Globalization Indices

| Variable | Description | Source |
|---|---|---|
| `KOFGI` | Overall KOF Globalization Index | KOF ETH Zürich |
| `KOFEcGI` | Economic Globalization sub-index | KOF ETH Zürich |
| `KOFSoGI` | Social Globalization sub-index | KOF ETH Zürich |
| `KOFPoGI` | Political Globalization sub-index | KOF ETH Zürich |

### Macroeconomic Controls (all lagged 1 period)

| Variable | Description | Source |
|---|---|---|
| `ln_gdppc` | Log GDP per capita | World Bank |
| `inflation_cpi` | CPI Inflation rate (%) | World Bank |
| `deficit` | Government fiscal deficit (% of GDP) | CPDS |
| `debt` | Government gross debt (% of GDP) | CPDS |
| `ln_population` | Log total population | World Bank |
| `dependency_ratio` | Old-age dependency ratio (%) | World Bank |

---

## Quickstart

### 1. Install Environment
```bash
pip install -e .[dev]
```

### 2. Install pre-commit hooks (auto-format on every commit)
```bash
pre-commit install
```

### 3. Run the full data pipeline
```bash
# Via Makefile
make data

# Or via CLI directly
python -m src.clean.pipeline --year-min 1980 --year-max 2022 --save-outputs
```

### 4. Run regressions
Open `notebooks/02_modern_pipeline.ipynb` and run all cells.

### 5. Compile the paper
```bash
make paper
```

### 6. Run everything end-to-end
```bash
make all
```

---

## Key Analysis Utilities

```python
from analysis.regression_utils import (
    prepare_regression_data,
    run_panel_ols,
    run_hausman_test,        # FE vs RE Hausman specification test
    generate_marginal_effects,
)
from analysis.robustness import export_stepwise_robustness_tables  # + spec curves
from clean.utils import load_config, setup_logging
```

### Running the Hausman Test
```python
hausman_result = run_hausman_test(ols_data, dep_var="sstran", exog_vars=exog_vars)
display(hausman_result)
# Output: Hausman Statistic | Degrees of Freedom | P-Value | Verdict (α=0.05)
```

---

## License

MIT License © Anton Ebsen
