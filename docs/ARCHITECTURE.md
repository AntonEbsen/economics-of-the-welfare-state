# Architecture

This is a short orientation for anyone (including future-you) opening
the repo and wondering where things live. It covers the module graph,
the data flow, and where to put new code.

## The data flow

```
           ┌─────────────────────────────────────┐
           │  data/raw/*.xlsx                    │  ← committed to repo
           │  (CPDS, KOF, World Bank × 4)        │    (~7.5 MB, SHA-256
           └──────────────┬──────────────────────┘     pinned in manifest)
                          │  scripts/download_raw_data.py (verify)
                          ▼
           ┌─────────────────────────────────────┐
           │  src/clean/                         │  ← per-source readers,
           │  {cpds,kofgi,gdppc,                 │    standardisers, and
           │   inflation,population,             │    country filters
           │   dependency_ratio}.py              │
           └──────────────┬──────────────────────┘
                          │  src/clean/pipeline.py::process_all_datasets()
                          ▼
           ┌─────────────────────────────────────┐
           │  data/processed/                    │  ← parquet + csv,
           │  <dataset>_32countries_YYYY_YYYY.*  │    one per source
           └──────────────┬──────────────────────┘
                          │  src/clean/merge.py::merge_all_datasets()
                          ▼
           ┌─────────────────────────────────────┐
           │  data/final/master_dataset.*        │  ← balanced panel,
           │  (parquet, csv, dta)                │    1,408 × 18
           └──────────────┬──────────────────────┘
                          │  src/analysis/{regression_utils, robustness,
                          │                export_web_data}.py
                          ▼
           ┌─────────────────────────────────────┐
           │  outputs/{tables,figures}/          │  ← LaTeX, CSV, PDF,
           │  paper/ (manuscript.tex)            │    PNG, SVG
           │  _site/ (Quarto HTML)               │
           │  web/ (Astro static site)           │
           └─────────────────────────────────────┘
```

## Package layout

### `src/clean/` — data cleaning and merging

The heavy lifting for turning heterogeneous source files into a balanced
country-year panel.

| Module | Responsibility |
|---|---|
| `pipeline.py` | `process_all_datasets()` orchestrator; `__main__` CLI |
| `cpds.py` | CPDS Excel → sstran/deficit/debt |
| `kofgi.py` | KOF index Excel → KOFGI/KOFEcGI/KOFSoGI/KOFPoGI |
| `gdppc.py`, `inflation.py`, `population.py`, `dependency_ratio.py` | World Bank indicators (each has its own Excel quirks) |
| `worldbank.py` | shared WB-wide-format → long-format transforms |
| `merge.py` | outer-join source frames, save master in multiple formats |
| `validation.py` | Pandera schemas + `validate_output()` assertions |
| `panel_utils.py` | `create_lags`, `create_differences`, `check_panel_balance`, `add_welfare_regimes` |
| `constants.py` | single source of truth for ISO3 codes, year defaults, country→ISO3 map |
| `metadata.py` | variable codebook (descriptions, units, sources) |
| `stats.py` | summary stat tables (LaTeX / markdown / HTML) |
| `quality.py` | auto-generated HTML data-quality report |
| `viz.py` | time-series / coverage / correlation plots |
| `subsets.py` | welfare-regime groupings |
| `structural_breaks.py` | Chow, QLR tests |
| `documentation.py` | auto-generated methods section |
| `export.py` | R / Stata script generators |
| `utils.py` | `load_config`, `setup_logging`, `map_country_to_iso3` |

### `src/analysis/` — econometrics

| Module | Responsibility |
|---|---|
| `regression_utils.py` | `run_panel_ols`, `run_hausman_test`, `run_event_study`, `run_placebo_test`, `prepare_regression_data`, `LATEX_LABEL_MAP` |
| `robustness.py` | baseline + regime-interaction tables, per-regime marginal-effects tables, post-communist exclusion robustness, stepwise specs, subperiod splits, heterogeneity by welfare regime, feedback (reverse-causality) regressions, KOF sub-component regressions |
| `correlations.py` | lagged-variable correlation matrix with significance stars (CSV + LaTeX export) |
| `trend_plots.py` | cross-country mean time-series figures (sstran + KOF indices) as PNG + PDF |
| `export_web_data.py` | JSON/CSV fixtures consumed by the Astro front-end in `web/` |
| `latex_injector.py` | swap generated tables into `paper/manuscript.tex` |

### `src/` — application layer

| File | Responsibility |
|---|---|
| `cli.py` | Typer-based `econ-clean` console script (`clean`, `analyze`, `verify-data`, `report`, `lock` subcommands) registered via `[project.scripts]` |
| `dashboard.py` | Streamlit data-quality dashboard (optional; not in CI) |
| `config.py` | pydantic settings wrapper around `config.yaml` |

### `tests/` — 109 tests

- `test_constants.py` — ISO3 coverage and year-range sanity
- `test_merge.py` — outer-join shape and column carry-through
- `test_panel_utils.py` — lags/differences/balance on synthetic frames
- `test_validation.py` — Pandera schemas + `validate_output()` branches
- `test_regression_utils.py` — PanelOLS recovers known β; Hausman shape; event study window; placebo distribution
- `test_pipeline_smoke.py` — end-to-end feed a 3×4 synthetic `results` dict through merge → schema → save
- `test_analysis.py`, `test_econometrics.py` — interaction terms and marginal effects
- `test_correlations.py` — correlation-matrix shape, stars, CSV + LaTeX export
- `test_feedback_regressions.py` — reverse-causality regressions + KOF sub-component (trade / finance / info / culture / interpersonal) regressions produce PanelResults per index and non-empty LaTeX tables
- `test_trend_plots.py` — cross-country mean of sstran and KOF indices aggregates correctly, writes PNG + PDF, skips missing indices, raises on empty input
- `test_baseline_interaction_regressions.py` — baseline (no interactions) and regime-interaction PanelOLS runs per index, per-regime marginal-effects LaTeX output per index, post-communist exclusion robustness, missing-index skip, ValueError on empty models, non-trivial LaTeX output
- `test_notebook_imports.py` — parses `02_modern_pipeline.ipynb`, verifies every top-level `from analysis.X import Y` still resolves (catches refactor drift) and that the nine thin-call markers remain intact
- `test_utils.py` — map_country_to_iso3, save_dataframe (parquet/csv/error), filter_to_target_countries (case normalisation, NaN drop), filter_to_year_range, load_config, setup_logging
- `test_latex_injector.py` — placeholder substitution, separate output path, missing template, unmatched placeholders left intact

## Dataset contract

The balanced master panel satisfies:

- **Key**: `(iso3, year)`, unique
- **Coverage**: 32 ISO3 codes × 44 years (1980–2023) = 1,408 rows
- **Schema**: `src/clean/validation.py::master_schema` (Pandera)
- **Dependent**: `sstran` (social security transfers, % GDP)
- **Main independent**: `KOFGI` plus sub-indices `KOFEcGI`, `KOFSoGI`,
  `KOFPoGI`
- **Controls**: `ln_gdppc`, `inflation_cpi`, `deficit`, `debt`,
  `ln_population`, `dependency_ratio`

`src/clean/panel_utils.py::add_welfare_regimes` attaches `regime_*`
dummies (conservative, social-democratic, mediterranean, liberal,
post-communist).

## Entry points

| Command | What it does |
|---|---|
| `make env` | install `[dev]` deps following `pyproject.toml` bounds |
| `make env-locked` | install exact pinned stack from `requirements-lock.txt` |
| `make env-uv` | install the pinned stack via `uv sync` (same pins, faster) |
| `make lock` | regenerate `requirements-lock.txt` via pip-tools |
| `make lock-uv` | regenerate `uv.lock` via `uv lock` |
| `make verify-data` | SHA-256 check on `data/raw/*.xlsx` |
| `make data` | run the cleaning pipeline |
| `make analyze` | regenerate LaTeX tables + PNG/PDF figures from the master panel |
| `make test` | run the pytest suite |
| `make lint` | ruff + black check (mirrors CI) |
| `make format` | ruff + black auto-fix |
| `make paper` | compile `paper/manuscript.tex` to PDF |
| `make report` | `quarto render` → `_site/` |
| `make all` | env + format + data + paper + report |
| `./run_analysis.sh` | one-command fresh-clone workflow (Unix) |
| `run_analysis.bat` | one-command fresh-clone workflow (Windows) |
| `econ-clean <cmd>` | installed console script; subcommands `clean`, `analyze`, `verify-data`, `report`, `lock` |
| `python -m src.clean.pipeline` | direct pipeline CLI (supports `--year-min`, `--year-max`, `--repo-root`, `--no-save`) |

## CI

`.github/workflows/ci.yml` runs three jobs on every push and PR:

1. **Lint (ruff + black + pre-commit)** — static checks on `src/` and
   `tests/`. pre-commit/action also runs the full hook list defined in
   `.pre-commit-config.yaml`.
2. **Tests + Coverage** — `pytest --cov=src`, uploads `coverage.xml` as
   an artifact.
3. **Smoke (real pipeline, subset years)** — verifies the raw data
   manifest, runs the pipeline on 2015–2022, asserts the expected
   parquet files are produced. Catches Excel-schema drift and merge
   regressions that unit tests don't.

A concurrency group keyed on `github.ref` cancels superseded runs so
push-then-PR-sync doesn't double-queue.

`.github/workflows/deploy-quarto.yml` publishes the Quarto site to
GitHub Pages on every push to `main`.

## Where to put new code

- **New data source** → add a module in `src/clean/<source>.py` with
  `read_…`, `standardize_…`, `filter_…`, `save_processed` following the
  existing per-source pattern. Wire it into
  `src/clean/pipeline.py::process_all_datasets` and
  `src/clean/merge.py::merge_all_datasets`. Add a test feeding a
  synthetic fixture into the new module.
- **New regression specification** → add to `src/analysis/robustness.py`
  if it's a variant of an existing table; add to
  `src/analysis/regression_utils.py` if it introduces a new estimator.
  Put LaTeX labels in `LATEX_LABEL_MAP` — do not duplicate the map.
- **Exploratory notebook** → new cell in `notebooks/02_modern_pipeline.ipynb`
  (or a new `03_*.ipynb`). Keep heavy logic in `src/analysis/`; the
  notebook should be thin orchestration + narrative.
- **Documentation** → user-facing guides go in `docs/`; the paper skeleton
  lives under `paper/`. Treat `docs/ARCHITECTURE.md` (this file) as
  source-of-truth for the module graph and refresh it when modules move.

## Related docs

- [`docs/GETTING_STARTED.md`](GETTING_STARTED.md) — first-run walkthrough
- [`docs/replication_notes.md`](replication_notes.md) — reproduction
  guide, runtime expectations, known gotchas
- [`docs/README.md`](README.md) — package feature overview
- [`paper/methods.md`](../paper/methods.md) — data section of the paper
