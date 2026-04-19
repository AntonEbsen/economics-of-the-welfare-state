# Replication Notes

This document describes what a reader needs to reproduce the tables and
figures in the paper from a fresh clone of this repository. It is meant
to answer the questions that show up in the first hour of a replication
attempt: what do I need, how long will it take, what can go wrong.

## System requirements

- **Python** 3.11 or newer (other versions are not tested; `pyproject.toml`
  enforces `requires-python = ">=3.11"`).
- **Disk**: ~100 MB after a clean clone. Raw inputs live in `data/raw/`
  (7.5 MB of committed `.xlsx`); derived outputs in `data/processed/`
  (~5 MB) and `data/final/` (<1 MB); figures and tables in `outputs/`
  (~3.3 MB).
- **Memory**: the pipeline peaks at roughly **500 MB** of resident RAM.
  The `master_dataset` is 1,408 rows × 18 columns, ~0.2 MB in pandas
  memory — small enough that this project runs comfortably on a laptop.
- **Network**: not required. All raw inputs are committed. `pip install`
  is the only external dependency fetch.
- **Operating system**: Linux, macOS, and Windows all supported.
  `run_analysis.sh` is the Unix/macOS entry point; `run_analysis.bat` is
  the Windows peer. CI runs the full lint/test/smoke suite on Ubuntu.

## One-command reproduction

```bash
git clone https://github.com/AntonEbsen/economics-of-the-welfare-state
cd economics-of-the-welfare-state

# Unix / macOS
./run_analysis.sh

# Windows
run_analysis.bat

# Or via make (any platform)
make all
```

Either path runs:

1. `pip install -e .[dev]` — installs dependencies following the
   lower-bound specs in `pyproject.toml`, registering the
   `econ-clean` console script.
2. `econ-clean verify-data` — verifies SHA-256 checksums of the six
   raw `.xlsx` files against the manifest.
3. `econ-clean clean` — cleans and standardises each source, merges
   into `data/final/master_dataset.parquet`.
4. `econ-clean analyze` — exports regression tables and figures into
   `outputs/`.

## Pinned reproduction

The lower-bound deps in `pyproject.toml` accept newer releases, and
numerical output of econometric libraries can drift across minor
versions. For an exact reproduction of the published numbers:

```bash
make env-locked   # installs the resolved pins from requirements-lock.txt
make data         # rebuilds master dataset with the pinned stack
```

Or, if you prefer the (much faster) `uv` resolver:

```bash
make env-uv       # uv sync against uv.lock — same pins, ~10× faster install
make data
```

Both `requirements-lock.txt` (pip-compile) and `uv.lock` (uv) are
committed and describe the **same** resolved pin graph. `uv.lock` is
the preferred lock format going forward; `requirements-lock.txt` is
kept for users without `uv` on PATH. Maintainers must regenerate them
in lockstep after bumping bounds:

```bash
make lock      # refresh requirements-lock.txt via pip-compile
make lock-uv   # refresh uv.lock via uv
```

The locked stack pins the full transitive graph (pandas, numpy,
linearmodels, statsmodels, pandera, matplotlib, etc.).

## Runtime expectations

On a mid-range 2024 laptop (quad-core, 16 GB RAM):

| Step | Wall clock |
|------|-----------|
| Checksum verify (`econ-clean verify-data`) | ~1 s |
| Data pipeline (`econ-clean clean`) | ~15–20 s |
| Regression / robustness exports | ~30–60 s |
| Full `make all` (env + data + paper + report) | ~5 min |

CI runs the pipeline in the `smoke-pipeline` job against the 2015–2022
subset in under 60 s on GitHub-hosted runners.

## Expected outputs

After `./run_analysis.sh` completes successfully, you should see:

- `data/processed/<dataset>_32countries_1980_2023.{parquet,csv}` — one
  pair per source (`cpds`, `population`, `gdppc`, `inflation`,
  `dependency_ratio`, `kofgi`).
- `data/final/master_dataset.{parquet,csv,dta}` — the merged panel.
- `outputs/tables/*.tex`, `outputs/tables/*.csv` — regression tables
  including the stepwise robustness, subperiod splits, and heterogeneity
  by welfare regime.
- `outputs/figures/*.{png,pdf,svg}` — specification curves, marginal
  effects plots, event studies.
- `pipeline.log` — full run log (also where validation warnings surface).

## Common gotchas

### "MISSING: <file>" from `download_raw_data.py`
Some `git clone` strategies (sparse checkout, partial clone, LFS without
tokens) can skip the `data/raw/*.xlsx` files. Run a full clone:
```bash
git clone https://github.com/AntonEbsen/economics-of-the-welfare-state
```
Or manually download the files listed with `econ-clean verify-data --show`.

### "CHECKSUM MISMATCH"
Source data vendors (CPDS, World Bank, KOF) do periodically re-issue
their Excel exports with revisions. If you've refreshed a file and the
hash no longer matches the manifest, results may drift. The published
paper's numbers correspond to the checksums committed with this repo.

### `RuntimeWarning: 'src.clean.pipeline' found in sys.modules`
A benign warning from Python's `-m` runner when `src` is also on
`sys.path`. Safe to ignore.

### Pandera deprecation warnings
`pandera` 0.17+ warns about the `pandas`-namespace import shim. Our
schemas import via `pandera.pandas` which is the forward-compatible path.

### Stata export requires `pyreadstat`
`save_master_dataset(..., formats=["stata"])` needs the `pyreadstat`
wheel, which is a dependency of pandas' Stata I/O. It's included in
`[dev]` via pandas but flagged in older resolvers.

### LaTeX build needs `latexmk` on PATH
`make paper` invokes `latexmk -pdf paper/main.tex`. If you don't need
the PDF, skip this target — tables and figures render fine standalone.

### Quarto rendering needs Quarto
`make report` calls `quarto render`; install it from
<https://quarto.org>. The rendered HTML site publishes to GitHub Pages
via `.github/workflows/deploy-quarto.yml`.

## Verifying a successful run

After `make data`, smoke-test the output:

```bash
python -c "
import pandas as pd
df = pd.read_parquet('data/final/master_dataset.parquet')
assert df.shape == (1408, 18), df.shape
assert df['iso3'].nunique() == 32
assert df['year'].min() == 1980 and df['year'].max() == 2023
print('master dataset OK')
"
```

For a more exhaustive check, run the test suite: `pytest -q`
(should report `49 passed`).

## Getting help

- File an issue: <https://github.com/AntonEbsen/economics-of-the-welfare-state/issues>
- Architecture overview: [`docs/ARCHITECTURE.md`](ARCHITECTURE.md)
- Getting-started walkthrough: [`docs/GETTING_STARTED.md`](GETTING_STARTED.md)
