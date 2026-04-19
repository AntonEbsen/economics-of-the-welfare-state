# `analysis/` — legacy reference implementations

This directory holds **reference** scripts for running the baseline
panel specifications in other statistical environments. They are not
part of the main Python pipeline and are **not wired into CI or the
test suite.** Treat them as documentation that happens to be executable.

## Files

| File | What it is | Status |
|---|---|---|
| `analysis.R` | fixest/plm fixed-effects + Hausman example on `master.csv` | reference only |
| `analysis.do` | Stata `xtreg` fixed-effects + Hausman example on `master.dta` | reference only |
| `label_variables.do` | Stata variable labels emitted by `src/clean/stats.py::export_stata_labels` | auto-generated |
| `master.csv` | snapshot of `data/final/master_dataset.csv` for the above | checked-in copy |
| `master.dta` | snapshot of `data/final/master_dataset.dta` for the above | checked-in copy |

## Running

### R

```r
setwd("analysis")
source("analysis.R")
```

Requires `fixest`, `plm`, and `did`. Produces the same baseline
coefficients as `src/analysis/regression_utils.py::run_panel_ols` up
to numerical precision.

### Stata

```stata
cd analysis
do analysis.do
```

Requires Stata 15 or newer. Expects `master.dta` in the same directory.

## Why keep them?

- They provide an independent check on the Python implementation: if
  the two-way FE coefficient from `fixest::feols` or Stata `xtreg,
  fe` differs from the `linearmodels.PanelOLS` estimate, something is
  wrong with the data or the specification (not the library).
- Reviewers and replicators sometimes prefer R or Stata output; having
  ready-to-run entry points lowers the bar.

## Refreshing after a data update

The snapshots `master.csv` and `master.dta` are pinned copies that
match the published results. To regenerate them after a pipeline
change, run the Python pipeline and then copy:

```bash
make data
cp data/final/master_dataset.csv  analysis/master.csv
cp data/final/master_dataset.dta  analysis/master.dta
```

## Scope

These files are intentionally minimal. Do not expand them into a
second parallel analysis pipeline — new robustness checks belong in
`src/analysis/robustness.py`. If you need R/Stata tooling for a new
check, export the coefficient table via `src/clean/export.py` and cite
it from the paper.
