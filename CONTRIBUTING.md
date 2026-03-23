# Contributing Guide

Welcome! This guide explains how to work with the project's code and data even if you are not a programmer.

---

## If You Just Want to Re-Run the Analysis

**Step 1:** Open the `economics-of-the-welfare-state` folder on your computer.

**Step 2:** Double-click the file called **`run_analysis.bat`**.

A black terminal window will appear and run everything automatically. When it says **ALL DONE**, close the window.

**Step 2:** Your results are waiting for you:
- 📄 **Regression tables** → `outputs/tables/` (`.tex` files for LaTeX)
- 📊 **Specification curve figures** → `outputs/figures/` (`.png` image files)
- 📝 **Execution log** → `pipeline.log` (open in Notepad if something went wrong)

---

## If You Want to Change a Variable or Setting

All regression settings live in one single file: **`config.yaml`** (open it with Notepad or any text editor).

```yaml
# config.yaml

indices:             # The globalization indices to test
  - KOFGI
  - KOFEcGI

controls:            # Macroeconomic control variables
  - ln_gdppc
  - inflation_cpi
  - deficit
  - debt
  - ln_population
  - dependency_ratio

dependent_var: sstran  # The outcome variable

lags: [1]            # How many years to lag the controls
```

**Examples:**
- To **add a new control variable**, add a new line under `controls:` with the variable name.
- To **test only one index**, remove the lines you don't need under `indices:`.
- After saving `config.yaml`, re-run `run_analysis.bat` to regenerate results.

---

## If You Want to Read the Outputs in Excel

The `.tex` table files are for LaTeX. If you'd like to view them in Excel, ask Anton to run the `export_to_excel()` helper, which saves an `.xlsx` alongside the `.tex`.

---

## Understanding the Log File

If the `run_analysis.bat` script fails, open `pipeline.log` in any text editor. Search for the word **ERROR** to find what went wrong and share the relevant lines with Anton.

---

## Project Structure (Quick Map)

| Folder / File | What is it? |
|---|---|
| `config.yaml` | All model settings (edit here!) |
| `run_analysis.bat` | Run everything with a double-click |
| `notebooks/02_modern_pipeline.ipynb` | Main analysis notebook (for Anton) |
| `data/raw/` | Original data files (do not edit) |
| `data/processed/` | Cleaned data (auto-generated) |
| `outputs/tables/` | Regression tables in `.tex` format |
| `outputs/figures/` | Charts and plots in `.png` format |
| `src/` | Python source code (for Anton) |
| `pipeline.log` | Execution log (check here if something fails) |

---

## Questions?

Contact **Anton Ebsen** — he maintains the code and can re-run or adjust the analysis.
