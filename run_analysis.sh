#!/usr/bin/env bash
# Unix/macOS analysis pipeline. Mirror of run_analysis.bat so contributors
# on non-Windows platforms have the same one-command entry point. For
# incremental development prefer `make data` / `make test` / `make all`
# — this script is the batteries-included path for a fresh clone.

set -euo pipefail

cd "$(dirname "$0")"

echo "============================================="
echo "   Economics of the Welfare State"
echo "   Automated Analysis Pipeline"
echo "============================================="

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found on PATH." >&2
  echo "Install Python 3.11+ from https://www.python.org/ and retry." >&2
  exit 1
fi

echo "[1/4] Verifying raw data manifest..."
python3 scripts/download_raw_data.py

echo
echo "[2/4] Installing / updating Python dependencies..."
python3 -m pip install -e ".[dev]" --quiet
echo "    Done."

echo
echo "[3/4] Running the data cleaning pipeline..."
python3 -m src.clean.pipeline
echo "    Done."

echo
echo "[4/4] Generating regression tables and figures..."
python3 - <<'PY'
from pathlib import Path
import pandas as pd

from clean.utils import load_config, setup_logging
from clean.panel_utils import add_welfare_regimes
from analysis.robustness import (
    export_stepwise_robustness_tables,
    export_subperiod_regressions,
    export_subperiod_heterogeneity_regressions,
)

setup_logging("pipeline.log")
config = load_config()
master_path = Path("data/final/master_dataset.parquet")
if not master_path.exists():
    raise SystemExit(f"master dataset not found at {master_path} — did the pipeline merge step run?")

master = pd.read_parquet(master_path)
master_regimes = add_welfare_regimes(master)
export_stepwise_robustness_tables(master_regimes, config)
export_subperiod_regressions(master_regimes, config)
export_subperiod_heterogeneity_regressions(master_regimes, config)
print("Tables and figures saved to outputs/")
PY
echo "    Done."

echo
echo "============================================="
echo "   ALL DONE!"
echo "   - Tables:  outputs/tables/"
echo "   - Figures: outputs/figures/"
echo "   - Log:     pipeline.log"
echo "============================================="
