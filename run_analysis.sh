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

echo "[1/4] Installing / updating Python dependencies..."
python3 -m pip install -e ".[dev]" --quiet
echo "    Done."

echo
echo "[2/4] Verifying raw data manifest..."
econ-clean verify-data

echo
echo "[3/4] Running the data cleaning pipeline..."
econ-clean clean
echo "    Done."

echo
echo "[4/4] Generating regression tables and figures..."
econ-clean analyze
echo "    Done."

echo
echo "============================================="
echo "   ALL DONE!"
echo "   - Tables:  outputs/tables/"
echo "   - Figures: outputs/figures/"
echo "   - Log:     pipeline.log"
echo "============================================="
