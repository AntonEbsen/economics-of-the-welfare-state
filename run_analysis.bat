@echo off
title Economics of the Welfare State - Analysis Pipeline
color 0A

echo =============================================
echo    Economics of the Welfare State
echo    Automated Analysis Pipeline
echo =============================================
echo.

REM Step 1: Check Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not on PATH.
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Installing / updating Python dependencies...
pip install -e .[dev] -q
if errorlevel 1 (
    echo ERROR: Dependency installation failed. Check your internet connection.
    pause
    exit /b 1
)
echo     Done.
echo.

echo [2/3] Running the data cleaning pipeline...
python -m src.clean.pipeline
if errorlevel 1 (
    echo ERROR: Data pipeline failed. See pipeline.log for details.
    pause
    exit /b 1
)
echo     Done.
echo.

echo [3/3] Generating regression tables and figures...
python -c "
import sys, pathlib
sys.path.insert(0, str(pathlib.Path('src')))
from clean.utils import load_config, setup_logging
from clean.panel_utils import add_welfare_regimes
from analysis.robustness import export_stepwise_robustness_tables, export_subperiod_regressions, export_subperiod_heterogeneity_regressions
import pandas as pd

setup_logging('pipeline.log')
config = load_config()
master = pd.read_parquet('data/final/master_dataset.parquet')
master_regimes = add_welfare_regimes(master)
export_stepwise_robustness_tables(master_regimes, config)
export_subperiod_regressions(master_regimes, config)
export_subperiod_heterogeneity_regressions(master_regimes, config)
print('Tables and figures saved to outputs/')
"
if errorlevel 1 (
    echo ERROR: Regression pipeline failed. See pipeline.log for details.
    pause
    exit /b 1
)
echo     Done.
echo.

echo =============================================
echo    ALL DONE!
echo    - Tables: outputs\tables\
echo    - Figures: outputs\figures\
echo    - Log:     pipeline.log
echo =============================================
echo.
pause
