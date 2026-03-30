import json
from pathlib import Path

notebook_path = Path("notebooks/02_modern_pipeline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The problematic cell content
new_source = [
    "from analysis.regression_utils import run_hausman_test, prepare_regression_data\n",
    "from clean import create_lags\n",
    "\n",
    "# ── Ensure 'reg_data' for KOFGI is specifically prepared ────────────\n",
    "idx_name = 'KOFGI'\n",
    "current_ctrl_vars = [idx_name] + config.get('controls')\n",
    "reg_data_hausman = create_lags(master_regimes, current_ctrl_vars, lags=[1])\n",
    "\n",
    "g_var = f'{idx_name}_lag1'\n",
    "lagged_ctrls_hausman = [f'{v}_lag1' for v in current_ctrl_vars if v != idx_name]\n",
    "\n",
    "# Prepare data and run the test\n",
    "ols_data, exog_vars = prepare_regression_data(reg_data_hausman, 'sstran', g_var, lagged_ctrls_hausman)\n",
    "hausman_result = run_hausman_test(ols_data, dep_var='sstran', exog_vars=exog_vars)\n",
    "display(hausman_result)",
]

# Update the cell at index 50
nb["cells"][50]["source"] = new_source

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Updated Hausman test cell in notebooks/02_modern_pipeline.ipynb")
