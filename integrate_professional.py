import json
import yaml

NOTEBOOK_PATH = 'notebooks/02_modern_pipeline.ipynb'
CONFIG_PATH = 'config.yaml'

def update_notebook_professional():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
            
        src = ''.join(cell['source'])
        
        # 1. Update Imports and Config Loading (Top of notebook)
        if 'import pandas as pd' in src and 'import numpy as np' in src and 'os.chdir' in src:
            cell['source'].insert(0, "import yaml\n")
            cell['source'].append("\n# Load configuration\n")
            cell['source'].append("with open('../config.yaml', 'r') as f:\n")
            cell['source'].append("    config = yaml.safe_load(f)\n")
            
        # 2. Add Data Validation after merging
        if 'master_regimes = add_welfare_regimes' in src:
            cell['source'].append("\n# ── Data Validation ──────────────────────────\n")
            cell['source'].append("from clean.validation import validate_master_data\n")
            cell['source'].append("master_regimes = validate_master_data(master_regimes)\n")
            
        # 3. Use Config for Regressions and Add Plotting
        if 'indices = ["KOFGI", "KOFEcGI", "KOFSoGI", "KOFPoGI"]' in src:
            if 'baseline_results = {}' in src:
                cell['source'] = [
                    "from analysis.regression_utils import prepare_regression_data, run_panel_ols, plot_coefficients, LATEX_LABEL_MAP\n",
                    "from linearmodels.panel import compare\n",
                    "\n",
                    "baseline_results = {}\n",
                    "indices = config['indices']\n",
                    "lagged_ctrls = [f\"{v}_lag1\" for v in config['controls']]\n",
                    "\n",
                    "for idx_name in indices:\n",
                    "    reg_data = create_lags(master_regimes, [idx_name] + config['controls'], lags=config['lags'])\n",
                    "    g_var = f\"{idx_name}_lag1\"\n",
                    "    \n",
                    "    ols_data, exog_vars = prepare_regression_data(reg_data, config['dependent_var'], g_var, lagged_ctrls, interactions=False)\n",
                    "    baseline_results[idx_name] = run_panel_ols(ols_data, config['dependent_var'], exog_vars)\n",
                    "    \n",
                    "    plot_coefficients(baseline_results[idx_name], title=f\"Baseline: {idx_name} on Social Spending\")\n",
                    "\n",
                    "baseline_comparison = compare(baseline_results, stars=True)\n",
                    "display(baseline_comparison)\n"
                ]
            elif 'interaction_results = {}' in src:
                cell['source'] = [
                    "from analysis.regression_utils import prepare_regression_data, run_panel_ols, generate_marginal_effects, plot_coefficients, LATEX_LABEL_MAP\n",
                    "from linearmodels.panel import compare\n",
                    "\n",
                    "interaction_results = {}\n",
                    "indices = config['indices']\n",
                    "lagged_ctrls = [f\"{v}_lag1\" for v in config['controls']]\n",
                    "\n",
                    "for idx_name in indices:\n",
                    "    reg_data = create_lags(master_regimes, [idx_name] + config['controls'], lags=config['lags'])\n",
                    "    g_var = f\"{idx_name}_lag1\"\n",
                    "    \n",
                    "    ols_data, exog_vars = prepare_regression_data(reg_data, config['dependent_var'], g_var, lagged_ctrls, interactions=True)\n",
                    "    interaction_results[idx_name] = run_panel_ols(ols_data, config['dependent_var'], exog_vars)\n",
                    "    \n",
                    "    me_table = generate_marginal_effects(interaction_results[idx_name], g_var)\n",
                    "    display(me_table.round(4))\n",
                    "    plot_coefficients(interaction_results[idx_name], title=f\"Interaction: {idx_name} with Welfare Regimes\")\n",
                    "\n",
                    "interaction_comparison = compare(interaction_results, stars=True)\n",
                    "display(interaction_comparison)\n"
                ]

    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
        
    print("Notebook professionalized with YAML, Validation, and Plots.")

if __name__ == "__main__":
    update_notebook_professional()
