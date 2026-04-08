import json

file_path = "notebooks/02_modern_pipeline.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_source = [
    "from linearmodels.panel import compare\n",
    "from analysis.regression_utils import prepare_regression_data, run_panel_ols, generate_marginal_effects, LATEX_LABEL_MAP\n",
    "from clean.utils import load_config\n",
    "import os\n",
    "\n",
    "config = load_config()\n",
    "indices = ['KOFGI', 'KOFEcGI', 'KOFSoGI', 'KOFPoGI']\n",
    "interaction_results = {}\n",
    "\n",
    "for idx_name in indices:\n",
    "    if idx_name not in master_regimes.columns:\n",
    "        continue\n",
    "    current_ctrl_vars = [idx_name] + config.get('controls')\n",
    "    reg_data = create_lags(master_regimes, current_ctrl_vars, lags=[1])\n",
    "    g_var = idx_name + '_lag1'\n",
    "    lagged_ctrls = [v + '_lag1' for v in current_ctrl_vars if v != idx_name]\n",
    "    # Social Democrat = reference; Post-Communist excluded\n",
    "    reg_data['int_conservative'] = reg_data[g_var] * reg_data['regime_conservative']\n",
    "    reg_data['int_mediterranean'] = reg_data[g_var] * reg_data['regime_mediterranean']\n",
    "    reg_data['int_liberal'] = reg_data[g_var] * reg_data['regime_liberal']\n",
    "    custom_lagged_ctrls = ['int_conservative', 'int_mediterranean', 'int_liberal'] + lagged_ctrls\n",
    "    ols_data, exog_vars = prepare_regression_data(reg_data, 'sstran', g_var, custom_lagged_ctrls, interactions=False)\n",
    "    header_name = LATEX_LABEL_MAP.get(g_var, idx_name).replace('_{t-1}', '').replace('$', '')\n",
    "    interaction_results[header_name] = run_panel_ols(ols_data, 'sstran', exog_vars)\n",
    "    sep = '=' * 20\n",
    "    print('\\n' + sep + ' ' + header_name + ' ' + sep)\n",
    "    me_table = generate_marginal_effects(interaction_results[header_name], g_var)\n",
    "    display(me_table.round(4))\n",
    "\n",
    "interaction_comparison = compare(interaction_results, stars=True)\n",
    "display(interaction_comparison)\n",
    "\n",
    "output_file = '../outputs/tables/interaction_regression_table.tex'\n",
    "os.makedirs(os.path.dirname(output_file), exist_ok=True)\n",
    "latex_str = interaction_comparison.summary.as_latex()\n",
    "for old, new in LATEX_LABEL_MAP.items():\n",
    "    latex_str = latex_str.replace(old, new)\n",
    "with open(output_file, 'w', encoding='utf-8') as f:\n",
    "    f.write(latex_str)\n",
    "print('Interaction table saved to: ' + output_file)\n",
]

# Replace cell 59 unconditionally
nb["cells"][59]["source"] = new_source
nb["cells"][59]["outputs"] = []
nb["cells"][59]["execution_count"] = None
print("Patched cell 59")

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Done.")
