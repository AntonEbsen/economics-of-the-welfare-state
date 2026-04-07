import json
from pathlib import Path


def update_notebook():
    nb_path = Path("notebooks/02_modern_pipeline.ipynb")
    if not nb_path.exists():
        print(f"Error: {nb_path} not found.")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find index of Part 8: Feedback Regressions
    idx_part8 = -1
    for i, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and "## Part 8" in source:
            idx_part8 = i
            break

    if idx_part8 == -1:
        print("Error: Could not find Part 8 section in notebook.")
        return

    # Check if section already exists to avoid duplicates
    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        if "Part 7.5: Unbundling Globalization" in source:
            print("Section already exists. Skipping.")
            return

    # Create Markdown cell
    md_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Part 7.5: Unbundling Globalization - Component Analysis\n",
            "\n",
            "In this section, we decompose the aggregate Globalization indices into their specific sub-dimensions to identify the precise drivers of the welfare state relationship:\n",
            "- **Economic**: Trade (`KOFTrGI`) vs. Financial (`KOFFiGI`)\n",
            "- **Social**: Interpersonal (`KOFIpGI`), Informational (`KOFInGI`), and Cultural (`KOFCuGI`)\n",
            "\n",
            'This "unbundling" helps distinguish between the **Race-to-the-Bottom** hypothesis (usually linked to trade and tax competition) and the **Compensation Hypothesis** (linked to economic insecurity from financial volatility).\n',
        ],
    }

    # Create Code cell
    code_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Component Comparison: Driscoll-Kraay Baseline Regressions\n",
            "component_results = {}\n",
            'sub_components = ["KOFTrGI", "KOFFiGI", "KOFIpGI", "KOFInGI", "KOFCuGI"]\n',
            "\n",
            "for comp in sub_components:\n",
            "    # Prepare variables with lags (1 year) using create_lags utility\n",
            '    current_ctrl_vars = [comp] + config.get("controls")\n',
            "    reg_data = create_lags(master, current_ctrl_vars, lags=[1])\n",
            "    \n",
            '    dep_var = config.get("dependent_var", "sstran")\n',
            '    indep_var = f"{comp}_lag1"\n',
            '    lagged_ctrls = [f"{v}_lag1" for v in config.get("controls")]\n',
            "    \n",
            "    # Unified preparation and Driscoll-Kraay estimation\n",
            "    ols_data, exog_vars = prepare_regression_data(reg_data, dep_var, indep_var, lagged_ctrls, interactions=False)\n",
            "    component_results[comp] = run_panel_ols(ols_data, dep_var, exog_vars)\n",
            "\n",
            "# Display Side-by-Side Comparison\n",
            "component_comparison = compare(component_results, stars=True)\n",
            'print("\\n" + "="*60)\n',
            'print("📊 GLOBALIZATION COMPONENT COMPARISON (DRISCOLL-KRAAY)")\n',
            'print("="*60)\n',
            "display(component_comparison)\n",
            "\n",
            "# Export Summary Table to LaTeX\n",
            'output_file = "../outputs/tables/component_regression_table.tex"\n',
            "os.makedirs(os.path.dirname(output_file), exist_ok=True)\n",
            'with open(output_file, "w", encoding="utf-8") as f:\n',
            "    f.write(component_comparison.summary.as_latex())\n",
            'print(f"\\n✅ Component comparison table saved to: {output_file}")\n',
        ],
    }

    # Insert cells
    nb["cells"].insert(idx_part8, md_cell)
    nb["cells"].insert(idx_part8 + 1, code_cell)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully updated notebook with Component Analysis section.")


if __name__ == "__main__":
    update_notebook()
