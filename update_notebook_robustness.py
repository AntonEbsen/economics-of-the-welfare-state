import json
from pathlib import Path

notebook_path = Path("notebooks/02_modern_pipeline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Robustness section
robustness_idx = -1
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and "Robustness Checks" in "".join(cell["source"]):
        robustness_idx = i
        break

if robustness_idx != -1:
    summary_markdown = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Enhanced Robustness Analysis\n",
            "\n",
            "We can now automatically generate high-resolution **Specification Curves** and a **Robustness Summary**.\n",
            "The summary table shows the percentage of model specifications where each control variable remained statistically significant.",
        ],
    }

    summary_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "from clean.utils import load_config\n",
            "from analysis.robustness import export_stepwise_robustness_tables\n",
            "\n",
            "config = load_config()\n",
            "\n",
            "# This will now also log a Robustness Summary and save 300-DPI figures\n",
            "export_stepwise_robustness_tables(\n",
            "    master, \n",
            "    config, \n",
            "    out_dir=REPO_ROOT / 'outputs' / 'tables'\n",
            ")",
        ],
    }

    # Insert after the Robustness Checks header code cell
    # Assuming the next cell is the code to run robustness
    nb["cells"].insert(robustness_idx + 1, summary_markdown)
    nb["cells"].insert(robustness_idx + 2, summary_code)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Successfully added robustness enhancements section")
