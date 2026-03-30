import json
from pathlib import Path

notebook_path = Path("notebooks/02_modern_pipeline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Define the diagnostic cell
diagnostic_markdown = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Part 1.5: Research Diagnostics (Stationarity & Normality)\n",
        "\n",
        "Before proceeding to regressions, we must verify the statistical properties of our variables. \n",
        "The **Im-Pesaran-Shin (IPS)** test is used to check for unit roots in panel data. \n",
        "- **I(0)**: Stationary (Good for levels regression)\n",
        "- **I(1)**: Non-stationary (May require first-differencing)\n",
        "\n",
        "We also run the **Jarque-Bera** test to check for normality.",
    ],
}

diagnostic_code = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "from clean import generate_diagnostic_report\n",
        "from clean.utils import load_config\n",
        "\n",
        "config = load_config()\n",
        'key_vars = ["sstran"] + config.get("controls", [])\n',
        "\n",
        "# Generate diagnostic report for all key variables\n",
        "diagnostic_df = generate_diagnostic_report(master, key_vars)",
    ],
}

# Insert after Cell 8 (where master dataset is merged)
# Looking at cell_output.txt, Cell 8 was lag creation.
# Let's find Part 2 (Cell 12) and insert before it.
target_idx = 11
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and "## Part 2" in "".join(cell["source"]):
        target_idx = i
        break

nb["cells"].insert(target_idx, diagnostic_markdown)
nb["cells"].insert(target_idx + 1, diagnostic_code)

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"✅ Successfully added diagnostic sections before index {target_idx}")
