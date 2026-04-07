import json
from pathlib import Path


def complete_fix():
    nb_path = Path("notebooks/02_modern_pipeline.ipynb")
    if not nb_path.exists():
        print(f"Error: {nb_path} not found.")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    fixed_count = 0
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            source_text = "".join(cell.get("source", []))
            if "# Component Comparison: Driscoll-Kraay Baseline Regressions" in source_text:
                if "from analysis.regression_utils import" not in source_text:
                    print("Fixing cell with missing imports...")
                    # Properly insert imports at the top
                    cell["source"] = [
                        "from analysis.regression_utils import prepare_regression_data, run_panel_ols, LATEX_LABEL_MAP\n",
                        "from clean.panel_utils import create_lags\n",
                        "from linearmodels.panel import compare\n",
                        "import os\n",
                        "\n",
                        "# Component Comparison: Driscoll-Kraay Baseline Regressions\n",
                    ] + (
                        cell["source"][1:] if cell["source"][0].startswith("#") else cell["source"]
                    )
                    fixed_count += 1

    if fixed_count > 0:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"Successfully fixed {fixed_count} cells in the notebook.")
    else:
        print("No cells needed fixing or target cells not found.")


if __name__ == "__main__":
    complete_fix()
