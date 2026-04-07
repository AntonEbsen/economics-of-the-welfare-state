import json
from pathlib import Path


def fix_notebook():
    nb_path = Path("notebooks/02_modern_pipeline.ipynb")
    if not nb_path.exists():
        print(f"Error: {nb_path} not found.")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    fixed = False
    for cell in nb["cells"]:
        source_text = "".join(cell.get("source", []))
        if "# Component Comparison: Driscoll-Kraay Baseline Regressions" in source_text:
            # Check if imports are already there
            if "from analysis.regression_utils import" in source_text:
                print("Imports already present.")
                continue

            print("Adding missing imports to the cell...")
            cell["source"] = [
                "from analysis.regression_utils import prepare_regression_data, run_panel_ols\n",
                "from clean.panel_utils import create_lags\n",
                "\n",
                "# Component Comparison: Driscoll-Kraay Baseline Regressions\n",
            ] + cell["source"][
                1:
            ]  # Keep the rest of the cell
            fixed = True
            break

    if fixed:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("Successfully updated notebook cell.")
    else:
        print("Target cell not found or already fixed.")


if __name__ == "__main__":
    fix_notebook()
