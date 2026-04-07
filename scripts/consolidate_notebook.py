import json
from pathlib import Path


def consolidate_notebook():
    nb_path = Path("notebooks/02_modern_pipeline.ipynb")
    if not nb_path.exists():
        print(f"Error: {nb_path} not found.")
        return

    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    new_cells = []
    seen_headers = set()
    seen_code_snippets = set()

    for cell in nb["cells"]:
        source_text = "".join(cell.get("source", []))

        # Identify headers to avoid duplicates
        if cell["cell_type"] == "markdown":
            header = source_text.strip().split("\n")[0]
            if header in ["## Part 7.5: Unbundling Globalization - Component Analysis"]:
                if header in seen_headers:
                    print(f"Removing duplicate markdown header: {header}")
                    continue
                seen_headers.add(header)

        # Identify code snippets to avoid duplicates
        if cell["cell_type"] == "code":
            snippet = source_text.strip().split("\n")[0:3]
            snippet_str = "\n".join(snippet)
            if "# Component Comparison: Driscoll-Kraay Baseline Regressions" in snippet_str:
                if snippet_str in seen_code_snippets:
                    print(f"Removing duplicate code snippet: {snippet_str[:50]}...")
                    continue
                seen_code_snippets.add(snippet_str)

                # Ensure the one we keep HAS THE IMPORTS
                if "from analysis.regression_utils import" not in source_text:
                    print("Adding missing imports to the retained cell...")
                    cell["source"] = [
                        "from analysis.regression_utils import prepare_regression_data, run_panel_ols\n",
                        "from clean.panel_utils import create_lags\n",
                        "from linearmodels.panel import compare\n",
                        "import os\n",
                        "\n",
                        "# Component Comparison: Driscoll-Kraay Baseline Regressions\n",
                    ] + (
                        cell["source"][1:] if cell["source"][0].startswith("#") else cell["source"]
                    )

        new_cells.append(cell)

    nb["cells"] = new_cells

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Successfully consolidated notebook and fixed imports.")


if __name__ == "__main__":
    consolidate_notebook()
