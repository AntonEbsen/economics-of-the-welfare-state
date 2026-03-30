import json
from pathlib import Path

notebook_path = Path("notebooks/02_modern_pipeline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Keep track of cells to delete (any robustness cell before the actual robustness section)
# Let's say any cell before index 30 that calls export_stepwise_robustness_tables
new_cells = []
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))

    # Is it the markdown header we accidentally added at the top?
    is_bad_markdown = False
    if cell["cell_type"] == "markdown" and "Enhanced Robustness Analysis" in src:
        # If it's before the halfway point of the notebook, it's definitely in the wrong place
        if i < len(nb["cells"]) // 2:
            is_bad_markdown = True

    # Is it the code cell we accidentally added at the top?
    is_bad_code = False
    if cell["cell_type"] == "code" and "export_stepwise_robustness_tables" in src:
        if i < len(nb["cells"]) // 2:
            is_bad_code = True

    if not is_bad_markdown and not is_bad_code:
        new_cells.append(cell)
    else:
        print(f"🗑️ Deleting misplaced cell at original index {i}")

nb["cells"] = new_cells

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Cleaned up mistakenly placed robustness cells from the top of the notebook.")
