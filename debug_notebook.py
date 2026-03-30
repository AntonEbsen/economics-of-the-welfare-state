import json
from pathlib import Path

notebook_path = Path("notebooks/02_modern_pipeline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if "export_stepwise_robustness_tables" in "".join(cell.get("source", [])):
        print(f"Cell {i} source:")
        print("".join(cell["source"]))
        print("-" * 20)
