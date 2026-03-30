import json
from pathlib import Path

notebook_path = Path("notebooks/02_modern_pipeline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Update Part 1.5 Diagnostics (Cell index 28 - need to find it by content)
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "generate_diagnostic_report(master" in src:
        # Make it robust to both master and master_regimes
        new_src = cell["source"]
        for j, line in enumerate(new_src):
            if "generate_diagnostic_report(master," in line:
                new_src[j] = line.replace(
                    "master,", 'master_regimes if "master_regimes" in locals() else master,'
                )
        cell["source"] = new_src

# Update Robustness section (Cell index 68)
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell["source"])
    if "export_stepwise_robustness_tables(master," in src:
        new_src = cell["source"]
        for j, line in enumerate(new_src):
            if "export_stepwise_robustness_tables(master," in line:
                new_src[j] = line.replace(
                    "master,", 'master_regimes if "master_regimes" in locals() else master,'
                )
        cell["source"] = new_src

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Fixed NameError in the newly added cells.")
