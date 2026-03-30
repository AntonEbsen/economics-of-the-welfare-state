import json
from pathlib import Path

notebook_path = Path("notebooks/02_modern_pipeline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        # Target the multi-line call specifically
        if "export_stepwise_robustness_tables(" in src and "    master," in src:
            new_src = []
            for line in cell["source"]:
                if "    master," in line:
                    new_src.append(
                        line.replace(
                            "master,", 'master_regimes if "master_regimes" in locals() else master,'
                        )
                    )
                else:
                    new_src.append(line)
            cell["source"] = new_src

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Fixed the multi-line NameError in notebooks/02_modern_pipeline.ipynb")
