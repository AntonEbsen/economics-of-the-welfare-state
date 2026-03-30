import json
from pathlib import Path

notebook_path = Path("notebooks/02_modern_pipeline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "generate_diagnostic_report(" in "".join(cell["source"]):
        src = cell["source"]

        # Check if already has it
        if "export_vif_latex" not in "".join(src):
            # Find imports
            for idx, line in enumerate(src):
                if "from clean import " in line:
                    src[idx] = line.replace(
                        "export_diagnostics_latex", "export_diagnostics_latex, export_vif_latex"
                    )
                    break

            # Add call
            src.append("\n")
            src.append(
                "# Check for Multicollinearity across different model specifications (VIF > 10 is problematic)\n"
            )
            src.append(
                'export_vif_latex(master, config.get("indices", []), config.get("controls", []), out_dir=REPO_ROOT / "outputs" / "tables")\n'
            )

        cell["source"] = src
        break

# also update the markdown explaining the section
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and "Research Diagnostics" in "".join(cell["source"]):
        src = cell["source"]
        if "Variance Inflation Factor" not in "".join(src):
            src.append(
                "\nWe also calculate **Variance Inflation Factor (VIF)** to check for Multicollinearity among our regressors."
            )
            cell["source"] = src
        break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Updated the diagnostic cell to include the VIF LaTeX export.")
