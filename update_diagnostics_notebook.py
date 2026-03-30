import json
from pathlib import Path

notebook_path = Path("notebooks/02_modern_pipeline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code" and "generate_diagnostic_report(" in "".join(cell["source"]):
        src = cell["source"]

        # Check if already has it
        if "export_diagnostics_latex" not in "".join(src):
            # Find imports
            for idx, line in enumerate(src):
                if "from clean import generate_diagnostic_report" in line:
                    src[idx] = line.replace(
                        "generate_diagnostic_report",
                        "generate_diagnostic_report, export_diagnostics_latex",
                    )
                    break

            # Add call
            src.append("\n")
            src.append("# Also generate and save a publication-ready LaTeX table for Overleaf\n")
            src.append(
                'export_diagnostics_latex(master, key_vars, out_dir=REPO_ROOT / "outputs" / "tables")\n'
            )

        cell["source"] = src
        break

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("✅ Updated the diagnostic cell to include the LaTeX export.")
