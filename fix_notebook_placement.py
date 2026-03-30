import json
from pathlib import Path

notebook_path = Path("notebooks/02_modern_pipeline.ipynb")
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# The cells we added were:
# 1. Markdown: "### Enhanced Robustness Analysis..."
# 2. Code: "from clean.utils import load_config\nfrom analysis.robustness import export_stepwise_robustness_tables\n..."

# Find the mistakenly placed robustness cells
cells_to_remove = []
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and "### Enhanced Robustness Analysis" in "".join(
        cell.get("source", [])
    ):
        cells_to_remove.append(i)

if len(cells_to_remove) > 0:
    wrong_idx = cells_to_remove[0]
    # We also need to get the code cell right after it
    extracted_markdown = nb["cells"][wrong_idx]
    extracted_code = nb["cells"][wrong_idx + 1]

    # Remove them from their current position
    nb["cells"].pop(wrong_idx + 1)
    nb["cells"].pop(wrong_idx)

    # Now find the ACTUAL Robustness Checks section.
    # We want a markdown cell that contains "Robustness Checks:" but is NOT the table of contents.
    # The ToC is usually near the beginning. The actual one is deeper in the notebook.
    correct_idx = -1
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "markdown":
            src = "".join(cell.get("source", []))
            if (
                "Robustness Checks:" in src
                and "Table of Contents" not in src
                and "table of contents" not in src.lower()
            ):
                # Make sure it's the actual header, e.g. starts with "###"
                if src.strip().startswith("### Robustness Checks"):
                    correct_idx = i
                    break

    if correct_idx != -1:
        # Insert after the correct header and the cell that follows it (which might be the original robustness test)
        # Actually, let's insert it right after the header itself
        nb["cells"].insert(correct_idx + 1, extracted_markdown)
        nb["cells"].insert(correct_idx + 2, extracted_code)

        with open(notebook_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)
        print(f"✅ Successfully moved robustness cells from index {wrong_idx} to {correct_idx + 1}")
    else:
        print("❌ Could not find the actual Robustness section!")
