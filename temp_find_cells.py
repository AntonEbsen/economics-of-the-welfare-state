import json

with open("notebooks/02_modern_pipeline.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Cell 21 (the Hausman test cell)
code_cell_idx = 0
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        if code_cell_idx == 21:
            print(f"Notebook cell index: {i}")
            print(f"Source: {''.join(c['source'])}")
            break
        code_cell_idx += 1
