import json

nb_path = 'notebooks/02_modern_pipeline.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'baseline_results = {}' in src:
            print(f"Cell {i} is Baseline.")
            # Check for incorrect interaction code
            if 'interactions=True' in src:
                print(f"Cell {i} HAS INCORRECT interaction=True")
        if 'interaction_results = {}' in src:
            print(f"Cell {i} is Interaction.")
