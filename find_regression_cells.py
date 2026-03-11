import json

nb_path = 'notebooks/02_modern_pipeline.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = ''.join(c['source'][:5])
        if 'prepare_regression_data' in src:
            print(f"Cell {i}: {src[:100]}...")
