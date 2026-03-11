import json

NOTEBOOK_PATH = 'notebooks/02_modern_pipeline.ipynb'

def cleanup_notebook():
    with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    # indices to REMOVE: 51, 57-62
    to_remove = {51, 57, 58, 59, 60, 61, 62}
    
    for i, cell in enumerate(nb['cells']):
        if i in to_remove:
            print(f"Removing redundant cell {i}")
            continue
            
        # Fix Baseline regression (Cell 54 in previous check, but index may shift if we don't handle it)
        # Better: match by content
        if cell['cell_type'] == 'code':
            src = ''.join(cell['source'])
            if 'baseline_results = {}' in src and 'interactions=True' in src:
                print(f"Fixing Baseline regression in cell {i} (interactions=True -> False)")
                cell['source'] = [s.replace('interactions=True', 'interactions=False') for s in cell['source']]
        
        new_cells.append(cell)

    nb['cells'] = new_cells
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook cleanup complete.")

if __name__ == "__main__":
    cleanup_notebook()
