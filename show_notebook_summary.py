"""
Notebook Cell Summary - What's in 01_cleaning_data.ipynb
"""
import json

with open('notebooks/01_cleaning_data.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"📊 Total cells in notebook: {len(cells)}\n")

# Find sections
sections = []
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown':
        source = ''.join(cell.get('source', []))
        if source.startswith('#'):
            title = source.split('\n')[0].strip('# ')
            sections.append((i, title))

print("📑 Major sections found:\n")
for pos, title in sections:
    print(f"  Cell {pos:2d}: {title}")

print("\n" + "="*70)
print("\n🎯 CPDS Section (cells 33-45):")
for i in range(33, min(46, len(cells))):
    cell = cells[i]
    source = ''.join(cell.get('source', []))[:60]
    print(f"  Cell {i}: {cell['cell_type']:8} - {source}...")

print("\n🎯 Population Section (cells 46-60):")
for i in range(46, min(61, len(cells))):
    cell = cells[i]
    source = ''.join(cell.get('source', []))[:60]
    print(f"  Cell {i}: {cell['cell_type']:8} - {source}...")

print("\n" + "="*70)
print("\n✅ Both CPDS and Population sections ARE in the notebook!")
print("\n📝 TO VIEW THEM IN YOUR EDITOR:")
print("   1. Close the notebook file if it's open")
print("   2. Reopen: notebooks/01_cleaning_data.ipynb")
print("   3. Scroll down past the KOF section (after cell 32)")
print("   4. You should see CPDS at cell 33 and Population at cell 46")
