import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import traceback
import sys
import os

notebook_filename = 'notebooks/02_modern_pipeline.ipynb'

print(f"Executing {notebook_filename}...")

if not os.path.exists(notebook_filename):
    print(f"Error: {notebook_filename} not found.")
    sys.exit(1)

with open(notebook_filename, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
try:
    # Set the working directory to notebooks/ so that ../config.yaml works
    ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
    print("Notebook executed successfully.")
except Exception as e:
    print("Execution failed. Traceback:")
    traceback.print_exc()

with open(notebook_filename, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)
