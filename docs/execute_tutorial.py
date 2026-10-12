"""Execute docs/tutorial.ipynb in place so its outputs stay current.

Usage: python docs/execute_tutorial.py [--check]
With --check the notebook is executed but not written (used in CI).
"""

import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

path = Path(__file__).with_name("tutorial.ipynb")
notebook = nbformat.read(path, as_version=4)
start = time.time()
NotebookClient(notebook, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(path.parent)}}).execute()
print(f"executed {path.name} in {time.time() - start:.0f}s")
if "--check" not in sys.argv:
    for cell in notebook.cells:
        cell.get("metadata", {}).pop("execution", None)
    nbformat.write(notebook, path)
