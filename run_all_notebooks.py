"""Run every notebook in a local Jupyter kernel and save cell outputs."""

import os
from pathlib import Path
import traceback

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
FOLDERS = (
    ROOT / "Basic to Advance Text Summarisation Models",
    ROOT / "Models Used in Proposed Architecture",
)
NOTEBOOKS = [path for folder in FOLDERS for path in sorted(folder.glob("*.ipynb"))]

# Keep Jupyter, NLTK, and Hugging Face state inside the repository workspace.
os.environ.setdefault("IPYTHONDIR", str(ROOT / ".ipython"))
os.environ.setdefault("JUPYTER_CONFIG_DIR", str(ROOT / ".jupyter_config"))
os.environ.setdefault("JUPYTER_DATA_DIR", str(ROOT / ".jupyter_data"))
os.environ.setdefault("JUPYTER_PATH", str(ROOT / ".venv" / "share" / "jupyter"))
os.environ.setdefault("JUPYTER_RUNTIME_DIR", str(ROOT / ".jupyter_runtime"))
os.environ.setdefault("HF_HOME", str(ROOT / ".model_cache"))
os.environ.setdefault("NLTK_DATA", str(ROOT / ".nltk_data"))

def run_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    client = NotebookClient(
        notebook,
        timeout=1800,
        startup_timeout=180,
        kernel_name="text-summarization-models",
        allow_errors=True,
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        client.execute()
    except Exception:
        # Keep all emitted output and surface runner-level failures in the terminal.
        traceback.print_exc()
        return notebook, ["Notebook runner failed before completion"]
    errors = []
    for index, cell in enumerate(notebook.cells, 1):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                errors.append(f"cell {index}: {output.ename}: {output.evalue}")
    nbformat.write(notebook, path)
    return notebook, errors

def main():
    failures = []
    for index, path in enumerate(NOTEBOOKS, 1):
        print(f"[{index}/{len(NOTEBOOKS)}] Running {path.relative_to(ROOT)}", flush=True)
        _, errors = run_notebook(path)
        if errors:
            failures.append((path.relative_to(ROOT), errors))
            print(f"  ERRORS: {'; '.join(errors)}", flush=True)
        else:
            print("  PASS", flush=True)
    print(f"\nExecuted {len(NOTEBOOKS)} notebooks; {len(failures)} notebook(s) contain cell errors.", flush=True)
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
