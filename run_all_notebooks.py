"""Execute and evaluate all 30 notebooks in a repository-local Jupyter kernel."""

import csv
import os
from pathlib import Path
import re
import traceback

import nbformat
from ipykernel.kernelspec import install as install_kernel
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
FOLDERS = (
    ROOT / "Basic and Advanced Text Summarisation Models",
    ROOT / "Pegasus Checkpoint Models",
)
EXPECTED_COUNTS = (18, 12)
KERNEL_NAME = "text-summarization-models"
RESULTS_PATH = ROOT / "evaluation_results.csv"
NOTEBOOKS = [path for folder in FOLDERS for path in sorted(folder.glob("*.ipynb"))]

# Keep Jupyter, NLTK, and Hugging Face state inside the repository workspace.
os.environ["IPYTHONDIR"] = str(ROOT / ".ipython")
os.environ["JUPYTER_CONFIG_DIR"] = str(ROOT / ".jupyter_config")
os.environ["JUPYTER_DATA_DIR"] = str(ROOT / ".jupyter_data")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT / ".jupyter_runtime")
os.environ.setdefault("HF_HOME", str(ROOT / ".model_cache"))
os.environ["NLTK_DATA"] = str(ROOT / ".nltk_data")
local_jupyter_path = str(ROOT / ".venv" / "share" / "jupyter")
existing_jupyter_path = os.environ.get("JUPYTER_PATH", "")
os.environ["JUPYTER_PATH"] = os.pathsep.join(
    [part for part in (local_jupyter_path, existing_jupyter_path) if part]
)


def ensure_kernel():
    """Register the current Python as the project kernel when it is not present."""
    if KERNEL_NAME in KernelSpecManager().find_kernel_specs():
        return
    install_kernel(
        kernel_name=KERNEL_NAME,
        display_name="Python (Text Summarization Models)",
        prefix=str(ROOT / ".venv"),
    )


def _output_text(notebook):
    chunks = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                value = output.get("text", "")
                chunks.append(value if isinstance(value, str) else "".join(value))
            elif output.output_type in {"display_data", "execute_result"}:
                value = output.get("data", {}).get("text/plain", "")
                chunks.append(value if isinstance(value, str) else "".join(value))
    return "\n".join(chunks)


def _metrics(notebook, text):
    """Read the metric values printed by the notebook's existing evaluator."""
    shared = re.findall(r"Metrics:\s*([^\r\n]+)", text)
    if shared:
        values = dict(re.findall(r"([a-zA-Z0-9_]+)=([0-9]+(?:\.[0-9]+)?)", shared[-1]))
        return {
            "rouge1_f1": values.get("rouge1_f1", ""),
            "rouge2_f1": values.get("rouge2_f1", ""),
            "rougeL_f1": values.get("rougeL_f1", ""),
            "bleu4": values.get("bleu4", ""),
            "legacy_rouge1_f1": "",
            "notebook_bleu": "",
            "metric_method": "shared ROUGE-1/2/L F1 and smoothed BLEU-4",
        }

    f1 = re.findall(r"F1-Score:\s*([0-9]+(?:\.[0-9]+)?)", text)
    bleu = re.findall(r"BLEU Score:\s*([0-9]+(?:\.[0-9]+)?)", text)
    if f1 and bleu:
        return {
            "rouge1_f1": "",
            "rouge2_f1": "",
            "rougeL_f1": "",
            "bleu4": "",
            "legacy_rouge1_f1": f1[-1],
            "notebook_bleu": bleu[-1],
            "metric_method": "notebook ROUGE-1 F1 and unsmoothed NLTK sentence BLEU",
        }
    return {
        "rouge1_f1": "",
        "rouge2_f1": "",
        "rougeL_f1": "",
        "bleu4": "",
        "legacy_rouge1_f1": "",
        "notebook_bleu": "",
        "metric_method": "missing metric output",
    }


def run_notebook(path):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    runner_error = ""
    try:
        client = NotebookClient(
            notebook,
            timeout=1800,
            startup_timeout=180,
            kernel_name=KERNEL_NAME,
            allow_errors=True,
            resources={"metadata": {"path": str(ROOT)}},
        )
        client.execute()
    except Exception as error:
        runner_error = f"Notebook runner failed: {type(error).__name__}: {error}"
        traceback.print_exc()
    finally:
        # Preserve all outputs, including partial output from a runner-level failure.
        nbformat.write(notebook, path)

    text = _output_text(notebook)
    errors = []
    for index, cell in enumerate(notebook.cells, 1):
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "error":
                errors.append(f"cell {index}: {output.ename}: {output.evalue}")
    metrics = _metrics(notebook, text)
    fallback = bool(re.search(r"inference unavailable|using a TextRank fallback", text, re.IGNORECASE))
    if runner_error or errors:
        status = "error"
    elif fallback:
        status = "fallback"
    elif metrics["metric_method"] == "missing metric output":
        status = "missing_metrics"
    else:
        status = "passed"
    return {
        "notebook": str(path.relative_to(ROOT)).replace("\\", "/"),
        "family": "Pegasus" if path.parent.name == FOLDERS[1].name else "Basic/advanced",
        "status": status,
        **metrics,
        "details": "; ".join(([runner_error] if runner_error else []) + errors),
    }


def write_results(rows):
    columns = [
        "notebook", "family", "status", "rouge1_f1", "rouge2_f1", "rougeL_f1",
        "bleu4", "legacy_rouge1_f1", "notebook_bleu", "metric_method", "details",
    ]
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    actual_counts = tuple(len(list(folder.glob("*.ipynb"))) for folder in FOLDERS)
    if actual_counts != EXPECTED_COUNTS:
        print(f"Expected 18 basic/advanced + 12 Pegasus notebooks; found {actual_counts}.", flush=True)
        return 1
    ensure_kernel()
    rows = []
    for index, path in enumerate(NOTEBOOKS, 1):
        print(f"[{index}/{len(NOTEBOOKS)}] Running {path.relative_to(ROOT)}", flush=True)
        row = run_notebook(path)
        rows.append(row)
        if row["status"] == "passed":
            print("  PASS (metrics captured)", flush=True)
        else:
            details = row["details"] or row["metric_method"]
            print(f"  {row['status'].upper()}: {details}", flush=True)
    write_results(rows)
    counts = {status: sum(row["status"] == status for row in rows) for status in ("passed", "fallback", "error", "missing_metrics")}
    print(f"\nExecuted {len(NOTEBOOKS)} notebooks; results saved to {RESULTS_PATH.name}.", flush=True)
    print("Statuses: " + ", ".join(f"{key}={value}" for key, value in counts.items()), flush=True)
    return 0 if counts["fallback"] == counts["error"] == counts["missing_metrics"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
