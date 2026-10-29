"""Execute the repository's Python notebook cells and save their outputs."""

import contextlib
import io
import json
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parent
NOTEBOOKS = sorted(ROOT.rglob("*.ipynb"))

def run_notebook(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace = {"__name__": "__notebook__"}
    execution_count = 0
    errors = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        execution_count += 1
        source = "".join(cell.get("source", []))
        stdout, stderr = io.StringIO(), io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile(source, f"{path.name}:cell{execution_count}", "exec"), namespace)
            cell["execution_count"] = execution_count
            outputs = []
            for name, content in (("stdout", stdout.getvalue()), ("stderr", stderr.getvalue())):
                if content:
                    outputs.append({"output_type": "stream", "name": name, "text": content.splitlines(keepends=True)})
            cell["outputs"] = outputs
        except Exception as error:  # Preserve a complete notebook traceback for inspection.
            errors.append(f"{type(error).__name__}: {error}")
            cell["execution_count"] = execution_count
            cell["outputs"] = [{
                "output_type": "error",
                "ename": type(error).__name__,
                "evalue": str(error),
                "traceback": traceback.format_exc().splitlines(),
            }]
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return errors

def main():
    failures = []
    for path in NOTEBOOKS:
        errors = run_notebook(path)
        if errors:
            failures.append((path.relative_to(ROOT), errors))
            print(f"FAIL {path.relative_to(ROOT)}: {'; '.join(errors)}")
        else:
            print(f"OK   {path.relative_to(ROOT)}")
    print(f"\nExecuted {len(NOTEBOOKS)} notebooks; {len(failures)} notebook(s) had errors.")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
