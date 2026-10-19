"""Run one notebook for one seed with isolated artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 456)


def prepare_notebook(notebook_path: Path, seed: int, run_dir: Path) -> Path:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    dataset_path = str((ROOT / "data" / "QuRE.csv").resolve())

    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = source.replace("RANDOM_SEED = 42", f"RANDOM_SEED = {seed}")
        source = source.replace("'data/QuRE.csv'", repr(dataset_path))
        source = source.replace('"data/QuRE.csv"', repr(dataset_path))
        source = source.replace("'../data/QuRE.csv'", repr(dataset_path))
        cell["source"] = source.splitlines(keepends=True)

    patched_path = run_dir / f"{notebook_path.stem}_seed_{seed}_input.ipynb"
    patched_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return patched_path


def run_notebook(notebook: Path, seed: int) -> int:
    run_dir = ROOT / "notebook_runs" / f"seed_{seed}" / notebook.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    patched = prepare_notebook(notebook, seed, run_dir)
    executed_name = f"{notebook.stem}_seed_{seed}_executed.ipynb"
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=-1",
        "--output",
        executed_name,
        patched.name,
    ]

    process = subprocess.run(command, cwd=run_dir, check=False)
    status = "COMPLETED" if process.returncode == 0 else "FAILED"
    print(f"\n{status}: {notebook.name} seed={seed}")
    return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", required=True, help="Notebook path relative to repository root")
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    args = parser.parse_args()

    notebook = (ROOT / args.notebook).resolve()
    if not notebook.exists() or notebook.suffix != ".ipynb":
        parser.error(f"Notebook not found: {args.notebook}")
    return run_notebook(notebook, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
