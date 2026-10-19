"""Run one notebook for one seed with isolated artifacts and centralized logging."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "results" / "notebook_run_logs.csv"
SEEDS = (42, 123, 456)
LOG_FIELDS = [
    "run_id",
    "started_at",
    "finished_at",
    "notebook",
    "seed",
    "status",
    "run_dir",
    "executed_notebook",
    "log_file",
    "duration_seconds",
    "return_code",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def append_log(row: dict[str, str]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    exists = LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_notebook(notebook: Path, seed: int) -> int:
    started = time.monotonic()
    started_at = utc_now()
    run_dir = ROOT / "notebook_runs" / f"seed_{seed}" / notebook.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    patched = prepare_notebook(notebook, seed, run_dir)
    executed_name = f"{notebook.stem}_seed_{seed}_executed.ipynb"
    executed_path = run_dir / executed_name
    log_file = run_dir / "run.log"
    run_id = str(uuid.uuid4())
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

    status = "COMPLETED"
    return_code = 0
    try:
        completed = subprocess.run(
            command,
            cwd=run_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        return_code = completed.returncode
        log_file.write_text(
            f"$ {' '.join(command)}\n\n{completed.stdout}\n{completed.stderr}",
            encoding="utf-8",
        )
        if return_code != 0:
            status = "FAILED"
    except Exception as exc:
        status = "FAILED"
        return_code = -1
        log_file.write_text(f"Runner error: {exc}\n", encoding="utf-8")

    append_log({
        "run_id": run_id,
        "started_at": started_at,
        "finished_at": utc_now(),
        "notebook": str(notebook.relative_to(ROOT)).replace("\\", "/"),
        "seed": str(seed),
        "status": status,
        "run_dir": str(run_dir.relative_to(ROOT)).replace("\\", "/"),
        "executed_notebook": str(executed_path.relative_to(ROOT)).replace("\\", "/"),
        "log_file": str(log_file.relative_to(ROOT)).replace("\\", "/"),
        "duration_seconds": f"{time.monotonic() - started:.1f}",
        "return_code": str(return_code),
    })
    print(f"{status}: {notebook.name} seed={seed}; log={log_file}")
    return return_code


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
