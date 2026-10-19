"""Cell-level logging for notebooks executed interactively or by nbconvert."""

from __future__ import annotations

import csv
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "run_id",
    "timestamp",
    "notebook",
    "seed",
    "event",
    "execution_count",
    "status",
    "error_type",
    "error_message",
]


def _root_dir() -> Path:
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "src").is_dir() and (candidate / "results").is_dir():
            return candidate
    return Path.cwd()


def _append(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def install_notebook_logging(notebook_name: str) -> str:
    """Register a post-cell hook and return the run ID for this kernel session."""
    shell = get_ipython()  # type: ignore[name-defined]
    previous_callback = shell.user_ns.get("_NOTEBOOK_LOG_CALLBACK")
    if previous_callback is not None:
        shell.events.unregister("post_run_cell", previous_callback)
    run_id = str(uuid.uuid4())
    log_path = _root_dir() / "results" / "notebook_cell_logs.csv"
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    _append(log_path, {
        "run_id": run_id,
        "timestamp": started,
        "notebook": notebook_name,
        "seed": str(shell.user_ns.get("RANDOM_SEED", os.getenv("RANDOM_SEED", "unknown"))),
        "event": "RUN_STARTED",
        "execution_count": "",
        "status": "STARTED",
        "error_type": "",
        "error_message": "",
    })

    def record_cell(result) -> None:
        error = getattr(result, "error_in_exec", None)
        _append(log_path, {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "notebook": notebook_name,
            "seed": str(shell.user_ns.get("RANDOM_SEED", os.getenv("RANDOM_SEED", "unknown"))),
            "event": "CELL_FINISHED",
            "execution_count": str(getattr(result, "execution_count", "")),
            "status": "FAILED" if error else "OK",
            "error_type": type(error).__name__ if error else "",
            "error_message": str(error) if error else "",
        })

    shell.events.register("post_run_cell", record_cell)
    shell.user_ns["_NOTEBOOK_LOG_CALLBACK"] = record_cell
    print(f"Notebook logging enabled: {log_path} (run_id={run_id})")
    return run_id
