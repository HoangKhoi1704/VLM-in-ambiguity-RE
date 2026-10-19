"""Aggregate cell-level notebook logs into one row per notebook run."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

FIELDS = [
    "run_id",
    "notebook",
    "seed",
    "started_at",
    "finished_at",
    "status",
    "cells_finished",
    "failed_cells",
    "first_error",
]


def aggregate(input_path: Path, output_path: Path) -> int:
    if not input_path.exists():
        raise FileNotFoundError(f"Log file not found: {input_path}")

    groups = defaultdict(list)
    with input_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[row["run_id"]].append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for rows in groups.values():
            started = rows[0]
            cell_rows = [row for row in rows if row["event"] == "CELL_FINISHED"]
            failed = [row for row in cell_rows if row["status"] == "FAILED"]
            writer.writerow({
                "run_id": started["run_id"],
                "notebook": started["notebook"],
                "seed": started["seed"],
                "started_at": started["timestamp"],
                "finished_at": rows[-1]["timestamp"],
                "status": "FAILED" if failed else "COMPLETED",
                "cells_finished": str(len(cell_rows)),
                "failed_cells": str(len(failed)),
                "first_error": failed[0]["error_message"] if failed else "",
            })
    return len(groups)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/notebook_cell_logs.csv")
    parser.add_argument("--output", default="results/notebook_run_summary.csv")
    args = parser.parse_args()
    count = aggregate(Path(args.input), Path(args.output))
    print(f"Aggregated {count} notebook run(s) into {args.output}")


if __name__ == "__main__":
    main()
