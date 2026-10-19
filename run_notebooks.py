"""Run all notebooks for seeds 42, 123, and 456 on local Python or Google Colab."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

NOTEBOOKS = (
    "notebooks/SRS_LLM_Zero_Shot.ipynb",
    "notebooks/SRS_VLM_Zero_Shot.ipynb",
    "notebooks/SRS_LLM_Finetune.ipynb",
    "notebooks/SRS_VLM_Finetune.ipynb",
)
SEEDS = (42, 123, 456)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", choices=NOTEBOOKS, help="Run only this notebook")
    parser.add_argument("--seed", choices=[str(seed) for seed in SEEDS], help="Run only this seed")
    parser.add_argument("--stop-on-error", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    notebooks = (args.notebook,) if args.notebook else NOTEBOOKS
    seeds = (int(args.seed),) if args.seed else SEEDS
    failures = 0

    for notebook in notebooks:
        for seed in seeds:
            command = [
                sys.executable,
                "src/notebook_runner.py",
                "--notebook",
                notebook,
                "--seed",
                str(seed),
            ]
            print(f"\nRunning {notebook} with seed {seed}", flush=True)
            completed = subprocess.run(command, cwd=root, check=False)
            if completed.returncode != 0:
                failures += 1
                if args.stop_on_error:
                    return completed.returncode

    print(f"\nCompleted with {failures} failure(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
