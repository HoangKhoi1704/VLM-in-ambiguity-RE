"""
Root entrypoint for running exact McNemar tests comparing LLM and VLM fine-tuned predictions.
Delegates execution to McNemar_verified_files/src/mcnemar_analysis.py.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MCNEMAR_SRC = os.path.join(ROOT_DIR, "McNemar_verified_files", "src")
if MCNEMAR_SRC not in sys.path:
    sys.path.insert(0, MCNEMAR_SRC)

from mcnemar_analysis import main

if __name__ == "__main__":
    main()
