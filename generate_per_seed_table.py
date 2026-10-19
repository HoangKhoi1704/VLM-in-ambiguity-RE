"""
Root entrypoint for generating per-seed evaluation tables for LLM and VLM models.
Delegates execution to src/generate_per_seed_table.py.
"""
from __future__ import annotations

import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.generate_per_seed_table import main

if __name__ == "__main__":
    main()
