"""
Root entrypoint for aggregating experimental results and statistical significance testing.
Delegates execution to src/aggregate_results.py.
"""
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.aggregate_results import summarize

if __name__ == "__main__":
    summarize()
