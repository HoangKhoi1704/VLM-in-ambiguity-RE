"""
Root entrypoint for running baseline experiments.
Delegates execution to src/run_baselines.py.
"""
import os
import sys
import argparse

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.run_baselines import run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/QuRE.csv", help="Path to dataset file (QuRE.csv / datasetRE.csv)")
    args = parser.parse_args()
    run_experiment(dataset_path=args.dataset)
