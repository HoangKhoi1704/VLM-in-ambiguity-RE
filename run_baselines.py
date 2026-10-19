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
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456], help="Random seeds to run")
    parser.add_argument("--output-dir", default="results", help="Directory for metrics and predictions")
    parser.add_argument("--no-predictions", action="store_true", help="Do not write prediction CSV files")
    args = parser.parse_args()
    run_experiment(
        dataset_path=args.dataset,
        seeds=args.seeds,
        output_dir=args.output_dir,
        save_predictions=not args.no_predictions,
    )
