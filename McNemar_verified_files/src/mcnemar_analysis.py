#!/usr/bin/env python3
"""Run exact McNemar tests comparing LLM and VLM fine-tuned predictions per seed.

The script requires paired predictions for the same test IDs and ground-truth labels.
It reports the exact two-sided binomial McNemar p-value and Holm-adjusted p-values
across the three seed-specific tests. It does not pool repeated test samples across seeds.
"""
from pathlib import Path
import argparse
import json

import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests


REQUIRED_COLUMNS = {"id", "defect", "predicted"}


def load_and_validate(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing prediction file: {path}")
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError(f"{path}: IDs must be present and unique.")
    if df[["defect", "predicted"]].isna().any().any():
        raise ValueError(f"{path}: ground-truth or predicted labels contain missing values.")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results",
                        help="Path to the repository results directory (default: results)")
    parser.add_argument("--output", default="results/mcnemar_results.json",
                        help="Output JSON path (default: results/mcnemar_results.json)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    predictions_dir = results_dir / "predictions"
    rows = []

    for seed in (42, 123, 456):
        llm_path = predictions_dir / f"predictions_llm_finetune_seed_{seed}.csv"
        vlm_path = predictions_dir / f"predictions_vlm_finetune_seed_{seed}.csv"
        llm = load_and_validate(llm_path)
        vlm = load_and_validate(vlm_path)

        # Align by ID rather than assuming row order.
        if set(llm["id"]) != set(vlm["id"]):
            raise ValueError(f"Seed {seed}: LLM and VLM test IDs do not match.")
        vlm = vlm.set_index("id").loc[llm["id"]].reset_index()
        if not llm["defect"].astype(str).equals(vlm["defect"].astype(str)):
            raise ValueError(f"Seed {seed}: ground-truth labels differ for paired IDs.")

        truth = llm["defect"].astype(str)
        llm_correct = llm["predicted"].astype(str).eq(truth)
        vlm_correct = vlm["predicted"].astype(str).eq(truth)

        # b: VLM correct and LLM wrong; c: LLM correct and VLM wrong.
        b = int(((~llm_correct) & vlm_correct).sum())
        c = int((llm_correct & (~vlm_correct)).sum())
        discordant = b + c
        p_value = (
            float(binomtest(min(b, c), n=discordant, p=0.5,
                            alternative="two-sided").pvalue)
            if discordant else 1.0
        )
        rows.append({
            "comparison": "VLM Fine-Tuned (QLoRA) vs LLM Fine-Tuned (QLoRA)",
            "seed": seed,
            "n_paired_test_samples": int(len(llm)),
            "llm_accuracy": float(llm_correct.mean()),
            "vlm_accuracy": float(vlm_correct.mean()),
            "vlm_correct_llm_wrong_b": b,
            "llm_correct_vlm_wrong_c": c,
            "discordant_pairs": discordant,
            "mcnemar_exact_two_sided_p": p_value,
        })

    adjusted = multipletests(
        [row["mcnemar_exact_two_sided_p"] for row in rows],
        method="holm"
    )[1]
    for row, p_adjusted in zip(rows, adjusted):
        row["holm_adjusted_p_across_three_seeds"] = float(p_adjusted)
        row["significant_at_0_05_after_holm"] = bool(p_adjusted < 0.05)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "test": "Exact two-sided McNemar test (binomial formulation)",
        "multiple_testing_correction": "Holm correction across three seed-specific tests",
        "comparison": "VLM Fine-Tuned (QLoRA) vs LLM Fine-Tuned (QLoRA)",
        "results": rows,
        "interpretation": (
            "These tests compare paired correctness on each seed's test split. "
            "They do not establish that multimodal pretraining caused any difference."
        )
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output_path}")
    for row in rows:
        print(
            f"seed={row['seed']}: b={row['vlm_correct_llm_wrong_b']}, "
            f"c={row['llm_correct_vlm_wrong_c']}, "
            f"p={row['mcnemar_exact_two_sided_p']:.6f}, "
            f"Holm p={row['holm_adjusted_p_across_three_seeds']:.6f}"
        )


if __name__ == "__main__":
    main()
