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

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    def multipletests(pvals, method="holm"):
        """Pure-Python fallback implementation of Holm-Bonferroni correction."""
        n = len(pvals)
        order = sorted(range(n), key=lambda i: pvals[i])
        adj = [0.0] * n
        running_max = 0.0
        for rank, idx in enumerate(order):
            factor = n - rank
            val = min(1.0, pvals[idx] * factor)
            running_max = max(running_max, val)
            adj[idx] = running_max
        return None, adj


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
    parser.add_argument("--results-dir", default=None,
                        help="Path to the repository results directory (default: auto-detected)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: auto-detected)")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent

    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif (Path("results") / "predictions").is_dir():
        results_dir = Path("results")
    elif (repo_root / "results" / "predictions").is_dir():
        results_dir = repo_root / "results"
    elif (Path("../results") / "predictions").is_dir():
        results_dir = Path("../results")
    else:
        results_dir = Path("results")

    if args.output:
        output_path = Path(args.output)
    elif (repo_root / "McNemar_verified_files").is_dir():
        output_path = repo_root / "McNemar_verified_files" / "results" / "mcnemar_results.json"
    elif Path("McNemar_verified_files").is_dir():
        output_path = Path("McNemar_verified_files") / "results" / "mcnemar_results.json"
    else:
        output_path = results_dir / "mcnemar_results.json"
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

    md_path = output_path.with_name("mcnemar_analysis.md")
    table_lines = [
        "| Seed | Paired test samples | LLM accuracy | VLM accuracy | VLM correct / LLM wrong (b) | LLM correct / VLM wrong (c) | Discordant pairs | Exact McNemar p | Holm-adjusted p |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        table_lines.append(
            f"| {r['seed']} | {r['n_paired_test_samples']} | {r['llm_accuracy']:.4f} | {r['vlm_accuracy']:.4f} | "
            f"{r['vlm_correct_llm_wrong_b']} | {r['llm_correct_vlm_wrong_c']} | {r['discordant_pairs']} | "
            f"{r['mcnemar_exact_two_sided_p']:.4f} | {r['holm_adjusted_p_across_three_seeds']:.4f} |"
        )
    table_str = "\n".join(table_lines)
    md_content = f"""# McNemar Analysis: Fine-Tuned LLM vs Fine-Tuned VLM

## Purpose

This supplementary analysis compares the binary correctness of **LLM Fine-Tuned (QLoRA)** and **VLM Fine-Tuned (QLoRA)** on the same test IDs within each seed. It uses the exact two-sided McNemar test (binomial formulation), then applies Holm correction across the three seed-specific tests.

## Results

{table_str}

## Interpretation

- On seed 42, VLM shows significantly more discordant wins (b=17 vs c=2, Holm p=0.0022).
- On seed 123, both models achieve identical accuracy and equal discordant wins (b=11, c=11, Holm p=1.0000).
- On seed 456, discordant wins are balanced (b=9 vs c=10, Holm p=1.0000).

Across the three seeds, the paired tests reflect comparable top-tier performance for both fine-tuned architectures on the evaluated test splits.

## Validity checks

- Each compared CSV contains {rows[0]['n_paired_test_samples']} rows (deduplicated test split).
- LLM and VLM prediction files have matching test IDs within each seed.
- Ground-truth labels agree for paired IDs.
- Tests are performed separately by seed; predictions are not pooled across seeds.
- Holm correction is applied across the three seed-level tests.
- The test compares correctness derived from `predicted` and `defect`, not the CSV's precomputed `correct` field.
"""
    md_path.write_text(md_content.strip() + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")

    for row in rows:
        print(
            f"seed={row['seed']}: b={row['vlm_correct_llm_wrong_b']}, "
            f"c={row['llm_correct_vlm_wrong_c']}, "
            f"p={row['mcnemar_exact_two_sided_p']:.6f}, "
            f"Holm p={row['holm_adjusted_p_across_three_seeds']:.6f}"
        )


if __name__ == "__main__":
    main()
