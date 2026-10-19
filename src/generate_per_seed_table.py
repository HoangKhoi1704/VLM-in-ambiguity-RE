"""
=============================================================================
GENERATE PER-SEED DETAILED EVALUATION TABLE FOR LLM AND VLM MODELS
=============================================================================
Generates comprehensive evaluation tables displaying individual metrics for each
random seed (42, 123, 456) along with the aggregate Mean ± Std for the 4 core models:
  1. LLM Zero-Shot (Qwen2.5-3B-Instruct)
  2. VLM Zero-Shot (Qwen2.5-VL-3B-Instruct)
  3. LLM Fine-Tuned (QLoRA 4-bit)
  4. VLM Fine-Tuned (QLoRA 4-bit)

Outputs saved to:
  - results/per_seed_results_table.md   (Markdown format for reports/documentation)
  - results/per_seed_results_table.tex  (LaTeX multirow table for academic papers)
  - results/per_seed_results.csv        (CSV format for Excel / Google Sheets)
  - results/per_seed_results.json       (Machine-readable structured JSON)
=============================================================================
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

CORE_MODELS = [
    ("llm_zero_shot", "LLM Zero-Shot (Qwen2.5-3B)"),
    ("vlm_zero_shot", "VLM Zero-Shot (Qwen2.5-VL-3B)"),
    ("llm_finetune", "LLM Fine-Tuned (QLoRA)"),
    ("vlm_finetune", "VLM Fine-Tuned (QLoRA)"),
]

BASELINE_MODELS = [
    ("majority_classifier", "Majority Classifier"),
    ("tfidf_logistic_regression", "TF-IDF + Logistic Regression"),
    ("tfidf_linear_svm", "TF-IDF + Linear SVM"),
]

METRICS = [
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Bal. Acc"),
    ("f1_macro", "F1-Macro"),
    ("f1_defect", "F1-Defect"),
    ("f1_ok", "F1-OK"),
    ("mcc", "MCC"),
    ("roc_auc", "ROC-AUC"),
    ("pr_auc", "PR-AUC"),
]

SEEDS = [42, 123, 456]


def find_metrics_files(metrics_dir: str | Path) -> list[str]:
    candidate_patterns = [
        os.path.join(str(metrics_dir), "metrics_*_seed_*.json"),
        os.path.join(str(metrics_dir), "metrics_*.json"),
        "results/metrics/metrics_*_seed_*.json",
        "../results/metrics/metrics_*_seed_*.json",
    ]
    for pattern in candidate_patterns:
        files = glob.glob(pattern)
        if files:
            return sorted(files)
    return []


def load_metrics_data(metrics_dir: str | Path) -> dict[str, dict[int, dict[str, Any]]]:
    files = find_metrics_files(metrics_dir)
    data: dict[str, dict[int, dict[str, Any]]] = {}

    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                d = json.load(fp)
            model = d.get("model", "unknown")
            seed = int(d.get("seed", 0))
            if model not in data:
                data[model] = {}
            data[model][seed] = d
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")

    # Fallback to check notebook_runs if any files were not found
    for seed in SEEDS:
        for m_key, _ in CORE_MODELS:
            if m_key not in data or seed not in data[m_key]:
                run_pattern = f"notebook_runs/seed_{seed}/**/metrics_{m_key}_seed_{seed}.json"
                run_files = glob.glob(run_pattern, recursive=True)
                if run_files:
                    try:
                        with open(run_files[0], "r", encoding="utf-8") as fp:
                            d = json.load(fp)
                        if m_key not in data:
                            data[m_key] = {}
                        data[m_key][seed] = d
                    except Exception:
                        pass
    return data


def format_value(val: float | None, precision: int = 4) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{precision}f}"


def build_per_seed_table(
    data: dict[str, dict[int, dict[str, Any]]],
    include_baselines: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_models = list(CORE_MODELS)
    if include_baselines:
        target_models = list(BASELINE_MODELS) + target_models

    table_rows: list[dict[str, Any]] = []
    json_export: dict[str, Any] = {}

    for model_key, model_display in target_models:
        seed_dict = data.get(model_key, {})
        json_export[model_key] = {
            "display_name": model_display,
            "seeds": {},
            "mean": {},
            "std": {},
        }

        # 1. Individual seed rows
        metric_series: dict[str, list[float]] = {m_key: [] for m_key, _ in METRICS}

        for seed in SEEDS:
            row: dict[str, Any] = {
                "model_key": model_key,
                "model_name": model_display,
                "seed": str(seed),
            }
            seed_data = seed_dict.get(seed, {})
            json_export[model_key]["seeds"][str(seed)] = {}

            for m_key, _ in METRICS:
                val = seed_data.get(m_key)
                if val is not None and not np.isnan(float(val)):
                    float_val = float(val)
                    row[m_key] = float_val
                    metric_series[m_key].append(float_val)
                    json_export[model_key]["seeds"][str(seed)][m_key] = float_val
                else:
                    row[m_key] = None
                    json_export[model_key]["seeds"][str(seed)][m_key] = None

            table_rows.append(row)

        # 2. Mean ± Std summary row
        mean_row: dict[str, Any] = {
            "model_key": model_key,
            "model_name": model_display,
            "seed": "Mean ± Std",
        }
        for m_key, _ in METRICS:
            vals = metric_series[m_key]
            if vals:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                mean_row[m_key] = f"{mean_val:.4f} ± {std_val:.4f}"
                json_export[model_key]["mean"][m_key] = mean_val
                json_export[model_key]["std"][m_key] = std_val
            else:
                mean_row[m_key] = "N/A"
                json_export[model_key]["mean"][m_key] = None
                json_export[model_key]["std"][m_key] = None

        table_rows.append(mean_row)

    return table_rows, json_export


def generate_markdown(table_rows: list[dict[str, Any]]) -> str:
    headers = ["Model", "Seed"] + [m_name for _, m_name in METRICS]
    alignments = [":---", ":---:"] + [":---:" for _ in METRICS]
    
    lines = [
        "# Detailed Performance Breakdown per Seed",
        "",
        "This table reports individual test performance across 3 random seeds (42, 123, 456) and overall Mean ± Std.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(alignments) + " |",
    ]

    current_model = None
    for row in table_rows:
        model_name = row["model_name"]
        seed_label = row["seed"]
        is_summary = seed_label == "Mean ± Std"

        # Display model name only on first seed row for visual hierarchy
        disp_model = f"**{model_name}**" if model_name != current_model else ""
        current_model = model_name

        disp_seed = f"**{seed_label}**" if is_summary else seed_label

        cols = [disp_model, disp_seed]
        for m_key, _ in METRICS:
            v = row.get(m_key)
            if isinstance(v, float):
                cols.append(f"{v:.4f}")
            elif isinstance(v, str):
                cols.append(f"**{v}**" if is_summary else v)
            else:
                cols.append("N/A")

        lines.append("| " + " | ".join(cols) + " |")

    return "\n".join(lines) + "\n"


def generate_latex(table_rows: list[dict[str, Any]]) -> str:
    lines = [
        "% LaTeX Table: Detailed Performance Breakdown per Seed",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\caption{Detailed ambiguity detection performance per random seed and overall Mean $\\pm$ Std.}",
        "\\label{tab:per_seed_results}",
        "\\begin{tabular}{llcccccccc}",
        "\\hline",
        "\\textbf{Model} & \\textbf{Seed} & \\textbf{Accuracy} & \\textbf{Bal. Acc} & \\textbf{F1-Macro} & \\textbf{F1-Defect} & \\textbf{F1-OK} & \\textbf{MCC} & \\textbf{ROC-AUC} & \\textbf{PR-AUC} \\\\",
        "\\hline",
    ]

    # Group by model
    models_seen: dict[str, list[dict[str, Any]]] = {}
    for r in table_rows:
        models_seen.setdefault(r["model_name"], []).append(r)

    for model_name, rows in models_seen.items():
        clean_model = model_name.replace("_", "\\_").replace("&", "\\&")
        for idx, r in enumerate(rows):
            is_summary = r["seed"] == "Mean ± Std"
            model_cell = f"\\multirow{{{len(rows)}}}{{*}}{{{clean_model}}}" if idx == 0 else ""
            seed_cell = f"\\textbf{{{r['seed']}}}" if is_summary else r["seed"]

            metric_cells = []
            for m_key, _ in METRICS:
                v = r.get(m_key)
                if isinstance(v, float):
                    metric_cells.append(f"{v:.4f}")
                elif isinstance(v, str) and "±" in v:
                    tex_v = v.replace("±", "$\\pm$")
                    metric_cells.append(f"\\textbf{{{tex_v}}}")
                else:
                    metric_cells.append(str(v))

            line = f"{model_cell} & {seed_cell} & " + " & ".join(metric_cells) + " \\\\"
            lines.append(line)
        lines.append("\\hline")

    lines.extend([
        "\\end{tabular}",
        "\\end{table*}",
        "",
    ])
    return "\n".join(lines)


def generate_csv(table_rows: list[dict[str, Any]]) -> str:
    flat_rows = []
    for r in table_rows:
        item = {
            "model": r["model_key"],
            "model_name": r["model_name"],
            "seed": r["seed"],
        }
        for m_key, _ in METRICS:
            v = r.get(m_key)
            item[m_key] = v
        flat_rows.append(item)
    df = pd.DataFrame(flat_rows)
    return df.to_csv(index=False, encoding="utf-8")


def print_terminal_table(table_rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 115)
    print("              DETAILED PERFORMANCE BREAKDOWN PER SEED (4 CORE MODELS)")
    print("=" * 115)
    header = f"{'Model':<28} | {'Seed':<12} | {'Accuracy':<10} | {'Bal. Acc':<10} | {'F1-Macro':<10} | {'MCC':<8} | {'ROC-AUC':<9} | {'PR-AUC':<9}"
    print(header)
    print("-" * 115)

    last_model = None
    for r in table_rows:
        is_summary = r["seed"] == "Mean ± Std"
        m_disp = r["model_name"] if r["model_name"] != last_model else ""
        last_model = r["model_name"]

        def f_str(key: str) -> str:
            v = r.get(key)
            if isinstance(v, float):
                return f"{v:.4f}"
            if isinstance(v, str):
                return v
            return "N/A"

        if is_summary:
            row_str = f"{'':<28} | {r['seed']:<12} | {f_str('accuracy'):<10} | {f_str('balanced_accuracy'):<10} | {f_str('f1_macro'):<10} | {f_str('mcc'):<8} | {f_str('roc_auc'):<9} | {f_str('pr_auc'):<9}"
            print(row_str)
            print("-" * 115)
        else:
            row_str = f"{m_disp:<28} | {r['seed']:<12} | {f_str('accuracy'):<10} | {f_str('balanced_accuracy'):<10} | {f_str('f1_macro'):<10} | {f_str('mcc'):<8} | {f_str('roc_auc'):<9} | {f_str('pr_auc'):<9}"
            print(row_str)


def run_generate(
    metrics_dir: str = "results/metrics",
    output_dir: str = "results",
    include_baselines: bool = False,
) -> None:
    data = load_metrics_data(metrics_dir)
    if not data:
        print(f"Error: No metric JSON files found in {metrics_dir}!")
        return

    table_rows, json_export = build_per_seed_table(data, include_baselines=include_baselines)

    # Print to console
    print_terminal_table(table_rows)

    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, "per_seed_results_table.md")
    tex_path = os.path.join(output_dir, "per_seed_results_table.tex")
    csv_path = os.path.join(output_dir, "per_seed_results.csv")
    json_path = os.path.join(output_dir, "per_seed_results.json")

    # Save Markdown
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(generate_markdown(table_rows))

    # Save LaTeX
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(generate_latex(table_rows))

    # Save CSV
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(generate_csv(table_rows))

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_export, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 115)
    print("SUCCESSFULLY GENERATED PER-SEED TABLES IN 4 FORMATS:")
    print(f"   1. Markdown : {md_path}")
    print(f"   2. LaTeX    : {tex_path}")
    print(f"   3. CSV      : {csv_path}")
    print(f"   4. JSON     : {json_path}")
    print("=" * 115)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", default="results/metrics", help="Directory containing metrics JSONs")
    parser.add_argument("--output-dir", default="results", help="Directory to save generated tables")
    parser.add_argument("--all-models", action="store_true", help="Include classical baselines along with 4 neural models")
    args = parser.parse_args()

    run_generate(
        metrics_dir=args.metrics_dir,
        output_dir=args.output_dir,
        include_baselines=args.all_models,
    )


if __name__ == "__main__":
    main()
