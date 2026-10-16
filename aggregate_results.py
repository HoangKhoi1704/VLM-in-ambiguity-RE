"""
=============================================================================
AGGREGATE MULTI-SEED RESULTS & STATISTICAL SIGNIFICANCE TESTING
Reads all `metrics_<model>_seed_<seed>.json` files in the directory.
Calculates:
  1. Mean ± Std for all metrics across seeds
  2. Paired t-test (p-value) comparing VLM Fine-Tuned vs other models
  3. Formatted Markdown and LaTeX tables for direct inclusion in the paper.
Saves results automatically to:
  - paper_results_summary.json
  - paper_results_table.md
  - paper_results_table.tex
=============================================================================
"""

import os
import sys
import glob
import json

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
from scipy import stats

METRIC_KEYS = [
    'accuracy',
    'balanced_accuracy',
    'f1_macro',
    'f1_weighted',
    'f1_defect',
    'mcc',
    'roc_auc',
    'pr_auc'
]

def load_all_metrics():
    files = glob.glob("metrics_*_seed_*.json")
    if not files:
        print("⚠️  Chưa tìm thấy file metrics JSON nào dạng `metrics_<model>_seed_<seed>.json`.")
        print("    Sau khi chạy các notebook/baselines, các file kết quả sẽ tự động được sinh ra.")
        return {}

    data_by_model = {}
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                d = json.load(fp)
            model = d.get('model', 'unknown')
            seed = d.get('seed', 0)
            if model not in data_by_model:
                data_by_model[model] = {}
            data_by_model[model][seed] = d
        except Exception as e:
            print(f"Lỗi đọc file {f}: {e}")

    return data_by_model

def summarize():
    data = load_all_metrics()
    if not data:
        return

    print("=" * 95)
    print("                      BẢNG TỔNG HỢP KẾT QUẢ ĐA SEED (Mean ± Std)")
    print("=" * 95)

    summary_table = {}
    json_export = {}

    for model, seed_dict in data.items():
        summary_table[model] = {}
        json_export[model] = {}
        seeds = sorted(seed_dict.keys())
        print(f"\n📌 Model: {model} (Đã chạy {len(seeds)} seeds: {seeds})")
        for m in METRIC_KEYS:
            vals = [seed_dict[s][m] for s in seeds if seed_dict[s].get(m) is not None]
            if vals:
                mean_val = float(np.mean(vals))
                std_val = float(np.std(vals))
                summary_table[model][m] = (mean_val, std_val, vals)
                json_export[model][m] = {
                    'mean': mean_val,
                    'std': std_val,
                    'raw_values': [float(v) for v in vals]
                }
                print(f"   {m:<20}: {mean_val:.4f} ± {std_val:.4f}  (Values: {[round(v, 4) for v in vals]})")

    # Generate Markdown Table
    print("\n" + "=" * 95)
    print("📋 MARKDOWN TABLE CHO BÀI BÁO (Paper Results Table)")
    print("=" * 95)
    md_lines = [
        "# Results Summary (Ambiguity Detection in SRS)",
        "",
        "| Model | Accuracy | Balanced Acc | F1-Macro | MCC | Standard ROC-AUC | PR-AUC |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    ]
    for model, m_dict in summary_table.items():
        def fmt(key):
            if key in m_dict:
                return f"{m_dict[key][0]:.3f}±{m_dict[key][1]:.3f}"
            return "N/A"
        row = f"| **{model}** | {fmt('accuracy')} | {fmt('balanced_accuracy')} | {fmt('f1_macro')} | {fmt('mcc')} | {fmt('roc_auc')} | {fmt('pr_auc')} |"
        md_lines.append(row)
        print(row)

    # Statistical Significance (Paired t-test)
    ref_model = "vlm_finetune"
    ttest_results = {}
    if ref_model in summary_table:
        print("\n" + "=" * 95)
        print(f"🔬 KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (Paired t-test so với {ref_model})")
        print("=" * 95)
        md_lines.extend(["", "## Statistical Significance (Paired t-test vs. vlm_finetune)", ""])
        ref_f1_macro = summary_table[ref_model].get('f1_macro', (None, None, []))[2]
        for comp_model, m_dict in summary_table.items():
            if comp_model == ref_model: continue
            comp_f1_macro = m_dict.get('f1_macro', (None, None, []))[2]
            if len(ref_f1_macro) == len(comp_f1_macro) and len(ref_f1_macro) >= 2:
                t_stat, p_val = stats.ttest_rel(ref_f1_macro, comp_f1_macro)
                sig_str = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s. (not significant)"))
                ttest_results[comp_model] = {
                    't_stat': float(t_stat),
                    'p_value': float(p_val),
                    'significance': sig_str
                }
                line = f"  • {ref_model} vs {comp_model}: F1-Macro t-stat = {t_stat:.4f}, p-value = {p_val:.4e} ({sig_str})"
                print(line)
                md_lines.append(f"- **{ref_model} vs {comp_model}**: $t = {t_stat:.4f}$, $p = {p_val:.4e}$ ({sig_str})")

    json_export['_ttest_vs_vlm_finetune'] = ttest_results

    # Generate LaTeX Table for Paper
    latex_lines = [
        "% LaTeX Table for IEEE / ACM Camera-Ready Paper",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Performance comparison across 3 random seeds (Mean $\\pm$ Std).}",
        "\\label{tab:main_results}",
        "\\begin{tabular}{lcccccc}",
        "\\hline",
        "\\textbf{Model} & \\textbf{Accuracy} & \\textbf{Bal. Acc} & \\textbf{F1-Macro} & \\textbf{MCC} & \\textbf{ROC-AUC} & \\textbf{PR-AUC} \\\\",
        "\\hline"
    ]
    for model, m_dict in summary_table.items():
        def fmt_tex(key):
            if key in m_dict:
                return f"{m_dict[key][0]:.3f}$\\pm${m_dict[key][1]:.3f}"
            return "N/A"
        clean_name = model.replace("_", "\\_")
        latex_lines.append(f"{clean_name} & {fmt_tex('accuracy')} & {fmt_tex('balanced_accuracy')} & {fmt_tex('f1_macro')} & {fmt_tex('mcc')} & {fmt_tex('roc_auc')} & {fmt_tex('pr_auc')} \\\\")
    latex_lines.extend([
        "\\hline",
        "\\end{tabular}",
        "\\end{table*}"
    ])

    # Save to files
    with open("paper_results_summary.json", 'w', encoding='utf-8') as f:
        json.dump(json_export, f, indent=2, ensure_ascii=False)

    with open("paper_results_table.md", 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines) + "\n")

    with open("paper_results_table.tex", 'w', encoding='utf-8') as f:
        f.write("\n".join(latex_lines) + "\n")

    print("\n" + "=" * 95)
    print("💾 ĐÃ LƯU THÀNH CÔNG CÁC FILE KẾT QUẢ TỔNG HỢP:")
    print("   1. paper_results_summary.json  (Dữ liệu thô JSON đầy đủ Mean, Std & t-test)")
    print("   2. paper_results_table.md      (Bảng Markdown trực quan cho báo cáo)")
    print("   3. paper_results_table.tex     (Bảng LaTeX chuẩn để copy thẳng vào Overleaf/bài báo)")
    print("=" * 95)

if __name__ == "__main__":
    summarize()
