"""
=============================================================================
AGGREGATE MULTI-SEED RESULTS & STATISTICAL SIGNIFICANCE TESTING
Reads all `metrics_<model>_seed_<seed>.json` files in the directory.
Calculates:
  1. Mean ± Std for all metrics across seeds
  2. Paired t-test (p-value) comparing VLM Fine-Tuned vs other models
  3. Formatted Markdown and LaTeX tables for direct inclusion in the paper.
=============================================================================
"""

import os
import glob
import json
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
    for model, seed_dict in data.items():
        summary_table[model] = {}
        seeds = sorted(seed_dict.keys())
        print(f"\n📌 Model: {model} (Đã chạy {len(seeds)} seeds: {seeds})")
        for m in METRIC_KEYS:
            vals = [seed_dict[s][m] for s in seeds if seed_dict[s].get(m) is not None]
            if vals:
                mean_val = np.mean(vals)
                std_val = np.std(vals)
                summary_table[model][m] = (mean_val, std_val, vals)
                print(f"   {m:<20}: {mean_val:.4f} ± {std_val:.4f}  (Values: {[round(v, 4) for v in vals]})")

    # Generate Markdown Table
    print("\n" + "=" * 95)
    print("📋 MARKDOWN TABLE CHO BÀI BÁO (Paper Results Table)")
    print("=" * 95)
    md_header = "| Model | Accuracy | Balanced Acc | F1-Macro | MCC | Standard ROC-AUC | PR-AUC |"
    md_sep    = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |"
    print(md_header)
    print(md_sep)
    for model, m_dict in summary_table.items():
        def fmt(key):
            if key in m_dict:
                return f"{m_dict[key][0]:.3f}±{m_dict[key][1]:.3f}"
            return "N/A"
        row = f"| **{model}** | {fmt('accuracy')} | {fmt('balanced_accuracy')} | {fmt('f1_macro')} | {fmt('mcc')} | {fmt('roc_auc')} | {fmt('pr_auc')} |"
        print(row)

    # Statistical Significance (Paired t-test)
    ref_model = "vlm_finetune"
    if ref_model in summary_table:
        print("\n" + "=" * 95)
        print(f"🔬 KIỂM ĐỊNH Ý NGHĨA THỐNG KÊ (Paired t-test so với {ref_model})")
        print("=" * 95)
        ref_f1_macro = summary_table[ref_model].get('f1_macro', (None, None, []))[2]
        for comp_model, m_dict in summary_table.items():
            if comp_model == ref_model: continue
            comp_f1_macro = m_dict.get('f1_macro', (None, None, []))[2]
            if len(ref_f1_macro) == len(comp_f1_macro) and len(ref_f1_macro) >= 2:
                t_stat, p_val = stats.ttest_rel(ref_f1_macro, comp_f1_macro)
                sig_str = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s. (not significant)"))
                print(f"  • {ref_model} vs {comp_model}:")
                print(f"      F1-Macro t-stat = {t_stat:.4f}, p-value = {p_val:.4e} ({sig_str})")

if __name__ == "__main__":
    summarize()
