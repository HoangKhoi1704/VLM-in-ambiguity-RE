# Detecting Ambiguity in Software Requirements Specifications (SRS): Comparing LLMs and VLMs

This repository contains the empirical code, benchmark dataset, notebooks, and multi-seed experimental results for detecting ambiguity defects in Software Requirements Specifications (SRS). We evaluate Large Language Models (`Qwen/Qwen2.5-3B-Instruct`) and Vision-Language Models (`Qwen/Qwen2.5-VL-3B-Instruct`) under both Zero-Shot and 4-bit QLoRA Fine-Tuning regimes on the QuRE dataset.

---

## Repository Structure

```
├── data/
│   ├── QuRE.csv                      # Benchmark dataset (2,187 requirements annotated for defects)
│   └── datasetRE.csv                 # Dataset mirror
├── notebooks/
│   ├── SRS_LLM_Zero_Shot.ipynb       # LLM Zero-Shot local GPU inference
│   ├── SRS_VLM_Zero_Shot.ipynb       # VLM Zero-Shot local GPU inference
│   ├── SRS_LLM_Finetune.ipynb        # QLoRA fine-tuning for Qwen2.5-3B-Instruct
│   └── SRS_VLM_Finetune.ipynb        # QLoRA fine-tuning for Qwen2.5-VL-3B-Instruct
├── src/
│   ├── run_baselines.py              # Evaluates Majority Classifier, TF-IDF + LR, TF-IDF + SVM
│   └── aggregate_results.py          # Aggregates multi-seed results and computes paired t-tests
├── results/
│   ├── metrics/                      # JSON metrics across all 21 runs (7 models x 3 seeds)
│   ├── predictions/                  # Hold-out test set predictions CSV with P(defect) probabilities
│   ├── figures/                      # Evaluation dashboards, ROC and PR curves
│   ├── paper_results_summary.json    # Complete statistical summary (Mean, Std, t-test p-values)
│   ├── paper_results_table.md        # Formatted Markdown results table
│   └── paper_results_table.tex       # Camera-ready LaTeX table for paper inclusion
├── McNemar_verified_files/           # Reproducible paired LLM/VLM significance analysis
│   ├── src/                           # McNemar analysis code
│   ├── results/                       # JSON result and report for paired tests
│   ├── README.md                      # Data flow and artifact ownership
│   └── MCNEMAR_VALIDATION_NOTE.md     # Reproducibility and provenance note
├── aggregate_results.py              # Root runner for results aggregation
├── run_baselines.py                  # Root runner for classical ML baselines
└── README.md
```

---

## Experimental Results

All experiments are conducted across **three independent random seeds** (`42`, `123`, `456`) on stratified 80/10/10 splits. Standard ROC-AUC and PR-AUC are derived from continuous posterior probabilities $P(\text{defect}) \in [0, 1]$ extracted from next-token logits.

### Performance Comparison (Mean ± Std)

| Model | Accuracy | Balanced Acc | F1-Macro | MCC | Standard ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Classifier** | 0.703±0.000 | 0.500±0.000 | 0.413±0.000 | 0.000±0.000 | 0.500±0.000 | 0.648±0.000 |
| **TF-IDF + Logistic Regression** | 0.776±0.010 | 0.645±0.018 | 0.660±0.022 | 0.410±0.033 | 0.869±0.029 | 0.726±0.060 |
| **TF-IDF + Linear SVM** | 0.807±0.030 | 0.740±0.034 | 0.753±0.037 | 0.515±0.077 | 0.866±0.022 | 0.732±0.059 |
| **LLM Zero-Shot** | 0.464±0.028 | 0.487±0.030 | 0.453±0.027 | -0.024±0.056 | 0.478±0.031 | 0.348±0.016 |
| **VLM Zero-Shot** | 0.571±0.019 | 0.462±0.016 | 0.458±0.016 | -0.080±0.032 | 0.425±0.008 | 0.267±0.004 |
| **LLM Fine-Tuned (QLoRA)** | 0.858±0.010 | 0.824±0.006 | 0.828±0.008 | 0.659±0.018 | 0.925±0.010 | 0.851±0.020 |
| **VLM Fine-Tuned (QLoRA)** | **0.865±0.021** | **0.824±0.051** | **0.830±0.037** | **0.673±0.057** | **0.932±0.007** | **0.852±0.013** |

### Statistical Significance (Paired $t$-test vs. VLM Fine-Tuned)
* vs. **TF-IDF + Linear SVM**: $t = 61.72$, $p = 2.62 \times 10^{-4}$ ($p < 0.001$, extremely statistically significant)
* vs. **TF-IDF + Logistic Regression**: $t = 9.95$, $p = 9.95 \times 10^{-3}$ ($p < 0.01$)
* vs. **LLM Zero-Shot**: $t = 29.52$, $p = 1.15 \times 10^{-3}$ ($p < 0.01$)
* vs. **VLM Zero-Shot**: $t = 21.94$, $p = 2.07 \times 10^{-3}$ ($p < 0.01$)
* vs. **LLM Fine-Tuned**: $t = 0.05$, $p = 0.961$ (comparable top-tier performance)

---

## Quickstart & Replication

### 1. Run Classical Baselines
```bash
python run_baselines.py
```

### 2. Aggregate Multi-Seed Results
```bash
python aggregate_results.py
```
This updates `results/paper_results_summary.json`, `results/paper_results_table.md`, and `results/paper_results_table.tex`.

### 3. Run the Verified McNemar Analysis
```bash
python McNemar_verified_files/src/mcnemar_analysis.py \
	--results-dir results \
	--output McNemar_verified_files/results/mcnemar_results.json
```
This compares paired correctness for the fine-tuned LLM and VLM on the same test IDs for seeds 42, 123, and 456. The result is supplementary to the aggregate metrics: all Holm-adjusted p-values are above 0.05, and the direction of the discordant pairs changes across seeds. See `McNemar_verified_files/README.md` for the artifact map.

### 4. Notebook Execution on Google Colab / Local GPU
Open any notebook inside `notebooks/`:
* `notebooks/SRS_LLM_Zero_Shot.ipynb`
* `notebooks/SRS_VLM_Zero_Shot.ipynb`
* `notebooks/SRS_LLM_Finetune.ipynb`
* `notebooks/SRS_VLM_Finetune.ipynb`
Configure `RANDOM_SEED` (`42`, `123`, or `456`) in Cell 3 and run all cells sequentially.
