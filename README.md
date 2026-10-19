# Detecting Ambiguity in Software Requirements Specifications (SRS): Comparing LLMs and VLMs

This repository contains the empirical code, benchmark dataset, notebooks, and multi-seed experimental results for detecting ambiguity defects in Software Requirements Specifications (SRS). We evaluate Large Language Models (`Qwen/Qwen2.5-3B-Instruct`) and Vision-Language Models (`Qwen/Qwen2.5-VL-3B-Instruct`) under both Zero-Shot and 4-bit QLoRA Fine-Tuning regimes on the QuRE dataset.

---

## Repository Structure

```
├── data/
│   ├── QuRE.csv                      # Benchmark dataset (2,109 deduplicated requirements annotated for defects)
│   └── datasetRE.csv                 # Dataset mirror (2,109 deduplicated requirements)
├── notebooks/
│   ├── SRS_LLM_Zero_Shot.ipynb       # LLM Zero-Shot local GPU inference
│   ├── SRS_VLM_Zero_Shot.ipynb       # VLM Zero-Shot local GPU inference
│   ├── SRS_LLM_Finetune.ipynb        # QLoRA fine-tuning for Qwen2.5-3B-Instruct
│   └── SRS_VLM_Finetune.ipynb        # QLoRA fine-tuning for Qwen2.5-VL-3B-Instruct
├── notebook_runs/                     # Isolated notebook outputs grouped by seed
│   └── README.md
├── src/
│   ├── run_baselines.py              # Evaluates Majority Classifier, TF-IDF + LR, TF-IDF + SVM
│   ├── notebook_runner.py             # Runs one notebook/seed with isolated folder
│   ├── aggregate_results.py           # Aggregates multi-seed results and computes paired t-tests
│   └── generate_per_seed_table.py     # Generates per-seed breakdown table across all seeds
├── results/
│   ├── metrics/                      # JSON metrics across all 21 runs (7 models x 3 seeds)
│   ├── predictions/                  # Hold-out test predictions for models and seeds
│   ├── figures/                      # Evaluation dashboards, ROC and PR curves
│   ├── paper_results_summary.json    # Complete statistical summary (Mean, Std, t-test p-values)
│   ├── paper_results_table.md        # Formatted Markdown results table
│   ├── paper_results_table.tex       # Camera-ready LaTeX table for paper inclusion
│   ├── per_seed_results_table.md     # Detailed per-seed metric breakdown (Markdown)
│   └── per_seed_results_table.tex    # Camera-ready per-seed LaTeX multirow table
├── McNemar_verified_files/           # Reproducible paired LLM/VLM significance analysis
│   ├── src/                           # McNemar analysis code
│   ├── results/                       # JSON result and report for paired tests
│   ├── README.md                      # Data flow and artifact ownership
│   └── MCNEMAR_VALIDATION_NOTE.md     # Reproducibility and provenance note
├── aggregate_results.py              # Root runner for results aggregation
├── generate_per_seed_table.py        # Root runner for per-seed results breakdown
├── run_baselines.py                  # Root runner for classical ML baselines
├── run_mcnemar.py                    # Root runner for McNemar test analysis
├── run_notebooks.py                  # Portable batch runner for local Python/Colab
└── README.md
```

Notebook runs are isolated under `notebook_runs/seed_<seed>/`.
Source notebooks in `notebooks/` remain unchanged.

---

## Experimental Results

All experiments are conducted across **three independent random seeds** (`42`, `123`, `456`) on stratified 80/10/10 splits. Standard ROC-AUC and PR-AUC are derived from continuous posterior probabilities $P(\text{defect}) \in [0, 1]$ extracted from next-token logits.

### Performance Comparison (Mean ± Std)

| Model                                  |        Accuracy        |      Balanced Acc      |        F1-Macro        |          MCC          |    Standard ROC-AUC    |         PR-AUC         |
| :------------------------------------- | :--------------------: | :--------------------: | :--------------------: | :--------------------: | :--------------------: | :--------------------: |
| **Majority Classifier**          |      0.706±0.000      |      0.500±0.000      |      0.414±0.000      |      0.000±0.000      |      0.500±0.000      |      0.647±0.000      |
| **TF-IDF + Logistic Regression** |      0.774±0.015      |      0.631±0.026      |      0.641±0.036      |      0.397±0.051      |      0.891±0.023      |      0.762±0.034      |
| **TF-IDF + Linear SVM**          |      0.823±0.035      |      0.749±0.055      |      0.766±0.054      |      0.547±0.098      |      0.893±0.036      |      0.780±0.051      |
| **LLM Zero-Shot**                |      0.490±0.032      |      0.510±0.035      |      0.476±0.032      |      0.018±0.064      |      0.496±0.046      |      0.373±0.042      |
| **VLM Zero-Shot**                |      0.603±0.027      |      0.492±0.033      |      0.488±0.035      |     -0.018±0.071      |      0.455±0.059      |      0.273±0.037      |
| **LLM Fine-Tuned (QLoRA)**       |      0.840±0.010      |      0.831±0.028      |      0.815±0.012      |      0.641±0.028      |      0.925±0.007      |      0.839±0.020      |
| **VLM Fine-Tuned (QLoRA)**       | **0.863±0.027** | **0.843±0.048** | **0.836±0.036** | **0.675±0.076** | **0.937±0.012** | **0.863±0.028** |

### Statistical Significance (Paired $t$-test vs. VLM Fine-Tuned)

* vs. **TF-IDF + Linear SVM**: $t = 1.15$, $p = 0.369$ (n.s.)
* vs. **TF-IDF + Logistic Regression**: $t = 3.87$, $p = 0.061$ (n.s.)
* vs. **Majority Classifier**: $t = 16.65$, $p = 3.59 \times 10^{-3}$ ($p < 0.01$)
* vs. **LLM Zero-Shot**: $t = 17.45$, $p = 3.27 \times 10^{-3}$ ($p < 0.01$)
* vs. **VLM Zero-Shot**: $t = 81.17$, $p = 1.52 \times 10^{-4}$ ($p < 0.001$, statistically significant)
* vs. **LLM Fine-Tuned**: $t = 0.81$, $p = 0.505$ (n.s., comparable top-tier performance)

---

## Quickstart & Replication

### 1. Run Classical Baselines

```bash
python run_baselines.py
```

The baseline runner evaluates the Majority Classifier, TF-IDF + Logistic Regression,
and TF-IDF + Linear SVM on stratified 80/10/10 splits for seeds `42`, `123`, and
`456`. It writes one metrics JSON and one prediction CSV per model and seed:

```text
results/metrics/metrics_<model>_seed_<seed>.json
results/predictions/predictions_<model>_seed_<seed>.csv
```

Useful options:

```bash
# Run selected seeds on a different dataset
python run_baselines.py --dataset data/datasetRE.csv --seeds 42 123

# Write artifacts to a different directory
python run_baselines.py --output-dir results/my_baselines

# Write metrics only
python run_baselines.py --no-predictions
```

Each prediction CSV contains `id`, `requirement`, `defect`, `weak_word`,
`length`, `predicted`, `p_defect`, and `correct`. The `p_defect` column is the
model score for the `defect` class and can be used for ROC-AUC or PR-AUC analysis.

### 2. Aggregate Multi-Seed Results

```bash
python aggregate_results.py
```

This updates `results/paper_results_summary.json`, `results/paper_results_table.md`, and `results/paper_results_table.tex`.

### 3. Generate Per-Seed Detailed Breakdown Table

```bash
python generate_per_seed_table.py
```

This generates detailed metric tables breaking down results across every individual seed (`42`, `123`, `456`) and overall `Mean ± Std` for the 4 core models (LLM Zero-Shot, VLM Zero-Shot, LLM Fine-Tuned, VLM Fine-Tuned). Outputs are saved automatically in 4 formats:
- `results/per_seed_results_table.md` (Markdown for reports)
- `results/per_seed_results_table.tex` (Camera-ready LaTeX multirow table)
- `results/per_seed_results.csv` (CSV for spreadsheets)
- `results/per_seed_results.json` (Structured JSON)

To include classical ML baselines as well:
```bash
python generate_per_seed_table.py --all-models
```

### 4. Run the Verified McNemar Analysis

```bash
python run_mcnemar.py
```

Alternatively, run with custom options:
```bash
python McNemar_verified_files/src/mcnemar_analysis.py \
	--results-dir results \
	--output McNemar_verified_files/results/mcnemar_results.json
```

This compares paired correctness for the fine-tuned LLM and VLM on the same test IDs for seeds `42`, `123`, and `456`. The result is supplementary to the aggregate metrics: on seed 42, VLM demonstrates significantly more discordant wins ($b=17$ vs. $c=2$, Holm-adjusted $p = 0.0022$), while on seeds 123 and 456 the models exhibit comparable/balanced correctness (Holm-adjusted $p = 1.0000$). See `McNemar_verified_files/README.md` and `McNemar_verified_files/results/mcnemar_analysis.md` for the complete audit trail.

### 5. Notebook Execution on Google Colab / Local GPU

Open any notebook inside `notebooks/`:

* `notebooks/SRS_LLM_Zero_Shot.ipynb`
* `notebooks/SRS_VLM_Zero_Shot.ipynb`
* `notebooks/SRS_LLM_Finetune.ipynb`
* `notebooks/SRS_VLM_Finetune.ipynb`
	Configure `RANDOM_SEED` (`42`, `123`, or `456`) in the seed configuration cell and run all cells sequentially.

For reproducible local runs with isolated folders, use the runner from the
repository root. The original notebook is copied to a seed-specific run folder,
its seed is set automatically, and the executed notebook is kept there:

```powershell
python src/notebook_runner.py --notebook notebooks/SRS_LLM_Zero_Shot.ipynb --seed 42
python src/notebook_runner.py --notebook notebooks/SRS_LLM_Zero_Shot.ipynb --seed 123
python src/notebook_runner.py --notebook notebooks/SRS_LLM_Zero_Shot.ipynb --seed 456
```

To run all four notebooks for all three seeds:

```powershell
./run_notebooks.ps1
```

The same workflow can run on Google Colab after cloning the repository and
installing the notebook dependencies. Colab uses the portable Python runner:

```python
!git clone <repository-url>
%cd VLM-in-ambiguity-RE
# Run dependency installation cell at the top of the notebook before calling the runner.
!python run_notebooks.py
```

To test one notebook/seed first:

```python
!python run_notebooks.py --notebook notebooks/SRS_LLM_Zero_Shot.ipynb --seed 42
```

Colab must have a GPU runtime for the LLM/VLM notebooks. Fine-tuning all 12
runs may exceed one runtime session; use one notebook or one seed at a time if
the runtime disconnects. Outputs are saved
in the cloned Colab workspace, so download or push them after the runs finish.
