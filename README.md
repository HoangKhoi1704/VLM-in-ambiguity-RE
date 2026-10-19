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
├── notebook_runs/                     # Isolated notebook outputs grouped by seed
│   └── README.md
├── src/
│   ├── run_baselines.py              # Evaluates Majority Classifier, TF-IDF + LR, TF-IDF + SVM
│   ├── notebook_runner.py             # Runs one notebook/seed and appends the run log
│   ├── notebook_logging.py            # Logs each notebook cell and its errors
│   └── aggregate_results.py           # Aggregates multi-seed results and computes paired t-tests
├── aggregate_notebook_logs.py         # Aggregates cell logs into run summaries
├── results/
│   ├── metrics/                      # JSON metrics across all 21 runs (7 models x 3 seeds)
│   ├── predictions/                  # Hold-out test predictions for models and seeds
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
├── run_notebooks.py                  # Portable batch runner for local Python/Colab
└── README.md
```

Notebook runs are isolated under `notebook_runs/seed_<seed>/<notebook>/`. The
central run table is `results/notebook_run_logs.csv`; it records the notebook,
seed, status, duration, executed notebook path, and `run.log` path for every
attempt.

---

## Experimental Results

All experiments are conducted across **three independent random seeds** (`42`, `123`, `456`) on stratified 80/10/10 splits. Standard ROC-AUC and PR-AUC are derived from continuous posterior probabilities $P(\text{defect}) \in [0, 1]$ extracted from next-token logits.

### Performance Comparison (Mean ± Std)

| Model                                  |        Accuracy        |      Balanced Acc      |        F1-Macro        |          MCC          |    Standard ROC-AUC    |         PR-AUC         |
| :------------------------------------- | :--------------------: | :--------------------: | :--------------------: | :--------------------: | :--------------------: | :--------------------: |
| **Majority Classifier**          |      0.703±0.000      |      0.500±0.000      |      0.413±0.000      |      0.000±0.000      |      0.500±0.000      |      0.648±0.000      |
| **TF-IDF + Logistic Regression** |      0.775±0.009      |      0.641±0.017      |      0.654±0.022      |      0.406±0.031      |      0.868±0.028      |      0.731±0.052      |
| **TF-IDF + Linear SVM**          |      0.807±0.028      |      0.737±0.031      |      0.752±0.035      |      0.513±0.072      |      0.864±0.022      |      0.730±0.053      |
| **LLM Zero-Shot**                |      0.464±0.028      |      0.487±0.030      |      0.453±0.027      |     -0.024±0.056     |      0.478±0.031      |      0.348±0.016      |
| **VLM Zero-Shot**                |      0.571±0.019      |      0.462±0.016      |      0.458±0.016      |     -0.080±0.032     |      0.425±0.008      |      0.267±0.004      |
| **LLM Fine-Tuned (QLoRA)**       |      0.858±0.010      |      0.824±0.006      |      0.828±0.008      |      0.659±0.018      |      0.925±0.010      |      0.851±0.020      |
| **VLM Fine-Tuned (QLoRA)**       | **0.863±0.019** | **0.821±0.048** | **0.828±0.035** | **0.668±0.053** | **0.930±0.009** | **0.849±0.018** |

### Statistical Significance (Paired $t$-test vs. VLM Fine-Tuned)

* vs. **TF-IDF + Linear SVM**: $t = 20.43$, $p = 2.39 \times 10^{-3}$ ($p < 0.01$)
* vs. **TF-IDF + Logistic Regression**: $t = 18.94$, $p = 2.78 \times 10^{-3}$ ($p < 0.01$)
* vs. **Majority Classifier**: $t = 17.00$, $p = 3.44 \times 10^{-3}$ ($p < 0.01$)
* vs. **LLM Zero-Shot**: $t = 34.86$, $p = 8.22 \times 10^{-4}$ ($p < 0.001$, statistically significant)
* vs. **VLM Zero-Shot**: $t = 23.03$, $p = 1.88 \times 10^{-3}$ ($p < 0.01$)
* vs. **LLM Fine-Tuned**: $t = -0.01$, $p = 0.993$ (comparable top-tier performance)

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
	Configure `RANDOM_SEED` (`42`, `123`, or `456`) in the seed configuration cell and run all cells sequentially.

When running a notebook directly with **Run all**, the first cell registers
cell-level logging automatically. Logs are appended to:

```text
results/notebook_cell_logs.csv
```

After one or more notebook runs, create a compact summary table with:

```powershell
python aggregate_notebook_logs.py
```

The summary is written to `results/notebook_run_summary.csv`. A failed cell is
recorded with `status=FAILED`, `error_type`, and `error_message`, so the
corresponding notebook run can be debugged without rerunning all notebooks.

For reproducible local runs with isolated folders, use the runner from the
repository root. The original notebook is copied to a seed-specific run folder,
its seed is set automatically, and the executed notebook plus all relative
artifacts are kept there:

```powershell
python src/notebook_runner.py --notebook notebooks/SRS_LLM_Zero_Shot.ipynb --seed 42
python src/notebook_runner.py --notebook notebooks/SRS_LLM_Zero_Shot.ipynb --seed 123
python src/notebook_runner.py --notebook notebooks/SRS_LLM_Zero_Shot.ipynb --seed 456
```

Replace the notebook path to run the VLM or fine-tuning notebooks. Each command
appends one row to `results/notebook_run_logs.csv`.

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
the runtime disconnects. Outputs and `results/notebook_run_logs.csv` are saved
in the cloned Colab workspace, so download or push them after the runs finish.
