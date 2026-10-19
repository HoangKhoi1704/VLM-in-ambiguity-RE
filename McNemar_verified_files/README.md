# McNemar Verified Analysis

This directory is the reproducible audit package for the paired comparison of the fine-tuned LLM and VLM.

## Data flow

```text
results/predictions/
  predictions_llm_finetune_seed_{42,123,456}.csv
  predictions_vlm_finetune_seed_{42,123,456}.csv
        |
        v
src/mcnemar_analysis.py
        |
        v
results/mcnemar_results.json
results/mcnemar_analysis.md
```

Each input CSV must contain unique `id`, ground-truth `defect`, and model `predicted` columns. The analysis pairs rows by `id`, verifies matching labels, and derives correctness from `predicted` versus `defect`.

## Artifact ownership

| Location | Role | Status |
| --- | --- | --- |
| `../results/predictions/` | Paired LLM/VLM prediction inputs, 211 rows per seed | Source data |
| `src/` | McNemar implementation | Executable code |
| `results/mcnemar_results.json` | Exact p-values and Holm-adjusted p-values | Machine-readable output |
| `results/mcnemar_analysis.md` | Table, interpretation, validity checks, manuscript text | Human-readable report |
| `MCNEMAR_VALIDATION_NOTE.md` | Reproducibility and provenance note | Audit note |
| `results/McNemar_MANUSCRIPT_UPDATE_PLAN.md` | How to incorporate the result into the paper | Manuscript support |

## Current evidence

- Three seed-specific paired tests are run on 211 test samples each (stratified deduplicated test split).
- Holm-adjusted p-values are `0.0022` (seed `42`, VLM > LLM), `1.0000` (seed `123`, identical), and `1.0000` (seed `456`, LLM 10 vs VLM 9).
- Overall across seeds, paired binary correctness remains comparable between the fine-tuned LLM and VLM.

This analysis is supplementary to the main metrics in `results/`. It does not replace the aggregate accuracy/F1/ROC-AUC/PR-AUC tables and does not test causal effects of multimodal pretraining.

## Reproduce from repository root

```bash
python run_mcnemar.py
```

Or directly via the submodule script:
```bash
python McNemar_verified_files/src/mcnemar_analysis.py \
  --results-dir results \
  --output McNemar_verified_files/results/mcnemar_results.json
```
