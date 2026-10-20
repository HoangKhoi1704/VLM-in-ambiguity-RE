# McNemar Analysis: Reproducibility Note

## Evidence and procedure

The calculations were checked against the six model prediction result files in the main project repository under `results/predictions/`, covering the fine-tuned LLM and fine-tuned VLM for seeds 42, 123, and 456. Within each seed, predictions were paired by `id`; paired ground-truth labels were required to agree. Correctness was recomputed by comparing each `predicted` value with its corresponding `defect` label.

The exact two-sided McNemar test was calculated for each seed. The three resulting p-values were adjusted using the Holm procedure. The sample counts, discordant-pair counts, accuracies, and p-values are recorded in `results/mcnemar_results.json` and summarized in `results/mcnemar_analysis.md`.

## Results

| Seed | Paired observations | VLM correct / LLM incorrect (b) | LLM correct / VLM incorrect (c) | Exact two-sided p | Holm-adjusted p | Significant after correction (α = 0.05) |
|---:|---:|---:|---:|---:|---:|:---:|
| 42 | 211 | 17 | 2 | 0.000729 | 0.002186 | Yes |
| 123 | 211 | 11 | 11 | 1.000000 | 1.000000 | No |
| 456 | 211 | 9 | 10 | 1.000000 | 1.000000 | No |

For seed 42, the paired correctness difference is statistically significant after Holm correction, with more discordant observations favoring the VLM. Seeds 123 and 456 do not show a statistically significant paired difference after correction. The direction and strength of the observed differences are therefore not consistent across all three seed-specific evaluations.

## Reproduction

This analysis package contains the script and reports, but not the six source prediction CSVs. To reproduce the calculations, place this package in the main project repository as `McNemar_verified_files/`, ensure the six CSVs are present under `results/predictions/`, and run the following command from the main project repository root:

```bash
python McNemar_verified_files/src/mcnemar_analysis.py --results-dir results --output McNemar_verified_files/results/mcnemar_results.json
```

The script regenerates `mcnemar_results.json` and `mcnemar_analysis.md`. It requires `pandas` and `scipy`; `statsmodels` is used for Holm correction when available, with a pure-Python fallback otherwise.

