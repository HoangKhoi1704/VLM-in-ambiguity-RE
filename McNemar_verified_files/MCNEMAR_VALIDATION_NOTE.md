# McNemar results — reproducibility note

The included results were rerun from the repository prediction CSVs for LLM Fine-Tuned and VLM Fine-Tuned, separately for seeds 42, 123, and 456.

Validation performed by the script:
- Each pair of CSVs contains 219 rows.
- IDs are unique and the LLM/VLM ID sets match within each seed.
- Ground-truth labels agree for matched IDs.
- McNemar's exact two-sided binomial test is computed from paired correctness (not from aggregate metrics).
- Holm correction is applied across the three seed-specific tests.

Reproduced results:
- Seed 42: b=11, c=4, exact p=0.118469, Holm-adjusted p=0.355408.
- Seed 123: b=13, c=10, exact p=0.677639, Holm-adjusted p=0.677639.
- Seed 456: b=4, c=11, exact p=0.118469, Holm-adjusted p=0.355408.

All adjusted p-values exceed 0.05. The result supports no statistically significant difference in paired binary correctness for these seed-specific test splits. It does not establish causal effects of multimodal pretraining, and it does not test F1-Macro, ROC-AUC, or PR-AUC.

Reproduce from repository root with:
```bash
python McNemar_verified_files/src/mcnemar_analysis.py --results-dir results --output McNemar_verified_files/results/mcnemar_results.json
```

This confirms reproducibility against the prediction CSVs included in the provided repository snapshot. It does not independently establish the provenance of the model predictions or checkpoints.
