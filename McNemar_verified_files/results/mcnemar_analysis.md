# McNemar Analysis: Fine-Tuned LLM vs Fine-Tuned VLM

## Purpose

This supplementary analysis compares the binary correctness of **LLM Fine-Tuned (QLoRA)** and **VLM Fine-Tuned (QLoRA)** on the same test IDs within each seed. It uses the exact two-sided McNemar test (binomial formulation), then applies Holm correction across the three seed-specific tests.

## Results

| Seed | Paired test samples | LLM accuracy | VLM accuracy | VLM correct / LLM wrong (b) | LLM correct / VLM wrong (c) | Discordant pairs | Exact McNemar p | Holm-adjusted p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 219 | 0.8539 | 0.8858 | 11 | 4 | 15 | 0.1185 | 0.3554 |\n| 123 | 219 | 0.8493 | 0.8630 | 13 | 10 | 23 | 0.6776 | 0.6776 |\n| 456 | 219 | 0.8721 | 0.8402 | 4 | 11 | 15 | 0.1185 | 0.3554 |\n
## Interpretation

None of the three seed-specific comparisons is statistically significant at the 0.05 level after Holm correction. The observed direction is not consistent across seeds: VLM has more discordant wins on seeds 42 and 123, while LLM has more on seed 456. These results do **not** provide statistically significant evidence of a difference in binary correctness between the two fine-tuned models on the evaluated splits.

McNemar's test evaluates paired correctness, not differences in F1-Macro, ROC-AUC, or PR-AUC. It also cannot establish that multimodal pretraining caused a performance difference.

## Validity checks

- Each compared CSV contains 219 rows.
- LLM and VLM prediction files have matching test IDs within each seed.
- Ground-truth labels agree for paired IDs.
- Tests are performed separately by seed; predictions are not pooled across seeds.
- Holm correction is applied across the three seed-level tests.
- The test compares correctness derived from `predicted` and `defect`, not the CSV's precomputed `correct` field.

## Reproduction

From the repository root, run:

```bash
python McNemar_verified_files/src/mcnemar_analysis.py --results-dir results --output McNemar_verified_files/results/mcnemar_results.json
```

The script reads `results/predictions/` and writes `McNemar_verified_files/results/mcnemar_results.json`. It requires `pandas`, `scipy`, and `statsmodels`.

## Suggested manuscript text

> We additionally compared the fine-tuned LLM and VLM using exact two-sided McNemar tests on paired test predictions for each seed. After Holm correction across the three seed-specific tests, none of the comparisons was statistically significant (adjusted p-values: 0.3554, 0.6776, and 0.3554 for seeds 42, 123, and 456, respectively). The direction of discordant prediction differences varied across seeds. Therefore, these tests do not establish a statistically significant difference in binary correctness between the fine-tuned models.

Use this text only if the manuscript's experimental setup and reported artifact versions correspond to the prediction files audited here.
