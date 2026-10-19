# McNemar Analysis: Fine-Tuned LLM vs Fine-Tuned VLM

## Purpose

This supplementary analysis compares the binary correctness of **LLM Fine-Tuned (QLoRA)** and **VLM Fine-Tuned (QLoRA)** on the same test IDs within each seed. It uses the exact two-sided McNemar test (binomial formulation), then applies Holm correction across the three seed-specific tests.

## Results

| Seed | Paired test samples | LLM accuracy | VLM accuracy | VLM correct / LLM wrong (b) | LLM correct / VLM wrong (c) | Discordant pairs | Exact McNemar p | Holm-adjusted p |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 211 | 0.8294 | 0.9005 | 17 | 2 | 19 | 0.0007 | 0.0022 |
| 123 | 211 | 0.8389 | 0.8389 | 11 | 11 | 22 | 1.0000 | 1.0000 |
| 456 | 211 | 0.8531 | 0.8483 | 9 | 10 | 19 | 1.0000 | 1.0000 |

## Interpretation

- On seed 42, VLM shows significantly more discordant wins (b=17 vs c=2, Holm p=0.0022).
- On seed 123, both models achieve identical accuracy and equal discordant wins (b=11, c=11, Holm p=1.0000).
- On seed 456, discordant wins are balanced (b=9 vs c=10, Holm p=1.0000).

Across the three seeds, the paired tests reflect comparable top-tier performance for both fine-tuned architectures on the evaluated test splits.

## Validity checks

- Each compared CSV contains 211 rows (deduplicated test split).
- LLM and VLM prediction files have matching test IDs within each seed.
- Ground-truth labels agree for paired IDs.
- Tests are performed separately by seed; predictions are not pooled across seeds.
- Holm correction is applied across the three seed-level tests.
- The test compares correctness derived from `predicted` and `defect`, not the CSV's precomputed `correct` field.
