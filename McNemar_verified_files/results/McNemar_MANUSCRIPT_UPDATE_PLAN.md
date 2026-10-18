# Manuscript Update Plan: Restore McNemar Analysis

## Repository changes completed

- Added `src/mcnemar_analysis.py` to run exact two-sided McNemar tests on paired correctness for the fine-tuned LLM and VLM, separately for seeds 42, 123, and 456.
- Added `results/mcnemar_results.json` with machine-readable results.
- Added `results/mcnemar_analysis.md` with the table, interpretation, validation checks, and manuscript-ready text.
- Updated `README.md` with the purpose and reproduction command.

## Evidence from the current prediction artifacts

The fine-tuned LLM and VLM prediction CSVs each contain 219 rows per seed. Within each seed, IDs and ground-truth labels match between the two models. Exact two-sided McNemar p-values are 0.1185 (seed 42), 0.6776 (seed 123), and 0.1185 (seed 456). After Holm correction, adjusted p-values are 0.3554, 0.6776, and 0.3554. None is significant at alpha = 0.05. The direction of discordant wins varies across seeds.

## Manuscript edits to make

The full manuscript source (`.tex` or `.docx`) was not included in this repository archive. Therefore, the manuscript itself has not been modified here. To restore McNemar transparently:

1. Add a short paragraph to the Statistical Analysis / Evaluation Metrics subsection stating that McNemar compares paired binary correctness on the same test IDs.
2. Add the results paragraph and table from `results/mcnemar_analysis.md` to the Results section, or move the detailed per-seed table to an appendix.
3. In the Discussion, state that none of the three seed-specific tests remains significant after Holm correction and that the direction varies across seeds.
4. Do not restore the previously mentioned unsupported McNemar p-value (for example, `p = 0.0192`) unless it can be independently reproduced from the exact paired prediction artifacts and documented test setup.
5. Do not claim that McNemar proves a causal effect of multimodal pretraining. It tests paired correctness only.
6. Ensure the manuscript identifies the exact artifact version and uses the same test IDs and labels as the analysis script.

## Reproduction

From the repository root:

```bash
python McNemar_verified_files/src/mcnemar_analysis.py --results-dir results --output McNemar_verified_files/results/mcnemar_results.json
```

Dependencies: `pandas`, `scipy`, `statsmodels`.

## Limitations

- These are three seed-specific tests; test splits differ between seeds and may overlap in their underlying samples.
- Holm correction controls family-wise error across the three reported tests, but the seed-wise results are not independent replications on entirely disjoint data.
- McNemar does not test differences in F1-Macro, ROC-AUC, or PR-AUC.
- The repository archive does not contain the full manuscript source, so only the code-side analysis and manuscript insertion plan can be updated here.
