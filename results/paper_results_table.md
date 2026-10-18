# Results Summary (Ambiguity Detection in SRS)

| Model | Accuracy | Balanced Acc | F1-Macro | MCC | Standard ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **llm_finetune** | 0.858±0.010 | 0.824±0.006 | 0.828±0.008 | 0.659±0.018 | 0.925±0.010 | 0.851±0.020 |
| **llm_zero_shot** | 0.464±0.028 | 0.487±0.030 | 0.453±0.027 | -0.024±0.056 | 0.478±0.031 | 0.348±0.016 |
| **majority_classifier** | 0.703±0.000 | 0.500±0.000 | 0.413±0.000 | 0.000±0.000 | 0.500±0.000 | 0.648±0.000 |
| **tfidf_linear_svm** | 0.807±0.030 | 0.740±0.034 | 0.753±0.037 | 0.515±0.077 | 0.866±0.022 | 0.732±0.059 |
| **tfidf_logistic_regression** | 0.776±0.010 | 0.645±0.018 | 0.660±0.022 | 0.410±0.033 | 0.869±0.029 | 0.726±0.060 |
| **vlm_finetune** | 0.863±0.019 | 0.821±0.048 | 0.828±0.035 | 0.668±0.053 | 0.930±0.009 | 0.849±0.018 |
| **vlm_zero_shot** | 0.571±0.019 | 0.462±0.016 | 0.458±0.016 | -0.080±0.032 | 0.425±0.008 | 0.267±0.004 |

## Statistical Significance (Paired t-test vs. vlm_finetune)

- **vlm_finetune vs llm_finetune**: $t = -0.0100$, $p = 9.9292e-01$ (n.s. (not significant))
- **vlm_finetune vs llm_zero_shot**: $t = 34.8596$, $p = 8.2190e-04$ (***)
- **vlm_finetune vs majority_classifier**: $t = 16.9997$, $p = 3.4425e-03$ (**)
- **vlm_finetune vs tfidf_linear_svm**: $t = 37.7164$, $p = 7.0223e-04$ (***)
- **vlm_finetune vs tfidf_logistic_regression**: $t = 11.1412$, $p = 7.9602e-03$ (**)
- **vlm_finetune vs vlm_zero_shot**: $t = 23.0312$, $p = 1.8799e-03$ (**)
