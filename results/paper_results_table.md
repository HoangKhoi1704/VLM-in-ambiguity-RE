# Results Summary (Ambiguity Detection in SRS)

| Model | Accuracy | Balanced Acc | F1-Macro | MCC | Standard ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **llm_finetune** | 0.858±0.010 | 0.824±0.006 | 0.828±0.008 | 0.659±0.018 | 0.925±0.010 | 0.851±0.020 |
| **llm_zero_shot** | 0.464±0.028 | 0.487±0.030 | 0.453±0.027 | -0.024±0.056 | 0.478±0.031 | 0.348±0.016 |
| **majority_classifier** | 0.703±0.000 | 0.500±0.000 | 0.413±0.000 | 0.000±0.000 | 0.500±0.000 | 0.648±0.000 |
| **tfidf_linear_svm** | 0.807±0.030 | 0.740±0.034 | 0.753±0.037 | 0.515±0.077 | 0.866±0.022 | 0.732±0.059 |
| **tfidf_logistic_regression** | 0.776±0.010 | 0.645±0.018 | 0.660±0.022 | 0.410±0.033 | 0.869±0.029 | 0.726±0.060 |
| **vlm_finetune** | 0.865±0.021 | 0.824±0.051 | 0.830±0.037 | 0.673±0.057 | 0.932±0.007 | 0.852±0.013 |
| **vlm_zero_shot** | 0.571±0.019 | 0.462±0.016 | 0.458±0.016 | -0.080±0.032 | 0.425±0.008 | 0.267±0.004 |

## Statistical Significance (Paired t-test vs. vlm_finetune)

- **vlm_finetune vs llm_finetune**: $t = 0.0548$, $p = 9.6130e-01$ (n.s. (not significant))
- **vlm_finetune vs llm_zero_shot**: $t = 29.5154$, $p = 1.1459e-03$ (**)
- **vlm_finetune vs majority_classifier**: $t = 16.1072$, $p = 3.8323e-03$ (**)
- **vlm_finetune vs tfidf_linear_svm**: $t = 61.7208$, $p = 2.6240e-04$ (***)
- **vlm_finetune vs tfidf_logistic_regression**: $t = 9.9516$, $p = 9.9471e-03$ (**)
- **vlm_finetune vs vlm_zero_shot**: $t = 21.9441$, $p = 2.0702e-03$ (**)
