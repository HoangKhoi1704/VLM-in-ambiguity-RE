# Results Summary (Ambiguity Detection in SRS)

| Model | Accuracy | Balanced Acc | F1-Macro | MCC | Standard ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **llm_finetune** | 0.847±0.018 | 0.824±0.026 | 0.818±0.016 | 0.648±0.034 | 0.933±0.006 | 0.856±0.016 |
| **llm_zero_shot** | 0.464±0.028 | 0.487±0.030 | 0.453±0.027 | -0.024±0.056 | 0.478±0.031 | 0.348±0.016 |
| **majority_classifier** | 0.703±0.000 | 0.500±0.000 | 0.413±0.000 | 0.000±0.000 | 0.500±0.000 | 0.648±0.000 |
| **tfidf_linear_svm** | 0.807±0.028 | 0.737±0.031 | 0.752±0.035 | 0.513±0.072 | 0.864±0.022 | 0.730±0.053 |
| **tfidf_logistic_regression** | 0.775±0.009 | 0.641±0.017 | 0.654±0.022 | 0.406±0.031 | 0.868±0.028 | 0.731±0.052 |
| **vlm_finetune** | 0.868±0.025 | 0.838±0.065 | 0.836±0.043 | 0.687±0.075 | 0.941±0.011 | 0.875±0.019 |
| **vlm_zero_shot** | 0.571±0.019 | 0.462±0.016 | 0.458±0.016 | -0.080±0.032 | 0.425±0.008 | 0.267±0.004 |

## Statistical Significance (Paired t-test vs. vlm_finetune)

- **vlm_finetune vs llm_finetune**: $t = 0.4580$, $p = 6.9190e-01$ (n.s. (not significant))
- **vlm_finetune vs llm_zero_shot**: $t = 20.4677$, $p = 2.3785e-03$ (**)
- **vlm_finetune vs majority_classifier**: $t = 13.8218$, $p = 5.1937e-03$ (**)
- **vlm_finetune vs tfidf_linear_svm**: $t = 13.6916$, $p = 5.2921e-03$ (**)
- **vlm_finetune vs tfidf_logistic_regression**: $t = 11.0668$, $p = 8.0663e-03$ (**)
- **vlm_finetune vs vlm_zero_shot**: $t = 18.4205$, $p = 2.9342e-03$ (**)
