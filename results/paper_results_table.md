# Results Summary (Ambiguity Detection in SRS)

| Model | Accuracy | Balanced Acc | F1-Macro | MCC | Standard ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **llm_finetune** | 0.840±0.010 | 0.831±0.028 | 0.815±0.012 | 0.641±0.028 | 0.925±0.007 | 0.839±0.020 |
| **llm_zero_shot** | 0.490±0.032 | 0.510±0.035 | 0.476±0.032 | 0.018±0.064 | 0.496±0.046 | 0.373±0.042 |
| **majority_classifier** | 0.706±0.000 | 0.500±0.000 | 0.414±0.000 | 0.000±0.000 | 0.500±0.000 | 0.647±0.000 |
| **tfidf_linear_svm** | 0.823±0.035 | 0.749±0.055 | 0.766±0.054 | 0.547±0.098 | 0.893±0.036 | 0.780±0.051 |
| **tfidf_logistic_regression** | 0.774±0.015 | 0.631±0.026 | 0.641±0.036 | 0.397±0.051 | 0.891±0.023 | 0.762±0.034 |
| **vlm_finetune** | 0.863±0.027 | 0.843±0.048 | 0.836±0.036 | 0.675±0.076 | 0.937±0.012 | 0.863±0.028 |
| **vlm_zero_shot** | 0.603±0.027 | 0.492±0.033 | 0.488±0.035 | -0.018±0.071 | 0.455±0.059 | 0.273±0.037 |

## Statistical Significance (Paired t-test vs. vlm_finetune)

- **vlm_finetune vs llm_finetune**: $t = 0.8059$, $p = 5.0490e-01$ (n.s. (not significant))
- **vlm_finetune vs llm_zero_shot**: $t = 17.4461$, $p = 3.2694e-03$ (**)
- **vlm_finetune vs majority_classifier**: $t = 16.6539$, $p = 3.5861e-03$ (**)
- **vlm_finetune vs tfidf_linear_svm**: $t = 1.1497$, $p = 3.6919e-01$ (n.s. (not significant))
- **vlm_finetune vs tfidf_logistic_regression**: $t = 3.8653$, $p = 6.0883e-02$ (n.s. (not significant))
- **vlm_finetune vs vlm_zero_shot**: $t = 81.1725$, $p = 1.5173e-04$ (***)
