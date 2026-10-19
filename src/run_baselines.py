"""
=============================================================================
BASELINE EXPERIMENTS FOR SOFTWARE REQUIREMENTS SPECIFICATION (SRS) AMBIGUITY
=============================================================================
Objective:
  Evaluate 3 classical baseline models for comparison against LLMs and VLMs:
    1. Majority Classifier (Zero-Rule): Predicts the majority class 'ok' (Empirical lower bound)
    2. TF-IDF + Logistic Regression: Linear statistical classification based on n-gram frequencies
    3. TF-IDF + Linear SVM (calibrated with Platt Scaling): Strong classical ML baseline

Experimental Rigor:
  - Repeated across 3 independent random seeds: [42, 123, 456]
  - Stratified 80/10/10 split: 1,749 Train / 219 Val / 219 Test
  - Standardized clean_requirement preprocessing with whitespace normalization
  - Strict prevention of data leakage: TF-IDF fitted solely on Train, transformed on Test
  - Computes ROC-AUC and PR-AUC from posterior defect probabilities P(defect)
=============================================================================
"""

import os
import re
import sys
import json
import argparse

sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc as calc_auc, matthews_corrcoef, confusion_matrix
)

# =============================================================================
# 1. DATA PREPROCESSING (Standardized across LLM and VLM pipelines)
# =============================================================================
MAX_REQ_CHARS = 800

def clean_requirement(text: str) -> str:
    """
    Standardize requirement text:
    - Replace consecutive whitespaces, tabs, and newlines with a single space.
    - Strip leading and trailing whitespace.
    - Truncate text if length exceeds MAX_REQ_CHARS (800 characters) for visual parity with VLMs.
    """
    text = re.sub(r'\s+', ' ', str(text)).strip()
    if len(text) > MAX_REQ_CHARS:
        text = text[:MAX_REQ_CHARS] + "..."
    return text

# =============================================================================
# 2. EVALUATION METRICS
# =============================================================================
def evaluate_predictions(y_true_bin, y_pred_bin, y_scores):
    """
    Compute comprehensive classification metrics on the hold-out test set:
      - y_true_bin: Ground-truth binary labels (0 = ok, 1 = defect)
      - y_pred_bin: Predicted binary labels (0 or 1)
      - y_scores  : Continuous posterior probabilities for defect P(defect) in [0, 1]
    """
    # 1. Overall Accuracy
    acc = accuracy_score(y_true_bin, y_pred_bin)
    
    # 2. Balanced Accuracy: Arithmetic mean of sensitivity and specificity (vital for 70/30 imbalance)
    bacc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    
    # 3. Precision, Recall, F1 for minority 'defect' class
    prec_d = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    rec_d = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    f1_d = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    
    # 4. Precision, Recall, F1 for majority 'ok' class
    prec_ok = precision_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    rec_ok = recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    f1_ok = f1_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    
    # 5. Macro-averaged and Weighted F1-scores
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    f1_weight = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    
    # 6. Matthews Correlation Coefficient (MCC): Gold standard for imbalanced binary classification
    mcc = matthews_corrcoef(y_true_bin, y_pred_bin)

    # 7. ROC-AUC: Area under the ROC curve computed from continuous probabilities
    try:
        roc_auc = roc_auc_score(y_true_bin, y_scores)
    except Exception:
        roc_auc = float('nan')

    # 8. PR-AUC: Area under the Precision-Recall curve
    try:
        pr_p, pr_r, _ = precision_recall_curve(y_true_bin, y_scores)
        pr_auc = calc_auc(pr_r, pr_p)
    except Exception:
        pr_auc = float('nan')

    return {
        'accuracy': float(acc),
        'balanced_accuracy': float(bacc),
        'precision_defect': float(prec_d),
        'recall_defect': float(rec_d),
        'f1_defect': float(f1_d),
        'precision_ok': float(prec_ok),
        'recall_ok': float(rec_ok),
        'f1_ok': float(f1_ok),
        'f1_macro': float(f1_macro),
        'f1_weighted': float(f1_weight),
        'mcc': float(mcc),
        'roc_auc': float(roc_auc) if not np.isnan(roc_auc) else None,
        'pr_auc': float(pr_auc) if not np.isnan(pr_auc) else None,
    }

# =============================================================================
# 3. TRAINING & EVALUATION PIPELINE
# =============================================================================
def run_experiment(dataset_path="data/QuRE.csv", seeds=None, output_dir="results", save_predictions=True):
    if seeds is None:
        seeds = [42, 123, 456]

    # Resolve candidate dataset paths
    candidate_paths = [
        dataset_path,
        "data/QuRE.csv",
        "data/datasetRE.csv",
        "../data/QuRE.csv",
        "../data/datasetRE.csv",
        "QuRE.csv",
        "datasetRE.csv"
    ]
    resolved_path = None
    for p in candidate_paths:
        if p and os.path.exists(p):
            resolved_path = p
            break

    if not resolved_path:
        print("⚠️  Dataset file not found (QuRE.csv / datasetRE.csv).")
        print("    Please check the data/ directory or current path.")
        return

    print(f"Loading dataset: {resolved_path}")
    df = pd.read_csv(resolved_path)
    
    # Step 1: Text cleaning
    df['clean_text'] = df['requirement'].apply(clean_requirement)
    
    # Step 2: Convert target labels to binary: 'ok' -> 0, 'defect' -> 1
    label2int = {'ok': 0, 'defect': 1}
    df['target'] = df['defect'].map(label2int)

    # Initialize baseline models
    models_to_test = {
        'majority_classifier': lambda: DummyClassifier(strategy='most_frequent'),
        'tfidf_logistic_regression': lambda seed: LogisticRegression(C=1.0, max_iter=1000, random_state=seed),
        'tfidf_linear_svm': lambda seed: CalibratedClassifierCV(LinearSVC(C=1.0, random_state=seed, max_iter=2000), cv=3),
    }

    all_results = {name: [] for name in models_to_test}

    # Iterate over independent random seeds
    for seed in seeds:
        print(f"\n{'='*60}\nRUNNING BASELINES FOR SEED: {seed}\n{'='*60}")
        
        # Stratified 80/10/10 split
        df_trainval, df_test = train_test_split(df, test_size=0.10, random_state=seed, stratify=df['target'])
        df_train, df_val = train_test_split(df_trainval, test_size=0.1111, random_state=seed, stratify=df_trainval['target'])

        X_train_raw = df_train['clean_text'].tolist()
        y_train = df_train['target'].values
        X_test_raw = df_test['clean_text'].tolist()
        y_test = df_test['target'].values

        # TF-IDF Feature extraction (unigrams + bigrams, up to 5,000 features)
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        
        # Fit strictly on train set to prevent data leakage
        X_train_tfidf = tfidf.fit_transform(X_train_raw)
        X_test_tfidf = tfidf.transform(X_test_raw)

        # Train and evaluate models
        for model_name, model_fn in models_to_test.items():
            clf = model_fn(seed) if model_name != 'majority_classifier' else model_fn()
            
            if model_name == 'majority_classifier':
                clf.fit(X_train_raw, y_train)
                y_pred = clf.predict(X_test_raw)
                majority_class = clf.classes_[np.argmax(clf.class_prior_)]
                y_scores = np.full(len(y_test), 1.0 if majority_class == 1 else 0.0)
            else:
                clf.fit(X_train_tfidf, y_train)
                y_pred = clf.predict(X_test_tfidf)
                y_scores = clf.predict_proba(X_test_tfidf)[:, 1]

            metrics = evaluate_predictions(y_test, y_pred, y_scores)
            metrics['seed'] = seed
            metrics['model'] = model_name
            all_results[model_name].append(metrics)

            out_metrics_dir = os.path.join(output_dir, "metrics")
            out_predictions_dir = os.path.join(output_dir, "predictions")
            os.makedirs(out_metrics_dir, exist_ok=True)
            if save_predictions:
                os.makedirs(out_predictions_dir, exist_ok=True)

            fname = os.path.join(out_metrics_dir, f"metrics_{model_name}_seed_{seed}.json")
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

            if save_predictions:
                predictions = df_test[['id', 'requirement', 'defect', 'weak_word']].copy()
                predictions['length'] = predictions['requirement'].astype(str).str.len()
                predictions['predicted'] = np.where(y_pred == 1, 'defect', 'ok')
                predictions['p_defect'] = y_scores
                predictions['correct'] = predictions['predicted'].eq(predictions['defect'])
                prediction_path = os.path.join(
                    out_predictions_dir,
                    f"predictions_{model_name}_seed_{seed}.csv"
                )
                predictions.to_csv(prediction_path, index=False, encoding='utf-8')

            print(f"[{model_name.upper()}] Acc: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f} | ROC-AUC: {metrics['roc_auc']} | PR-AUC: {metrics['pr_auc']}")

    baseline_all_path = os.path.join(out_metrics_dir, "baseline_results_all.json")
    with open(baseline_all_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print("\n" + "="*80)
    print("           BASELINE RESULTS SUMMARY ACROSS 3 SEEDS (Mean ± Std)")
    print("="*80)
    header = f"{'Model':<28} | {'Accuracy':<14} | {'F1-Macro':<14} | {'ROC-AUC':<14} | {'PR-AUC':<14}"
    print(header)
    print("-" * 88)
    for model_name, runs in all_results.items():
        accs = [r['accuracy'] for r in runs]
        f1s = [r['f1_macro'] for r in runs]
        rocs = [r['roc_auc'] for r in runs if r['roc_auc'] is not None]
        prs = [r['pr_auc'] for r in runs if r['pr_auc'] is not None]

        acc_str = f"{np.mean(accs):.4f}±{np.std(accs):.4f}"
        f1_str = f"{np.mean(f1s):.4f}±{np.std(f1s):.4f}"
        roc_str = f"{np.mean(rocs):.4f}±{np.std(rocs):.4f}" if rocs else "N/A"
        pr_str = f"{np.mean(prs):.4f}±{np.std(prs):.4f}" if prs else "N/A"
        print(f"{model_name:<28} | {acc_str:<14} | {f1_str:<14} | {roc_str:<14} | {pr_str:<14}")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/QuRE.csv", help="Path to dataset file (QuRE.csv or datasetRE.csv)")
    args = parser.parse_args()
    run_experiment(dataset_path=args.dataset)
