"""
=============================================================================
RUN BASELINES FOR SRS AMBIGUITY DETECTION
Evaluates:
  1. Majority Class Classifier (Zero-Rule)
  2. TF-IDF + Logistic Regression
  3. TF-IDF + Linear Support Vector Machine (Linear SVM with probability calibration)

Across 3 random seeds: [42, 123, 456]
Using the exact same Stratified 80/10/10 Split & clean_requirement preprocessing.
=============================================================================
"""

import os
import re
import json
import argparse
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

# --- Preprocessing identical to LLM/VLM models ---
MAX_REQ_CHARS = 800

def clean_requirement(text: str) -> str:
    """Chuẩn hóa whitespace và truncate nếu quá dài."""
    text = re.sub(r'\s+', ' ', str(text)).strip()
    if len(text) > MAX_REQ_CHARS:
        text = text[:MAX_REQ_CHARS] + "..."
    return text

def evaluate_predictions(y_true_bin, y_pred_bin, y_scores):
    acc = accuracy_score(y_true_bin, y_pred_bin)
    bacc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    prec_d = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    rec_d = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    f1_d = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    prec_ok = precision_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    rec_ok = recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    f1_ok = f1_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    f1_weight = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    mcc = matthews_corrcoef(y_true_bin, y_pred_bin)

    try:
        roc_auc = roc_auc_score(y_true_bin, y_scores)
    except:
        roc_auc = float('nan')

    try:
        pr_p, pr_r, _ = precision_recall_curve(y_true_bin, y_scores)
        pr_auc = calc_auc(pr_r, pr_p)
    except:
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

def run_experiment(dataset_path="datasetRE.csv", seeds=[42, 123, 456]):
    if not os.path.exists(dataset_path):
        print(f"⚠️  Không tìm thấy file {dataset_path} tại thư mục hiện tại.")
        print("    Vui lòng copy file datasetRE.csv vào đây và chạy lại lệnh: python run_baselines.py")
        return

    print(f"📂 Đang tải dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df['clean_text'] = df['requirement'].apply(clean_requirement)
    label2int = {'ok': 0, 'defect': 1}
    df['target'] = df['defect'].map(label2int)

    models_to_test = {
        'majority_classifier': lambda: DummyClassifier(strategy='most_frequent'),
        'tfidf_logistic_regression': lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'tfidf_linear_svm': lambda: CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=2000), cv=3),
    }

    all_results = {name: [] for name in models_to_test}

    for seed in seeds:
        print(f"\n{'='*60}\n🎲 RUNNING WITH SEED: {seed}\n{'='*60}")
        # Stratified 80/10/10 split
        df_trainval, df_test = train_test_split(df, test_size=0.10, random_state=seed, stratify=df['target'])
        df_train, df_val = train_test_split(df_trainval, test_size=0.1111, random_state=seed, stratify=df_trainval['target'])

        X_train_raw = df_train['clean_text'].tolist()
        y_train = df_train['target'].values
        X_test_raw = df_test['clean_text'].tolist()
        y_test = df_test['target'].values

        # TF-IDF Vectorizer
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        X_train_tfidf = tfidf.fit_transform(X_train_raw)
        X_test_tfidf = tfidf.transform(X_test_raw)

        for model_name, model_fn in models_to_test.items():
            clf = model_fn()
            if model_name == 'majority_classifier':
                clf.fit(X_train_raw, y_train)
                y_pred = clf.predict(X_test_raw)
                # Majority prob proxy
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

            # Save individual seed metrics
            fname = f"metrics_{model_name}_seed_{seed}.json"
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

            print(f"[{model_name.upper()}] Acc: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f} | ROC-AUC: {metrics['roc_auc']} | PR-AUC: {metrics['pr_auc']}")

    # Save summary
    with open("baseline_results_all.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    # Print summary table Mean ± Std
    print("\n" + "="*80)
    print("           TỔNG HỢP KẾT QUẢ BASELINES TRÊN 3 SEEDS (Mean ± Std)")
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
    parser.add_argument("--dataset", default="datasetRE.csv", help="Đường dẫn tới file datasetRE.csv")
    args = parser.parse_args()
    run_experiment(dataset_path=args.dataset)
