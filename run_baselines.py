"""
=============================================================================
CHƯƠNG TRÌNH CHẠY BASELINES CHO BÀI TOÁN PHÁT HIỆN YÊU CẦU MƠ HỒ (SRS AMBIGUITY)
=============================================================================
Mục đích:
  Đánh giá 3 mô hình Baseline truyền thống làm cơ sở so sánh với LLM và VLM:
    1. Majority Classifier (Zero-Rule): Luôn đoán nhãn đa số 'ok' (Mức sàn tối thiểu)
    2. TF-IDF + Logistic Regression: Phân loại tuyến tính thống kê tần suất từ
    3. TF-IDF + Linear SVM (có bọc hiệu chỉnh xác suất Platt Scaling): Mô hình ML cổ điển mạnh nhất

Thiết kế thực nghiệm chuẩn mực:
  - Lặp lại trên 3 Random Seeds: [42, 123, 456]
  - Cùng bộ chia phân tầng (Stratified 80/10/10 Split: 1749 Train / 219 Val / 219 Test)
  - Cùng hàm tiền xử lý clean_requirement chuẩn hóa khoảng trắng
  - Chống rò rỉ dữ liệu (No Data Leakage): TF-IDF chỉ fit trên Train, chỉ transform Test
  - Tính toán ROC-AUC và PR-AUC trên xác suất thực tế P(defect)
=============================================================================
"""

import os
import re
import sys
import json
import argparse

# Thiết lập encoding UTF-8 để in tiếng Việt chuẩn trên Terminal Windows
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

# Các công cụ chia dữ liệu, trích xuất đặc trưng và mô hình từ scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier

# Các độ đo đánh giá hiệu năng (Metrics)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc as calc_auc, matthews_corrcoef, confusion_matrix
)

# =============================================================================
# 1. TIỀN XỬ LÝ DỮ LIỆU (Giống hệt 100% trong notebook của LLM và VLM)
# =============================================================================
MAX_REQ_CHARS = 800

def clean_requirement(text: str) -> str:
    """
    Chuẩn hóa văn bản yêu cầu phần mềm:
    - Thay thế các khoảng trắng thừa, tab, xuống dòng liên tiếp bằng 1 khoảng trắng duy nhất.
    - Cắt tỉa khoảng trắng ở hai đầu (strip).
    - Cắt ngắn nếu độ dài vượt quá MAX_REQ_CHARS (800 ký tự) để đồng nhất với định dạng ảnh của VLM.
    """
    text = re.sub(r'\s+', ' ', str(text)).strip()
    if len(text) > MAX_REQ_CHARS:
        text = text[:MAX_REQ_CHARS] + "..."
    return text

# =============================================================================
# 2. HÀM TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)
# =============================================================================
def evaluate_predictions(y_true_bin, y_pred_bin, y_scores):
    """
    Tính toán đầy đủ các chỉ số hiệu năng trên tập kiểm thử (Test Set):
      - y_true_bin: Nhãn thực tế (0 = ok, 1 = defect)
      - y_pred_bin: Nhãn dự đoán nhị phân (0 hoặc 1)
      - y_scores  : Xác suất liên tục mô hình dự đoán nhãn defect (P(defect) thuộc [0, 1])
    """
    # 1. Accuracy: Tỷ lệ đoán đúng tổng thể
    acc = accuracy_score(y_true_bin, y_pred_bin)
    
    # 2. Balanced Accuracy: Trung bình Recall của 2 lớp (rất quan trọng khi dữ liệu lệch 70/30)
    bacc = balanced_accuracy_score(y_true_bin, y_pred_bin)
    
    # 3. Precision, Recall, F1 cho lớp 'defect' (lớp lỗi thiểu số quan trọng nhất)
    prec_d = precision_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    rec_d = recall_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    f1_d = f1_score(y_true_bin, y_pred_bin, pos_label=1, zero_division=0)
    
    # 4. Precision, Recall, F1 cho lớp 'ok' (lớp hợp lệ đa số)
    prec_ok = precision_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    rec_ok = recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    f1_ok = f1_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)
    
    # 5. F1-Macro (Trung bình không trọng số của F1-ok và F1-defect) và F1-Weighted
    f1_macro = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    f1_weight = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    
    # 6. Matthews Correlation Coefficient (MCC): Thước đo chuẩn vàng đánh giá chất lượng phân loại nhị phân
    mcc = matthews_corrcoef(y_true_bin, y_pred_bin)

    # 7. ROC-AUC: Diện tích dưới đường cong ROC tính trên xác suất P(defect)
    try:
        roc_auc = roc_auc_score(y_true_bin, y_scores)
    except:
        roc_auc = float('nan')

    # 8. PR-AUC: Diện tích dưới đường cong Precision-Recall (đặc biệt chuẩn xác cho dữ liệu mất cân bằng)
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

# =============================================================================
# 3. QUY TRÌNH THỰC NGHIỆM CHÍNH (TRAINING & EVALUATION PIPELINE)
# =============================================================================
def run_experiment(dataset_path="datasetRE.csv", seeds=[42, 123, 456]):
    # Kiểm tra tự động file dataset (hỗ trợ cả QuRE.csv và datasetRE.csv)
    if not os.path.exists(dataset_path):
        if os.path.exists("QuRE.csv"):
            dataset_path = "QuRE.csv"
        else:
            print(f"⚠️  Không tìm thấy file {dataset_path} hoặc QuRE.csv tại thư mục hiện tại.")
            print("    Vui lòng copy file dataset vào đây và chạy lại lệnh: python run_baselines.py")
            return

    print(f"📂 Đang tải dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Bước 1: Làm sạch văn bản yêu cầu bằng hàm chuẩn
    df['clean_text'] = df['requirement'].apply(clean_requirement)
    
    # Bước 2: Chuyển đổi nhãn dạng chuỗi sang nhị phân: 'ok' -> 0, 'defect' -> 1
    label2int = {'ok': 0, 'defect': 1}
    df['target'] = df['defect'].map(label2int)

    # Khởi tạo 3 mô hình Baseline:
    models_to_test = {
        # Baseline 1: DummyClassifier luôn đoán nhãn xuất hiện nhiều nhất ('ok')
        'majority_classifier': lambda: DummyClassifier(strategy='most_frequent'),
        
        # Baseline 2: Hồi quy Logistic trên ma trận đặc trưng từ vựng TF-IDF
        'tfidf_logistic_regression': lambda: LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        
        # Baseline 3: Linear SVM bọc bằng CalibratedClassifierCV (Platt Scaling)
        # Giúp SVM biến đổi khoảng cách siêu phẳng thành xác suất thực tế [0, 1] để tính ROC-AUC
        'tfidf_linear_svm': lambda: CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=2000), cv=3),
    }

    all_results = {name: [] for name in models_to_test}

    # =========================================================================
    # LẶP QUA TỪNG RANDOM SEED ĐỂ ĐẢM BẢO TÍNH ĐỘC LẬP & TÁI LẬP (REPRODUCIBILITY)
    # =========================================================================
    for seed in seeds:
        print(f"\n{'='*60}\n🎲 BẮT ĐẦU CHẠY VỚI SEED: {seed}\n{'='*60}")
        
        # Bước 3: Phân chia tập dữ liệu 80/10/10 có Stratified (giữ nguyên tỷ lệ 70% ok : 30% defect)
        # - Tách 10% ra làm Test Set (219 mẫu)
        df_trainval, df_test = train_test_split(df, test_size=0.10, random_state=seed, stratify=df['target'])
        # - Trong 90% còn lại, lấy 11.11% làm Validation Set (0.9 * 0.1111 = 0.10 => 219 mẫu), 80% còn lại làm Train Set (1749 mẫu)
        df_train, df_val = train_test_split(df_trainval, test_size=0.1111, random_state=seed, stratify=df_trainval['target'])

        X_train_raw = df_train['clean_text'].tolist()
        y_train = df_train['target'].values
        X_test_raw = df_test['clean_text'].tolist()
        y_test = df_test['target'].values

        # Bước 4: Trích xuất đặc trưng TF-IDF (1-gram và 2-gram, lấy tối đa 5000 từ/cụm từ quan trọng nhất)
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), sublinear_tf=True)
        
        # QUAN TRỌNG (CHỐNG RÒ RỈ DỮ LIỆU):
        # - fit_transform: Chỉ học từ vựng và tần suất trên tập TRAIN
        X_train_tfidf = tfidf.fit_transform(X_train_raw)
        # - transform: Tập TEST chỉ biến đổi theo từ vựng đã học từ TRAIN (tuyệt đối không học từ TEST)
        X_test_tfidf = tfidf.transform(X_test_raw)

        # Bước 5: Huấn luyện và kiểm thử từng mô hình
        for model_name, model_fn in models_to_test.items():
            clf = model_fn()
            
            if model_name == 'majority_classifier':
                # Majority chỉ cần đếm tần suất nhãn trên tập Train
                clf.fit(X_train_raw, y_train)
                y_pred = clf.predict(X_test_raw)
                # Xác suất đoán nhãn defect (luôn bằng 0 vì lớp đa số là 'ok')
                majority_class = clf.classes_[np.argmax(clf.class_prior_)]
                y_scores = np.full(len(y_test), 1.0 if majority_class == 1 else 0.0)
            else:
                # Huấn luyện mô hình (LR hoặc SVM) trên ma trận TF-IDF của tập Train
                clf.fit(X_train_tfidf, y_train)
                # Dự đoán nhãn phân loại (0 hoặc 1) trên tập Test
                y_pred = clf.predict(X_test_tfidf)
                # Dự đoán xác suất liên tục P(defect) [cột index 1] để vẽ đường cong ROC/PR
                y_scores = clf.predict_proba(X_test_tfidf)[:, 1]

            # Tính toán toàn bộ các metric
            metrics = evaluate_predictions(y_test, y_pred, y_scores)
            metrics['seed'] = seed
            metrics['model'] = model_name
            all_results[model_name].append(metrics)

            # Lưu file kết quả JSON riêng lẻ cho từng seed (ví dụ: metrics_tfidf_logistic_regression_seed_42.json)
            fname = f"metrics_{model_name}_seed_{seed}.json"
            with open(fname, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2)

            print(f"[{model_name.upper()}] Acc: {metrics['accuracy']:.4f} | F1-macro: {metrics['f1_macro']:.4f} | ROC-AUC: {metrics['roc_auc']} | PR-AUC: {metrics['pr_auc']}")

    # Bước 6: Lưu tổng hợp kết quả của cả 3 mô hình trên cả 3 seeds vào 1 file duy nhất
    with open("baseline_results_all.json", 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)

    # =========================================================================
    # IN BẢNG TỔNG HỢP KHOA HỌC: TRUNG BÌNH ± ĐỘ LỆCH CHUẨN (MEAN ± STD)
    # =========================================================================
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
    parser.add_argument("--dataset", default="datasetRE.csv", help="Đường dẫn tới file datasetRE.csv hoặc QuRE.csv")
    args = parser.parse_args()
    run_experiment(dataset_path=args.dataset)
