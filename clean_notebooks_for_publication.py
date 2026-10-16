"""
Script to refine Jupyter Notebooks for formal academic publication:
  1. Strips informal code comments from Python code cells.
  2. Replaces informal/casual markdown descriptions with IEEE/ACM-grade academic English.
  3. Preserves all cell execution counts, outputs, widgets, and metadata intact.
"""

import json
import re
import io
import tokenize

def remove_comments_from_code(source: str) -> str:
    """Removes comment lines and trailing comments while preserving strings and magic commands."""
    lines = source.split('\n')
    cleaned_lines = []
    
    for line in lines:
        stripped = line.strip()
        # Keep shell commands / magics untouched
        if stripped.startswith('!') or stripped.startswith('%'):
            cleaned_lines.append(line)
            continue
        # Drop pure comment lines
        if stripped.startswith('#') and not stripped.startswith('#!'):
            continue
        
        # If line contains code and a comment, strip trailing comment safely
        if '#' in line:
            try:
                io_obj = io.StringIO(line)
                tokens = list(tokenize.generate_tokens(io_obj.readline))
                for tok in tokens:
                    if tok.type == tokenize.COMMENT:
                        line = line[:tok.start[1]].rstrip()
                        break
            except Exception:
                pass
        
        if line.strip():
            cleaned_lines.append(line)
        elif cleaned_lines and cleaned_lines[-1] != '':
            cleaned_lines.append('')
            
    # Remove leading/trailing empty lines
    while cleaned_lines and cleaned_lines[0] == '':
        cleaned_lines.pop(0)
    while cleaned_lines and cleaned_lines[-1] == '':
        cleaned_lines.pop()
        
    return '\n'.join(cleaned_lines) + '\n'


# =============================================================================
# ACADEMIC MARKDOWN SPECIFICATIONS (IEEE / ACM / Camera-Ready Standard)
# =============================================================================

MD_LLM_ZERO_SHOT = {
    0: (
        "# Zero-Shot Ambiguity Detection in Software Requirements Specifications (SRS) using Qwen2.5-3B-Instruct\n\n"
        "**Abstract & Overview:**\n"
        "Requirements ambiguity is a critical precursor to software defects and implementation failures. "
        "This experimental notebook assesses the out-of-the-box, zero-shot classification performance of the "
        "`Qwen/Qwen2.5-3B-Instruct` Large Language Model (LLM) on the benchmark QuRE dataset (2,187 industrial requirements).\n\n"
        "The model is evaluated on local GPU infrastructure in half-precision (FP16). To ensure rigorous scientific evaluation, "
        "continuous posterior probabilities $P(\\text{defect}) \\in [0, 1]$ are extracted directly from the next-token logits over "
        "the binary target vocabulary (`ok` vs. `defect`), enabling authentic Receiver Operating Characteristic (ROC-AUC) and "
        "Precision-Recall (PR-AUC) analysis without reliance on discrete label proxies."
    ),
    1: "## 1. Environment and Dependencies\nInitializes the runtime environment and installs required libraries including PyTorch, HuggingFace Transformers, Accelerate, and Scikit-Learn.",
    3: "## 2. Dataset Ingestion and Standardization\nLoads the empirical QuRE dataset (2,187 requirements). Text preprocessing is standardized via whitespace normalization and a context boundary constraint (800 characters) to ensure strict parity across text and visual modality experiments.",
    5: "## 3. Stratified Dataset Partitioning\nApplies stratified dataset partitioning (80% Train, 10% Validation, 10% Test) to preserve class balance (~70.2% `ok`, ~29.8% `defect`). Evaluation is performed exclusively on the 10% hold-out test set across configurable random seeds (42, 123, 456).",
    7: "## 4. Prompt Engineering and Chat Template Formulation\nFormulates the classification instruction adhering to the Qwen2.5 ChatML format. The model is constrained to binary prediction (`ok` vs. `defect`) given an unannotated requirement input.",
    9: "## 5. Model Initialization and Precision Configuration\nLoads `Qwen/Qwen2.5-3B-Instruct` in native FP16 precision with automatic hardware device placement for reproducible local GPU inference.",
    11: "## 6. Forward Pass and Logits-Based Probability Extraction\nExecutes forward passes across all test instances. Next-token logits corresponding to target tokens are normalized via softmax over the candidate pair to extract continuous probabilities $P(\\text{defect}) \\in [0, 1]$, enabling rigorous ROC and PR curve derivation.",
    13: "## 7. Formal Quantitative Performance Evaluation\nComputes comprehensive classification metrics: Accuracy, Balanced Accuracy, Precision, Recall, Class-specific F1-scores, Macro F1, Matthews Correlation Coefficient (MCC), Standard ROC-AUC, and PR-AUC. Generates formal publication-grade evaluation dashboards.",
    15: "## 8. Failure Mode and Error Analysis\nIdentifies False Positive (Type I) and False Negative (Type II) misclassifications to analyze linguistic patterns and syntactic structures that induce zero-shot errors.",
    17: "## 9. Artifact Serialization and Multi-Seed Results Export\nSerializes hold-out test predictions and formal evaluation metrics into structured CSV and JSON files indexed by random seed to facilitate multi-seed statistical aggregation.",
    19: "## 10. Qualitative Inference Routine\nProvides an interactive verification routine for inspecting model predictions and posterior probability distributions on arbitrary SRS statements."
}

MD_VLM_ZERO_SHOT = {
    0: (
        "# Zero-Shot Ambiguity Detection in Software Requirements Specifications (SRS) using Qwen2.5-VL-3B-Instruct\n\n"
        "**Abstract & Overview:**\n"
        "This experimental notebook evaluates the zero-shot capabilities of the `Qwen/Qwen2.5-VL-3B-Instruct` Vision-Language Model (VLM) "
        "on the QuRE benchmark dataset (2,187 industrial requirements). The experiment investigates whether multimodal vision-language architectures "
        "exhibit competitive zero-shot comprehension on technical text specifications prior to domain adaptation.\n\n"
        "Inference is executed locally on GPU hardware. Posterior probabilities $P(\\text{defect})$ are derived directly from next-token logits "
        "to compute true Standard ROC-AUC and PR-AUC metrics, maintaining methodological consistency across all comparative benchmarks."
    ),
    1: "## 1. Environment and Dependencies\nInitializes the execution environment and installs core dependencies for vision-language processing and evaluation.",
    3: "## 2. Dataset Ingestion and Preprocessing\nLoads and inspects the QuRE dataset with standardized text cleaning to eliminate preprocessing confounders.",
    5: "## 3. Stratified Dataset Partitioning\nPartitions the dataset using stratified 80/10/10 train/validation/test splits across configurable random seeds (42, 123, 456).",
    7: "## 4. Prompt Engineering and ChatML Schema\nStructures input prompts conforming to the multimodal chat template for binary ambiguity classification.",
    9: "## 5. Vision-Language Model Initialization\nLoads `Qwen/Qwen2.5-VL-3B-Instruct` and its multimodal processor in FP16 precision on local GPU memory.",
    11: "## 6. Forward Pass and Continuous Probability Extraction\nPerforms forward inference over the hold-out test set, extracting softmax-normalized probabilities $P(\\text{defect})$ from model logits at the decision step.",
    13: "## 7. Quantitative Performance Evaluation and Calibration\nGenerates comprehensive evaluation metrics and publication-ready diagnostic curves (Confusion Matrix, ROC, PR).",
    15: "## 8. Qualitative Error Diagnostics\nCategorizes Type I and Type II errors to evaluate semantic boundary challenges encountered by zero-shot VLM inference.",
    17: "## 9. Artifact Serialization and Export\nExports test predictions and evaluation metrics to structured JSON and CSV formats indexed by random seed.",
    19: "## 10. Qualitative Inference Routine\nInteractive test harness for evaluating model behavior on arbitrary requirement inputs."
}

MD_LLM_FINETUNE = {
    0: (
        "# Parameter-Efficient Fine-Tuning of Qwen2.5-3B-Instruct for SRS Ambiguity Detection\n\n"
        "**Abstract & Overview:**\n"
        "This study implements domain adaptation of `Qwen/Qwen2.5-3B-Instruct` for automated ambiguity detection in Software Requirements Specifications (SRS). "
        "Using 4-bit Quantized Low-Rank Adaptation (QLoRA), the model is fine-tuned on the benchmark QuRE dataset (2,187 requirements).\n\n"
        "The training pipeline uses cross-entropy loss computation restricted exclusively to response tokens (`ok` / `defect`) via attention masking. "
        "Evaluation across multiple independent random seeds (42, 123, 456) measures stability, generalization, and statistical significance against classical ML baselines."
    ),
    1: "## 1. Environment Setup and Dependency Installation\nInstalls HuggingFace Transformers, PEFT, TRL, BitsAndBytes, and Scikit-Learn for quantized parameter-efficient fine-tuning.",
    3: "## 2. Dataset Ingestion and Standardization\nLoads the empirical QuRE dataset, applying standardized whitespace normalization and context clipping (`MAX_REQ_CHARS = 800`).",
    5: "## 3. Stratified Dataset Partitioning\nPartitions data into stratified Train (80%, 1,749 samples), Validation (10%, 219 samples), and Test (10%, 219 samples) subsets under configurable random seeds.",
    7: "## 4. Prompt Engineering and Supervised Fine-Tuning Format\nConstructs ChatML instruction prompts pairing requirements with target supervisory labels (`ok` or `defect`).",
    9: "## 5. Model Loading with 4-bit QLoRA Configuration\nLoads `Qwen/Qwen2.5-3B-Instruct` in NF4 4-bit quantization with LoRA adapters targetting all linear projection layers ($r=16, \\alpha=32$).",
    11: "## 6. Dataset Tokenization and Supervised Loss Masking\nTokenizes input sequences and masks prompt tokens with label value `-100` so cross-entropy loss is computed exclusively on the target classification token.",
    13: "## 7. Supervised Fine-Tuning Execution\nExecutes SFTTrainer with cosine learning rate scheduling, mixed precision (FP16/BF16), and evaluation-loss checkpointing.",
    15: "## 8. Training Dynamics and Convergence Visualization\nPlots training and validation loss trajectories across optimization steps to verify convergence and monitor potential overfitting.",
    17: "## 9. Hold-Out Test Evaluation and Probability Logits Extraction\nPerforms forward inference on the hold-out test set to extract candidate logits and calibrated continuous probabilities $P(\\text{defect})$.",
    19: "## 10. Quantitative Performance Metrics & Calibration\nEvaluates Accuracy, Balanced Accuracy, Precision, Recall, Class-specific F1, Macro F1, MCC, Standard ROC-AUC, and PR-AUC.",
    21: "## 11. Qualitative Error Inspection\nPerforms automated error isolation on False Positives and False Negatives to evaluate residual ambiguity patterns.",
    23: "## 12. Model Artifact Serialization & Multi-Seed Results Export\nSaves trained LoRA adapter weights, configuration files, and evaluation metrics serialized with seed-specific identifiers.",
    25: "## 13. Qualitative Demonstration Routine\nInteractive interface for verifying fine-tuned model inference on arbitrary requirement statements."
}

MD_VLM_FINETUNE = {
    0: (
        "# Parameter-Efficient Fine-Tuning of Qwen2.5-VL-3B-Instruct for SRS Ambiguity Detection\n\n"
        "**Abstract & Overview:**\n"
        "This experimental notebook investigates domain-adapted Vision-Language Models (VLMs) for Software Requirements Specifications (SRS) ambiguity classification. "
        "We adapt `Qwen/Qwen2.5-VL-3B-Instruct` using 4-bit QLoRA on the QuRE dataset (2,187 requirements) across three random seeds (42, 123, 456).\n\n"
        "We evaluate whether multimodal model fine-tuning yields superior representation and sensitivity over text-only LLMs on software engineering artifacts. "
        "Standard ROC-AUC and PR-AUC are computed directly from continuous next-token logits."
    ),
    1: "## 1. Environment Setup and Dependency Installation\nInstalls HuggingFace Transformers, PEFT, TRL, BitsAndBytes, and Scikit-Learn configured for multimodal QLoRA fine-tuning.",
    3: "## 2. Dataset Ingestion and Standardization\nLoads and inspects the QuRE dataset with standardized text cleaning to ensure experimental parity.",
    5: "## 3. Stratified Dataset Partitioning\nConstructs stratified 80/10/10 splits across configurable random seeds (42, 123, 456) preserving the ~70.2% `ok` : ~29.8% `defect` distribution.",
    7: "## 4. Prompt Engineering and Chat Template Formulation\nConfigures instruction templates mapping requirement statements to single-token supervisory targets.",
    9: "## 5. Vision-Language Model Initialization with 4-bit QLoRA\nInitializes `Qwen/Qwen2.5-VL-3B-Instruct` in 4-bit NormalFloat quantization with LoRA adapter configuration ($r=16, \\alpha=32$).",
    11: "## 6. Dataset Tokenization and Loss Masking\nTokenizes multimodal input structures, masking non-target tokens to restrict gradient backpropagation to the classification token.",
    13: "## 7. Model Optimization and Training Protocol\nExecutes SFTTrainer with gradient accumulation and evaluation loss monitoring to select the optimal model checkpoint.",
    15: "## 8. Optimization Trajectory and Convergence Analysis\nVisualizes loss convergence across training epochs, assessing stability and generalization.",
    17: "## 9. Hold-Out Evaluation and Probability Extraction\nExecutes forward evaluation over the isolated test set to extract candidate logits and continuous posterior probabilities $P(\\text{defect})$.",
    19: "## 10. Quantitative Performance Metrics & Diagnostic Visualizations\nComputes formal classification metrics (Accuracy, Balanced Accuracy, Macro F1, MCC, Standard ROC-AUC, PR-AUC) and visualization curves.",
    21: "## 11. Residual Error Analysis\nIsolates False Positive and False Negative predictions for qualitative linguistic inspection.",
    23: "## 12. Adapter Serialization and Results Export\nExports trained LoRA adapters, evaluation metrics, and prediction CSVs with seed indexing to support statistical significance aggregation.",
    25: "## 13. Qualitative Demonstration Routine\nInteractive test harness for evaluating model predictions and posterior probability distributions on arbitrary requirement inputs."
}

NOTEBOOK_SPECS = {
    'SRS_LLM_Zero_Shot.ipynb': MD_LLM_ZERO_SHOT,
    'SRS_VLM_Zero_Shot.ipynb': MD_VLM_ZERO_SHOT,
    'SRS_LLM_Finetune.ipynb': MD_LLM_FINETUNE,
    'SRS_VLM_Finetune.ipynb': MD_VLM_FINETUNE,
}

def process_notebook(filepath: str, md_spec: dict):
    print(f"Refining: {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for idx, cell in enumerate(nb['cells']):
        cell_type = cell.get('cell_type')
        
        # 1. Update Markdown cells with academic English
        if cell_type == 'markdown' and idx in md_spec:
            new_text = md_spec[idx]
            # Convert to notebook source lines
            lines = [l + '\n' for l in new_text.split('\n')]
            if lines and lines[-1].endswith('\n'):
                lines[-1] = lines[-1][:-1]  # remove trailing newline on last line
            cell['source'] = lines
            
        # 2. Strip informal code comments from Code cells
        elif cell_type == 'code':
            raw_src = ''.join(cell.get('source', []))
            cleaned_src = remove_comments_from_code(raw_src)
            lines = [l + '\n' for l in cleaned_src.split('\n')]
            if lines and lines[-1].endswith('\n'):
                lines[-1] = lines[-1][:-1]
            cell['source'] = lines
            
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"  -> Successfully refined {filepath}")

def main():
    for nb_path, md_spec in NOTEBOOK_SPECS.items():
        process_notebook(nb_path, md_spec)
    print("\nAll 4 notebooks have been updated to IEEE/ACM Publication Quality!")

if __name__ == "__main__":
    main()
