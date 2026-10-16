"""
Update Cell 4 (Dataset Loading) across all 4 notebooks to support:
  1. data/QuRE.csv (running from repo root)
  2. ../data/QuRE.csv (running from notebooks/ directory)
  3. QuRE.csv (running in same directory)
  4. Auto-download from GitHub raw repository (when running in Google Colab)
  5. Fallback to files.upload()
"""

import json

NEW_LOADER_CODE = """import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

candidate_paths = [
    'data/QuRE.csv',
    '../data/QuRE.csv',
    'QuRE.csv',
    'data/datasetRE.csv',
    '../data/datasetRE.csv',
    'datasetRE.csv'
]

filename = None
for candidate in candidate_paths:
    if os.path.exists(candidate):
        filename = candidate
        break

if filename is None:
    try:
        url = 'https://raw.githubusercontent.com/HoangKhoi1704/VLM-in-ambiguity-RE/main/data/QuRE.csv'
        print('Fetching dataset from repository: ' + url)
        df_fetched = pd.read_csv(url)
        os.makedirs('data', exist_ok=True)
        filename = 'data/QuRE.csv'
        df_fetched.to_csv(filename, index=False)
    except Exception:
        pass

if filename is None:
    try:
        from google.colab import files
        print('Upload file QuRE.csv hoặc datasetRE.csv:')
        uploaded = files.upload()
        filename = list(uploaded.keys())[0]
    except ImportError:
        raise FileNotFoundError('Không tìm thấy QuRE.csv hoặc datasetRE.csv trong thư mục!')

df = pd.read_csv(filename)
print(f'\\n[Dataset] Loaded: {filename}')
print(f'   Total samples: {len(df)}')
print(f'   Columns      : {df.columns.tolist()}')
print(f'\\nClass Distribution:')
print(df['defect'].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

label_order = ['ok', 'defect']
counts = df['defect'].value_counts().reindex(label_order, fill_value=0)
colors = ['#2ecc71', '#e74c3c']

axes[0].bar(label_order, counts, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
axes[0].set_title('Class Distribution (Count)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Number of Samples')
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(counts):
    axes[0].text(i, v + 20, f'{v:,}', ha='center', fontweight='bold')

pcts = [counts['ok'] / len(df) * 100, counts['defect'] / len(df) * 100]
axes[1].pie(
    pcts,
    labels=[f'ok\\n({pcts[0]:.1f}%)', f'defect\\n({pcts[1]:.1f}%)'],
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'edgecolor': 'black', 'linewidth': 0.8}
)
axes[1].set_title('Class Proportion (%)', fontsize=12, fontweight='bold')

plt.suptitle('SRS Ambiguity Dataset Overview (QuRE)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
"""

notebooks = [
    'notebooks/SRS_LLM_Zero_Shot.ipynb',
    'notebooks/SRS_VLM_Zero_Shot.ipynb',
    'notebooks/SRS_LLM_Finetune.ipynb',
    'notebooks/SRS_VLM_Finetune.ipynb'
]

for nb_path in notebooks:
    print(f"Updating Cell 4 in {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    lines = [l + '\n' for l in NEW_LOADER_CODE.split('\n')]
    if lines and lines[-1].endswith('\n'):
        lines[-1] = lines[-1][:-1]
        
    nb['cells'][4]['source'] = lines
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"  -> Successfully updated {nb_path}")

print("\nAll 4 notebooks now have universal smart dataset loading!")
