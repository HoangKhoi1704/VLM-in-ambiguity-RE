"""
Clean decorative icons across the codebase while retaining essential status/warning indicators.
"""

import json
import re

# Decorative icons to remove (with optional trailing space)
DECORATIVE_REPLACEMENTS = {
    '🚀 ': '',
    '🚀': '',
    '🎲 ': '',
    '🎲': '',
    '🔮 ': '',
    '🔮': '',
    '🎯 ': '',
    '🎯': '',
    '📊 ': '',
    '📊': '',
    '📁 ': '',
    '📁': '',
    '📝 ': '',
    '📝': '',
    '🔄 ': '',
    '🔄': '',
    '🖥 ': '',
    '🖥': '',
    '💾 ': '',
    '💾': '',
    '🧪 ': '',
    '🧪': '',
    '🔴 ': '',
    '🔴': '',
    '🏆 ': '',
    '🏆': '',
    '📋 ': '',
    '📋': '',
    '📌 ': '',
    '📌': '',
    '📥 ': '',
    '📥': '',
    '🔬 ': '',
    '🔬': '',
    '⚙️ ': '',
    '⚙️': '',
    '🌟 ': '',
    '🌟': '',
    '💡 ': '',
    '💡': '',
    '📄 ': '',
    '📄': '',
    '📂 ': '',
    '📂': '',
}

def clean_text(text: str) -> str:
    for icon, repl in DECORATIVE_REPLACEMENTS.items():
        text = text.replace(icon, repl)
    return text

def clean_notebook(path: str):
    print(f"Cleaning decorative icons in: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for cell in nb['cells']:
        src = ''.join(cell.get('source', []))
        cleaned_src = clean_text(src)
        lines = [l + '\n' for l in cleaned_src.split('\n')]
        if lines and lines[-1].endswith('\n'):
            lines[-1] = lines[-1][:-1]
        cell['source'] = lines
        
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2, ensure_ascii=False)
    print(f"  -> Done {path}")

def clean_py_file(path: str):
    print(f"Cleaning decorative icons in: {path}")
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    cleaned = clean_text(content)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(cleaned)
    print(f"  -> Done {path}")

def main():
    notebooks = [
        'SRS_LLM_Zero_Shot.ipynb',
        'SRS_VLM_Zero_Shot.ipynb',
        'SRS_LLM_Finetune.ipynb',
        'SRS_VLM_Finetune.ipynb'
    ]
    for nb in notebooks:
        clean_notebook(nb)
        
    for py in ['aggregate_results.py', 'run_baselines.py']:
        clean_py_file(py)
        
    print("\nSuccessfully removed all decorative icons. Only essential functional indicators are preserved!")

if __name__ == "__main__":
    main()
