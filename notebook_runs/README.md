# Isolated Notebook Runs

Each notebook run is isolated by seed:

```text
notebook_runs/
  seed_42/
    SRS_LLM_Zero_Shot.ipynb
    SRS_VLM_Zero_Shot.ipynb
    SRS_LLM_Finetune.ipynb
    SRS_VLM_Finetune.ipynb
  seed_123/
  seed_456/
```

Run one notebook and seed from the repository root:

```powershell
python src/notebook_runner.py --notebook notebooks/SRS_LLM_Zero_Shot.ipynb --seed 42
```

Use the same command for seeds `123` and `456` and for the other notebooks.
