# Isolated Notebook Runs

Each notebook run is isolated by seed and notebook:

```text
notebook_runs/
  seed_42/
    SRS_LLM_Zero_Shot/
    SRS_VLM_Zero_Shot/
    SRS_LLM_Finetune/
    SRS_VLM_Finetune/
  seed_123/
  seed_456/
```

The runner creates an executed notebook and `run.log` in the selected directory. The source notebooks in `notebooks/` remain unchanged.

Run one notebook and seed from the repository root:

```powershell
python src/notebook_runner.py --notebook notebooks/SRS_LLM_Zero_Shot.ipynb --seed 42
```

Use the same command for seeds `123` and `456` and for the other notebooks. Every completed or failed run appends one row to `results/notebook_run_logs.csv`.
