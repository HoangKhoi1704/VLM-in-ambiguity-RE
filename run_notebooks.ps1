$notebooks = @(
    "notebooks/SRS_LLM_Zero_Shot.ipynb",
    "notebooks/SRS_VLM_Zero_Shot.ipynb",
    "notebooks/SRS_LLM_Finetune.ipynb",
    "notebooks/SRS_VLM_Finetune.ipynb"
)
$seeds = @(42, 123, 456)

foreach ($notebook in $notebooks) {
    foreach ($seed in $seeds) {
        Write-Host "Running $notebook with seed $seed"
        python src/notebook_runner.py --notebook $notebook --seed $seed
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Run failed: $notebook seed $seed"
        }
    }
}
