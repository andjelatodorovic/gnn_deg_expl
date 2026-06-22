$ErrorActionPreference = "Stop"

Write-Host "I'm computing faithfulness :)"
Write-Host "PowerShell PID: $PID"

$SPLITS = "id_val"
$SEEDS = "1"
$PRETRAIN = "degenerate"
$METRICS = "rfidm/rfidp/suff_cause/suff/nec/counter_fid"

$DATASETS = @(
    "BAColorGVIsol/basis/no_shift"
)

$BUDGETS = @(10, 20, 50, 80, 100, 150, 200, 300, 500, 1000)

foreach ($DATASET in $DATASETS) {
    foreach ($B in $BUDGETS) {

        Write-Host ""
        Write-Host "=============================================="
        Write-Host "Running GSAT | DATASET=$DATASET | B=$B"
        Write-Host "=============================================="

        goodtg --config_path "final_configs/$DATASET/GSAT.yaml" `
               --seeds $SEEDS `
               --task eval_metric `
               --metrics $METRICS `
               --splits $SPLITS `
               --pretrain $PRETRAIN `
               --backbone ACR2 `
               --expval_budget $B

        if ($LASTEXITCODE -ne 0) {
            throw "GSAT failed for DATASET=$DATASET B=$B"
        }

        Write-Host "DONE GSAT $PRETRAIN ACR2 expval_budget $B"

        Write-Host ""
        Write-Host "======================================================"
        Write-Host "Running GSATGNNs_MODIFIED | DATASET=$DATASET | B=$B"
        Write-Host "======================================================"

        goodtg --config_path "final_configs/$DATASET/GSATGNNs_MODIFIED.yaml" `
               --seeds $SEEDS `
               --task eval_metric `
               --metrics $METRICS `
               --splits $SPLITS `
               --backbone ACR2 `
               --expval_budget $B

        if ($LASTEXITCODE -ne 0) {
            throw "GSATGNNs_MODIFIED failed for DATASET=$DATASET B=$B"
        }

        Write-Host "DONE GSATGNNs_MODIFIED ACR2 expval_budget $B"
    }
}

Write-Host ""
Write-Host "DONE all :)"