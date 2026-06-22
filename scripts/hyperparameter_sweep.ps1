$ErrorActionPreference = "Stop"

$REPO_ROOT = "C:\Users\Cerebria\gnn_deg_expl"
Set-Location $REPO_ROOT

Write-Host "Running GSATGNNs_MODIFIED hyperparameter sweep"
Write-Host "PowerShell PID: $PID"

$DATASET = "BAColorGVIsol/basis/no_shift"

# Use the real working config path.
$CONFIG_REL = "final_configs/$DATASET/GSATGNNs_MODIFIED.yaml"
$CONFIG_ABS = Join-Path $REPO_ROOT "configs\$CONFIG_REL"
$BACKUP_ABS = "$CONFIG_ABS.backup"

$SEEDS = "1"
$BACKBONE = "ACR2"
$METRICS = "rfidm/rfidp/suff_cause/suff/nec/counter_fid"
$SPLITS = "id_val"
$EXPVAL_BUDGET = 10

# Small first sweep.
$Ks = @(5, 10, 20)
$EPSILONS = @(0.01, 0.05)

if (!(Test-Path $CONFIG_ABS)) {
    throw "Config not found: $CONFIG_ABS"
}

# Backup original config once.
Copy-Item $CONFIG_ABS $BACKUP_ABS -Force
Write-Host "Backed up original config to:"
Write-Host $BACKUP_ABS

try {
    foreach ($K in $Ks) {
        foreach ($EPS in $EPSILONS) {

            Write-Host ""
            Write-Host "=================================================="
            Write-Host "Training GSATGNNs_MODIFIED | K=$K | epsilon=$EPS"
            Write-Host "=================================================="

            # Restore original before each modification.
            Copy-Item $BACKUP_ABS $CONFIG_ABS -Force

            $yaml = Get-Content $CONFIG_ABS -Raw

            $newExtra = @"
  extra_param:
    - $K      # K
    - $EPS    # epsilon
    - 1       # game_iterations
    - 0.01    # merlin_lr
    - 1       # morgana_steps
    - 0.01    # morgana_lr
    - 1.0     # morgana_weight
"@

            # Replace only the indented ood.extra_param block.
            $pattern = "(?ms)^  extra_param:\s*\r?\n(?:    - .*\r?\n)+"

            if ($yaml -notmatch $pattern) {
                throw "Could not find ood.extra_param block in $CONFIG_ABS"
            }

            $yaml = [regex]::Replace($yaml, $pattern, $newExtra, 1)

            # Keep old GSAT spec loss disabled for modified model.
            $yaml = [regex]::Replace($yaml, "ood_param:\s*[0-9.]+", "ood_param: 0.0", 1)

            Set-Content -Path $CONFIG_ABS -Value $yaml -Encoding UTF8

            Write-Host "Current config extra_param:"
            Select-String -Path $CONFIG_ABS -Pattern "extra_param|ood_param" -Context 0,8

            goodtg --config_path $CONFIG_REL `
                   --seeds $SEEDS `
                   --task train `
                   --backbone $BACKBONE

            if ($LASTEXITCODE -ne 0) {
                throw "Training failed for K=$K epsilon=$EPS"
            }

            Write-Host "DONE training K=$K epsilon=$EPS"

            Write-Host ""
            Write-Host "Evaluating faithfulness | K=$K | epsilon=$EPS | budget=$EXPVAL_BUDGET"

            goodtg --config_path $CONFIG_REL `
                   --seeds $SEEDS `
                   --task eval_metric `
                   --metrics $METRICS `
                   --splits $SPLITS `
                   --backbone $BACKBONE `
                   --expval_budget $EXPVAL_BUDGET

            if ($LASTEXITCODE -ne 0) {
                throw "eval_metric failed for K=$K epsilon=$EPS"
            }

            Write-Host "DONE eval_metric K=$K epsilon=$EPS"
        }
    }
}
finally {
    # Always restore original config, even if a run fails.
    Copy-Item $BACKUP_ABS $CONFIG_ABS -Force
    Write-Host ""
    Write-Host "Restored original GSATGNNs_MODIFIED.yaml from backup."
}

Write-Host ""
Write-Host "DONE hyperparameter sweep :)"