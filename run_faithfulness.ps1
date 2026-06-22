$ErrorActionPreference = "Stop"

Write-Host "I'm computing faithfulness :)"
Write-Host "The PID of this script is: $PID"

$Splits = "id_val"
$Seeds = "1/2/3/4/5"
$Pretrain = "degenerate"

$Datasets = @(
    "BAColorGVIsol/basis/no_shift"
)

$Budgets = @(10, 20, 50, 80, 100, 150, 200, 300, 500, 1000)

foreach ($Dataset in $Datasets) {
    foreach ($B in $Budgets) {

        $Config = "final_configs/$Dataset/GSAT.yaml"

        Write-Host "Running GSAT budget $B"

        $Args = @(
            "--config_path", $Config,
            "--seeds", $Seeds,
            "--task", "eval_metric",
            "--metrics", "rfidm/rfidp/suff_cause/suff/nec/counter_fid",
            "--splits", $Splits,
            "--pretrain", $Pretrain,
            "--backbone", "ACR2",
            "--expval_budget", "$B"
        )

        & goodtg @Args

        if ($LASTEXITCODE -ne 0) {
            throw "goodtg failed for budget $B"
        }

        Write-Host "DONE GSAT $Pretrain ACR2 ablation expval_budget $B"
    }
}

Write-Host "DONE all :)"