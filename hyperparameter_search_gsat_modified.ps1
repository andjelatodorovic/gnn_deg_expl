param(
    [string]$RepoRoot = "C:\Users\Cerebria\gnn_deg_expl",
    [string]$Dataset = "BAColorGVIsol",
    [string]$Seeds = "1/2",
    [int]$MaxEpoch = 50,
    [int]$ExpvalBudget = 2,
    [switch]$Evaluate,
    [switch]$GenerateOnly
)

$ErrorActionPreference = "Continue"
Set-StrictMode -Version Latest

$ConfigDirectory = Join-Path `
    $RepoRoot `
    "configs\final_configs\$Dataset\basis\no_shift"

$LogDirectory = Join-Path `
    $RepoRoot `
    "hyperparameter_search_logs\$Dataset"

$BaseYaml = Join-Path $ConfigDirectory "base.yaml"

if (-not (Test-Path $RepoRoot)) {
    throw "Repository not found: $RepoRoot"
}

if (-not (Test-Path $BaseYaml)) {
    throw "Base YAML not found: $BaseYaml"
}

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null

function Format-YamlNumber {
    param([double]$Value)

    return $Value.ToString(
        "0.########",
        [System.Globalization.CultureInfo]::InvariantCulture
    )
}

function Format-NameNumber {
    param([double]$Value)

    return (Format-YamlNumber $Value).Replace(".", "p")
}

# Medium search: baseline plus one-factor variations.
# K below 1 is interpreted as the fraction of nodes selected per graph.
$Trials = @(
    [pscustomobject]@{
        Label = "baseline"
        K = 0.20
        Epsilon = 0.005
        GameIterations = 1
        MerlinLR = 0.0005
        MorganaSteps = 2
        MorganaLR = 0.0002
        MorganaWeight = 0.01
    },
    [pscustomobject]@{
        Label = "k10"
        K = 0.10
        Epsilon = 0.005
        GameIterations = 1
        MerlinLR = 0.0005
        MorganaSteps = 2
        MorganaLR = 0.0002
        MorganaWeight = 0.01
    },
    [pscustomobject]@{
        Label = "k30"
        K = 0.30
        Epsilon = 0.005
        GameIterations = 1
        MerlinLR = 0.0005
        MorganaSteps = 2
        MorganaLR = 0.0002
        MorganaWeight = 0.01
    },
    [pscustomobject]@{
        Label = "eps001"
        K = 0.20
        Epsilon = 0.001
        GameIterations = 1
        MerlinLR = 0.0005
        MorganaSteps = 2
        MorganaLR = 0.0002
        MorganaWeight = 0.01
    },
    [pscustomobject]@{
        Label = "eps01"
        K = 0.20
        Epsilon = 0.010
        GameIterations = 1
        MerlinLR = 0.0005
        MorganaSteps = 2
        MorganaLR = 0.0002
        MorganaWeight = 0.01
    },
    [pscustomobject]@{
        Label = "merlinlr01"
        K = 0.20
        Epsilon = 0.005
        GameIterations = 1
        MerlinLR = 0.0001
        MorganaSteps = 2
        MorganaLR = 0.0002
        MorganaWeight = 0.01
    },
    [pscustomobject]@{
        Label = "merlinlr1"
        K = 0.20
        Epsilon = 0.005
        GameIterations = 1
        MerlinLR = 0.001
        MorganaSteps = 2
        MorganaLR = 0.0002
        MorganaWeight = 0.01
    },
    [pscustomobject]@{
        Label = "morganasteps5"
        K = 0.20
        Epsilon = 0.005
        GameIterations = 1
        MerlinLR = 0.0005
        MorganaSteps = 5
        MorganaLR = 0.0002
        MorganaWeight = 0.01
    },
    [pscustomobject]@{
        Label = "morganalr05"
        K = 0.20
        Epsilon = 0.005
        GameIterations = 1
        MerlinLR = 0.0005
        MorganaSteps = 2
        MorganaLR = 0.0005
        MorganaWeight = 0.01
    },
    [pscustomobject]@{
        Label = "weight001"
        K = 0.20
        Epsilon = 0.005
        GameIterations = 1
        MerlinLR = 0.0005
        MorganaSteps = 2
        MorganaLR = 0.0002
        MorganaWeight = 0.001
    },
    [pscustomobject]@{
        Label = "weight005"
        K = 0.20
        Epsilon = 0.005
        GameIterations = 1
        MerlinLR = 0.0005
        MorganaSteps = 2
        MorganaLR = 0.0002
        MorganaWeight = 0.05
    },
    [pscustomobject]@{
        Label = "weight01"
        K = 0.20
        Epsilon = 0.005
        GameIterations = 1
        MerlinLR = 0.0005
        MorganaSteps = 2
        MorganaLR = 0.0002
        MorganaWeight = 0.10
    }
)

Push-Location $RepoRoot

try {
    foreach ($Trial in $Trials) {
        $K = Format-YamlNumber $Trial.K
        $Epsilon = Format-YamlNumber $Trial.Epsilon
        $MerlinLR = Format-YamlNumber $Trial.MerlinLR
        $MorganaLR = Format-YamlNumber $Trial.MorganaLR
        $MorganaWeight = Format-YamlNumber $Trial.MorganaWeight

        $RunName = "HP_search_$($Trial.Label)_k$(Format-NameNumber $Trial.K)"
        $YamlFile = Join-Path $ConfigDirectory "$RunName.yaml"

        $Yaml = @"
includes:
  - base.yaml

model:
  model_name: GSATGNNs_MODIFIED

ood:
  ood_alg: GSAT
  ood_param: 1.0
  extra_param:
    - $K
    - $Epsilon
    - $($Trial.GameIterations)
    - $MerlinLR
    - $($Trial.MorganaSteps)
    - $MorganaLR
    - $MorganaWeight

train:
  max_epoch: $MaxEpoch
  mile_stones:
    - 20

log_file: $RunName
clean_save: true
use_norm: none
mitigation_sampling: default
"@

        Set-Content -Path $YamlFile -Value $Yaml -Encoding ASCII

        Write-Host ""
        Write-Host "========================================" -ForegroundColor Cyan
        Write-Host "RUN: $RunName" -ForegroundColor Cyan
        Write-Host "K: $K | Seeds: $Seeds | Epochs per seed: $MaxEpoch"
        Write-Host "YAML: $YamlFile"
        Write-Host "========================================" -ForegroundColor Cyan

        if ($GenerateOnly) {
            continue
        }

        $RelativeConfig = `
            "final_configs/$Dataset/basis/no_shift/$RunName.yaml"

        $TrainLog = Join-Path $LogDirectory "$RunName-train.log"

        & goodtg `
            --config_path $RelativeConfig `
            --seeds $Seeds `
            --task train `
            --pretrain degenerate `
            --backbone ACR2 `
            --max_epoch $MaxEpoch 2>&1 |
            Tee-Object -FilePath $TrainLog

        $TrainExitCode = $LASTEXITCODE

        if ($TrainExitCode -ne 0) {
            Write-Host ""
            Write-Host "TRAINING FAILED: $RunName" -ForegroundColor Red
            Write-Host "Log: $TrainLog" -ForegroundColor Red
            break
        }

        if ($Evaluate) {
            $EvalLog = Join-Path $LogDirectory "$RunName-eval.log"

            Write-Host ""
            Write-Host "Evaluating $RunName..." -ForegroundColor Green

            & goodtg `
                --config_path $RelativeConfig `
                --seeds $Seeds `
                --task eval_metric `
                --metrics "rfidm/rfidp/suff_cause/suff/nec/counter_fid" `
                --splits "id_val" `
                --pretrain degenerate `
                --backbone ACR2 `
                --expval_budget $ExpvalBudget 2>&1 |
                Tee-Object -FilePath $EvalLog

            $EvalExitCode = $LASTEXITCODE

            if ($EvalExitCode -ne 0) {
                Write-Host ""
                Write-Host "EVALUATION FAILED: $RunName" -ForegroundColor Red
                Write-Host "Log: $EvalLog" -ForegroundColor Red
                break
            }
        }
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Hyperparameter search finished." -ForegroundColor Green
Write-Host "Configurations: $($Trials.Count)"
Write-Host "Seeds: $Seeds"
Write-Host "Epochs per configuration per seed: $MaxEpoch"
Write-Host "Logs: $LogDirectory"
