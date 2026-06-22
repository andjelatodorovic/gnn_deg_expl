$ErrorActionPreference = "Stop"

# ==============================
# GSAT Baseline vs Arthur-Morgana
# ==============================

$Seed = "1"
$Epochs = "200"

$Config = "final_configs/BAColorGVIsol/basis/no_shift/GSAT.yaml"
$Pretrain = "degenerate"
$Backbone = "ACR2"

$ResultDir = "results/arthur_morgana_comparison"
$LogDir = "$ResultDir/logs"
$TablePath = "$ResultDir/results_table.csv"

New-Item -ItemType Directory -Force $ResultDir | Out-Null
New-Item -ItemType Directory -Force $LogDir | Out-Null

if (!(Test-Path $Config)) {
    throw "Config file not found: $Config"
}

function Add-AccuracyRows {
    param(
        [string[]]$OutputLines,
        [string]$Variant,
        [string]$Phase,
        [string]$LogPath
    )

    $Rows = @()
    $InFinal = $false

    foreach ($Line in $OutputLines) {
        if ($Line -match "Final accuracies:") {
            $InFinal = $true
            continue
        }

        if ($InFinal -and $Line -match "^\s*([A-Z_]+)\s*=\s*([0-9\.]+)\s*\+-\s*([0-9\.]+)") {
            $Rows += [pscustomobject]@{
                timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
                variant   = $Variant
                phase     = $Phase
                seed      = $Seed
                epochs    = $Epochs
                config    = $Config
                pretrain  = $Pretrain
                backbone  = $Backbone
                metric    = $Matches[1]
                mean      = $Matches[2]
                std       = $Matches[3]
                log_file  = $LogPath
            }
        }
    }

    if ($Rows.Count -gt 0) {
        $Rows | Export-Csv -Path $TablePath -NoTypeInformation -Append
        Write-Host "Saved $($Rows.Count) rows to $TablePath"
    } else {
        Write-Host "No final accuracy rows found for $Variant / $Phase. Check log: $LogPath"
    }
}

function Invoke-GoodRun {
    param(
        [string]$Variant,
        [string]$Phase,
        [bool]$ArthurMorgana
    )

    $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $LogPath = "$LogDir/${Stamp}_${Variant}_${Phase}_seed${Seed}.log"

    Write-Host ""
    Write-Host "=================================================="
    Write-Host "Running $Variant / $Phase / seed=$Seed"
    Write-Host "Log: $LogPath"
    Write-Host "=================================================="

    $Args = @(
        "--config_path", $Config,
        "--seeds", $Seed,
        "--task", $Phase,
        "--pretrain", $Pretrain,
        "--backbone", $Backbone
    )

    if ($Phase -eq "train") {
        $Args += @("--epoch", $Epochs)
    }

    if ($ArthurMorgana) {
        $Args += @("--arthur_morgana_enabled", "true")
    }

    $Output = & goodtg @Args 2>&1 | Tee-Object -FilePath $LogPath

    if ($LASTEXITCODE -ne 0) {
        throw "goodtg failed for $Variant / $Phase. See log: $LogPath"
    }

    Add-AccuracyRows -OutputLines $Output -Variant $Variant -Phase $Phase -LogPath $LogPath
}

Write-Host "Starting GSAT baseline vs Arthur-Morgana comparison..."
Write-Host "Config: $Config"
Write-Host "Seed: $Seed"

Invoke-GoodRun -Variant "baseline" -Phase "train" -ArthurMorgana $false
Invoke-GoodRun -Variant "baseline" -Phase "test"  -ArthurMorgana $false

Invoke-GoodRun -Variant "arthur_morgana" -Phase "train" -ArthurMorgana $true
Invoke-GoodRun -Variant "arthur_morgana" -Phase "test"  -ArthurMorgana $true

Write-Host ""
Write-Host "DONE."
Write-Host "Results table saved to: $TablePath"
Write-Host "Logs saved to: $LogDir"
