param(
    [string]$TaskName = "TheSlowBrainDailyShadow",
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$At = "07:30",
    [int]$FeatureLimit = 5000,
    [switch]$FullUniverse
)

$ErrorActionPreference = "Stop"

$scriptPath = Join-Path $ProjectRoot "scripts\run_daily_slowbrain.py"
if (-not (Test-Path -LiteralPath $scriptPath)) {
    throw "Daily runner not found: $scriptPath"
}

$uvArgs = @(
    "run",
    "python",
    $scriptPath,
    "--project-root",
    $ProjectRoot,
    "--feature-limit",
    "$FeatureLimit"
)

if ($FullUniverse) {
    $uvArgs += "--full-universe"
}

$quotedArgs = $uvArgs | ForEach-Object {
    if ($_ -match "\s") {
        '"' + ($_.Replace('"', '\"')) + '"'
    } else {
        $_
    }
}
$argument = $quotedArgs -join " "

$action = New-ScheduledTaskAction -Execute "uv" -Argument $argument -WorkingDirectory $ProjectRoot
$trigger = New-ScheduledTaskTrigger -Daily -At ([datetime]::ParseExact($At, "HH:mm", $null))
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Hours 2)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited
$description = "Runs TheSlowBrain daily shadow workflow. Broker live execution remains blocked by the application."

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description $description `
    -Force | Out-Null

Write-Host "Registered scheduled task: $TaskName"
Write-Host "Project root: $ProjectRoot"
Write-Host "Command: uv $argument"
