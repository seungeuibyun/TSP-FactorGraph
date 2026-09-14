param(
    [string]$Python = "python",
    [int]$WaitForPid = 0
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$resultDir = Join-Path $repo "results\scalability"
New-Item -ItemType Directory -Path $resultDir -Force | Out-Null

if ($WaitForPid -gt 0) {
    while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $resultDir "scalability.mc.$stamp.stdout.log"
$stderr = Join-Path $resultDir "scalability.mc.$stamp.stderr.log"
$process = Start-Process `
    -FilePath $Python `
    -ArgumentList @(
        "-u",
        "-m",
        "scripts.run_experiment",
        "experiments/scalability/config.json"
    ) `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

$process.Id | Set-Content (Join-Path $resultDir "scalability.mc.pid")
$stdout | Set-Content (Join-Path $resultDir "scalability.mc.latest-log")
