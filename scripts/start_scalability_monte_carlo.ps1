param(
    [string]$Python = "python",
    [int]$WaitForPid = 0
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$resultDir = Join-Path $repo "results\scalability\spatial_mc"
New-Item -ItemType Directory -Path $resultDir -Force | Out-Null

$pidFile = Join-Path $resultDir "scalability.mc.pid"
if (Test-Path -LiteralPath $pidFile) {
    $existingPid = [int](Get-Content -LiteralPath $pidFile -ErrorAction Stop)
    if (Get-Process -Id $existingPid -ErrorAction SilentlyContinue) {
        throw "Scalability experiment is already running with PID $existingPid"
    }
}

if ($WaitForPid -gt 0) {
    while (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 30
    }
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stdout = Join-Path $resultDir "scalability.mc.$stamp.stdout.log"
$stderr = Join-Path $resultDir "scalability.mc.$stamp.stderr.log"
$env:NUMBA_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
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

$process.Id | Set-Content $pidFile
$stdout | Set-Content (Join-Path $resultDir "scalability.mc.latest-log")

$watchStdout = Join-Path $resultDir "scalability.watch.$stamp.stdout.log"
$watchStderr = Join-Path $resultDir "scalability.watch.$stamp.stderr.log"
$watcher = Start-Process `
    -FilePath $Python `
    -ArgumentList @(
        "-u",
        "-m",
        "scripts.watch_scalability_results",
        "--job-pid", $process.Id,
        "--shard-dir", (Join-Path $resultDir "shards"),
        "--output", (Join-Path $resultDir "scalability_runs.csv"),
        "--figure-dir", (Join-Path $repo "figures\scalability"),
        "--interval-s", "60"
    ) `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $watchStdout `
    -RedirectStandardError $watchStderr `
    -WindowStyle Hidden `
    -PassThru
$watcher.Id | Set-Content (Join-Path $resultDir "scalability.watch.pid")
$watchStdout | Set-Content (Join-Path $resultDir "scalability.watch.latest-log")
