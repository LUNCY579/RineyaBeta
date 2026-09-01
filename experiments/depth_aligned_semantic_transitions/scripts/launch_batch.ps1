# Launch a DAST run batch DETACHED, so it survives the launching session.
#
# Why this exists. A 9-run batch was started as a child of an interactive shell and
# was killed when that session exited -- mid-training, at step 1480 of 2000, with no
# completion record. Silence looked exactly like progress. A multi-hour batch must
# not be parented to whatever tool happened to start it.
#
# Start-Process detaches the batch from this shell, so it keeps running after the
# session ends. The work itself is driven by resume_runs.py, which is idempotent: a
# run counts as complete only when both dev_summary.json and generation_summary.json
# exist, so re-invoking this script after ANY interruption resumes exactly the runs
# that still need doing and re-runs nothing that finished.
#
# The live watcher (scripts/watch_runs.py under a Monitor) is a separate concern and
# IS session-scoped. Losing the watcher does not stop the batch; re-arm it and it
# picks up current state from the filesystem.
#
# Usage:
#   .\scripts\launch_batch.ps1 -Tags g3_s17_state_final_repeated,g3_s29_state_ordered
#   .\scripts\launch_batch.ps1 -Tags (Get-Content tags.txt)

param(
    [Parameter(Mandatory = $true)][string[]]$Tags,
    [string]$Python = "C:\Users\clclc\RineyaExperiment\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$pkg = Split-Path -Parent $PSScriptRoot

if (-not (Test-Path $Python)) { throw "interpreter not found: $Python" }

# Refuse to start a second batch on top of a running one: two trainers would contend
# for the same GPU and the same run directories.
$running = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -like '*resume_runs.py*' -or $_.CommandLine -like '*dast.cli*train*' })
if ($running.Count -gt 0) {
    throw "a batch is already running (PID $($running[0].ProcessId)); stop it first or wait"
}

$runs = Join-Path $pkg "runs"
New-Item -ItemType Directory -Force -Path $runs | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = Join-Path $runs "_batch_$stamp.log"
$err = Join-Path $runs "_batch_$stamp.err"

$env:PYTHONPATH = Join-Path $pkg "src"
$env:PYTHONHASHSEED = "0"
$env:PYTHONIOENCODING = "utf-8"

$argList = @("-u", "scripts\resume_runs.py", "--run", "--tags") + $Tags
$proc = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $pkg `
    -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru

Write-Output "launched detached PID $($proc.Id)"
Write-Output "  tags : $($Tags.Count)"
Write-Output "  log  : $log"
Write-Output "  err  : $err"
Write-Output ""
Write-Output "Arm the live watcher separately (it is session-scoped and does not"
Write-Output "affect the batch):"
Write-Output "  python scripts/watch_runs.py --interval 90 --tags $($Tags -join ' ')"
