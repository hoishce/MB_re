param(
    [Parameter(Mandatory=$true)][string]$run_id,
    [Parameter(Mandatory=$true)][string]$syn_dir,
    [string]$models_dir = "saved_models/ckpt/synthesizer",
    [int]$save_every = 1000,
    [int]$backup_every = 25000,
    [int]$log_every = 200,
    [switch]$force_restart,
    [string]$hparams = ""
)

# Usage: .\run_synth_train.ps1 -run_id myrun -syn_dir saved_models/synth_minidataset
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$env:PYTHONPATH = $repoRoot
$defaultPy = Join-Path $repoRoot ".venv\Scripts\python.exe"
$py = if ($env:MOCKINGBIRD_PYTHON) {
    $env:MOCKINGBIRD_PYTHON
} elseif (Test-Path $defaultPy) {
    $defaultPy
} else {
    "python"
}

$argsList = @(
    (Join-Path $repoRoot "train.py"),
    "--type",
    "synth",
    $run_id,
    $syn_dir,
    "-m",
    $models_dir,
    "-s",
    $save_every,
    "-b",
    $backup_every,
    "-l",
    $log_every
)

if ($force_restart) {
    $argsList += "--force_restart"
}

if ($hparams -ne '') {
    $argsList += @("--hparams", $hparams)
}

Write-Host "Running training command:"
Write-Host "$py $($argsList -join ' ')"
& $py @argsList
