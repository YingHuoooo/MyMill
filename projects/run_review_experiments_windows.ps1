$ErrorActionPreference = "Stop"

# Run from the projects directory:
#   cd E:\qyusong\Code\26-May\MyMill-main\projects
#   conda activate dm
#   powershell -ExecutionPolicy Bypass -File .\run_review_experiments_windows.ps1
#
# Optional: set DM_PYTHON to a specific interpreter before running:
#   $env:DM_PYTHON="G:\Anaconda\envs\dm\python.exe"

$Seeds = "123,456,789"
$Prefix = "paper_review_v1"
$Python = if ($env:DM_PYTHON) { $env:DM_PYTHON } else { "python" }

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
  throw "Cannot find Python command: $Python. Activate the dm environment or set DM_PYTHON."
}

function Invoke-DmPython {
  & $Python @args
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code $LASTEXITCODE`: $Python $args"
  }
}

Invoke-DmPython -c "import sys, torch; print('Python:', sys.executable); print('Torch:', torch.__version__)"

$Common = @(
  "--gpu", "0",
  "--depth", "5",
  "--model", "unet",
  "--conditioning", "concat",
  "--ckpt", "../pretrained/00840solver/00840.solver.tar",
  "--strict-load", "true",
  "--test-take", "-1",
  "--seeds", $Seeds,
  "--red-crc-alpha", "0.03",
  "--green-crc-alpha", "0.03",
  "--crc-max-threshold", "0.5",
  "--risk-rescue-budget", "0.002",
  "--risk-rescue-min-prob", "0.25",
  "--calibration-ratio", "0.2",
  "--skip-existing"
)

Invoke-DmPython run_paper_experiments.py `
  --suite multi_seed `
  --alias-prefix $Prefix `
  @Common

Invoke-DmPython run_paper_experiments.py `
  --suite threshold_multi_seed `
  --alias-prefix $Prefix `
  --thresholds "0.30,0.35,0.40,0.45,0.50" `
  @Common

Invoke-DmPython run_paper_experiments.py `
  --suite calibrated_fixed `
  --alias-prefix $Prefix `
  --calibrated-fixed-thresholds "0.30,0.35,0.40,0.45,0.50" `
  --calibrated-fixed-budget "0.002" `
  @Common

Invoke-DmPython summarize_paper_experiments.py `
  logs/paper_experiments/multi_seed_summary.csv `
  logs/paper_experiments/threshold_multi_seed_summary.csv `
  logs/paper_experiments/calibrated_fixed_summary.csv `
  --out logs/paper_experiments/review_decision_table.csv

Write-Host "Done. Key output:"
Write-Host "  logs/paper_experiments/review_decision_table.csv"
