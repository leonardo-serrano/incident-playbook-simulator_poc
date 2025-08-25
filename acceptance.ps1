<# 
 acceptance.ps1
 End-to-end demo + acceptance checks for the PoC.

 What it does:
 1) Clears previous traces under reports/traces (keeps other artifacts).
 2) Runs two canonical scenarios:
      - ALERT-1001 (CPU high on db01)
      - ALERT-1002 (HTTP 500 spike on checkout-api)
 3) Rebuilds aggregated metrics (reports/metrics.json).
 4) Scores the runs using eval/expected_actions.json -> reports/score.json.
 5) Executes automatic checks (utils/acceptance_check.py) and prints PASS/FAIL.

 Notes:
 - Works whether SIM_USE_LLM=true or false. With LLM=true, outputs may vary in text,
   but the structure (plan/retrieve/decision/HITL/executor/summarizer) remains.
 - Requires the virtualenv activated and Python on PATH.
#>

param(
  [string]$Python = "python",
  [string]$TracesDir = "reports\traces",
  [string]$Expected = "eval\expected_actions.json",
  [string]$ScoreOut = "reports\score.json"
)

Write-Host ">> Preparing folders..."
if (-not (Test-Path -Path "reports"))        { New-Item -ItemType Directory -Force reports | Out-Null }
if (-not (Test-Path -Path $TracesDir))       { New-Item -ItemType Directory -Force $TracesDir | Out-Null }

Write-Host ">> Cleaning old traces in $TracesDir ..."
Get-ChildItem $TracesDir -Filter *.jsonl -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

Write-Host ">> Running scenario ALERT-1001 (CPU high)..."
& $Python .\main.py --alert ALERT-1001

Write-Host ">> Running scenario ALERT-1002 (HTTP 500 spike)..."
& $Python .\main.py --alert ALERT-1002

Write-Host ">> Scoring runs against expected actions..."
& $Python -m utils.scoring --traces $TracesDir --expected $Expected --out $ScoreOut

Write-Host ">> Running acceptance checks..."
& $Python -m utils.acceptance_check --traces $TracesDir --metrics reports\metrics.json --score $ScoreOut --expected $Expected
