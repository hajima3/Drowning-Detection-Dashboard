# Drowning Detection Dashboard — Start Script
# Uses conda environment: env2

Write-Host ""
Write-Host "  Drowning Detection Dashboard" -ForegroundColor Cyan
Write-Host "  Starting on http://localhost:5000" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot\..
conda activate env2
python app.py
