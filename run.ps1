# PowerShell script to activate venv and run app
$venv = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"
if (Test-Path $venv) {
    & $venv
} else {
    Write-Host "No venv found. Create one: py -3.11 -m venv venv"
    exit 1
}
python app_enhanced.py
