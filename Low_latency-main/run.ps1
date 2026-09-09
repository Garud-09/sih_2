# PowerShell runner for ISRO PS 26172 ASR & Telemetry Server on Windows

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

# Check if .venv exists, if not create and install dependencies
if (-not (Test-Path ".venv")) {
    Write-Host "[*] Setting up virtual environment..." -ForegroundColor Cyan
    if (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv .venv --python 3.12
        $env:VIRTUAL_ENV = ".venv"
        uv pip install -r server/requirements.txt
    } else {
        python -m venv .venv
        & .venv\Scripts\python.exe -m pip install -r server/requirements.txt
    }
}

# Free up ports 8000 and 8765 if running on Windows
$ports = @(8000, 8765)
foreach ($port in $ports) {
    $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($connections) {
        foreach ($conn in $connections) {
            Write-Host "[*] Killing process on port $port (PID: $($conn.OwningProcess))..." -ForegroundColor Yellow
            Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        }
    }
}

Start-Sleep -Seconds 0.5

# Activate virtual environment
if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .venv\Scripts\Activate.ps1
}

# Launch the ASR & Telemetry server
Write-Host "[*] Launching ISRO PS 26172 ASR & Telemetry Server..." -ForegroundColor Green
python -u server/asr_server.py
