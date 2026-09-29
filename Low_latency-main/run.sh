#!/usr/bin/env bash
set -e

# Change directory to the repository root where this script resides
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if .venv exists, if not create and install dependencies
if [ ! -d ".venv" ]; then
    echo "[*] Setting up virtual environment..."
    uv venv .venv --python 3.12
    VIRTUAL_ENV=.venv uv pip install -r server/requirements.txt
fi

# Free up ports 8000 and 8765 if an old instance is lingering
fuser -k 8000/tcp 8765/tcp 2>/dev/null || true
sleep 0.5

# Activate virtual environment
source .venv/bin/activate

# Launch the ASR & Telemetry server
echo "[*] Launching ISRO PS 26172 ASR & Telemetry Server..."
python server/asr_server.py
