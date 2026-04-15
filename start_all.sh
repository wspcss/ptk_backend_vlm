#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "Starting all services... Logs in $LOG_DIR/"

cd "$SCRIPT_DIR/ptk_backend"     && node server.js                             2>&1 | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done | tee -a "$LOG_DIR/ptk_backend.log" &
cd "$SCRIPT_DIR/ptk_frontend"    && npm start                                  2>&1 | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done | tee -a "$LOG_DIR/ptk_frontend.log" &
cd "$SCRIPT_DIR/vivi_case_mgmt"  && npm start                                  2>&1 | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done | tee -a "$LOG_DIR/vivi_case_mgmt.log" &
cd "$SCRIPT_DIR/ptk_backend_vlm" && source venv/bin/activate && python main.py  2>&1 | while IFS= read -r line; do echo "[$(date '+%Y-%m-%d %H:%M:%S')] $line"; done | tee -a "$LOG_DIR/ptk_backend_vlm.log" &

trap "kill 0" EXIT
wait