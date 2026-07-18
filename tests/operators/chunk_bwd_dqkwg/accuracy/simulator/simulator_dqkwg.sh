#!/usr/bin/env bash
# Run after loading the CANN simulator and the installed chunk_bwd_dqkwg custom OPP.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v msprof >/dev/null 2>&1; then
    echo "[ERROR] msprof is unavailable; load the CANN simulator environment first." >&2
    exit 2
fi

export DQKWG_SIM_DATA_DIR="${DQKWG_SIM_DATA_DIR:-$SCRIPT_DIR/outputs/data}"
mkdir -p "$DQKWG_SIM_DATA_DIR"

echo "[INFO] Running chunk_bwd_dqkwg simulator"
echo "[INFO] data_dir=$DQKWG_SIM_DATA_DIR"
exec msprof op simulator python3 "$SCRIPT_DIR/pta_simulator.py"
