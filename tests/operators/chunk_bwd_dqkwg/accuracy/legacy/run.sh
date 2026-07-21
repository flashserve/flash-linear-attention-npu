#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASE_ID="${2:-${1:-case_22}}"

export TEST_DEVICE_ID="${TEST_DEVICE_ID:-0}"
python3 "${SCRIPT_DIR}/pta.py" "${CASE_ID}"
