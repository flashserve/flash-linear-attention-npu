#!/usr/bin/env bash
# prepare_wy_repr_bwd NPU vs GPU dump and CPU fp64 dual benchmark.
#
# Usage:
#   ./run_prepare_wy_repr_bwd_gpu_dump_dual.sh [DUMP_ROOT] [extra args...]
#   ./run_prepare_wy_repr_bwd_gpu_dump_dual.sh /path/to/008_prepare_wy_repr_bwd.pt
#   ./run_prepare_wy_repr_bwd_gpu_dump_dual.sh --pt /path/to/008_prepare_wy_repr_bwd.pt

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TEST_DEVICE_ID="${TEST_DEVICE_ID:-0}"
OP_TAG="prepare_wy_repr_bwd"
PY="${SCRIPT_DIR}/test_prepare_wy_repr_bwd_gpu_dump_dual.py"

DUMP_ROOT=""
declare -a PY_ARGS=()

if [[ $# -eq 0 ]]; then
  DUMP_ROOT="./GPU_DUMP"
  PY_ARGS=(--dump-root "$DUMP_ROOT")
elif [[ "${1:-}" == *.pt ]]; then
  DUMP_ROOT="$(cd "$(dirname "$1")" && pwd)"
  PY_ARGS=(--pt "$1" "${@:2}")
elif [[ "${1:-}" == --pt || "${1:-}" == --pts ]]; then
  if [[ "${1:-}" == --pt && -n "${2:-}" ]]; then
    DUMP_ROOT="$(cd "$(dirname "$2")" && pwd)"
  fi
  PY_ARGS=("$@")
elif [[ -d "${1:-}" ]]; then
  DUMP_ROOT="$(cd "$1" && pwd)"
  shift
  PY_ARGS=(--dump-root "$DUMP_ROOT" "$@")
else
  PY_ARGS=("$@")
fi

if [[ -n "$DUMP_ROOT" ]]; then
  LOG_DIR="${DUMP_ROOT}/logs"
else
  LOG_DIR="${SCRIPT_DIR}/logs"
fi
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/${OP_TAG}_gpu_dump_dual_${TS}.log"
LATEST_LINK="${LOG_DIR}/${OP_TAG}_gpu_dump_dual_latest.log"

echo "[INFO] TEST_DEVICE_ID=${TEST_DEVICE_ID}"
echo "[INFO] python: ${PY}"
echo "[INFO] args: ${PY_ARGS[*]}"
echo "[INFO] log: ${LOG_FILE}"

set +e
python -u "$PY" "${PY_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
STATUS="${PIPESTATUS[0]}"
set -e

if command -v ln >/dev/null 2>&1; then
  ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LINK" 2>/dev/null || cp "$LOG_FILE" "$LATEST_LINK"
else
  cp "$LOG_FILE" "$LATEST_LINK"
fi

exit "$STATUS"
