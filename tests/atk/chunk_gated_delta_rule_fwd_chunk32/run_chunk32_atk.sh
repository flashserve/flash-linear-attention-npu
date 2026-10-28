#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATK_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEVICE_ID="${1:-${NPU_DEVICE_ID:-0}}"
SCOPE="${2:-accuracy}"

exec bash "${ATK_ROOT}/run_test_cpu.sh" \
  -op=chunk_gated_delta_rule_fwd_chunk32 \
  -npu_device_id="${DEVICE_ID}" \
  -soc=ascend950 \
  -scope="${SCOPE}"
