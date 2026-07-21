#!/usr/bin/env bash
# bwd_dhu GVA 双标杆：随机输入直接测，无需 example dump
#
# 用法:
#   TEST_DEVICE_ID=0 bash tests/operators/chunk_gated_delta_rule_bwd_dhu/accuracy/run_legacy_gva.sh
# 调用前需由外部环境加载 CANN、custom OPP 和 Python 依赖。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEVICE="${TEST_DEVICE_ID:-0}"

export TEST_DEVICE_ID="$DEVICE"
export BWD_HU_OUT_DIR="${BWD_HU_OUT_DIR:-$SCRIPT_DIR/outputs/gva}"
export PYTHONUNBUFFERED=1

echo "device=$DEVICE out_dir=$BWD_HU_OUT_DIR"
cd "$SCRIPT_DIR"
exec python3 legacy_gva.py
