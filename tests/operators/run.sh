#!/bin/bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
device=""
op=""
soc="${FLA_NPU_SOC:-ascend910b}"
timeout_sec="${FLA_NPU_TEST_TIMEOUT:-1800}"
dry_run=false

usage() {
    echo "用法: bash tests/operators/run.sh --device <N> [--op <NAME>] [--soc <SOC>] [--timeout <秒>] [--mode dry-run]"
    echo ""
    echo "选项:"
    echo "  --device N       指定 NPU device id（必填）"
    echo "  --op NAME        只测试一个 tests/op_cases 中注册的 Ascend C 算子"
    echo "  --soc SOC        ascend910b、ascend910_93 或 ascend950"
    echo "  --timeout 秒     单算子超时时间，默认 1800"
    echo "  --mode dry-run   只打印统一精度入口将执行的命令"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            device="${2:-}"
            shift 2
            ;;
        --op)
            op="${2:-}"
            shift 2
            ;;
        --soc)
            soc="${2:-}"
            shift 2
            ;;
        --timeout)
            timeout_sec="${2:-}"
            shift 2
            ;;
        --mode)
            if [[ "${2:-}" != "dry-run" ]]; then
                echo "[ERROR] --mode 仅支持 dry-run" >&2
                usage >&2
                exit 2
            fi
            dry_run=true
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "[ERROR] 未知参数: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$device" ]]; then
    echo "[ERROR] --device 参数必填" >&2
    usage >&2
    exit 2
fi

case "$op" in
    gdn_fwd_o) op="chunk_fwd_o" ;;
    gdn_fwd_h) op="chunk_gated_delta_rule_fwd_h" ;;
esac

args=(--soc "$soc" --device "$device" --timeout "$timeout_sec")
if [[ -n "$op" ]]; then
    args+=(--ops "$op")
fi
if $dry_run; then
    args+=(--dry-run)
fi

cd "$repo_dir"
exec python3 ci/run_operator_accuracy.py "${args[@]}"
