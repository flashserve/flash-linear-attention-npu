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

# GDN 全量测试入口
# 用法:
#   bash test.sh --device 0              # 在 device 0 上运行全量测试
#   bash test.sh --device 0 --op causal_conv1d  # 只测指定算子，多个算子用逗号分隔
#   bash test.sh --device 0 --mode dry-run      # 只打印要执行的命令

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# 源码树内建 OPP（libcust_opapi.so）存在时才把源码路径 prepend 到 PYTHONPATH；
# 否则（例如只安装了 wheel 的用户）直接使用当前环境已安装的 fla_npu，
# 避免源码树骨架 OPP 遮蔽已安装 wheel 的内嵌 OPP。
SOURCE_OPP="$SCRIPT_DIR/../fla_npu/opp/vendors/fla_npu_transformer"
if [ -f "$SOURCE_OPP/op_api/lib/libcust_opapi.so" ]; then
    export PYTHONPATH="$(cd "$SCRIPT_DIR/.." && pwd):${PYTHONPATH:-}"
fi
TEST_DEVICE_ID=""
SINGLE_OP=""
DRY_RUN=false

usage() {
    echo "用法: bash test.sh --device <N> [--op <NAME>] [--mode dry-run]"
    echo ""
    echo "选项:"
    echo "  --device N    指定 NPU device id（必填）"
    echo "  --op NAME[,NAME...]  只测试指定算子（可选，默认全量）"
    echo "  --mode dry-run 只打印命令，不执行"
    echo ""
    echo "算子列表:"
    echo "  prepare_wy_repr_bwd_full, chunk_gated_delta_rule_bwd_dhu,"
    echo "  chunk_bwd_dv_local, causal_conv1d, prepare_wy_repr_bwd_da,"
    echo "  chunk_bwd_dqkwg, gdn_fwd_o, gdn_fwd_h, recompute_w_u_fwd,"
    echo "  chunk_local_cumsum, chunk_scaled_dot_kkt"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device) TEST_DEVICE_ID="$2"; shift 2 ;;
        --op)     SINGLE_OP="$2";     shift 2 ;;
        --mode)   [[ "$2" == "dry-run" ]] && DRY_RUN=true; shift 2 ;;
        *)        usage ;;
    esac
done

if [[ -z "$TEST_DEVICE_ID" ]]; then
    echo "[ERROR] --device 参数必填"
    usage
fi

export TEST_DEVICE_ID
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export TEST_LOG_DIR="${TEST_LOG_DIR:-$SCRIPT_DIR/test_output}"
mkdir -p "$TEST_LOG_DIR"

success_list=()
fail_list=()
timeout_list=()

TIMEOUT_SEC=300

op_selected() {
    local name="$1" requested
    [[ -z "$SINGLE_OP" ]] && return 0
    local requested_ops=()
    IFS=',' read -ra requested_ops <<< "$SINGLE_OP"
    for requested in "${requested_ops[@]}"; do
        requested="${requested#"${requested%%[![:space:]]*}"}"
        requested="${requested%"${requested##*[![:space:]]}"}"
        [[ "$requested" == "$name" ]] && return 0
    done
    return 1
}

print_test_failure_excerpt() {
    local log_file="$1"
    [[ -f "$log_file" ]] || return 0

    local excerpt
    excerpt=$(
        grep -iE -B2 -A4 \
            'Traceback \(most recent call last\)|[A-Za-z_][A-Za-z0-9_.]*(Error|Exception):|ACL_ERROR|segmentation fault|core dumped' \
            "$log_file" 2>/dev/null | head -24 || true
    )
    if [[ -n "$excerpt" ]]; then
        echo "    关键错误:"
        printf '%s\n' "$excerpt"
        return 0
    fi

    excerpt=$(
        grep -iE -B2 -A3 \
            '\[ERROR\]|(^|[^[:alpha:]])(NPU|kernel|device|stream).*(error|failed)' \
            "$log_file" 2>/dev/null | head -20 || true
    )
    echo "    日志尾部:"
    if [[ -n "$excerpt" ]]; then
        printf '%s\n' "$excerpt"
    else
        tail -20 "$log_file" 2>/dev/null || true
    fi
}

run_test() {
    local name=$1
    local cmd=$2

    if ! op_selected "$name"; then
        return
    fi

    local log_file="$TEST_LOG_DIR/${name}.log"

    if $DRY_RUN; then
        echo "[DRY-RUN] $name:  TEST_DEVICE_ID=$TEST_DEVICE_ID  $cmd  > $log_file 2>&1"
        return
    fi

    printf "[RUN   ] %-34s" "$name"
    local start_ts=$(date +%s)

    if timeout $TIMEOUT_SEC bash -c "cd \"$SCRIPT_DIR\" && $cmd" > "$log_file" 2>&1; then
        local end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        printf "  [PASS]  %3ds  log: %s\n" "$elapsed" "$log_file"
        success_list+=("$name")
        return 0
    elif [[ $? -eq 124 ]]; then
        printf "  [TIMEOUT]      log: %s\n" "$log_file"
        print_test_failure_excerpt "$log_file"
        timeout_list+=("$name")
        return 124
    else
        local end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        printf "  [FAIL]  %3ds  log: %s\n" "$elapsed" "$log_file"
        print_test_failure_excerpt "$log_file"
        fail_list+=("$name")
        return 1
    fi
}

echo "=========================================="
echo "GDN 测试"
echo "=========================================="
echo "device:    $TEST_DEVICE_ID"
echo "log_dir:   $TEST_LOG_DIR"
echo "timeout:   ${TIMEOUT_SEC}s/每个测试"
echo "=========================================="
echo ""

run_test "prepare_wy_repr_bwd_full"        "python3 test_npu_prepare_wy_repr_bwd_full.py"
run_test "chunk_gated_delta_rule_bwd_dhu"  "python3 test_npu_chunk_gated_delta_rule_bwd_dhu.py"
run_test "chunk_bwd_dv_local"              "python3 test_npu_chunk_bwd_dv_local.py"
run_test "causal_conv1d"                   "python3 test_npu_causal_conv1d.py"
run_test "prepare_wy_repr_bwd_da"          "python3 test_npu_prepare_wy_repr_bwd_da.py"
run_test "chunk_bwd_dqkwg"                 "python3 test_npu_chunk_bwd_dqkwg.py"
run_test "gdn_fwd_o"                       "bash run_gdn_fwd_o.sh"
run_test "gdn_fwd_h"                       "bash run_gdn_fwd_h.sh"
run_test "recompute_w_u_fwd"                "python3 test_npu_recompute_w_u_fwd.py"
run_test "chunk_local_cumsum"              "python3 test_npu_chunk_local_cumsum.py"
run_test "chunk_scaled_dot_kkt"            "python3 test_npu_chunk_scaled_dot_kkt.py"

echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="

total=$(( ${#success_list[@]} + ${#fail_list[@]} + ${#timeout_list[@]} ))

echo "PASSED  (${#success_list[@]}/$total):"
for t in "${success_list[@]}"; do
    printf "  [PASS]  %s\n" "$t"
done

echo ""
echo "FAILED  (${#fail_list[@]}/$total):"
for t in "${fail_list[@]}"; do
    printf "  [FAIL]  %s  →  log: %s/%s.log\n" "$t" "$TEST_LOG_DIR" "$t"
done

echo ""
echo "TIMEOUT (${#timeout_list[@]}/$total):"
for t in "${timeout_list[@]}"; do
    printf "  [TIMEOUT] %s  →  log: %s/%s.log\n" "$t" "$TEST_LOG_DIR" "$t"
done

echo "=========================================="

if [[ ${#fail_list[@]} -gt 0 || ${#timeout_list[@]} -gt 0 ]]; then
    exit 1
fi
exit 0
