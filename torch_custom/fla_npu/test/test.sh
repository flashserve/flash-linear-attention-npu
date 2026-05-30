#!/bin/bash
# GDN 全量测试入口
# 用法:
#   bash test.sh --device 0              # 在 device 0 上运行全量测试
#   bash test.sh --device 0 --op causal_conv1d  # 只测指定算子
#   bash test.sh --device 0 --mode dry-run      # 只打印要执行的命令

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DEVICE_ID=""
SINGLE_OP=""
DRY_RUN=false

usage() {
    echo "用法: bash test.sh --device <N> [--op <NAME>] [--mode dry-run]"
    echo ""
    echo "选项:"
    echo "  --device N    指定 NPU device id（必填）"
    echo "  --op NAME     只测试指定算子（可选，默认全量）"
    echo "  --mode dry-run 只打印命令，不执行"
    echo ""
    echo "算子列表:"
    echo "  prepare_wy_repr_bwd_full, chunk_gated_delta_rule_bwd_dhu,"
    echo "  chunk_bwd_dv_local, causal_conv1d, prepare_wy_repr_bwd_da,"
    echo "  chunk_bwd_dqkwg, gdn_fwd_o, gdn_fwd_h, recompute_wu_fwd"
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
export TEST_LOG_DIR="$SCRIPT_DIR/test_output"
mkdir -p "$TEST_LOG_DIR"

success_list=()
fail_list=()
timeout_list=()

TIMEOUT_SEC=300

run_test() {
    local name=$1
    local cmd=$2

    if [[ -n "$SINGLE_OP" && "$name" != "$SINGLE_OP" ]]; then
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
        timeout_list+=("$name")
        return 124
    else
        local end_ts=$(date +%s)
        local elapsed=$((end_ts - start_ts))
        printf "  [FAIL]  %3ds  log: %s\n" "$elapsed" "$log_file"
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
run_test "recompute_wu_fwd"                "python3 test_npu_recompute_w_u_fwd.py"

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
