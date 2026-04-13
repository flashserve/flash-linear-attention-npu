#!/bin/bash

success_list=()
fail_list=()

run_test() {
    name=$1
    cmd=$2

    echo "=============================="
    echo "[RUN] $name"
    echo "=============================="

    eval $cmd

    if [ $? -ne 0 ]; then
        echo "[FAIL] $name"
        fail_list+=("$name")
    else
        echo "[PASS] $name"
        success_list+=("$name")
    fi
}

# ------------------------------
# 运行测试
# ------------------------------

run_test "prepare_wy_repr_bwd_full" "python3 test_npu_prepare_wy_repr_bwd_full.py"

run_test "chunk_gated_delta_rule_bwd_dhu" "python3 test_npu_chunk_gated_delta_rule_bwd_dhu.py"

run_test "chunk_bwd_dv_local" "python3 test_npu_chunk_bwd_dv_local.py"

run_test "prepare_wy_repr_bwd_da" "python3 test_npu_prepare_wy_repr_bwd_da.py"

run_test "chunk_bwd_dqkwg" "python3 test_npu_chunk_bwd_dqkwg.py"

run_test "gdn_fwd_o" "bash run_gdn_fwd_o.sh"

run_test "gdn_fwd_h" "bash run_gdn_fwd_h.sh"


# ------------------------------
# 汇总结果
# ------------------------------

echo
echo "===================================="
echo "Test Summary"
echo "===================================="

echo "PASSED (${#success_list[@]}):"
for t in "${success_list[@]}"; do
    echo "  ✔ $t"
done

echo
echo "FAILED (${#fail_list[@]}):"
for t in "${fail_list[@]}"; do
    echo "  ✘ $t"
done

echo "===================================="