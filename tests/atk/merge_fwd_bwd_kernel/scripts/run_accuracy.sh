#!/usr/bin/env bash
set -euo pipefail

# merge_fwd_bwd_kernel 的 ATK 三路双标杆精度入口。
# npu_dut + cpu_lowp + --bm_device cpu（隐式 FP64 golden）。
# 不要用 run_test_cpu.sh 的 npu_dut + cpu_golden 两节点：那只会 nan_check golden，
# DUT 精度列是 "-"，报告文件名带 nan_check。

show_help() {
    cat <<'EOF'
运行 merge_fwd_bwd_kernel 的 cv_fused_double_benchmark 精度测试。

用法：
  bash tests/atk/merge_fwd_bwd_kernel/scripts/run_accuracy.sh [device]

环境变量：
  ATK_BIN              ATK 命令，默认 atk
  ATK_OUTPUT_ROOT      结果根目录，默认算子目录下 atk_output/double_benchmark
  ACCURACY_START/END   可选 case 范围，必须同时设置，且 0 <= start < end
  ATK_TIMEOUT          超时秒数，默认 14400
  CANN_ENV             可选 CANN set_env.sh
  FLA_NPU_ENV          可选 fla_npu_transformer set_env.bash
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
    exit 0
fi

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
op_dir=$(cd -- "$script_dir/.." && pwd)
device=${1:-0}
atk_bin=${ATK_BIN:-atk}
case_json=$op_dir/atk_merge_fwd_bwd_kernel.json
start=${ACCURACY_START:-}
end=${ACCURACY_END:-}

[[ "$device" =~ ^[0-9]+$ ]] || { echo "device 必须是非负整数" >&2; exit 2; }
if [[ -n "$start" || -n "$end" ]]; then
    [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ && "$end" -gt "$start" ]] || {
        echo "ACCURACY_START/END 必须同时设置，且满足 0 <= start < end" >&2
        exit 2
    }
fi
[[ -f "$case_json" ]] || { echo "找不到用例：$case_json" >&2; exit 2; }
command -v "$atk_bin" >/dev/null 2>&1 || { echo "找不到 ATK 命令：$atk_bin" >&2; exit 2; }

if [[ -n "${CANN_ENV:-}" ]]; then
    # shellcheck disable=SC1090
    source "$CANN_ENV"
fi
if [[ -n "${FLA_NPU_ENV:-}" ]]; then
    # shellcheck disable=SC1090
    source "$FLA_NPU_ENV"
fi

range_args=()
if [[ -n "$start" ]]; then
    range_args=(-s "$start" -e "$end")
fi

timestamp=$(date +%Y%m%d_%H%M%S)
output_root=${ATK_OUTPUT_ROOT:-$op_dir/atk_output/double_benchmark}
run_dir=$output_root/$timestamp
mkdir -p "$run_dir"

export TORCH_DEVICE_BACKEND_AUTOLOAD=${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}

node_yaml=$run_dir/node.yaml
cat >"$node_yaml" <<EOF
nodes:
  - backend: npu
    task: ['accuracy']
    devices: [$device]
    name: npu_dut
  - backend: cpu
    task: ['accuracy']
    name: cpu_lowp
EOF

cd "$run_dir"
"$atk_bin" task \
    -c "$case_json" \
    -n "$node_yaml" \
    -p "$op_dir/executor_merge_fwd_bwd_kernel.py" \
    --task accuracy \
    --bm_device cpu \
    --syc_dataset \
    --save_data output \
    --single_process \
    -mt 1 \
    --save_report_json \
    "${range_args[@]}" \
    -to "${ATK_TIMEOUT:-14400}"
echo "双标杆精度测试完成：$run_dir"
