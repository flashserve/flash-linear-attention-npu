#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<'EOF'
运行 chunk_gated_delta_rule_fwd 的 ATK 原生三路双标杆精度测试。

用法：
  bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_double_benchmark.sh [device]

环境变量：
  ATK_BIN              ATK 命令，默认 atk
  REQUIRED_ATK_VERSION ATK 最低版本，默认 26.8.8
  ATK_OUTPUT_ROOT      结果根目录，默认算子目录下 atk_output/double_benchmark
  GDN_ATK_RESULT_DIR   可选精确结果目录，供分片入口调用
  GDN_ATK_CASE_JSON    用例 JSON，默认正式 500 条矩阵
  ACCURACY_START/END   可选 case 范围，必须同时设置
  GDN_ATK_MAX_TASK     ATK -mt 并发度，默认 5
  GDN_ATK_SINGLE_PROCESS 设为 1 时增加 --single_process
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
case_json=${GDN_ATK_CASE_JSON:-$op_dir/atk_chunk_gated_delta_rule_fwd.json}
max_task=${GDN_ATK_MAX_TASK:-5}
single_process=${GDN_ATK_SINGLE_PROCESS:-0}
start=${ACCURACY_START:-}
end=${ACCURACY_END:-}

[[ "$device" =~ ^[0-9]+$ ]] || { echo "device 必须是非负整数" >&2; exit 2; }
[[ "$max_task" =~ ^[1-9][0-9]*$ ]] || { echo "GDN_ATK_MAX_TASK 必须是正整数" >&2; exit 2; }
[[ "$single_process" == 0 || "$single_process" == 1 ]] || {
    echo "GDN_ATK_SINGLE_PROCESS 只能是 0 或 1" >&2
    exit 2
}
if [[ -n "$start" || -n "$end" ]]; then
    [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ && "$end" -gt "$start" ]] || {
        echo "ACCURACY_START/END 必须同时设置，且满足 0 <= start < end" >&2
        exit 2
    }
fi
[[ -f "$case_json" ]] || { echo "找不到用例：$case_json" >&2; exit 2; }
command -v "$atk_bin" >/dev/null 2>&1 || { echo "找不到 ATK 命令：$atk_bin" >&2; exit 2; }
required_atk_version=${REQUIRED_ATK_VERSION:-26.8.8}
installed_atk_version=$("$atk_bin" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true)
[[ -n "$installed_atk_version" ]] || { echo "无法获取 ATK 版本" >&2; exit 2; }
if ! printf '%s\n%s\n' "$required_atk_version" "$installed_atk_version" | sort -V -C 2>/dev/null; then
    echo "ATK 版本过低：当前 $installed_atk_version，要求 >= $required_atk_version" >&2
    exit 2
fi

case_json=$(cd -- "$(dirname -- "$case_json")" && pwd)/$(basename -- "$case_json")
case_count=$(python3 - "$case_json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    cases = json.load(handle)
if not isinstance(cases, list) or not cases:
    raise SystemExit("case JSON 必须是非空列表")
print(len(cases))
PY
)
expected_cases=$case_count
range_args=()
if [[ -n "$start" ]]; then
    [[ "$end" -le "$case_count" ]] || { echo "ACCURACY_END 超出用例数 $case_count" >&2; exit 2; }
    range_args=(-s "$start" -e "$end")
    expected_cases=$((end - start))
fi

timestamp=$(date +%Y%m%d_%H%M%S)
output_root=${ATK_OUTPUT_ROOT:-$op_dir/atk_output/double_benchmark}
run_dir=${GDN_ATK_RESULT_DIR:-$output_root/$timestamp}
[[ ! -e "$run_dir" ]] || { echo "结果目录已存在，拒绝覆盖：$run_dir" >&2; exit 2; }
mkdir -p "$run_dir"
node_yaml="$run_dir/node.yaml"
cat >"$node_yaml" <<EOF
nodes:
  - backend: npu
    task: ['accuracy']
    devices: [$device]
    name: phase6
  - backend: npu
    task: ['accuracy']
    devices: [$device]
    name: gold
EOF

execution_args=()
if [[ "$single_process" == 1 ]]; then
    execution_args+=(--single_process)
fi

task_log="$run_dir/atk_task.log"
printf '%q ' "$atk_bin" task -c "$case_json" -n "$node_yaml" \
    -p "$op_dir/executor_chunk_gated_delta_rule_fwd.py" --task accuracy \
    -bd cpu --syc_dataset --save_data input --save_data output --save_data profile \
    -mt "$max_task" "${execution_args[@]}" "${range_args[@]}" >"$run_dir/command.txt"
printf '\n' >>"$run_dir/command.txt"

set +e
(
    cd "$run_dir"
    "$atk_bin" task \
        -c "$case_json" \
        -n "$node_yaml" \
        -p "$op_dir/executor_chunk_gated_delta_rule_fwd.py" \
        --task accuracy \
        -bd cpu \
        --syc_dataset \
        --save_data input \
        --save_data output \
        --save_data profile \
        -mt "$max_task" \
        "${execution_args[@]}" \
        "${range_args[@]}"
) >"$task_log" 2>&1
task_rc=$?
set -e

if (( task_rc != 0 )); then
    echo "ATK 执行失败，rc=$task_rc，日志：$task_log" >&2
    tail -120 "$task_log" >&2 || true
    exit "$task_rc"
fi

python3 "$script_dir/validate_runtime_roles.py" "$task_log" \
    --expected-cases "$expected_cases" \
    --json-out "$run_dir/runtime_role_contract.json"

report_file=$(find "$run_dir/atk_output" -type f -name '*.xlsx' -print -quit 2>/dev/null || true)
[[ -n "$report_file" ]] || { echo "ATK 未生成 XLSX 报告：$run_dir" >&2; exit 97; }
python3 "$script_dir/summarize_atk_report.py" "$report_file" \
    --json-out "$run_dir/summary.json" \
    --expected-cases "$expected_cases"

echo "双标杆精度测试完成：$run_dir"
