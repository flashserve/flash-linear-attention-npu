#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo "运行 chunk_gated_delta_rule_bwd 的 CPU 双标杆精度测试。"
    echo "角色：NPU DUT、CPU FP32 cpu_lowp、CPU FP64 golden。"
    exit 0
fi

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
op_dir=$(cd -- "$script_dir/.." && pwd)
device=${1:-0}
atk_bin=${ATK_BIN:-atk}
case_json=${GDN_ATK_CASE_JSON:-$op_dir/atk_chunk_gated_delta_rule_bwd.json}
run_root=${ATK_OUTPUT_ROOT:-$op_dir/atk_output/double_benchmark}
max_task=${GDN_ATK_MAX_TASK:-1}
single_process=${GDN_ATK_SINGLE_PROCESS:-1}
start=${ACCURACY_START:-}
end=${ACCURACY_END:-}

[[ "$device" =~ ^[0-9]+$ ]] || { echo "device 必须是非负整数" >&2; exit 2; }
[[ "$max_task" =~ ^[1-9][0-9]*$ ]] || { echo "GDN_ATK_MAX_TASK 必须是正整数" >&2; exit 2; }
[[ -f "$case_json" ]] || { echo "找不到用例：$case_json" >&2; exit 2; }
command -v "$atk_bin" >/dev/null 2>&1 || { echo "找不到 ATK 命令：$atk_bin" >&2; exit 2; }

if [[ -z "${GDN_ATK_BENCHMARK_ARCH:-}" ]]; then
    GDN_ATK_BENCHMARK_ARCH=$(python3 - "$device" <<'PY'
import sys
import torch
import torch_npu
name = torch.npu.get_device_name(int(sys.argv[1]))
if name.startswith("Ascend910"):
    print("arch22")
elif name.startswith("Ascend950"):
    print("arch35")
else:
    raise SystemExit(f"Unsupported benchmark target: {name}")
PY
)
fi
[[ "$GDN_ATK_BENCHMARK_ARCH" == arch22 || "$GDN_ATK_BENCHMARK_ARCH" == arch35 ]] || exit 2
export GDN_ATK_BENCHMARK_ARCH
echo "CPU FP32 benchmark: $GDN_ATK_BENCHMARK_ARCH"

required_atk_version=${REQUIRED_ATK_VERSION:-26.7.8}
installed_atk_version=$($atk_bin --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true)
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
range_args=()
expected_cases=$case_count
if [[ -n "$start" || -n "$end" ]]; then
    [[ "$start" =~ ^[0-9]+$ && "$end" =~ ^[0-9]+$ && "$end" -gt "$start" && "$end" -le "$case_count" ]] || exit 2
    range_args=(-s "$start" -e "$end")
    expected_cases=$((end - start))
fi

timestamp=$(date +%Y%m%d_%H%M%S)
run_dir=${GDN_ATK_RESULT_DIR:-$run_root/$timestamp}
[[ ! -e "$run_dir" ]] || { echo "结果目录已存在，拒绝覆盖：$run_dir" >&2; exit 2; }
mkdir -p "$run_dir"
node_yaml="$run_dir/node.yaml"
cat >"$node_yaml" <<EOF
nodes:
  - backend: npu
    task: ['accuracy']
    devices: [$device]
    name: chunk_gdn_bwd
  - backend: cpu
    task: ['accuracy']
    name: cpu_lowp
EOF

execution_args=()
if [[ "$single_process" == 1 ]]; then execution_args+=(--single_process); fi
task_log="$run_dir/atk_task.log"

set +e
(
    cd "$run_dir"
    "$atk_bin" task -c "$case_json" -n "$node_yaml" -p "$op_dir/executor_chunk_gated_delta_rule_bwd.py" --task accuracy -bd cpu --syc_dataset --save_data input --save_data output -mt "$max_task" -to "${ATK_TIMEOUT:-14400}" "${execution_args[@]}" "${range_args[@]}"
) >"$task_log" 2>&1
task_rc=$?
set -e

if (( task_rc != 0 )); then
    echo "ATK 双标杆执行失败，rc=$task_rc，日志：$task_log" >&2
    tail -120 "$task_log" >&2 || true
    exit "$task_rc"
fi

python3 - "$task_log" "$expected_cases" <<'PY'
import re
import sys
from collections import Counter
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
expected = int(sys.argv[2])
roles = re.findall(r"\[gdn-double-atk\].*?case=(\d+).*?role=(dut|benchmark|golden)\b", text)
counts = Counter(role for _, role in roles)
if len(roles) != expected * 3 or set(counts) != {"dut", "benchmark", "golden"}:
    raise SystemExit(f"三路角色数量异常：cases={expected}, observed={len(roles)}, roles={dict(counts)}")
print(f"三路角色校验通过：{expected} case，{dict(counts)}")
PY

report_file=$(find "$run_dir" -type f -name '*.xlsx' -print -quit 2>/dev/null || true)
[[ -n "$report_file" ]] || { echo "ATK 未生成 XLSX 报告：$run_dir" >&2; exit 97; }
python3 - "$report_file" "$expected_cases" <<'PY'
import sys
import openpyxl

workbook = openpyxl.load_workbook(sys.argv[1], read_only=True)
rows = iter(workbook["statistic"].values)
header = list(next(rows))
accuracy_columns = [i for i, name in enumerate(header) if name and name.endswith("_精度通过")]
result_column = header.index("运行结果")
results = list(rows)
expected = int(sys.argv[2])
if len(results) != expected or not accuracy_columns:
    raise SystemExit(f"报告用例数或精度列异常：expected={expected}, actual={len(results)}")
failed = []
for row in results:
    accuracy = [row[i] for i in accuracy_columns if row[i] not in (None, "", "-")]
    if row[result_column] != "SUCCESS" or not accuracy or any(value is not True for value in accuracy):
        failed.append(row[0])
workbook.close()
if failed:
    raise SystemExit(f"双标杆精度未通过，case={failed}")
print(f"双标杆精度报告校验通过：{expected} case")
PY
echo "双标杆精度测试完成：$run_dir"
