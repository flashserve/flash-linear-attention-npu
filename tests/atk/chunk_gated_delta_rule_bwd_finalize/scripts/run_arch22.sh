#!/usr/bin/env bash
set -euo pipefail

# 新增 A2/A3 用例的 ATK 入口；原统一脚本和 A5 路径保持不变。
scope=${1:-accuracy}
device=${2:-0}
soc=${3:-A2}
case "$scope" in accuracy|mssanitizer|determinism|keys) ;; *) echo "scope: accuracy|mssanitizer|determinism|keys" >&2; exit 2 ;; esac
[[ "$device" =~ ^[0-9]+$ ]] || { echo "device 必须为非负整数" >&2; exit 2; }
[[ "$soc" == A2 || "$soc" == A3 ]] || { echo "soc 必须为 A2 或 A3" >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
op_dir=$(cd -- "$script_dir/.." && pwd)
python=${ATK_PYTHON:-python3}
atk_bin=${ATK_BIN:-$(command -v atk)}
run_dir=${REVIEWFIX_RESULT_DIR:-$op_dir/atk_output/arch22_${scope}_$(date +%Y%m%d_%H%M%S)}
mkdir -p "$run_dir"
run_dir=$(cd -- "$run_dir" && pwd)
[[ ! -f "$run_dir/cases.json" && ! -f "$run_dir/summary.json" ]] || { echo "拒绝覆盖阶段结果：$run_dir" >&2; exit 2; }
: "${FINALIZE_SOURCE_COMMIT:?必须绑定被测源码提交}"
: "${FINALIZE_BINDING_PATH:?必须提供冻结 binding 路径}"
: "${FINALIZE_BINDING_SHA256:?必须提供冻结 binding 原始字节 SHA}"
export FINALIZE_CPU_EVIDENCE=${FINALIZE_CPU_EVIDENCE:-$run_dir/role-evidence}
select_args=(--scope "$scope" --soc "$soc" --output "$run_dir")
if [[ -n ${ARCH22_SOURCE_CASE_IDS+x} ]]; then
    select_args+=(--source-case-ids "$ARCH22_SOURCE_CASE_IDS")
fi
count=$("$python" -B "$script_dir/arch22_cases.py" "${select_args[@]}")
atk_cmd=("$python" -B "$atk_bin")
version=$("${atk_cmd[@]}" --version)
printf '%s\n' "$version" > "$run_dir/atk-version.txt"
"$python" -B - "$version" <<'PY'
import re,sys
m=re.search(r'(\d+)\.(\d+)\.(\d+)', sys.argv[1])
if not m or tuple(map(int,m.groups())) < (26,8,8):
    raise SystemExit('ATK 必须 >=26.8.8；mixed_tolerance_bm 不能降级')
PY
args=()
if [[ "$scope" == accuracy ]]; then
    cat > "$run_dir/node.yaml" <<EOF
nodes:
  - backend: npu
    task: ['accuracy']
    devices: [$device]
    name: finalize
  - backend: cpu
    task: ['accuracy']
    name: cpu_lowp
EOF
    args=(task -c "$run_dir/cases.json" -n "$run_dir/node.yaml" -p "$op_dir/executor_chunk_gated_delta_rule_bwd_finalize.py"
          --task accuracy -bd cpu --syc_dataset -mt 1 --single_process -to "${ATK_TIMEOUT:-14400}")
elif [[ "$scope" == mssanitizer ]]; then
    command -v mssanitizer >/dev/null
    args=(node --name finalize --backend npu --devices "$device" task -c "$run_dir/cases.json"
          -p "$op_dir/executor_chunk_gated_delta_rule_bwd_finalize.py" --task run --mssanitizer -msl "$run_dir/mssanitizer.log"
          -mt 1 --single_process -to "${ATK_TIMEOUT:-14400}")
elif [[ "$scope" == determinism ]]; then
    args=(node --name finalize --backend npu --devices "$device" task -c "$run_dir/cases.json"
          -p "$op_dir/executor_chunk_gated_delta_rule_bwd_finalize.py" --task accuracy_dc
          --dc_loop_nums "${DC_LOOP_NUMS:-50}" -to "${DC_TIMEOUT:-3600}")
else
    export ASCEND_GLOBAL_LOG_LEVEL=0
    export ASCEND_SLOG_PRINT_TO_STDOUT=1
    args=(node --name finalize --backend npu --devices "$device" task -c "$run_dir/cases.json"
          -p "$op_dir/executor_chunk_gated_delta_rule_bwd_finalize.py" --task run
          -mt 1 --single_process -to "${ATK_TIMEOUT:-3600}")
fi
command=("${atk_cmd[@]}" "${args[@]}")
if [[ "$scope" == mssanitizer ]]; then
    command=(mssanitizer --tool="${MSS_TOOL:-memcheck}" --log-file="$run_dir/mssanitizer.log" --log-level=info -- "${command[@]}")
fi
printf '%q ' "${command[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
set +e
(cd "$run_dir"; "${command[@]}") > "$run_dir/atk_task.log" 2>&1
task_rc=$?
set -e
"$python" -B "$script_dir/summarize_arch22.py" "$run_dir" --scope "$scope" --task-rc "$task_rc" --expected-cases "$count"
