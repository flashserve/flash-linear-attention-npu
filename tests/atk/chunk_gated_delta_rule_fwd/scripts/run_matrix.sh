#!/usr/bin/env bash
set -euo pipefail

show_help() {
    cat <<'EOF'
以可恢复分片方式运行 chunk_gated_delta_rule_fwd 双标杆精度矩阵。

用法：
  bash tests/atk/chunk_gated_delta_rule_fwd/scripts/run_matrix.sh [device]

环境变量：
  GDN_ATK_CASE_JSON    用例 JSON，默认正式 500 条矩阵
  GDN_ATK_MATRIX_LABEL 结果标签，默认 generalized500
  GDN_ATK_MATRIX_ROOT  精确矩阵结果目录
  GDN_ATK_SHARD_SIZE   每个 fresh ATK 进程的 case 数，默认 25
  GDN_ATK_MATRIX_START 从哪个 case 开始补跑，默认 0
  GDN_ATK_MAX_TASK     每分片并发度，默认 1
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    show_help
    exit 0
fi

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
op_dir=$(cd -- "$script_dir/.." && pwd)
device=${1:-0}
case_json=${GDN_ATK_CASE_JSON:-$op_dir/atk_chunk_gated_delta_rule_fwd.json}
matrix_label=${GDN_ATK_MATRIX_LABEL:-generalized500}
shard_size=${GDN_ATK_SHARD_SIZE:-25}
matrix_start=${GDN_ATK_MATRIX_START:-0}
timestamp=$(date +%Y%m%d_%H%M%S)
matrix_root_override=${GDN_ATK_MATRIX_ROOT:-}
matrix_root=${matrix_root_override:-$op_dir/atk_output/${matrix_label}_${timestamp}}

[[ "$matrix_label" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "GDN_ATK_MATRIX_LABEL 只能包含 ASCII 字母、数字、点、下划线和连字符" >&2
    exit 2
}
[[ "$shard_size" =~ ^[1-9][0-9]*$ ]] || { echo "GDN_ATK_SHARD_SIZE 必须是正整数" >&2; exit 2; }
[[ "$matrix_start" =~ ^[0-9]+$ ]] || { echo "GDN_ATK_MATRIX_START 必须是非负整数" >&2; exit 2; }
[[ -f "$case_json" ]] || { echo "找不到用例：$case_json" >&2; exit 2; }

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
(( matrix_start <= case_count )) || { echo "GDN_ATK_MATRIX_START 超出 $case_count" >&2; exit 2; }
if (( matrix_start > 0 )) && [[ -z "$matrix_root_override" ]]; then
    echo "从非零 case 续跑时必须通过 GDN_ATK_MATRIX_ROOT 指定已有矩阵目录" >&2
    exit 2
fi
mkdir -p "$matrix_root"

for ((start = matrix_start; start < case_count; start += shard_size)); do
    end=$((start + shard_size))
    if (( end > case_count )); then end=$case_count; fi
    shard_root="$matrix_root/shard_${start}_${end}"
    if [[ -f "$shard_root/summary.json" ]]; then
        python3 - "$shard_root/summary.json" "$start" "$end" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
start, end = map(int, sys.argv[2:])
summary = json.loads(summary_path.read_text(encoding="utf-8"))
ids = [int(value) for value in (summary.get("statistic") or {}).get("case_ids", [])]
expected = list(range(start, end))
if int(summary.get("expected_cases", -1)) != len(expected) or sorted(ids) != expected:
    raise SystemExit(
        f"已有 summary 与分片 {start}:{end} 不匹配："
        f"expected_cases={summary.get('expected_cases')}, case_ids={sorted(ids)}"
    )
PY
        echo "复用已有分片：$shard_root"
        continue
    fi
    [[ ! -e "$shard_root" ]] || { echo "不完整分片已存在，拒绝覆盖：$shard_root" >&2; exit 2; }
    set +e
    GDN_ATK_CASE_JSON="$case_json" \
    GDN_ATK_RESULT_DIR="$shard_root" \
    GDN_ATK_MAX_TASK=${GDN_ATK_MAX_TASK:-1} \
    GDN_ATK_SINGLE_PROCESS=${GDN_ATK_SINGLE_PROCESS:-1} \
    ACCURACY_START="$start" \
    ACCURACY_END="$end" \
        bash "$script_dir/run_double_benchmark.sh" "$device"
    shard_rc=$?
    set -e
    if (( shard_rc != 0 && shard_rc != 99 )); then
        echo "分片 $start:$end 执行失败，rc=$shard_rc" >&2
        exit "$shard_rc"
    fi
done

python3 "$script_dir/aggregate_matrix_summaries.py" "$matrix_root" \
    --expected-cases "$case_count" \
    --json-out "$matrix_root/aggregate_summary.json"
echo "矩阵运行完成：$matrix_root"
