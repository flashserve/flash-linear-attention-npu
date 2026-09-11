#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
以可恢复分片方式运行 ChunkKdaFwdPrepare 的精度、确定性或内存矩阵。

用法：
  bash tests/atk/chunk_kda_fwd_prepare/scripts/run_matrix.sh <scope> [device]

scope：
  accuracy       200 条精度矩阵，默认每片 25 条
  determinism    432 个 TilingKey 的确定性矩阵，默认每片 4 条
  mssanitizer    432 个 TilingKey 的内存矩阵，默认每片 4 条

环境变量：
  KDA_PREPARE_ATK_MATRIX_ROOT   指定或续跑结果目录
  KDA_PREPARE_ATK_SHARD_SIZE    覆盖默认分片大小
  KDA_PREPARE_ATK_MATRIX_START  从指定 case 继续，默认 0
  KDA_PREPARE_ATK_SOC           目标 SoC，默认 auto
  MSS_TOOL                      内存工具，默认 memcheck
  ATK_TIMEOUT/DC_TIMEOUT/MSS_TIMEOUT
                                单 case 超时，均默认 60 秒
  DC_LOOP_NUMS                  确定性循环次数，正式矩阵固定为 50
  ATK_GM_INIT_MODE              正式精度矩阵固定为 on
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  show_help
  exit 0
fi

scope=${1:-}
device=${2:-0}
case "$scope" in
  accuracy|determinism|mssanitizer) ;;
  *) show_help; echo "必须指定合法 scope" >&2; exit 2 ;;
esac

script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
op_dir=$(cd -- "$script_dir/.." && pwd)
repo_root=$(cd -- "$op_dir/../../.." && pwd)
runner="$repo_root/tests/atk/run_test_cpu.sh"
verifier="$script_dir/verify_matrix.py"
soc=${KDA_PREPARE_ATK_SOC:-auto}
matrix_start=${KDA_PREPARE_ATK_MATRIX_START:-0}
tool=""

case "$scope" in
  accuracy)
    case_file="$op_dir/atk_chunk_kda_fwd_prepare.json"
    default_shard_size=25
    ;;
  determinism)
    case_file="$op_dir/atk_chunk_kda_fwd_prepare_mss.json"
    default_shard_size=4
    ;;
  mssanitizer)
    case_file="$op_dir/atk_chunk_kda_fwd_prepare_mss.json"
    default_shard_size=4
    tool=${MSS_TOOL:-memcheck}
    case "$tool" in
      memcheck|racecheck|initcheck|synccheck) ;;
      *) echo "不支持的 MSS_TOOL：$tool" >&2; exit 2 ;;
    esac
    ;;
esac

source_checked_env() {
  local label=$1
  local path=$2
  [[ -f "$path" ]] || {
    echo "$label不存在：$path" >&2
    exit 2
  }
  # shellcheck disable=SC1090
  source "$path"
}

# 指纹校验与统一 runner 必须处于同一套环境，避免校验和执行加载不同 OPP。
if [[ -n "${ATK_ENV:-}" ]]; then
  source_checked_env "ATK 环境" "${ATK_ENV}/bin/activate"
fi
if [[ -n "${CANN_ENV:-}" ]]; then
  source_checked_env "CANN 环境" "$CANN_ENV"
fi
if [[ -n "${FLA_NPU_ENV:-${FLA_NPU_OPP_ENV:-}}" ]]; then
  source_checked_env \
    "fla_npu_transformer 环境" "${FLA_NPU_ENV:-${FLA_NPU_OPP_ENV:-}}"
fi

shard_size=${KDA_PREPARE_ATK_SHARD_SIZE:-$default_shard_size}
timestamp=$(date +%Y%m%d_%H%M%S)
label=$scope
if [[ -n "$tool" ]]; then
  label="${scope}_${tool}"
fi
matrix_root_override=${KDA_PREPARE_ATK_MATRIX_ROOT:-}
matrix_root=${matrix_root_override:-$op_dir/atk_output/${label}_${timestamp}}

validate_timeout() {
  local name=$1
  local value=$2
  [[ "$value" =~ ^[1-9][0-9]*$ ]] && (( value <= 60 )) || {
    echo "$name 必须是 1 到 60 秒之间的整数" >&2
    exit 2
  }
}

accuracy_timeout=${ATK_TIMEOUT:-60}
determinism_timeout=${DC_TIMEOUT:-60}
sanitizer_timeout=${MSS_TIMEOUT:-60}
determinism_loops=${DC_LOOP_NUMS:-50}
accuracy_gm_mode=${ATK_GM_INIT_MODE:-on}
case "$scope" in
  accuracy) validate_timeout ATK_TIMEOUT "$accuracy_timeout" ;;
  determinism) validate_timeout DC_TIMEOUT "$determinism_timeout" ;;
  mssanitizer) validate_timeout MSS_TIMEOUT "$sanitizer_timeout" ;;
esac
if [[ "$scope" == "determinism" && "$determinism_loops" != "50" ]]; then
  echo "正式确定性矩阵要求 DC_LOOP_NUMS=50" >&2
  exit 2
fi
if [[ "$scope" == "accuracy" && "$accuracy_gm_mode" != "on" ]]; then
  echo "正式精度矩阵要求 ATK_GM_INIT_MODE=on" >&2
  exit 2
fi

[[ "$shard_size" =~ ^[1-9][0-9]*$ ]] || {
  echo "KDA_PREPARE_ATK_SHARD_SIZE 必须是正整数" >&2
  exit 2
}
[[ "$matrix_start" =~ ^[0-9]+$ ]] || {
  echo "KDA_PREPARE_ATK_MATRIX_START 必须是非负整数" >&2
  exit 2
}
[[ -s "$case_file" ]] || { echo "找不到用例文件：$case_file" >&2; exit 2; }

case_count=$(python3 - "$case_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    cases = json.load(handle)
if not isinstance(cases, list) or not cases:
    raise SystemExit("用例文件必须是非空列表")
print(len(cases))
PY
)
expected_case_count=432
if [[ "$scope" == "accuracy" ]]; then
  expected_case_count=200
fi
(( case_count == expected_case_count )) || {
  echo "$scope 正式矩阵必须为 $expected_case_count 条，实际为 $case_count 条" >&2
  exit 2
}
(( matrix_start <= case_count )) || {
  echo "KDA_PREPARE_ATK_MATRIX_START 超出用例总数 $case_count" >&2
  exit 2
}
if (( matrix_start > 0 )) && [[ -z "$matrix_root_override" ]]; then
  echo "从非零 case 续跑时必须指定 KDA_PREPARE_ATK_MATRIX_ROOT" >&2
  exit 2
fi
mkdir -p "$matrix_root"
matrix_root=$(cd -- "$matrix_root" && pwd)

case "$scope" in
  accuracy)
    contract_timeout=$accuracy_timeout
    contract_loops=1
    contract_gm_init=on
    ;;
  determinism)
    contract_timeout=$determinism_timeout
    contract_loops=$determinism_loops
    contract_gm_init=not_applicable
    ;;
  mssanitizer)
    contract_timeout=$sanitizer_timeout
    contract_loops=1
    contract_gm_init=not_applicable
    ;;
esac

runtime_manifest="$matrix_root/runtime_manifest.json"
runtime_args=(
  runtime --case-file "$case_file" --soc "$soc"
  --output "$runtime_manifest"
)
if [[ "$scope" == "mssanitizer" ]]; then
  runtime_args+=(--require-sanitizer)
fi
if [[ "$scope" != "accuracy" ]]; then
  runtime_args+=(--require-complete-key-set)
fi
python3 "$verifier" "${runtime_args[@]}"

verifier_common=(
  --case-file "$case_file" --scope "$scope" --tool "$tool"
  --runtime-manifest "$runtime_manifest"
  --timeout "$contract_timeout" --loop-nums "$contract_loops"
  --gm-init-mode "$contract_gm_init"
)

require_tiling_args=()
if [[ "$scope" != "accuracy" ]]; then
  # 同时保留 Host 计算结果和实际 Launch 使用值，避免只验证静态期望矩阵。
  export ASCEND_GLOBAL_LOG_LEVEL=3
  export ASCEND_MODULE_LOG_LEVEL=OP=0
  export ASCEND_SLOG_PRINT_TO_STDOUT=1
  export ASCEND_LOG_SYNC_SAVE=1
  require_tiling_args=(--require-tiling-log)
fi

for ((start = matrix_start; start < case_count; start += shard_size)); do
  end=$((start + shard_size))
  if (( end > case_count )); then
    end=$case_count
  fi
  shard_root="$matrix_root/shard_${start}_${end}"
  summary="$shard_root/summary.json"
  console_log="$shard_root/console.log"

  if [[ -f "$summary" ]]; then
    # 每次续跑都重新解析原始报告和日志，不信任旧摘要本身。
    python3 "$verifier" shard \
      "${verifier_common[@]}" \
      --shard-root "$shard_root" --console-log "$console_log" \
      --start "$start" --end "$end" --summary "$summary" \
      "${require_tiling_args[@]}"
    continue
  fi
  [[ ! -e "$shard_root" ]] || {
    echo "不完整分片已存在，拒绝覆盖：$shard_root" >&2
    exit 2
  }
  mkdir -p "$shard_root"

  common_env=(
    ATK_OUTPUT_ROOT="$shard_root"
    CASE_START="$start"
    CASE_END="$end"
  )
  case "$scope" in
    accuracy)
      common_env+=(ATK_GM_INIT_MODE="$accuracy_gm_mode")
      common_env+=(ATK_TIMEOUT="$accuracy_timeout")
      ;;
    determinism)
      common_env+=(DC_LOOP_NUMS="$determinism_loops")
      common_env+=(DC_TIMEOUT="$determinism_timeout")
      ;;
    mssanitizer)
      common_env+=(MSS_TOOL="$tool")
      common_env+=(MSS_TIMEOUT="$sanitizer_timeout")
      common_env+=(MSS_LOG_PATH="$shard_root/${tool}.log")
      ;;
  esac

  set +e
  env "${common_env[@]}" bash "$runner" \
    -op=chunk_kda_fwd_prepare -npu_device_id="$device" \
    -soc="$soc" -scope="$scope" 2>&1 | tee "$console_log"
  shard_rc=${PIPESTATUS[0]}
  set -e
  if (( shard_rc != 0 )); then
    echo "分片 $start:$end 未通过，退出码 $shard_rc" >&2
    exit "$shard_rc"
  fi

  python3 "$verifier" shard \
    "${verifier_common[@]}" \
    --shard-root "$shard_root" --console-log "$console_log" \
    --start "$start" --end "$end" --summary "$summary" \
    "${require_tiling_args[@]}"
done

python3 "$verifier" aggregate \
  "${verifier_common[@]}" \
  --matrix-root "$matrix_root" --output "$matrix_root/aggregate_summary.json" \
  "${require_tiling_args[@]}"
echo "完整矩阵校验通过：$matrix_root"
