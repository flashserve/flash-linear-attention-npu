#!/usr/bin/env bash
set -euo pipefail

show_help() {
  cat <<'EOF'
以可恢复分片方式运行 ChunkKdaFwdPrepare 的精度、确定性或内存矩阵。

用法：
  bash tests/atk/chunk_kda_fwd_prepare/scripts/run_matrix.sh <scope> [device]

scope：
  accuracy       200 条精度矩阵，每个 ATK 进程只执行 1 条
  determinism    432 个 TilingKey 的确定性矩阵，每个进程只执行 1 条
  mssanitizer    432 个 TilingKey 的内存矩阵，每个进程只执行 1 条

环境变量：
  KDA_PREPARE_ATK_MATRIX_ROOT   指定或续跑结果目录
  KDA_PREPARE_ATK_SHARD_SIZE    正式矩阵固定为 1
  KDA_PREPARE_ATK_MATRIX_START  从指定 case 继续，默认 0
  KDA_PREPARE_ATK_SOC           目标 SoC，正式矩阵必须显式指定
  MSS_TOOL                      内存工具，默认 memcheck
  ATK_TIMEOUT/DC_TIMEOUT         单 case 超时，默认 60 秒，上限 60 秒
  MSS_TIMEOUT                    内存检查单 case 超时，默认 1000 秒，上限 1000 秒
  DC_LOOP_NUMS                  确定性循环次数，正式矩阵固定为 50
  ATK_GM_INIT_MODE              正式精度矩阵固定为 off
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
coverage_checker="$script_dir/check_coverage.py"
soc=${KDA_PREPARE_ATK_SOC:-}
matrix_start=${KDA_PREPARE_ATK_MATRIX_START:-0}
tool=""

case "$soc" in
  ""|auto)
    echo "正式矩阵必须显式设置 KDA_PREPARE_ATK_SOC，不能使用 auto" >&2
    exit 2
    ;;
  a2|A2|ascend910b) soc=ascend910b ;;
  a3|A3|ascend910_93) soc=ascend910_93 ;;
  a5|A5|ascend950) soc=ascend950 ;;
  *) echo "不支持的 SoC：$soc" >&2; exit 2 ;;
esac

case "$scope" in
  accuracy)
    case_file="$op_dir/atk_chunk_kda_fwd_prepare.json"
    default_shard_size=1
    ;;
  determinism)
    case_file="$op_dir/atk_chunk_kda_fwd_prepare_mss.json"
    default_shard_size=1
    ;;
  mssanitizer)
    case_file="$op_dir/atk_chunk_kda_fwd_prepare_mss.json"
    default_shard_size=1
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
  set +u
  source "$path"
  set -u
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

# 在第一条 case 下发前校验三份冻结 JSON、200 条多 seed 合同和 432-key 集合。
python3 "$coverage_checker"

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
  local upper_bound=$3
  [[ "$value" =~ ^[1-9][0-9]*$ ]] && (( value <= upper_bound )) || {
    echo "$name 必须是 1 到 $upper_bound 秒之间的整数" >&2
    exit 2
  }
}

accuracy_timeout=${ATK_TIMEOUT:-60}
determinism_timeout=${DC_TIMEOUT:-60}
sanitizer_timeout=${MSS_TIMEOUT:-1000}
determinism_loops=${DC_LOOP_NUMS:-50}
accuracy_gm_mode=${ATK_GM_INIT_MODE:-off}
case "$scope" in
  accuracy) validate_timeout ATK_TIMEOUT "$accuracy_timeout" 60 ;;
  determinism) validate_timeout DC_TIMEOUT "$determinism_timeout" 60 ;;
  mssanitizer) validate_timeout MSS_TIMEOUT "$sanitizer_timeout" 1000 ;;
esac
if [[ "$scope" == "determinism" && "$determinism_loops" != "50" ]]; then
  echo "正式确定性矩阵要求 DC_LOOP_NUMS=50" >&2
  exit 2
fi
if [[ "$scope" == "accuracy" && "$accuracy_gm_mode" != "off" ]]; then
  echo "正式精度矩阵要求 ATK_GM_INIT_MODE=off" >&2
  exit 2
fi

[[ "$shard_size" =~ ^[1-9][0-9]*$ ]] || {
  echo "KDA_PREPARE_ATK_SHARD_SIZE 必须是正整数" >&2
  exit 2
}
[[ "$shard_size" == "1" ]] || {
  echo "为保证 ATK 超时逐 case 生效，正式矩阵分片大小固定为 1" >&2
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
if (( matrix_start < case_count && matrix_start % shard_size != 0 )); then
  echo "KDA_PREPARE_ATK_MATRIX_START 必须位于当前分片边界" >&2
  exit 2
fi
if (( matrix_start > 0 )) && [[ -z "$matrix_root_override" ]]; then
  echo "从非零 case 续跑时必须指定 KDA_PREPARE_ATK_MATRIX_ROOT" >&2
  exit 2
fi
# 在创建或规范化前拒绝用户提供的根目录符号链接；否则 mkdir/cd 会把
# 结果写到链接目标，后续校验无法知道原始路径是否被替换。分别计算不
# 展开符号链接和物理 realpath，连同祖先目录的链接一并拒绝。
if [[ -L "$matrix_root" ]]; then
  echo "KDA_PREPARE_ATK_MATRIX_ROOT 不能是符号链接：$matrix_root" >&2
  exit 2
fi
matrix_root_nosymlink=$(realpath -m -s -- "$matrix_root") || {
  echo "无法规范化 KDA_PREPARE_ATK_MATRIX_ROOT：$matrix_root" >&2
  exit 2
}
matrix_root_realpath=$(realpath -m -- "$matrix_root") || {
  echo "无法解析 KDA_PREPARE_ATK_MATRIX_ROOT：$matrix_root" >&2
  exit 2
}
if [[ "$matrix_root_nosymlink" != "$matrix_root_realpath" ]]; then
  echo "KDA_PREPARE_ATK_MATRIX_ROOT 路径包含符号链接：$matrix_root" >&2
  exit 2
fi
mkdir -p "$matrix_root_realpath"
matrix_root="$matrix_root_realpath"

case "$scope" in
  accuracy)
    contract_timeout=$accuracy_timeout
    contract_loops=1
    contract_gm_init=off
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
test_artifacts=(
  "$op_dir/executor_chunk_kda_fwd_prepare.py"
  "$op_dir/chunk_kda_fwd_prepare.yaml"
  "$op_dir/gen_chunk_kda_fwd_prepare.py"
  "$repo_root/tests/op_cases/chunk_kda_fwd_prepare.json"
  "$repo_root/tests/atk/common/_ascendc_common_executor.py"
  "$repo_root/tests/atk/common/check_atk_result.py"
  "$repo_root/tests/atk/common/run_with_process_deadline.py"
  "$runner"
  "$coverage_checker"
  "$script_dir/run_matrix.sh"
  "$verifier"
)
runtime_args=(
  runtime --case-file "$case_file" --soc "$soc"
  --output "$runtime_manifest"
)
for artifact in "${test_artifacts[@]}"; do
  runtime_args+=(--test-artifact "$artifact")
done
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
  --single-process-mode off
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

# 续跑前先从原始报告和日志重验完整前缀，损坏证据不能带入新矩阵。
for ((start = 0; start < matrix_start; start += shard_size)); do
  end=$((start + shard_size))
  if (( end > matrix_start )); then
    end=$matrix_start
  fi
  shard_root="$matrix_root/shard_${start}_${end}"
  summary="$shard_root/summary.json"
  console_log="$shard_root/console.log"
  [[ -f "$summary" ]] || {
    echo "续跑前缀缺少完整分片：$shard_root" >&2
    exit 2
  }
  python3 "$verifier" shard \
    "${verifier_common[@]}" \
    --shard-root "$shard_root" --console-log "$console_log" \
    --start "$start" --end "$end" --summary "$summary" \
    "${require_tiling_args[@]}"
done

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
    ATK_SINGLE_PROCESS=off
  )
  case "$scope" in
    accuracy)
      common_env+=(ACCURACY_START="$start")
      common_env+=(ACCURACY_END="$end")
      common_env+=(ATK_GM_INIT_MODE="$accuracy_gm_mode")
      common_env+=(ATK_TIMEOUT="$accuracy_timeout")
      ;;
    determinism)
      common_env+=(DETERMINISM_START="$start")
      common_env+=(DETERMINISM_END="$end")
      common_env+=(DC_LOOP_NUMS="$determinism_loops")
      common_env+=(DC_TIMEOUT="$determinism_timeout")
      ;;
    mssanitizer)
      common_env+=(MSS_START="$start")
      common_env+=(MSS_END="$end")
      common_env+=(MSS_TOOL="$tool")
      common_env+=(MSS_TIMEOUT="$sanitizer_timeout")
      common_env+=(MSS_LOG_PATH="$shard_root/${tool}.log")
      # ATK -msl 与外层 mssanitizer 共用同一份原始日志。
      common_env+=(MSS_SANITIZER_LOG_PATH="$shard_root/${tool}.log")
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

aggregate_runtime_args=(--soc "$soc")
for artifact in "${test_artifacts[@]}"; do
  aggregate_runtime_args+=(--test-artifact "$artifact")
done
python3 "$verifier" aggregate \
  "${verifier_common[@]}" "${aggregate_runtime_args[@]}" \
  --matrix-root "$matrix_root" --output "$matrix_root/aggregate_summary.json" \
  "${require_tiling_args[@]}"
echo "完整矩阵校验通过：$matrix_root"
