#!/usr/bin/env bash
set -euo pipefail

# 单算子 ATK 混合容差单标杆一键验证脚本。
# 所有测试动作均由 ATK 发起；内存检测由 mssanitizer 包裹 ATK run 任务。

show_usage() {
  cat <<'EOF'
用法：
  bash tests/atk/run_test_cpu.sh -op=<算子名> [-npu_device_id=<NPU卡号>]

常用参数：
  -op=chunk_kda_fwd              ATK 算子目录名
  -npu_device_id=0               传给 ATK node --devices 的 NPU 卡号，默认 0；gen_cases 不需要
  -soc=ascend910b                可选：ascend910b/A2、ascend910_93/A3、ascend950/A5；默认 auto 自动探测
  -scope=all                     可选：all、accuracy、performance、determinism、mssanitizer、gen_cases
                                 all 包含 accuracy/performance/determinism/mssanitizer；
                                 gen_cases 只生成精度候选用例，不在 all 中

常用环境变量：
  ATK_ENV                        ATK 虚拟环境目录，设置后 source "$ATK_ENV/bin/activate"
  CANN_ENV                       CANN set_env.sh 路径，设置后 source
  FLA_NPU_ENV                    fla_npu_transformer set_env.bash 路径，设置后 source
  ATK_OUTPUT_ROOT                输出根目录，默认 ./atk_output
  ATK_GM_INIT_MODE               GM 数据初始化模式，默认 on（Prepare 默认 off）；可设 on/off
  ATK_SINGLE_PROCESS             是否向 ATK 传 -sp，默认 on（Prepare 默认 off）；可设 on/off
  ATK_TIMEOUT                    精度阶段超时，默认 14400（Prepare 默认 60）
  DC_LOOP_NUMS                   确定性循环次数，默认 50（与 ATK 一致）
  DC_TIMEOUT                     确定性阶段超时，默认 3600（Prepare 默认 60）
  PERFORMANCE_TIMEOUT            性能阶段超时，默认 2000（Prepare 默认 60）
  CASE_START/CASE_END            通用 case 顺序范围；不设置时不传 -s/-e，ATK 执行全部用例
  ACCURACY_START/ACCURACY_END    精度与 NaN 检测 case 范围
  PERFORMANCE_START/PERFORMANCE_END  性能 case 范围
  DETERMINISM_START/DETERMINISM_END  确定性 case 范围
  MSS_START/MSS_END              mssanitizer case 范围
  MSS_TOOL                       mssanitizer 工具，默认 memcheck
  MSS_TIMEOUT                    内存检测单用例超时；默认不设置（Prepare 默认 1000）
  MSS_LOG_PATH                   ATK -msl 日志路径，默认 ${ATK_OUTPUT_ROOT}/mssanitizer_<op>_<时间戳>.log
  MSS_SANITIZER_LOG_PATH         外层 mssanitizer 原始日志；Prepare 必须与 MSS_LOG_PATH 相同
  GEN_CASES_DTYPE_NUMBERS        生成用例时传给 atk case -dt，默认 100；双 dtype 算子生成 200 条
  GEN_CASES_EXTRA_NUMBERS        生成用例时传给 atk case -en，默认 0
  GEN_CASES_SEED                 生成用例随机种子，默认 20260813

示例：
  bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd
  bash tests/atk/run_test_cpu.sh -op=chunk_bwd_dqkwg -scope=performance
  bash tests/atk/run_test_cpu.sh -op=chunk_bwd_dqkwg -scope=gen_cases
  CASE_START=0 CASE_END=1 bash tests/atk/run_test_cpu.sh -op=chunk_bwd_dqkwg
EOF
}

log_info() {
  echo "[ATK CPU标杆验证] $*"
}

die() {
  echo "[ATK CPU标杆验证] 错误：$*" >&2
  exit 1
}

# 全局变量：记录已执行的测试类型与未通过项数
RAN_TYPES=()
RESULT_FAIL_COUNT=0
PERFORMANCE_RUN_START_NS=""

# 记录已执行的测试阶段，供最终汇总使用。
record_ran_type() {
  RAN_TYPES+=("$1")
}

# 统一检查已跑阶段的结果并输出汇总。
# 多项时输出每项结果 + 汇总行；单项时仅输出该项结果，不显示汇总行。
print_result_summary() {
  if [[ ! -f "$RESULT_CHECK_PY" ]]; then
    if [[ "$OP" == "chunk_kda_fwd_prepare" ]]; then
      die "Prepare 必须执行结果检查，但找不到 ${RESULT_CHECK_PY}"
    fi
    log_info "跳过结果检查：找不到 ${RESULT_CHECK_PY}"
    return 0
  fi
  local fail=0 ran="${#RAN_TYPES[@]}"
  [[ $ran -gt 0 ]] || return 0
  local pass=0
  for t in "${RAN_TYPES[@]}"; do
    local checker_args=(
      --type "$t"
      --output-root "$ATK_OUTPUT_ROOT"
      --op "$OP"
    )
    if [[ "$OP" == "chunk_kda_fwd_prepare" && "$t" == "accuracy" ]]; then
      checker_args+=(--require-accuracy-pass)
    fi
    if [[ "$OP" == "chunk_kda_fwd_prepare" && "$t" == "performance" ]]; then
      checker_args+=(--expected-case-file "$PERFORMANCE_CASE_FILE")
      checker_args+=(--require-performance-stability)
      if [[ -n "$PERFORMANCE_START" || -n "$PERFORMANCE_END" ]]; then
        checker_args+=(
          --expected-case-start "$PERFORMANCE_START"
          --expected-case-end "$PERFORMANCE_END"
        )
      fi
    fi
    if [[ "$t" == "performance" && -n "$PERFORMANCE_RUN_START_NS" ]]; then
      checker_args+=(--report-min-mtime-ns "$PERFORMANCE_RUN_START_NS")
    fi
    if python3 "$RESULT_CHECK_PY" "${checker_args[@]}"; then
      pass=$((pass + 1))
    else
      fail=$((fail + 1))
    fi
  done
  RESULT_FAIL_COUNT=$fail
  # 仅多项时打印汇总行，单项已由上面输出，无需重复
  if [[ $ran -gt 1 ]]; then
    log_info "结果汇总：${pass}/${ran} 通过, ${fail} 项失败"
  fi
}

source_env_file() {
  local label="$1"
  local file_path="$2"
  if [[ -f "$file_path" ]]; then
    log_info "加载${label}：${file_path}"
    set +u
    # shellcheck source=/dev/null
    source "$file_path"
    set -u
  fi
}

should_run() {
  local stage="$1"
  if [[ "$stage" == "gen_cases" ]]; then
    [[ "$RUN_SCOPE" == "gen_cases" ]]
    return
  fi
  [[ "$RUN_SCOPE" == "all" || "$RUN_SCOPE" == "$stage" ]]
}

validate_bounded_timeout() {
  local name="$1"
  local value="$2"
  local upper_bound="${3:-60}"
  [[ "$value" =~ ^[1-9][0-9]*$ ]] && (( value <= upper_bound )) || \
    die "${name} 必须是 1 到 ${upper_bound} 秒之间的整数"
}

# ATK 的 -to 是 worker soft timeout；原生 NPU 调用不响应 soft timeout 时，
# 仍需由进程级截止回收整个 ATK 进程组。额外 30 秒只留给进程启动与清理，
# 每条 case 的逻辑超时仍由传给 ATK 的 60/1000 秒控制。
run_with_prepare_hard_deadline() {
  local logical_timeout="$1"
  local case_start="$2"
  local case_end="$3"
  shift 3
  if [[ "$OP" != "chunk_kda_fwd_prepare" ]] ||
     [[ -z "$case_start" || -z "$case_end" ]] ||
     [[ ! "$case_start" =~ ^[0-9]+$ || ! "$case_end" =~ ^[0-9]+$ ]] ||
     (( case_end - case_start != 1 )); then
    "$@"
    return
  fi

  command -v timeout >/dev/null 2>&1 || \
    die "chunk_kda_fwd_prepare 正式测试需要 GNU timeout 提供进程级硬截止"
  local hard_timeout=$((logical_timeout + 30))
  local command_rc=0
  if timeout --signal=TERM --kill-after=10s "${hard_timeout}s" "$@"; then
    return 0
  else
    command_rc=$?
  fi
  if (( command_rc == 124 || command_rc == 137 )); then
    echo "chunk_kda_fwd_prepare 测试超过进程级硬截止 ${hard_timeout}s（case 合同 ${logical_timeout}s）" >&2
  fi
  return "$command_rc"
}

# Prepare 的正式矩阵要求超时真正作用于每条 worker case。统一入口也按
# 各 scope 的约定上限做硬校验，避免绕过算子专用分片脚本。
validate_chunk_kda_fwd_prepare_contract() {
  [[ "$OP" == "chunk_kda_fwd_prepare" ]] || return 0
  [[ "$RUN_SCOPE" == "gen_cases" ]] && return 0
  [[ "$RUN_SCOPE" != "all" ]] || \
    die "chunk_kda_fwd_prepare 的 all 不能闭合四种 sanitizer，请使用算子 README 中的 run_matrix.sh"

  case "$ATK_SINGLE_PROCESS" in
    off|disable|false|0) ;;
    *) die "chunk_kda_fwd_prepare 正式测试要求 ATK_SINGLE_PROCESS=off" ;;
  esac
  if should_run accuracy; then
    validate_bounded_timeout ATK_TIMEOUT "$ATK_TIMEOUT"
    case "$ATK_GM_INIT_MODE" in
      off|disable|false|0) ;;
      *) die "chunk_kda_fwd_prepare 正式精度测试要求 ATK_GM_INIT_MODE=off" ;;
    esac
  fi
  if should_run performance; then
    validate_bounded_timeout PERFORMANCE_TIMEOUT "$PERFORMANCE_TIMEOUT"
  fi
  if should_run determinism; then
    validate_bounded_timeout DC_TIMEOUT "$DC_TIMEOUT"
    [[ "$DC_LOOP_NUMS" == "50" ]] || \
      die "chunk_kda_fwd_prepare 正式确定性测试要求 DC_LOOP_NUMS=50"
  fi
  if should_run mssanitizer; then
    validate_bounded_timeout MSS_TIMEOUT "$MSS_TIMEOUT" 1000
  fi
}

validate_case_json() {
  local label="$1"
  local file_path="$2"
  [[ -s "$file_path" ]] || die "${label}不存在或为空：${file_path}"
  python3 - "$label" "$file_path" <<'PY'
import json
import sys

label, file_path = sys.argv[1:]
try:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise SystemExit(f"{label}无法解析：{file_path}：{error}")

if not isinstance(data, (dict, list)) or len(data) == 0:
    raise SystemExit(f"{label}没有可执行用例：{file_path}")
PY
}

set_case_range_args() {
  local label="$1"
  local start="$2"
  local end="$3"
  CASE_RANGE_ARGS=()
  if [[ -n "$start" || -n "$end" ]]; then
    [[ -n "$start" && -n "$end" ]] || die "${label} 需要同时设置 start 和 end"
    CASE_RANGE_ARGS=(-s "$start" -e "$end")
  fi
}

resolve_physical_device_id() {
  local logical_id="$1" visible physical_id
  [[ "$logical_id" =~ ^[0-9]+$ ]] || return 1
  visible="${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-}}"
  if [[ -z "$visible" ]]; then
    echo "$logical_id"
    return 0
  fi

  local -a physical_ids
  IFS=',' read -r -a physical_ids <<< "$visible"
  (( logical_id < ${#physical_ids[@]} )) || return 1
  physical_id="${physical_ids[$logical_id]}"
  physical_id="${physical_id//[[:space:]]/}"
  [[ "$physical_id" =~ ^[0-9]+$ ]] || return 1
  echo "$physical_id"
}

# 从 npu-smi 中只解析当前 ATK 逻辑卡对应的物理卡型号。
detect_soc_from_npu() {
  local logical_id="$1" physical_id name normalized
  physical_id=$(resolve_physical_device_id "$logical_id") || return 0
  name=$(npu-smi info 2>/dev/null | awk -F'|' -v target="$physical_id" '
      {
          first = $2;
          second = $3;
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", first);
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", second);
          split(first, fields, /[[:space:]]+/);
          if (fields[1] == target && fields[2] != "") {
              print fields[2];
              exit;
          }
          if (first == target && second != "") {
              print second;
              exit;
          }
      }
  ' || true)
  normalized=$(printf '%s' "$name" | tr '[:lower:]' '[:upper:]')

  if [[ "$normalized" == ASCEND950* || "$normalized" == 950* \
      || "$normalized" =~ ^957[0-9A-F]$ ]]; then
    echo "ascend950"
  elif [[ "$normalized" =~ 910[_[:space:]-]*93 ]]; then
    echo "ascend910_93"
  elif [[ "$normalized" =~ 910B[0-9]* ]]; then
    echo "ascend910b"
  else
    echo ""
  fi
}

# 校验 ATK 版本不低于 REQUIRED_ATK_VERSION（使用 GNU sort -V 比较）。
check_atk_version() {
  local required="$REQUIRED_ATK_VERSION"
  local installed
  installed="$("$ATK_BIN" --version 2>/dev/null | sed -nE \
    's/^[[:space:]]*([0-9]+\.[0-9]+\.[0-9]+)[[:space:]]*$/\1/p' | head -n1 || true)"
  [[ -n "$installed" ]] || die "无法获取 ATK 版本，请确认 atk 可正常执行（atk --version）"
  if printf '%s\n%s\n' "$required" "$installed" | sort -V -C 2>/dev/null; then
    log_info "ATK 版本：${installed}（要求 >= ${required}）"
  else
    die "ATK 版本过低：当前 ${installed}，要求 >= ${required}，请升级 ATK"
  fi
}

# 根据 ATK_GM_INIT_MODE 解析是否向 ATK 传入 --gm_init_flag。
# on（默认）：开启；off：关闭。
resolve_gm_init_args() {
  local mode="$ATK_GM_INIT_MODE"
  local enable=""
  case "$mode" in
    on|enable|true|1) enable="on" ;;
    off|disable|false|0) enable="off" ;;
    *) die "不支持的 ATK_GM_INIT_MODE：${mode}，请使用 on/off" ;;
  esac
  GM_INIT_ARGS=()
  if [[ "$enable" == "on" ]]; then
    GM_INIT_ARGS=(--gm_init_flag)
  fi
  log_info "GM 数据初始化（ATK_GM_INIT_MODE=${mode}）：${enable}"
}

resolve_single_process_args() {
  case "$ATK_SINGLE_PROCESS" in
    on|enable|true|1) SINGLE_PROCESS_ARGS=(-sp) ;;
    off|disable|false|0) SINGLE_PROCESS_ARGS=() ;;
    *) die "不支持的 ATK_SINGLE_PROCESS：${ATK_SINGLE_PROCESS}，请使用 on/off" ;;
  esac
}

OP=""
NPU_DEVICE_ID="${NPU_DEVICE_ID:-0}"
SOC="${SOC:-auto}"
RUN_SCOPE="${RUN_SCOPE:-all}"
ATK_GM_INIT_MODE="${ATK_GM_INIT_MODE:-}"
ATK_SINGLE_PROCESS="${ATK_SINGLE_PROCESS:-}"
REQUIRED_ATK_VERSION="${REQUIRED_ATK_VERSION:-26.8.8}"
ATK_TIMEOUT="${ATK_TIMEOUT:-}"
DC_LOOP_NUMS="${DC_LOOP_NUMS:-50}"
DC_TIMEOUT="${DC_TIMEOUT:-}"
PERFORMANCE_TIMEOUT="${PERFORMANCE_TIMEOUT:-}"
CASE_START="${CASE_START:-}"
CASE_END="${CASE_END:-}"
MSS_TOOL="${MSS_TOOL:-memcheck}"
MSS_TIMEOUT="${MSS_TIMEOUT:-}"
MSS_LOG_PATH="${MSS_LOG_PATH:-}"
MSS_SANITIZER_LOG_PATH="${MSS_SANITIZER_LOG_PATH:-}"
GEN_CASES_DTYPE_NUMBERS="${GEN_CASES_DTYPE_NUMBERS:-100}"
GEN_CASES_EXTRA_NUMBERS="${GEN_CASES_EXTRA_NUMBERS:-0}"
GEN_CASES_SEED="${GEN_CASES_SEED:-20260813}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -op=*) OP="${1#-op=}" ;;
    -op)
      shift
      [[ $# -gt 0 ]] || die "参数 -op 需要取值"
      OP="$1"
      ;;
    --op=*) OP="${1#--op=}" ;;
    --op)
      shift
      [[ $# -gt 0 ]] || die "参数 --op 需要取值"
      OP="$1"
      ;;
    -npu_device_id=*) NPU_DEVICE_ID="${1#-npu_device_id=}" ;;
    -npu_device_id)
      shift
      [[ $# -gt 0 ]] || die "参数 -npu_device_id 需要取值"
      NPU_DEVICE_ID="$1"
      ;;
    --npu_device_id=*) NPU_DEVICE_ID="${1#--npu_device_id=}" ;;
    --npu_device_id)
      shift
      [[ $# -gt 0 ]] || die "参数 --npu_device_id 需要取值"
      NPU_DEVICE_ID="$1"
      ;;
    -soc=*) SOC="${1#-soc=}" ;;
    -soc)
      shift
      [[ $# -gt 0 ]] || die "参数 -soc 需要取值"
      SOC="$1"
      ;;
    --soc=*) SOC="${1#--soc=}" ;;
    --soc)
      shift
      [[ $# -gt 0 ]] || die "参数 --soc 需要取值"
      SOC="$1"
      ;;
    -scope=*) RUN_SCOPE="${1#-scope=}" ;;
    -scope)
      shift
      [[ $# -gt 0 ]] || die "参数 -scope 需要取值"
      RUN_SCOPE="$1"
      ;;
    --scope=*) RUN_SCOPE="${1#--scope=}" ;;
    --scope)
      shift
      [[ $# -gt 0 ]] || die "参数 --scope 需要取值"
      RUN_SCOPE="$1"
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      show_usage
      die "未知参数：$1"
      ;;
  esac
  shift
done

[[ -n "$OP" ]] || die "必须传入 -op=<算子名>"

case "$RUN_SCOPE" in
  all|accuracy|performance|determinism|mssanitizer|gen_cases) ;;
  *) die "不支持的执行范围：${RUN_SCOPE}" ;;
esac

case "$SOC" in
  auto) ;;
  a2|A2|ascend910b) SOC="ascend910b" ;;
  a3|A3|ascend910_93) SOC="ascend910_93" ;;
  a5|A5|ascend950) SOC="ascend950" ;;
  *) die "不支持的 SOC：${SOC}，请使用 ascend910b/A2、ascend910_93/A3 或 ascend950/A5" ;;
esac

if [[ "$OP" == "chunk_kda_fwd_prepare" ]]; then
  # Prepare 的正式矩阵要求每条 worker case 使用 60 秒合同。
  ATK_GM_INIT_MODE="${ATK_GM_INIT_MODE:-off}"
  ATK_SINGLE_PROCESS="${ATK_SINGLE_PROCESS:-off}"
  ATK_TIMEOUT="${ATK_TIMEOUT:-60}"
  DC_TIMEOUT="${DC_TIMEOUT:-60}"
  PERFORMANCE_TIMEOUT="${PERFORMANCE_TIMEOUT:-60}"
  MSS_TIMEOUT="${MSS_TIMEOUT:-1000}"
else
  ATK_GM_INIT_MODE="${ATK_GM_INIT_MODE:-on}"
  ATK_SINGLE_PROCESS="${ATK_SINGLE_PROCESS:-on}"
  ATK_TIMEOUT="${ATK_TIMEOUT:-14400}"
  DC_TIMEOUT="${DC_TIMEOUT:-3600}"
  PERFORMANCE_TIMEOUT="${PERFORMANCE_TIMEOUT:-2000}"
fi

# Prepare 性能报告必须给出可验收的波动校验列；ATK 默认关闭该检查时会
# 将该列写成 None，无法区分未检查和通过。
PERFORMANCE_FLUCTUATION_ARGS=()
if [[ "$OP" == "chunk_kda_fwd_prepare" ]]; then
  PERFORMANCE_FLUCTUATION_ARGS=(--fluctuation_check)
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="${SCRIPT_DIR}/${OP}"
RESULT_CHECK_PY="${SCRIPT_DIR}/common/check_atk_result.py"

# NPU 后端固定为 npu
CASE_FILE="${OP_DIR}/atk_${OP}.json"
PERFORMANCE_CASE_FILE="${OP_DIR}/atk_${OP}_perf.json"
MSS_CASE_FILE="${OP_DIR}/atk_${OP}_mss.json"
EXECUTOR_FILE="${OP_DIR}/executor_${OP}.py"
YAML_FILE="${OP_DIR}/${OP}.yaml"
GEN_FILE="${OP_DIR}/gen_${OP}.py"
SANITIZER_VERIFIER_FILE="${OP_DIR}/scripts/verify_matrix.py"

[[ -d "$OP_DIR" ]] || die "找不到 ATK 算子目录：${OP_DIR}"
if should_run gen_cases; then
  [[ -f "$YAML_FILE" ]] || die "找不到 ATK YAML 文件：${YAML_FILE}"
  [[ -f "$GEN_FILE" ]] || die "找不到 ATK 生成器：${GEN_FILE}"
else
  [[ -f "$EXECUTOR_FILE" ]] || die "找不到 ATK 执行器：${EXECUTOR_FILE}"
fi

if [[ -n "${ATK_ENV:-}" ]]; then
  source_env_file "ATK虚拟环境" "${ATK_ENV}/bin/activate"
fi
if [[ -n "${CANN_ENV:-}" ]]; then
  source_env_file "CANN环境" "$CANN_ENV"
fi
if [[ -n "${FLA_NPU_ENV:-${FLA_NPU_OPP_ENV:-}}" ]]; then
  source_env_file "fla_npu_transformer环境" "${FLA_NPU_ENV:-${FLA_NPU_OPP_ENV:-}}"
fi

# 环境脚本可能设置同名变量；必须在全部环境加载后再执行正式合同校验，
# 防止 Prepare 的 60 秒、禁用 -sp/GM 初始化等约束被覆盖。
validate_chunk_kda_fwd_prepare_contract

ATK_BIN="$(command -v atk || true)"
[[ -n "$ATK_BIN" ]] || die "找不到 atk，请先安装并激活 ATK 环境"

case "$RUN_SCOPE" in
  all)
    validate_case_json "精度用例文件" "$CASE_FILE"
    validate_case_json "性能用例文件" "$PERFORMANCE_CASE_FILE"
    validate_case_json "内存检测与确定性用例文件" "$MSS_CASE_FILE"
    ;;
  accuracy) validate_case_json "精度用例文件" "$CASE_FILE" ;;
  performance) validate_case_json "性能用例文件" "$PERFORMANCE_CASE_FILE" ;;
  determinism|mssanitizer) validate_case_json "内存检测与确定性用例文件" "$MSS_CASE_FILE" ;;
esac

# 实际执行时必须校验物理 SoC，避免把一个平台的证据记到另一个平台名下。
if [[ "$RUN_SCOPE" != "gen_cases" ]]; then
  detected_soc="$(detect_soc_from_npu "$NPU_DEVICE_ID")"
  [[ -n "$detected_soc" ]] || die "无法从 npu-smi 解析实际 NPU SoC"
  if [[ "$SOC" == "auto" ]]; then
    SOC="$detected_soc"
  elif [[ "$SOC" != "$detected_soc" ]]; then
    die "目标 SoC 与实际 NPU 不一致：目标 ${SOC}，实际 ${detected_soc}"
  fi
  log_info "实际 NPU SoC：${detected_soc}"
fi

ACCURACY_START="${ACCURACY_START:-$CASE_START}"
ACCURACY_END="${ACCURACY_END:-$CASE_END}"
PERFORMANCE_START="${PERFORMANCE_START:-$CASE_START}"
PERFORMANCE_END="${PERFORMANCE_END:-$CASE_END}"
DETERMINISM_START="${DETERMINISM_START:-$CASE_START}"
DETERMINISM_END="${DETERMINISM_END:-$CASE_END}"
MSS_START="${MSS_START:-$CASE_START}"
MSS_END="${MSS_END:-$CASE_END}"

if [[ "$RUN_SCOPE" == "all" ]] &&
   [[ -n "$ACCURACY_START" || -n "$ACCURACY_END" ||
      -n "$PERFORMANCE_START" || -n "$PERFORMANCE_END" ||
      -n "$DETERMINISM_START" || -n "$DETERMINISM_END" ||
      -n "$MSS_START" || -n "$MSS_END" ]]; then
  die "正式验收的 all 必须执行全部用例，不能设置 case 范围"
fi

cd "$OP_DIR"
ATK_OUTPUT_ROOT="${ATK_OUTPUT_ROOT:-./atk_output}"
mkdir -p "${ATK_OUTPUT_ROOT}/accuracy" "${ATK_OUTPUT_ROOT}/perf" \
  "${ATK_OUTPUT_ROOT}/determinism" "${ATK_OUTPUT_ROOT}/mssanitizer"
# mssanitizer 日志路径：未显式指定时使用 ATK_OUTPUT_ROOT 下的带时间戳绝对路径
# ATK celery worker 工作目录与脚本不同，必须用绝对路径，否则无法找到日志文件
MSS_LOG_PATH="${MSS_LOG_PATH:-$(cd "${ATK_OUTPUT_ROOT}" && pwd)/mssanitizer_${OP}_$(date +%Y%m%d_%H%M%S).log}"
if [[ "$OP" == "chunk_kda_fwd_prepare" ]]; then
  # ATK 文档要求外层 mssanitizer 与 ATK -msl 共用同一个原始日志文件，
  # 否则 Start/Finish 记录可能被分散到不同文件，无法证明每个 kernel 闭环完成。
  MSS_SANITIZER_LOG_PATH="${MSS_SANITIZER_LOG_PATH:-$MSS_LOG_PATH}"
  if should_run mssanitizer; then
    # ATK worker 可能切换工作目录；日志路径必须先固定为绝对路径。
    # realpath -m 允许目标文件尚不存在，同时保留后续 verifier 的文件存在性检查。
    [[ ! -L "$MSS_LOG_PATH" ]] || die "MSS_LOG_PATH 不能是符号链接：${MSS_LOG_PATH}"
    [[ ! -L "$MSS_SANITIZER_LOG_PATH" ]] || \
      die "MSS_SANITIZER_LOG_PATH 不能是符号链接：${MSS_SANITIZER_LOG_PATH}"
    MSS_LOG_PATH="$(realpath -m -- "$MSS_LOG_PATH")" || \
      die "无法规范化 MSS_LOG_PATH：${MSS_LOG_PATH}"
    MSS_SANITIZER_LOG_PATH="$(realpath -m -- "$MSS_SANITIZER_LOG_PATH")" || \
      die "无法规范化 MSS_SANITIZER_LOG_PATH：${MSS_SANITIZER_LOG_PATH}"
    [[ "$MSS_SANITIZER_LOG_PATH" == "$MSS_LOG_PATH" ]] || \
      die "Prepare 的外层 sanitizer 日志必须与 ATK -msl 日志相同"
  fi
else
  MSS_SANITIZER_LOG_PATH="${MSS_SANITIZER_LOG_PATH:-$MSS_LOG_PATH}"
fi

log_info "算子：${OP}"
log_info "SOC：${SOC}"
if [[ "$RUN_SCOPE" != "gen_cases" ]]; then
  log_info "NPU 设备号：${NPU_DEVICE_ID}"
fi
log_info "ATK 路径：${ATK_BIN}"
log_info "输出根目录：${ATK_OUTPUT_ROOT}"
check_atk_version
resolve_gm_init_args
resolve_single_process_args

if should_run gen_cases; then
  log_info "开始生成精度候选用例：atk case -dt ${GEN_CASES_DTYPE_NUMBERS} -en ${GEN_CASES_EXTRA_NUMBERS}"
  "$ATK_BIN" case \
    -f "./${OP}.yaml" \
    -p "./gen_${OP}.py" \
    -dt "$GEN_CASES_DTYPE_NUMBERS" \
    -en "$GEN_CASES_EXTRA_NUMBERS" \
    -s "$GEN_CASES_SEED"
  log_info "完成精度候选用例生成：result/${OP}/json/all_${OP}.json"
fi

if should_run accuracy; then
  log_info "开始精度与 NaN 检测：mixed_tolerance_bm + CPU单标杆"
  set_case_range_args "精度与 NaN 检测 case 范围" "$ACCURACY_START" "$ACCURACY_END"
  run_with_prepare_hard_deadline \
    "$ATK_TIMEOUT" "$ACCURACY_START" "$ACCURACY_END" \
    "$ATK_BIN" node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
      --output_path "${ATK_OUTPUT_ROOT}/accuracy" \
    node --name cpu_golden --backend cpu \
      --output_path "${ATK_OUTPUT_ROOT}/accuracy" \
    task \
      -c "./atk_${OP}.json" \
      --task accuracy \
      --bm_device cpu \
      -p "./executor_${OP}.py" \
      "${CASE_RANGE_ARGS[@]}" \
      "${GM_INIT_ARGS[@]}" \
      "${SINGLE_PROCESS_ARGS[@]}" \
      -to "$ATK_TIMEOUT"
  log_info "完成精度与 NaN 检测"
  record_ran_type accuracy
fi

if should_run performance; then
  log_info "开始性能测试：performance_device"
  set_case_range_args "性能测试 case 范围" "$PERFORMANCE_START" "$PERFORMANCE_END"
  # 绑定本次性能报告，避免复用同一输出目录中历史成功的 xlsx。
  PERFORMANCE_RUN_START_NS="$(python3 -c 'import time; print(time.time_ns())')"
  run_with_prepare_hard_deadline \
    "$PERFORMANCE_TIMEOUT" "$PERFORMANCE_START" "$PERFORMANCE_END" \
    "$ATK_BIN" node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
      --output_path "${ATK_OUTPUT_ROOT}/perf" \
    task \
      -c "atk_${OP}_perf.json" \
      --task performance_device \
      -p "executor_${OP}.py" \
      "${CASE_RANGE_ARGS[@]}" \
      --save_data profile \
      "${SINGLE_PROCESS_ARGS[@]}" \
      "${PERFORMANCE_FLUCTUATION_ARGS[@]}" \
      -to "$PERFORMANCE_TIMEOUT"
  log_info "完成性能测试"
  record_ran_type performance
fi

if should_run determinism; then
  log_info "开始确定性测试：accuracy_dc（循环次数=${DC_LOOP_NUMS}，超时=${DC_TIMEOUT}s）"
  set_case_range_args "确定性测试 case 范围" "$DETERMINISM_START" "$DETERMINISM_END"
  run_with_prepare_hard_deadline \
    "$DC_TIMEOUT" "$DETERMINISM_START" "$DETERMINISM_END" \
    "$ATK_BIN" node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
    --output_path "${ATK_OUTPUT_ROOT}/determinism" \
    task \
      -c "atk_${OP}_mss.json" \
      -p "executor_${OP}.py" \
      --task accuracy_dc \
      --dc_loop_nums "$DC_LOOP_NUMS" \
      "${SINGLE_PROCESS_ARGS[@]}" \
      -to "$DC_TIMEOUT" \
      "${CASE_RANGE_ARGS[@]}"
  log_info "完成确定性测试"
  record_ran_type determinism
fi

if should_run mssanitizer; then
  command -v mssanitizer >/dev/null 2>&1 || die "找不到 mssanitizer，请先加载支持 sanitizer 的 CANN/调试环境"
  # PyTorch 内存池会隐藏真实 GM 分配边界，导致 memcheck 越界判断不准确。
  export PYTORCH_NO_NPU_MEMORY_CACHING=1
  log_info "开始内存检测：mssanitizer ${MSS_TOOL}"
  log_info "ATK mssanitizer 日志：${MSS_LOG_PATH}"
  set_case_range_args "内存检测 case 范围" "$MSS_START" "$MSS_END"
  MSS_TIMEOUT_ARGS=()
  if [[ -n "$MSS_TIMEOUT" ]]; then
    [[ "$MSS_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || \
      die "MSS_TIMEOUT 必须是正整数，当前值：${MSS_TIMEOUT}"
    MSS_TIMEOUT_ARGS=(-to "$MSS_TIMEOUT")
  fi
  # 每次执行先清空旧证据，避免复用路径时把上一次 clean finish 误当成本次结果。
  : > "$MSS_LOG_PATH"
  run_with_prepare_hard_deadline \
    "$MSS_TIMEOUT" "$MSS_START" "$MSS_END" \
    mssanitizer --tool="$MSS_TOOL" --log-file "$MSS_LOG_PATH" -- \
    "$ATK_BIN" node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
    --output_path "${ATK_OUTPUT_ROOT}/mssanitizer" \
    task \
      -c "atk_${OP}_mss.json" \
      -p "executor_${OP}.py" \
      --task run \
      --mssanitizer \
      -msl "$MSS_LOG_PATH" \
      "${SINGLE_PROCESS_ARGS[@]}" \
      "${MSS_TIMEOUT_ARGS[@]}" \
      "${CASE_RANGE_ARGS[@]}"
  if [[ "$OP" == "chunk_kda_fwd_prepare" ]]; then
    expected_finish_count=432
    if [[ -n "$MSS_START" || -n "$MSS_END" ]]; then
      [[ "$MSS_START" =~ ^[0-9]+$ && "$MSS_END" =~ ^[0-9]+$ ]] || \
        die "MSS_START/MSS_END 必须是非负整数"
      (( MSS_END > MSS_START )) || die "MSS_END 必须大于 MSS_START"
      expected_finish_count=$((MSS_END - MSS_START))
    fi
    python3 "$SANITIZER_VERIFIER_FILE" sanitizer-log \
      --tool "$MSS_TOOL" \
      --log "$MSS_LOG_PATH" \
      --expected-finish-count "$expected_finish_count"
  fi
  log_info "完成内存检测"
  record_ran_type mssanitizer
fi

log_info "请求的 ATK 测试动作已执行完成"
# 统一检查已跑阶段的结果；多项输出汇总，单项仅输出该项结果
print_result_summary
if [[ "$RESULT_FAIL_COUNT" -gt 0 ]]; then
  exit 1
fi
