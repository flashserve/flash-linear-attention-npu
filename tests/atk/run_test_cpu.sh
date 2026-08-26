#!/usr/bin/env bash
set -euo pipefail

# 单算子 ATK CPU 标杆一键验证脚本。
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
                                 all 包含 accuracy/determinism/mssanitizer；
                                 performance 需单独指定；gen_cases 不在 all 中

常用环境变量：
  ATK_ENV                        ATK 虚拟环境目录，设置后 source "$ATK_ENV/bin/activate"
  CANN_ENV                       CANN set_env.sh 路径，设置后 source
  FLA_NPU_ENV                    fla_npu_transformer set_env.bash 路径，设置后 source
  ATK_OUTPUT_ROOT                输出根目录，默认 ./atk_output
  ATK_GM_INIT_MODE               GM 数据初始化模式，默认 on；可设 on/off
  ATK_TIMEOUT                    精度阶段超时，默认 14400
  DC_LOOP_NUMS                   确定性循环次数，默认 50（与 ATK 一致）
  DC_TIMEOUT                     确定性阶段超时，默认 3600
  PERFORMANCE_TIMEOUT            性能阶段超时，默认 2000
  PERFORMANCE_CASE_FILE          性能阶段用例 JSON，默认优先使用算子目录下的性能子集
  CASE_START/CASE_END            通用 case 顺序范围；不设置时不传 -s/-e，ATK 执行全部用例
  ACCURACY_START/ACCURACY_END    精度与 NaN 检测 case 范围
  PERFORMANCE_START/END          性能 case 范围
  DETERMINISM_START/END          确定性 case 范围
  MSS_START/MSS_END              mssanitizer case 范围
  MSS_TOOL                       mssanitizer 工具，默认 memcheck
  MSS_LOG_PATH                   ATK -msl 日志路径，默认 ${ATK_OUTPUT_ROOT}/mssanitizer_<op>_<时间戳>.log
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

# 记录已执行的测试阶段，供最终汇总使用。
record_ran_type() {
  RAN_TYPES+=("$1")
}

# 统一检查已跑阶段的结果并输出汇总。
# 多项时输出每项结果 + 汇总行；单项时仅输出该项结果，不显示汇总行。
print_result_summary() {
  if [[ ! -f "$RESULT_CHECK_PY" ]]; then
    log_info "跳过结果检查：找不到 ${RESULT_CHECK_PY}"
    return 0
  fi
  local fail=0 ran="${#RAN_TYPES[@]}"
  [[ $ran -gt 0 ]] || return 0
  local pass=0
  for t in "${RAN_TYPES[@]}"; do
    if python3 "$RESULT_CHECK_PY" --type "$t" \
        --output-root "$ATK_OUTPUT_ROOT" --op "$OP"; then
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
  # performance 不包含在 all 中，需单独指定 -scope=performance
  if [[ "$stage" == "performance" ]]; then
    [[ "$RUN_SCOPE" == "performance" ]]
    return
  fi
  [[ "$RUN_SCOPE" == "all" || "$RUN_SCOPE" == "$stage" ]]
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

# 从 npu-smi 探测真实 SOC，返回 ascend910b/ascend910_93/ascend950；探测失败返回空。
detect_soc_from_npu() {
  local name
  name=$(npu-smi info 2>/dev/null | awk -F'|' '/^\|[[:space:]]*[0-9]+[[:space:]]+[^[:space:]]+[[:space:]]*\|/ {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2);
      split($2, fields, /[[:space:]]+/);
      print fields[2];
      exit;
  }' || true)
  case "$name" in
    *910B*|*910b*) echo "ascend910b" ;;
    *910_93*|*910*93*) echo "ascend910_93" ;;
    *950*|*Ascend950*) echo "ascend950" ;;
    *) echo "" ;;
  esac
}

# 校验 ATK 版本不低于 REQUIRED_ATK_VERSION（使用 GNU sort -V 比较）。
check_atk_version() {
  local required="$REQUIRED_ATK_VERSION"
  local installed
  installed="$("$ATK_BIN" --version 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1 || true)"
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

OP=""
NPU_DEVICE_ID="${NPU_DEVICE_ID:-0}"
SOC="${SOC:-auto}"
RUN_SCOPE="${RUN_SCOPE:-all}"
ATK_GM_INIT_MODE="${ATK_GM_INIT_MODE:-on}"
REQUIRED_ATK_VERSION="${REQUIRED_ATK_VERSION:-26.7.8}"
ATK_TIMEOUT="${ATK_TIMEOUT:-14400}"
DC_LOOP_NUMS="${DC_LOOP_NUMS:-50}"
DC_TIMEOUT="${DC_TIMEOUT:-3600}"
PERFORMANCE_TIMEOUT="${PERFORMANCE_TIMEOUT:-2000}"
CASE_START="${CASE_START:-}"
CASE_END="${CASE_END:-}"
MSS_TOOL="${MSS_TOOL:-memcheck}"
MSS_LOG_PATH="${MSS_LOG_PATH:-}"
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OP_DIR="${SCRIPT_DIR}/${OP}"
RESULT_CHECK_PY="${SCRIPT_DIR}/common/check_atk_result.py"

# NPU 后端固定为 npu
CASE_FILE="${OP_DIR}/atk_${OP}.json"
PERFORMANCE_CASE_FILE="${PERFORMANCE_CASE_FILE:-${OP_DIR}/atk_${OP}_performance.json}"
EXECUTOR_FILE="${OP_DIR}/executor_${OP}.py"
YAML_FILE="${OP_DIR}/${OP}.yaml"
GEN_FILE="${OP_DIR}/gen_${OP}.py"

[[ -d "$OP_DIR" ]] || die "找不到 ATK 算子目录：${OP_DIR}"
if should_run gen_cases; then
  [[ -f "$YAML_FILE" ]] || die "找不到 ATK YAML 文件：${YAML_FILE}"
  [[ -f "$GEN_FILE" ]] || die "找不到 ATK 生成器：${GEN_FILE}"
else
  [[ -f "$CASE_FILE" ]] || die "找不到 ATK 用例文件：${CASE_FILE}"
  [[ -f "$EXECUTOR_FILE" ]] || die "找不到 ATK 执行器：${EXECUTOR_FILE}"
fi
if should_run performance && [[ ! -f "$PERFORMANCE_CASE_FILE" ]]; then
  PERFORMANCE_CASE_FILE="${OP_DIR}/atk_${OP}_perf.json"
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

ATK_BIN="$(command -v atk || true)"
[[ -n "$ATK_BIN" ]] || die "找不到 atk，请先安装并激活 ATK 环境"

# SOC 为 auto 时探测真实芯片，用于 ATK_GM_INIT_MODE 判定与日志展示
if [[ "$SOC" == "auto" ]]; then
  detected_soc="$(detect_soc_from_npu)"
  if [[ -n "$detected_soc" ]]; then
    SOC="$detected_soc"
  else
    log_info "无法从 npu-smi 解析 NPU 型号，按 ascend910b 处理 GM 初始化"
    SOC="ascend910b"
  fi
fi

ACCURACY_START="${ACCURACY_START:-$CASE_START}"
ACCURACY_END="${ACCURACY_END:-$CASE_END}"
PERFORMANCE_START="${PERFORMANCE_START:-$CASE_START}"
PERFORMANCE_END="${PERFORMANCE_END:-$CASE_END}"
DETERMINISM_START="${DETERMINISM_START:-$CASE_START}"
DETERMINISM_END="${DETERMINISM_END:-$CASE_END}"
MSS_START="${MSS_START:-$CASE_START}"
MSS_END="${MSS_END:-$CASE_END}"

cd "$OP_DIR"
ATK_OUTPUT_ROOT="${ATK_OUTPUT_ROOT:-./atk_output}"
mkdir -p "${ATK_OUTPUT_ROOT}/cpu_dual_reference" "${ATK_OUTPUT_ROOT}/perf"
# mssanitizer 日志路径：未显式指定时使用 ATK_OUTPUT_ROOT 下的带时间戳绝对路径
# ATK celery worker 工作目录与脚本不同，必须用绝对路径，否则无法找到日志文件
MSS_LOG_PATH="${MSS_LOG_PATH:-$(cd "${ATK_OUTPUT_ROOT}" && pwd)/mssanitizer_${OP}_$(date +%Y%m%d_%H%M%S).log}"

log_info "算子：${OP}"
log_info "SOC：${SOC}"
if [[ "$RUN_SCOPE" != "gen_cases" ]]; then
  log_info "NPU 设备号：${NPU_DEVICE_ID}"
fi
log_info "ATK 路径：${ATK_BIN}"
log_info "输出根目录：${ATK_OUTPUT_ROOT}"
check_atk_version
resolve_gm_init_args

if should_run gen_cases; then
  log_info "开始生成泛化用例：atk case -dt ${GEN_CASES_DTYPE_NUMBERS} -en ${GEN_CASES_EXTRA_NUMBERS}"
  "$ATK_BIN" case \
    -f "./${OP}.yaml" \
    -p "./gen_${OP}.py" \
    -dt "$GEN_CASES_DTYPE_NUMBERS" \
    -en "$GEN_CASES_EXTRA_NUMBERS" \
    -s "$GEN_CASES_SEED"
  log_info "完成泛化用例生成：result/${OP}/json/all_${OP}.json"
fi

if should_run accuracy; then
  log_info "开始精度与 NaN 检测：accuracy + CPU高精度标杆 + CPU同精度标杆 + GM 初始化"
  set_case_range_args "精度与 NaN 检测 case 范围" "$ACCURACY_START" "$ACCURACY_END"
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
      --output_path "${ATK_OUTPUT_ROOT}/cpu_dual_reference" \
    node --name cpu_reference --backend cpu \
      --output_path "${ATK_OUTPUT_ROOT}/cpu_dual_reference" \
    task \
      -c "./atk_${OP}.json" \
      --task accuracy \
      --bm_device cpu \
      -p "./executor_${OP}.py" \
      "${CASE_RANGE_ARGS[@]}" \
      "${GM_INIT_ARGS[@]}" \
      -to "$ATK_TIMEOUT"
  log_info "完成精度与 NaN 检测"
  record_ran_type accuracy
fi

if should_run performance; then
  log_info "开始性能测试：performance_device"
  log_info "性能用例文件：${PERFORMANCE_CASE_FILE}"
  set_case_range_args "性能测试 case 范围" "$PERFORMANCE_START" "$PERFORMANCE_END"
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
      --output_path "${ATK_OUTPUT_ROOT}/perf" \
    task \
      -c "$PERFORMANCE_CASE_FILE" \
      --task performance_device \
      -p "executor_${OP}.py" \
      "${CASE_RANGE_ARGS[@]}" \
      --save_data profile \
      -sp \
      -to "$PERFORMANCE_TIMEOUT"
  log_info "完成性能测试"
fi

if should_run determinism; then
  log_info "开始确定性测试：accuracy_dc（循环次数=${DC_LOOP_NUMS}，超时=${DC_TIMEOUT}s）"
  set_case_range_args "确定性测试 case 范围" "$DETERMINISM_START" "$DETERMINISM_END"
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
    task \
      -c "atk_${OP}_mss.json" \
      -p "executor_${OP}.py" \
      --task accuracy_dc \
      --dc_loop_nums "$DC_LOOP_NUMS" \
      -to "$DC_TIMEOUT" \
      "${CASE_RANGE_ARGS[@]}"
  log_info "完成确定性测试"
  record_ran_type determinism
fi

if should_run mssanitizer; then
  command -v mssanitizer >/dev/null 2>&1 || die "找不到 mssanitizer，请先加载支持 sanitizer 的 CANN/调试环境"
  log_info "开始内存检测：mssanitizer ${MSS_TOOL}"
  log_info "ATK mssanitizer 日志：${MSS_LOG_PATH}"
  set_case_range_args "内存检测 case 范围" "$MSS_START" "$MSS_END"
  touch "$MSS_LOG_PATH"
  mssanitizer --tool="$MSS_TOOL" -- \
    "$ATK_BIN" node --name npu_dut --backend npu --devices "$NPU_DEVICE_ID" \
    task \
      -c "atk_${OP}_mss.json" \
      -p "executor_${OP}.py" \
      --task run \
      --mssanitizer \
      -msl "$MSS_LOG_PATH" \
      "${CASE_RANGE_ARGS[@]}"
  log_info "完成内存检测"
  record_ran_type mssanitizer
fi

log_info "请求的 ATK 测试动作已执行完成"
# 统一检查已跑阶段的结果；多项输出汇总，单项仅输出该项结果
print_result_summary
if [[ "$RESULT_FAIL_COUNT" -gt 0 ]]; then
  exit 1
fi
