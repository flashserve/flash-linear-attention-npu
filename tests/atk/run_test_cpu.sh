#!/usr/bin/env bash
set -euo pipefail

# 单算子 ATK CPU 标杆一键验证脚本。
# 所有测试动作均由 ATK 发起；内存检测由 mssanitizer 包裹 ATK run 任务。

show_usage() {
  cat <<'EOF'
用法：
  bash tests/atk/run_test_cpu.sh -op=<算子名> -npu_device_id=<NPU卡号>

常用参数：
  -op=chunk_kda_fwd              ATK 算子目录名
  -npu_device_id=6               物理 NPU 卡号；脚本会映射为进程内逻辑 0 传给 ATK
  -soc=ascend910b                可选：ascend910b/A2、ascend910_93/A3、ascend950/A5
  -scope=all                     可选：all、accuracy、performance、determinism、mssanitizer、gen_cases

常用环境变量：
  ATK_ENV                        ATK 虚拟环境目录，设置后 source "$ATK_ENV/bin/activate"
  CANN_ENV                       CANN set_env.sh 路径，设置后 source
  FLA_NPU_ENV                    fla_npu_transformer set_env.bash 路径，设置后 source
  ATK_OUTPUT_ROOT                输出根目录，默认 ./atk_output
  ATK_TIMEOUT                    精度阶段超时，默认 14400
  PERFORMANCE_TIMEOUT            性能阶段超时，默认 2000
  CASE_START/CASE_END            通用 case 顺序范围；不设置时不传 -s/-e，ATK 执行全部用例
  ACCURACY_START/ACCURACY_END    精度与 NaN 检测 case 范围
  PERFORMANCE_START/END          性能 case 范围
  DETERMINISM_START/END          确定性 case 范围
  MSS_START/MSS_END              mssanitizer case 范围
  MSS_TOOL                       mssanitizer 工具，默认 memcheck
  MSS_LOG_PATH                   ATK -msl 日志路径，默认写入当前算子的 mindstudio_sanitizer_log
  GEN_CASES_DTYPE_NUMBERS        生成用例时传给 atk case -dt，默认 100；双 dtype 算子生成 200 条
  GEN_CASES_EXTRA_NUMBERS        生成用例时传给 atk case -en，默认 0
  GEN_CASES_SEED                 生成用例随机种子，默认 20260813

示例：
  bash tests/atk/run_test_cpu.sh -op=chunk_kda_fwd -npu_device_id=6
  bash tests/atk/run_test_cpu.sh -op=chunk_bwd_dqkwg -scope=gen_cases
  CASE_START=0 CASE_END=1 bash tests/atk/run_test_cpu.sh -op=chunk_bwd_dqkwg -npu_device_id=6
EOF
}

log_info() {
  echo "[ATK CPU标杆验证] $*"
}

die() {
  echo "[ATK CPU标杆验证] 错误：$*" >&2
  exit 1
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

OP=""
NPU_DEVICE_ID="${NPU_DEVICE_ID:-}"
SOC="${SOC:-auto}"
RUN_SCOPE="${RUN_SCOPE:-all}"
ATK_TIMEOUT="${ATK_TIMEOUT:-14400}"
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
if [[ "$RUN_SCOPE" != "gen_cases" ]]; then
  [[ -n "$NPU_DEVICE_ID" ]] || die "必须传入 -npu_device_id=<NPU卡号>"
fi

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
CASE_FILE="${OP_DIR}/atk_${OP}.json"
EXECUTOR_FILE="${OP_DIR}/executor_${OP}.py"
YAML_FILE="${OP_DIR}/${OP}.yaml"
GEN_FILE="${OP_DIR}/gen_${OP}.py"
MSS_LOG_PATH="${MSS_LOG_PATH:-${OP_DIR}/mindstudio_sanitizer_log/atk_mssanitizer.log}"

[[ -d "$OP_DIR" ]] || die "找不到 ATK 算子目录：${OP_DIR}"
if should_run gen_cases; then
  [[ -f "$YAML_FILE" ]] || die "找不到 ATK YAML 文件：${YAML_FILE}"
  [[ -f "$GEN_FILE" ]] || die "找不到 ATK 生成器：${GEN_FILE}"
else
  [[ -f "$CASE_FILE" ]] || die "找不到 ATK 用例文件：${CASE_FILE}"
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

ATK_DEVICE_ID="$NPU_DEVICE_ID"
if [[ "$RUN_SCOPE" != "gen_cases" ]]; then
  export ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICE_ID"
  ATK_DEVICE_ID=0
fi

ATK_BIN="$(command -v atk || true)"
[[ -n "$ATK_BIN" ]] || die "找不到 atk，请先安装并激活 ATK 环境"

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
mkdir -p "$(dirname "$MSS_LOG_PATH")"

log_info "算子：${OP}"
log_info "SOC：${SOC}"
if [[ "$RUN_SCOPE" != "gen_cases" ]]; then
  log_info "NPU 设备号：${NPU_DEVICE_ID}"
  log_info "ATK 逻辑设备号：${ATK_DEVICE_ID}"
fi
log_info "ATK 路径：${ATK_BIN}"
log_info "输出根目录：${ATK_OUTPUT_ROOT}"
"$ATK_BIN" --version || die "atk --version 执行失败"

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
  log_info "开始精度与 NaN 检测：accuracy + CPU高精度标杆 + CPU同精度标杆 + --gm_init_flag"
  set_case_range_args "精度与 NaN 检测 case 范围" "$ACCURACY_START" "$ACCURACY_END"
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$ATK_DEVICE_ID" \
      --output_path "${ATK_OUTPUT_ROOT}/cpu_dual_reference" \
    node --name cpu_reference --backend cpu \
      --output_path "${ATK_OUTPUT_ROOT}/cpu_dual_reference" \
    task \
      -c "./atk_${OP}.json" \
      --task accuracy \
      --bm_device cpu \
      -p "./executor_${OP}.py" \
      "${CASE_RANGE_ARGS[@]}" \
      --gm_init_flag \
      -sp \
      -mt 1 \
      -to "$ATK_TIMEOUT"
  log_info "完成精度与 NaN 检测"
fi

if should_run performance; then
  log_info "开始性能测试：performance_device"
  set_case_range_args "性能测试 case 范围" "$PERFORMANCE_START" "$PERFORMANCE_END"
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$ATK_DEVICE_ID" \
      --output_path "${ATK_OUTPUT_ROOT}/perf" \
    task \
      -c "atk_${OP}.json" \
      --task performance_device \
      -p "executor_${OP}.py" \
      "${CASE_RANGE_ARGS[@]}" \
      --save_data profile \
      -sp \
      -to "$PERFORMANCE_TIMEOUT"
  log_info "完成性能测试"
fi

if should_run determinism; then
  log_info "开始确定性测试：accuracy_dc"
  set_case_range_args "确定性测试 case 范围" "$DETERMINISM_START" "$DETERMINISM_END"
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$ATK_DEVICE_ID" \
    task \
      -c "atk_${OP}.json" \
      -p "executor_${OP}.py" \
      --task accuracy_dc \
      "${CASE_RANGE_ARGS[@]}"
  log_info "完成确定性测试"
fi

if should_run mssanitizer; then
  command -v mssanitizer >/dev/null 2>&1 || die "找不到 mssanitizer，请先加载支持 sanitizer 的 CANN/调试环境"
  log_info "开始内存检测：mssanitizer ${MSS_TOOL}"
  log_info "ATK mssanitizer 日志：${MSS_LOG_PATH}"
  : > "$MSS_LOG_PATH"
  set_case_range_args "内存检测 case 范围" "$MSS_START" "$MSS_END"
  mssanitizer --tool="$MSS_TOOL" --log-file="$MSS_LOG_PATH" -- \
    "$ATK_BIN" node --name npu_dut --backend npu --devices "$ATK_DEVICE_ID" \
    task \
      -c "atk_${OP}.json" \
      -p "executor_${OP}.py" \
      --task run \
      --mssanitizer \
      -msl "$MSS_LOG_PATH" \
      "${CASE_RANGE_ARGS[@]}"
  log_info "完成内存检测"
fi

log_info "请求的 ATK 测试动作已执行完成"
