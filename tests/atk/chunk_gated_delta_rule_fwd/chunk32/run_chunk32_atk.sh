#!/usr/bin/env bash
# Chunk32 cases live beside the fused-forward suite and are not a top-level -op.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATK_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEVICE_ID="${1:-${NPU_DEVICE_ID:-0}}"
SCOPE="${2:-accuracy}"
OP="chunk_gated_delta_rule_fwd_chunk32"
RESULT_CHECK_PY="${ATK_ROOT}/common/check_atk_result.py"
ATK_TIMEOUT="${ATK_TIMEOUT:-14400}"
DC_LOOP_NUMS="${DC_LOOP_NUMS:-50}"
DC_TIMEOUT="${DC_TIMEOUT:-3600}"
PERFORMANCE_TIMEOUT="${PERFORMANCE_TIMEOUT:-2000}"
MSS_TOOL="${MSS_TOOL:-memcheck}"
ATK_OUTPUT_ROOT="${ATK_OUTPUT_ROOT:-${SCRIPT_DIR}/atk_output}"

log_info() { echo "[chunk32 ATK] $*"; }
die() { echo "[chunk32 ATK] 错误：$*" >&2; exit 1; }

case "$SCOPE" in
  accuracy|double|performance|determinism|mssanitizer|all) ;;
  *) die "scope 只能是 accuracy、double、performance、determinism、mssanitizer 或 all" ;;
esac

if [[ -n "${ATK_ENV:-}" && -f "${ATK_ENV}/bin/activate" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${ATK_ENV}/bin/activate"
  set -u
fi
if [[ -n "${CANN_ENV:-}" && -f "${CANN_ENV}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${CANN_ENV}"
  set -u
fi
if [[ -n "${FLA_NPU_ENV:-${FLA_NPU_OPP_ENV:-}}" && -f "${FLA_NPU_ENV:-${FLA_NPU_OPP_ENV}}" ]]; then
  set +u
  # shellcheck source=/dev/null
  source "${FLA_NPU_ENV:-${FLA_NPU_OPP_ENV}}"
  set -u
fi

ATK_BIN="$(command -v atk || true)"
[[ -n "$ATK_BIN" ]] || die "找不到 atk"

cd "$SCRIPT_DIR"
mkdir -p "${ATK_OUTPUT_ROOT}/accuracy" "${ATK_OUTPUT_ROOT}/double" "${ATK_OUTPUT_ROOT}/perf"
MSS_LOG_PATH="${MSS_LOG_PATH:-$(cd "${ATK_OUTPUT_ROOT}" && pwd)/mssanitizer_${OP}_$(date +%Y%m%d_%H%M%S).log}"
GM_INIT_ARGS=(--gm_init_flag)
if [[ "${ATK_GM_INIT_MODE:-on}" == "off" ]]; then
  GM_INIT_ARGS=()
fi

run_accuracy() {
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$DEVICE_ID" \
      --output_path "${ATK_OUTPUT_ROOT}/accuracy" \
    node --name cpu_golden --backend cpu \
      --output_path "${ATK_OUTPUT_ROOT}/accuracy" \
    task \
      -c "./atk_${OP}.json" \
      --task accuracy \
      --bm_device cpu \
      -p "./executor_${OP}.py" \
      "${GM_INIT_ARGS[@]}" \
      -to "$ATK_TIMEOUT"
}

run_double() {
  local gpu_host="${GPU_HOST:-}"
  local gpu_port="${GPU_HOST_PORT:-9090}"
  local gpu_device="${GPU_DEVICE_ID:-0}"
  [[ -n "$gpu_host" ]] || die "双标杆需要 GPU_HOST，GPU 节点跑 gpu_triton_fwd.py"
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$DEVICE_ID" \
      --output_path "${ATK_OUTPUT_ROOT}/double" \
    node --name gold --backend gpu --host "$gpu_host" --port "$gpu_port" \
      --devices "$gpu_device" --is_compare true \
      --output_path "${ATK_OUTPUT_ROOT}/double" \
    task \
      -c "./atk_${OP}_double.json" \
      --task accuracy \
      -bd cpu \
      -p "./executor_${OP}.py" \
      --syc_dataset \
      "${GM_INIT_ARGS[@]}" \
      -mt "${DOUBLE_MAX_TASK:-1}" \
      -to "$ATK_TIMEOUT"
}

run_performance() {
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$DEVICE_ID" \
      --output_path "${ATK_OUTPUT_ROOT}/perf" \
    task \
      -c "atk_${OP}_perf.json" \
      --task performance_device \
      -p "executor_${OP}.py" \
      --save_data profile \
      -sp \
      -to "$PERFORMANCE_TIMEOUT"
}

run_determinism() {
  "$ATK_BIN" node --name npu_dut --backend npu --devices "$DEVICE_ID" \
    task \
      -c "atk_${OP}_mss.json" \
      -p "executor_${OP}.py" \
      --task accuracy_dc \
      --dc_loop_nums "$DC_LOOP_NUMS" \
      -to "$DC_TIMEOUT"
}

run_mss() {
  command -v mssanitizer >/dev/null 2>&1 || die "找不到 mssanitizer"
  touch "$MSS_LOG_PATH"
  mssanitizer --tool="$MSS_TOOL" -- \
    "$ATK_BIN" node --name npu_dut --backend npu --devices "$DEVICE_ID" \
    task \
      -c "atk_${OP}_mss.json" \
      -p "executor_${OP}.py" \
      --task run \
      --mssanitizer \
      -msl "$MSS_LOG_PATH"
}

FAIL=0
check_one() {
  local kind="$1"
  if [[ -f "$RESULT_CHECK_PY" ]]; then
    python3 "$RESULT_CHECK_PY" --type "$kind" --output-root "$ATK_OUTPUT_ROOT" --op "$OP" || FAIL=$((FAIL + 1))
  fi
}

if [[ "$SCOPE" == "all" || "$SCOPE" == "accuracy" ]]; then
  log_info "混合容差"
  run_accuracy
  check_one accuracy
fi
if [[ "$SCOPE" == "double" ]]; then
  log_info "双标杆"
  run_double
fi
if [[ "$SCOPE" == "all" || "$SCOPE" == "performance" ]]; then
  log_info "性能"
  run_performance
fi
if [[ "$SCOPE" == "all" || "$SCOPE" == "determinism" ]]; then
  log_info "确定性"
  run_determinism
  check_one determinism
fi
if [[ "$SCOPE" == "all" || "$SCOPE" == "mssanitizer" ]]; then
  log_info "内存检测"
  run_mss
  check_one mssanitizer
fi

[[ "$FAIL" -eq 0 ]] || exit 1
