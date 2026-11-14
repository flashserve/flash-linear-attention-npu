#!/usr/bin/env bash
# Activate the matching Python/CANN environment and install the current wheel first.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir/.."
scope=${1:-accuracy}
device=${NPU_DEVICE_ID:-0}
output=${ATK_OUTPUT_ROOT:-./arch22_results}
mkdir -p "$output"
output=$(cd "$output" && pwd)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
case_file=atk_recurrent_gated_delta_rule_arch22.json
case_args=()
task=accuracy
case "$scope" in
  accuracy) ;;
  determinism)
    task=accuracy_dc
    case_file=atk_recurrent_gated_delta_rule_arch22_mss.json
    case_args=(--dc_loop_nums 50)
    ;;
  performance)
    task=performance_device
    case_file=atk_recurrent_gated_delta_rule_arch22_perf.json
    case_args=(--save_data profile)
    ;;
  memory)
    touch "$output/sanitizer.log"
    mssanitizer --tool=memcheck --leak-check=yes --kernel-name=RecurrentGatedDeltaRule \
      --log-file "$output/sanitizer.log" -- \
      atk node --name npu_dut --backend npu --devices "$device" --output_path "$output" task \
      -c atk_recurrent_gated_delta_rule_arch22_mss.json -p executor_recurrent_gated_delta_rule.py \
      --task run --mssanitizer -msl "$output/sanitizer.log" -sp -to 2000
    exit $?
    ;;
  *) printf 'Unknown scope: %s\n' "$scope" >&2; exit 2 ;;
esac
if [[ "$task" == accuracy ]]; then
  atk node --name npu_dut --backend npu --devices "$device" --output_path "$output" \
    node --name cpu_golden --backend cpu --output_path "$output" task \
    -c "$case_file" --task accuracy --bm_device cpu -p executor_recurrent_gated_delta_rule.py \
    --gm_init_flag "${GM_INIT_MB:-512}" -sp -to 2000
else
  atk node --name npu_dut --backend npu --devices "$device" --output_path "$output" task \
    -c "$case_file" --task "$task" -p executor_recurrent_gated_delta_rule.py \
    "${case_args[@]}" -sp -to 2000
fi
