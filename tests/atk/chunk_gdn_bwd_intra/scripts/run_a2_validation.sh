#!/usr/bin/env bash
# A2/A3（DAV_2201）运行时 G 桶（G_KEY=0）用例的执行入口。
# 与 950 主流程分离：case 来自 atk_chunk_gdn_bwd_intra_arch22.json
# （5 条 ascend910b，G=5/6/8/12/16）。A3 与 A2 同为 DAV_2201，共用同一
# 构建产物与用例集，不再单独维护 ascend910_93 用例；如需复核，可在
# 对应 SoC 机器上原样执行本脚本。须在对应 SoC 机器上运行；
# 使用前需安装包含本算子的 wheel 并激活 CANN 环境。
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$script_dir/.."

scope=${1:-accuracy}
device=${NPU_DEVICE_ID:-0}
output=${ATK_OUTPUT_ROOT:-./a2_results}
mkdir -p "$output"
output=$(cd "$output" && pwd)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

case_args=()
task=accuracy
case "$scope" in
  accuracy) ;;
  performance)
    task=performance_device
    case_args=(--save_data profile)
    ;;
  *) printf 'Unknown scope: %s\n' "$scope" >&2; exit 2 ;;
esac

case_file=atk_chunk_gdn_bwd_intra_arch22.json
if [[ "$task" == accuracy ]]; then
  atk node --name npu_dut --backend npu --devices "$device" --output_path "$output" \
    node --name cpu_golden --backend cpu --output_path "$output" task \
    -c "$case_file" --task accuracy --bm_device cpu \
    -p executor_chunk_gdn_bwd_intra.py --gm_init_flag "${GM_INIT_MB:-512}" -sp -to 2000
else
  atk node --name npu_dut --backend npu --devices "$device" --output_path "$output" task \
    -c "$case_file" --task "$task" -p executor_chunk_gdn_bwd_intra.py \
    "${case_args[@]}" -sp -to 2000
fi
