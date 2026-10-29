#!/usr/bin/env bash
# Run the official ATK chunk32 suites against the public chunk_kda_fwd V2 entry.
set -euo pipefail

ACTION=${1:-accuracy}
DEVICE=${2:-4}
SUITE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ATK=${ATK:-atk}
OUT=${ATK_OUTPUT_ROOT:-$SUITE_DIR/atk_output}/$ACTION

if [[ -n "${CANN_ENV:-}" ]]; then
    # shellcheck disable=SC1090
    source "$CANN_ENV"
fi
export OPENBLAS_NUM_THREADS=1 GOTO_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export KDA_ATK_REFERENCE_WORKERS=1
export KDA_ATK_VISIBLE_OUTPUTS=attn_out,final_state

run_atk() {
    local suite=$1 task=$2 output=$3
    shift 3
    "$ATK" node --name npu_dut --backend npu --devices "$DEVICE" -o "$output" \
        node --name cpu_reference --backend cpu -o "$output" \
        task -c "$SUITE_DIR/$suite" --task "$task" --bm_device cpu \
        -p "$SUITE_DIR/executor_chunk_kda_fwd_chunk32.py" -sp -mt 1 -to 3600 "$@"
}

case "$ACTION" in
    accuracy)
        run_atk atk_chunk_kda_fwd_chunk32.json accuracy "$OUT"
        ;;
    performance)
        run_atk atk_chunk_kda_fwd_chunk32_perf.json performance_device "$OUT"
        ;;
    mssanitizer)
        mssanitizer "$ATK" node --name npu_dut --backend npu --devices "$DEVICE" -o "$OUT" \
            node --name cpu_reference --backend cpu -o "$OUT" \
            task -c "$SUITE_DIR/atk_chunk_kda_fwd_chunk32_mss.json" --task run --bm_device cpu \
            -p "$SUITE_DIR/executor_chunk_kda_fwd_chunk32.py" -sp -mt 1 -to 3600
        ;;
    *)
        echo "usage: bash run_chunk32_atk.sh [accuracy|performance|mssanitizer] [device]" >&2
        exit 2
        ;;
esac
