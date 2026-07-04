#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CASES_JSON="${SCRIPT_DIR}/test_prepare_wy_repr_bwd.json"
PROF_DIR="${SCRIPT_DIR}/prof_output"
GEN_REPORT_PY="${SCRIPT_DIR}/gen_perf_report.py"

MODE=""
DEVICE=0
VIZ=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --precision)
            MODE="precision"
            shift
            ;;
        --performance)
            MODE="performance"
            shift
            ;;
        --json)
            CASES_JSON="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --viz)
            VIZ="--viz"
            shift
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: $0 --precision|--performance [--json <path>] [--device <id>] [--viz]"
            exit 1
            ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "[ERROR] Must specify --precision or --performance"
    echo "Usage: $0 --precision|--performance [--json <path>] [--device <id>] [--viz]"
    exit 1
fi

if [ ! -f "$CASES_JSON" ]; then
    echo "[ERROR] JSON case file not found: $CASES_JSON"
    exit 1
fi

if [ "$MODE" = "precision" ]; then
    PY_SCRIPT="$SCRIPT_DIR/test_prepare_wy_repr_bwd.py"
    if [ ! -f "$PY_SCRIPT" ]; then
        echo "[ERROR] Script not found: $PY_SCRIPT"
        exit 1
    fi

    echo "=========================================="
    echo " prepare_wy_repr_bwd precision test"
    echo " Script: $PY_SCRIPT"
    echo " Cases:  $CASES_JSON"
    echo "=========================================="

    python3 "$PY_SCRIPT" --json "$CASES_JSON" --device "$DEVICE" $VIZ
else
    PY_SCRIPT="$SCRIPT_DIR/test_prepare_wy_repr_bwd_performance.py"
    if [ ! -f "$PY_SCRIPT" ]; then
        echo "[ERROR] Script not found: $PY_SCRIPT"
        exit 1
    fi

    rm -rf "$PROF_DIR"

    echo "=========================================="
    echo " prepare_wy_repr_bwd performance test"
    echo " Script: $PY_SCRIPT"
    echo " Cases:  $CASES_JSON"
    echo " Device: $DEVICE"
    echo " Prof:   $PROF_DIR"
    echo "=========================================="

    msprof --output="$PROF_DIR" python3 "$PY_SCRIPT" --json "$CASES_JSON" --device "$DEVICE" || true

    REPORT_CSV="$SCRIPT_DIR/perf_report.csv"
    python3 "$GEN_REPORT_PY" --prof-dir "$PROF_DIR" --json "$CASES_JSON" --output "$REPORT_CSV"
fi
