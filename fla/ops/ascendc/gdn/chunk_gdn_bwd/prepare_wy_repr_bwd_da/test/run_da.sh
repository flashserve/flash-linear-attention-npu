#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CASES_JSON="${SCRIPT_DIR}/test_da_cases.json"
PROF_DIR="${SCRIPT_DIR}/prof_output"

MODE=""
DEVICE=0
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
        *)
            echo "[ERROR] Unknown argument: $1"
            echo "Usage: $0 --precision|--performance [--json <path>] [--device <id>]"
            exit 1
            ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "[ERROR] Must specify --precision or --performance"
    echo "Usage: $0 --precision|--performance [--json <path>] [--device <id>]"
    exit 1
fi

if [ ! -f "$CASES_JSON" ]; then
    echo "[ERROR] JSON case file not found: $CASES_JSON"
    exit 1
fi

if [ "$MODE" = "precision" ]; then
    PY_SCRIPT="$SCRIPT_DIR/test_da.py"
    if [ ! -f "$PY_SCRIPT" ]; then
        echo "[ERROR] Script not found: $PY_SCRIPT"
        exit 1
    fi

    echo "=========================================="
    echo " prepare_wy_repr_bwd_da precision test"
    echo " Script: $PY_SCRIPT"
    echo " Cases:  $CASES_JSON"
    echo "=========================================="

    python3 "$PY_SCRIPT" --json "$CASES_JSON"
else
    PY_SCRIPT="$SCRIPT_DIR/test_da_performance.py"
    if [ ! -f "$PY_SCRIPT" ]; then
        echo "[ERROR] Script not found: $PY_SCRIPT"
        exit 1
    fi

    rm -rf "$PROF_DIR"

    echo "=========================================="
    echo " prepare_wy_repr_bwd_da performance test"
    echo " Script: $PY_SCRIPT"
    echo " Cases:  $CASES_JSON"
    echo " Device: $DEVICE"
    echo " Prof:   $PROF_DIR"
    echo "=========================================="

    msprof --output="$PROF_DIR" python3 "$PY_SCRIPT" --json "$CASES_JSON" --device "$DEVICE" || true

    REPORT_CSV="$SCRIPT_DIR/perf_report.csv"
    python3 "$SCRIPT_DIR/gen_perf_report.py" --prof-dir "$PROF_DIR" --json "$CASES_JSON" --output "$REPORT_CSV"
fi
