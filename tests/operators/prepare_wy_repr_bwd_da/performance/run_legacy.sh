#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
GENERATED_DIR="${SCRIPT_DIR}/generated"
CASES_JSON="${GENERATED_DIR}/prepare_wy_repr_bwd_da_cases.json"
PROF_DIR="${GENERATED_DIR}/prof_output"
MODE=""
DEVICE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --precision) MODE="precision"; shift ;;
        --performance) MODE="performance"; shift ;;
        --device) DEVICE="$2"; shift 2 ;;
        *) echo "Usage: $0 --precision|--performance [--device <id>]"; exit 1 ;;
    esac
done
[[ -n "${MODE}" ]] || { echo "Specify --precision or --performance"; exit 1; }

cd "${REPO_ROOT}"
python3 -m tests.operators._shared.legacy_cases materialize \
    --op prepare_wy_repr_bwd_da \
    --suite source_matrix \
    --output "${CASES_JSON}"

if [[ "${MODE}" == "precision" ]]; then
    TEST_DEVICE_ID="${DEVICE}" \
        python3 tests/operators/prepare_wy_repr_bwd_da/accuracy/test_da.py --json "${CASES_JSON}"
else
    rm -rf "${PROF_DIR}"
    msprof --output="${PROF_DIR}" \
        python3 tests/operators/prepare_wy_repr_bwd_da/performance/legacy_profile.py \
        --json "${CASES_JSON}" --device "${DEVICE}"
    python3 tests/operators/prepare_wy_repr_bwd_da/performance/gen_legacy_report.py \
        --prof-dir "${PROF_DIR}" \
        --json "${CASES_JSON}" \
        --output "${GENERATED_DIR}/perf_report.csv"
fi
