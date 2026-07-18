#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

mapfile -t CASE_IDS < <(
    cd "${REPO_ROOT}"
    python3 -m tests.operators._shared.legacy_cases list \
        --op chunk_bwd_dqkwg \
        --suite direct_regression
)

for case_id in "${CASE_IDS[@]}"; do
    echo "========== chunk_bwd_dqkwg: ${case_id} =========="
    bash "${SCRIPT_DIR}/run.sh" "${case_id}"
done
