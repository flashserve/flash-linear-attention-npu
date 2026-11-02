#!/usr/bin/env bash
# Minimal CP error-reporting checks; no accuracy/performance matrix or kernels.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
: "${ASCEND_HOME_PATH:?source CANN before running}"
export DFX_BUILD_DIR="${DFX_BUILD_DIR:-$REPO/build}"
export DFX_DEVICE="${DFX_DEVICE:-0}"
TESTS="$REPO/tests/operators/pre_process_bwd_kernel_merged"
python "$TESTS/test_host_fault_contract.py" -v \
  CpHostFaultContractTest.test_output_pointer_guards_precede_executor_construction \
  CpHostFaultContractTest.test_execution_failure_keeps_public_status_and_first_error
CP_DFX_CASES=1,8,18,19,25,39 \
python "$REPO/tests/atk/pre_process_bwd_kernel_merged/scripts/test_tiling_dfx.py" -v
python "$TESTS/test_native_executor_contract.py" -v
printf '%s\n' 'PASS CP DFX smoke: 14 subcases (4 boundary + 6 processor + 4 native); no model matrix executed'
