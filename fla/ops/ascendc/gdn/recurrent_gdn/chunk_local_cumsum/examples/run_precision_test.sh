#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if git -C "${SCRIPT_DIR}" rev-parse --show-toplevel >/dev/null 2>&1; then
  REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
else
  REPO_ROOT=$(cd "${SCRIPT_DIR}/../../../../../../.." && pwd)
fi
CANN_HOME=${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-9.0.0}

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f "${CANN_HOME}/set_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${CANN_HOME}/set_env.sh"
fi
CANN_HOME=${ASCEND_HOME_PATH:-${CANN_HOME}}

CASE_DIR=${CASE_DIR:-"${SCRIPT_DIR}/testdata"}
BUILD_DIR=${BUILD_DIR:-"${REPO_ROOT}/build/chunk_local_cumsum_precision"}
OPP_ROOT="${BUILD_DIR}/opp"
VENDOR_ROOT="${OPP_ROOT}/vendors/custom_transformer"

find_raw_vendor_root() {
  local pkg_root="${REPO_ROOT}/build/_CPack_Packages/Linux/External"
  if [ ! -d "${pkg_root}" ]; then
    return 1
  fi
  find "${pkg_root}" -path "*/packages/vendors/custom_transformer" -type d 2>/dev/null | sort | tail -n 1
}

has_chunk_local_cumsum_kernel() {
  [ -n "${RAW_VENDOR_ROOT:-}" ] && compgen -G "${RAW_VENDOR_ROOT}/op_impl/ai_core/tbe/kernel/ascend910b/chunk_local_cumsum/*.o" >/dev/null
}

if [ "${SKIP_TRITON_GENERATE:-0}" = "1" ]; then
  if [ ! -f "${CASE_DIR}/cases.txt" ]; then
    echo "SKIP_TRITON_GENERATE=1 but ${CASE_DIR}/cases.txt does not exist" >&2
    exit 1
  fi
else
  python3 "${SCRIPT_DIR}/generate_triton_cumsum_cases.py" --out-dir "${CASE_DIR}"
fi

RAW_VENDOR_ROOT=$(find_raw_vendor_root || true)
if [ -z "${RAW_VENDOR_ROOT}" ] || \
   [ ! -f "${RAW_VENDOR_ROOT}/op_api/lib/libcust_opapi.so" ] || \
   ! has_chunk_local_cumsum_kernel; then
  (cd "${REPO_ROOT}" && bash build.sh --pkg --soc=ascend910b --ops=chunk_local_cumsum -j16)
  RAW_VENDOR_ROOT=$(find_raw_vendor_root || true)
fi

if [ -z "${RAW_VENDOR_ROOT}" ]; then
  echo "Cannot find packages/vendors/custom_transformer under ${REPO_ROOT}/build/_CPack_Packages/Linux/External" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
rm -rf "${OPP_ROOT}"
mkdir -p "${OPP_ROOT}/vendors"
cp -a "${RAW_VENDOR_ROOT}" "${OPP_ROOT}/vendors/"
printf 'load_priority=custom_transformer\n' > "${OPP_ROOT}/vendors/config.ini"

g++ "${SCRIPT_DIR}/test_aclnn_chunk_local_cumsum.cpp" \
  -std=gnu++17 \
  -I "${VENDOR_ROOT}/op_api/include" \
  -I "${CANN_HOME}/aarch64-linux/include" \
  -L "${VENDOR_ROOT}/op_api/lib" \
  -L "${CANN_HOME}/aarch64-linux/lib64" \
  -Wl,-rpath,"${VENDOR_ROOT}/op_api/lib" \
  -Wl,-rpath,"${CANN_HOME}/aarch64-linux/lib64" \
  -lcust_opapi -lascendcl -lnnopbase -lc_sec \
  -o "${BUILD_DIR}/test_aclnn_chunk_local_cumsum"

export ASCEND_CUSTOM_OPP_PATH="${VENDOR_ROOT}"
export LD_LIBRARY_PATH="${VENDOR_ROOT}/op_api/lib:${CANN_HOME}/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"
"${BUILD_DIR}/test_aclnn_chunk_local_cumsum" "${CASE_DIR}/cases.txt"
