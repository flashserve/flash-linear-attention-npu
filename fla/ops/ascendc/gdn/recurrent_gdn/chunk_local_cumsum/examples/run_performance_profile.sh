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

if [ "${EXTRA_FGD_CASE:-0}" = "1" ] && [ -z "${CASE_DIR:-}" ]; then
  CASE_DIR="${SCRIPT_DIR}/perfdata"
else
  CASE_DIR=${CASE_DIR:-"${SCRIPT_DIR}/testdata"}
fi
BUILD_DIR=${BUILD_DIR:-"${REPO_ROOT}/build/chunk_local_cumsum_precision"}
OP_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
PROFILE_ROOT=${PROFILE_ROOT:-"${OP_DIR}/profiling/perf_compare"}
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

GEN_ARGS=(--out-dir "${CASE_DIR}")
if [ "${EXTRA_FGD_CASE:-0}" = "1" ]; then
  GEN_ARGS+=(--extra-fgd-case)
fi
if [ "${SKIP_TRITON_GENERATE:-0}" = "1" ]; then
  if [ ! -f "${CASE_DIR}/cases.txt" ]; then
    echo "SKIP_TRITON_GENERATE=1 but ${CASE_DIR}/cases.txt does not exist" >&2
    exit 1
  fi
else
  python3 "${SCRIPT_DIR}/generate_triton_cumsum_cases.py" "${GEN_ARGS[@]}"
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
export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64:${VENDOR_ROOT}/op_api/lib:${CANN_HOME}/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}"

if [ "${PERF_CASES:-}" = "all" ] || [ -z "${PERF_CASES:-}" ]; then
  mapfile -t CASES < <(awk 'NR > 1 {print $1}' "${CASE_DIR}/cases.txt")
else
  # shellcheck disable=SC2206
  CASES=(${PERF_CASES})
fi

if [ "${KEEP_PROFILE_ROOT:-0}" != "1" ]; then
  rm -rf "${PROFILE_ROOT}"
fi
mkdir -p "${PROFILE_ROOT}/triton" "${PROFILE_ROOT}/ascendc"

for case_name in "${CASES[@]}"; do
  echo "[ChunkLocalCumsum][Triton] profiling ${case_name}"
  msprof op \
    --application="python3 ${SCRIPT_DIR}/profile_triton_cumsum_case.py --case ${case_name} --case-dir ${CASE_DIR}" \
    --output="${PROFILE_ROOT}/triton/${case_name}" \
    --aic-metrics=BasicInfo \
    --launch-count=1 \
    --warm-up=0

  echo "[ChunkLocalCumsum][AscendC] profiling ${case_name}"
  msprof op \
    --application="${BUILD_DIR}/test_aclnn_chunk_local_cumsum ${CASE_DIR}/cases.txt ${case_name}" \
    --output="${PROFILE_ROOT}/ascendc/${case_name}" \
    --aic-metrics=BasicInfo \
    --launch-count=1 \
    --warm-up=0
done

python3 "${SCRIPT_DIR}/parse_msprof_cumsum_perf.py" \
  --profile-root "${PROFILE_ROOT}" \
  --case-dir "${CASE_DIR}" \
  --out-json "${OP_DIR}/performance_comparison.json" \
  --out-md "${OP_DIR}/performance_comparison.md" \
  --cases "${CASES[@]}"
