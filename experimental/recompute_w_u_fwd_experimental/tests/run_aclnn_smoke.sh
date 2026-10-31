#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 4 ]]; then
    echo "Usage: $0 <isolated-opp-install-root> [device-id] [value-width] [chunk-size]" >&2
    exit 2
fi

install_root="$(realpath "$1")"
device_id="${2:-0}"
value_width="${3:-128}"
chunk_size="${4:-64}"
vendor_dir="${install_root}/vendors/fla_npu_transformer"
if [[ ! -f "${vendor_dir}/op_api/lib/libcust_opapi.so" ]]; then
    echo "Experimental OPP not found under ${install_root}" >&2
    exit 1
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "Source the CANN toolkit set_env.sh first" >&2
    exit 1
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="${script_dir}/../../../build/aclnn_smoke_experimental"
g++ -std=c++17 "${script_dir}/aclnn_smoke.cpp" \
    -I"${ASCEND_HOME_PATH}/include" \
    -I"${vendor_dir}/op_api/include" \
    -L"${ASCEND_HOME_PATH}/lib64" \
    -L"${vendor_dir}/op_api/lib" \
    -Wl,-rpath,"${vendor_dir}/op_api/lib" \
    -lcust_opapi -lascendcl -lnnopbase -o "${binary}"

export ASCEND_CUSTOM_OPP_PATH="${vendor_dir}:${vendor_dir}/op_api/lib"
export LD_LIBRARY_PATH="${vendor_dir}/op_api/lib:${LD_LIBRARY_PATH:-}"
"${binary}" "${device_id}" "${value_width}" "${chunk_size}"
