#!/usr/bin/env bash
set -euo pipefail

OP_NAME="chunk_gated_delta_rule_bwd_dhu"
OP_TYPE="ChunkGatedDeltaRuleBwdDhu"
VENDOR_NAME="fla_npu"

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
PROFILE_SCRIPT="${SCRIPT_DIR}/profile_chunk_gated_delta_rule_bwd_dhu_perf.py"

TARGET="auto"
DEVICE=""
BRANCH="both"
REPEAT="5"
TAG=$(date +%Y%m%d_%H%M%S)
OUT_DIR=""
SKIP_BUILD="0"

usage() {
    cat <<'EOF'
Usage:
  bash torch_custom/fla_npu/test/run_chunk_gated_delta_rule_bwd_dhu_perf.sh --target a2 --device <physical_npu_id>
  bash torch_custom/fla_npu/test/run_chunk_gated_delta_rule_bwd_dhu_perf.sh --target a5 --device <physical_npu_id>

Options:
  --target a2|a5|auto      Target server profile. Default: auto.
  --device ID              Physical NPU id for ASCEND_RT_VISIBLE_DEVICES. Required.
  --branch g|gK|both       Branches to profile. Default: both.
  --repeat N               Repeats per branch. Default: 5.
  --tag TAG                Artifact tag. Default: current timestamp.
  --out-dir DIR            Artifact directory. Default: artifacts/dhu_perf_<target>_release_<branch><repeat>_<tag>.
  --skip-build             Reuse installed package, still validates wrapper and runs msprof.
EOF
}

log() {
    printf '[dhu-perf] %s\n' "$*"
}

die() {
    printf '[dhu-perf][ERROR] %s\n' "$*" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            TARGET="${2:?missing --target value}"
            shift 2
            ;;
        --device)
            DEVICE="${2:?missing --device value}"
            shift 2
            ;;
        --branch)
            BRANCH="${2:?missing --branch value}"
            shift 2
            ;;
        --repeat)
            REPEAT="${2:?missing --repeat value}"
            shift 2
            ;;
        --tag)
            TAG="${2:?missing --tag value}"
            shift 2
            ;;
        --out-dir)
            OUT_DIR="${2:?missing --out-dir value}"
            shift 2
            ;;
        --skip-build)
            SKIP_BUILD="1"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

[[ -n "${DEVICE}" ]] || die "--device is required"
[[ "${BRANCH}" == "g" || "${BRANCH}" == "gK" || "${BRANCH}" == "both" ]] || die "--branch must be g, gK, or both"
[[ "${REPEAT}" =~ ^[1-9][0-9]*$ ]] || die "--repeat must be a positive integer"

if [[ "${TARGET}" == "auto" ]]; then
    if [[ -f /home/npu_user7/zhangshuolei/Ascend/ascend-toolkit/set_env.sh ]]; then
        TARGET="a5"
    elif [[ -f /workspace/zs/Ascend/ascend-toolkit/set_env.sh ]]; then
        TARGET="a2"
    else
        die "unable to auto-detect target; pass --target a2 or --target a5"
    fi
fi

case "${TARGET}" in
    a2)
        CONDA_SH="/data/miniconda3/etc/profile.d/conda.sh"
        CANN_ENV="/workspace/zs/Ascend/ascend-toolkit/set_env.sh"
        SOC="ascend910b"
        ;;
    a5)
        CONDA_SH="/home/npu_user7/miniconda3/etc/profile.d/conda.sh"
        CANN_ENV="/home/npu_user7/zhangshuolei/Ascend/ascend-toolkit/set_env.sh"
        SOC="ascend950"
        ;;
    *)
        die "--target must be a2, a5, or auto"
        ;;
esac

[[ -f "${CONDA_SH}" ]] || die "conda profile not found: ${CONDA_SH}"
[[ -f "${CANN_ENV}" ]] || die "CANN env not found: ${CANN_ENV}"
[[ -f "${PROFILE_SCRIPT}" ]] || die "profile script not found: ${PROFILE_SCRIPT}"

if [[ -z "${OUT_DIR}" ]]; then
    OUT_DIR="${REPO_ROOT}/artifacts/dhu_perf_${TARGET}_release_${BRANCH}${REPEAT}_${TAG}"
fi
PROFILE_OUT="${OUT_DIR}/profiler"
mkdir -p "${OUT_DIR}" "${PROFILE_OUT}"

exec > >(tee "${OUT_DIR}/run.log") 2>&1

log "repo=${REPO_ROOT}"
log "target=${TARGET} soc=${SOC} device=${DEVICE} branch=${BRANCH} repeat=${REPEAT}"
log "out_dir=${OUT_DIR}"

cd "${REPO_ROOT}"
set +u
source "${CONDA_SH}"
conda activate zsl
source "${CANN_ENV}"
set -u
export FLA_NPU_SOC="${SOC}"

log "npu-smi snapshot"
npu-smi info | tee "${OUT_DIR}/npu_smi.txt"

verify_release_build() {
    [[ -f build/CMakeCache.txt ]] || die "build/CMakeCache.txt not found"
    grep -nE 'CMAKE_BUILD_TYPE:STRING|BISHENG_FLAGS|OP_DEBUG_CONFIG|ENABLE_OOM' build/CMakeCache.txt \
        | tee "${OUT_DIR}/release_cmake_flags.txt" || true

    grep -q '^CMAKE_BUILD_TYPE:STRING=Release$' build/CMakeCache.txt \
        || die "CMAKE_BUILD_TYPE is not Release"
    grep -q '^BISHENG_FLAGS:STRING=$' build/CMakeCache.txt \
        || die "BISHENG_FLAGS is not empty"
    grep -q '^ENABLE_OOM:BOOL=OFF$' build/CMakeCache.txt \
        || die "ENABLE_OOM is not OFF"
    grep -q '^OP_DEBUG_CONFIG:STRING=false$' build/CMakeCache.txt \
        || die "OP_DEBUG_CONFIG is not false"

    if find "${OUT_DIR}/build.log" build/CMakeCache.txt build/binary/"${SOC}" -type f 2>/dev/null \
        | xargs grep -nE 'ccec_g|dump_cce|sanitizer|--op_debug_config|CMAKE_BUILD_TYPE=Debug| -g --cce-enable-oom|BISHENG_FLAGS=.*[^[:space:]]' \
        > "${OUT_DIR}/debug_flag_hits.txt"; then
        cat "${OUT_DIR}/debug_flag_hits.txt"
        die "debug or sanitizer compile flags were found"
    fi
    : > "${OUT_DIR}/debug_flag_hits.txt"
}

if [[ "${SKIP_BUILD}" == "0" ]]; then
    log "clean build outputs"
    rm -rf build build_out dist

    log "check NPU build environment"
    python3 scripts/check_npu_env.py --build-only 2>&1 | tee "${OUT_DIR}/check_npu_env.log"

    log "build Release run package without debug flags"
    bash build.sh \
        --pkg \
        --soc="${SOC}" \
        --vendor_name="${VENDOR_NAME}" \
        --ops="${OP_NAME}" \
        --build-type=Release \
        -j64 2>&1 | tee "${OUT_DIR}/build.log"

    verify_release_build

    log "build Python wheel from the Release run package"
    FLA_NPU_SOC="${SOC}" \
    FLA_NPU_OPS="${OP_NAME}" \
    FLA_NPU_SKIP_RUN_BUILD=1 \
        python3 -m pip wheel --no-build-isolation --no-deps . -w dist 2>&1 | tee "${OUT_DIR}/wheel.log"

    log "install wheel"
    python3 -m pip install --force-reinstall --no-deps dist/flash_linear_attention_npu-*.whl \
        2>&1 | tee "${OUT_DIR}/pip_install.log"

    log "validate packaged wheel API"
    python3 scripts/check_packaged_wheel_api.py 2>&1 | tee "${OUT_DIR}/check_packaged_wheel_api.log"
else
    log "skip build requested; validating installed wrapper only"
fi

OPP_ROOT=$(python3 - <<'PY'
import os
import fla_npu
print(os.path.join(os.path.dirname(fla_npu.__file__), "opp"))
PY
)
export FLA_NPU_OPP_PATH="${OPP_ROOT}"
export ASCEND_CUSTOM_OPP_PATH="${OPP_ROOT}:${OPP_ROOT}/vendors/fla_npu_transformer:${ASCEND_CUSTOM_OPP_PATH:-}"
export LD_LIBRARY_PATH="${OPP_ROOT}/vendors/fla_npu_transformer/op_api/lib:${LD_LIBRARY_PATH:-}"
export ASCEND_RT_VISIBLE_DEVICES="${DEVICE}"
export TEST_DEVICE_ID=0
export TORCH_EXTENSIONS_DIR="/tmp/torch_ext_dhu_perf_${TARGET}_${TAG}"
export TORCHINDUCTOR_CACHE_DIR="/tmp/torchinductor_dhu_perf_${TARGET}_${TAG}"
export TRITON_CACHE_DIR="/tmp/triton_dhu_perf_${TARGET}_${TAG}"

log "runtime package check"
python3 - <<'PY' 2>&1 | tee "${OUT_DIR}/runtime_package.log"
import os
import fla_npu
from fla_npu.ops import ascendc
print("fla_npu", fla_npu.__file__)
print("has_dhu", hasattr(ascendc, "npu_chunk_gated_delta_rule_bwd_dhu"))
print("FLA_NPU_OPP_PATH", os.environ.get("FLA_NPU_OPP_PATH"))
print("ASCEND_CUSTOM_OPP_PATH", os.environ.get("ASCEND_CUSTOM_OPP_PATH"))
print("ASCEND_RT_VISIBLE_DEVICES", os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
PY

log "run msprof"
rm -rf "${PROFILE_OUT}"
mkdir -p "${PROFILE_OUT}"
msprof --output="${PROFILE_OUT}" \
    python3 -u "${PROFILE_SCRIPT}" --device 0 --branch "${BRANCH}" --repeat "${REPEAT}" \
    2>&1 | tee "${OUT_DIR}/msprof.log"

OP_SUMMARY=$(find "${PROFILE_OUT}" -path '*/mindstudio_profiler_output/op_summary_*.csv' -print | sort | tail -n 1)
[[ -n "${OP_SUMMARY}" ]] || die "op_summary csv was not generated"

for name in api_statistic op_statistic op_summary task_time; do
    src=$(find "${PROFILE_OUT}" -path "*/mindstudio_profiler_output/${name}_*.csv" -print | sort | tail -n 1)
    [[ -n "${src}" ]] || die "${name} csv was not generated"
    cp "${src}" "${OUT_DIR}/${name}.csv"
done

python3 - "${OUT_DIR}/op_summary.csv" "${BRANCH}" "${REPEAT}" "${OUT_DIR}/dhu_perf_summary.csv" "${OP_TYPE}" <<'PY'
import csv
import statistics
import sys

op_summary, branch_arg, repeat_s, out_csv, op_type = sys.argv[1:]
repeat = int(repeat_s)
branches = ["g", "gK"] if branch_arg == "both" else [branch_arg]
expected = len(branches) * repeat
fields = [
    "Task Duration(us)",
    "aicore_time(us)",
    "aic_mac_time(us)",
    "aic_scalar_time(us)",
    "aic_mte1_time(us)",
    "aic_mte2_time(us)",
    "aic_mte3_time(us)",
    "aic_fixpipe_time(us)",
    "aiv_time(us)",
    "aiv_vec_time(us)",
    "aiv_scalar_time(us)",
    "aiv_mte2_time(us)",
    "aiv_mte3_time(us)",
    "cube_utilization(%)",
]

with open(op_summary, newline="") as f:
    rows = [row for row in csv.DictReader(f) if row.get("OP Type") == op_type]

if len(rows) != expected:
    raise SystemExit(f"expected {expected} {op_type} rows, got {len(rows)}")

records = []
idx = 0
for branch in branches:
    for repeat_idx in range(1, repeat + 1):
        row = rows[idx]
        idx += 1
        record = {"branch": branch, "repeat": str(repeat_idx)}
        for field in fields:
            record[field] = row.get(field, "")
        records.append(record)

with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["branch", "repeat", *fields])
    writer.writeheader()
    writer.writerows(records)

print("DHU PERF SUMMARY")
for branch in branches:
    values = [float(record["Task Duration(us)"]) for record in records if record["branch"] == branch]
    print(
        f"{branch}: values={','.join(f'{value:.3f}' for value in values)} "
        f"mean={statistics.mean(values):.3f} min={min(values):.3f} max={max(values):.3f}"
    )
PY

(
    cd "${OUT_DIR}"
    sha256sum api_statistic.csv op_statistic.csv op_summary.csv task_time.csv dhu_perf_summary.csv \
        > SHA256SUMS
)

log "artifacts=${OUT_DIR}"
log "summary=${OUT_DIR}/dhu_perf_summary.csv"
