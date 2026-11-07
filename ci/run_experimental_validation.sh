#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

soc="${CI_SOC:-ascend950}"
op="recompute_w_u_fwd_experimental"
device_id="${CI_EXPERIMENTAL_DEVICE_ID:-0}"
jobs="${CI_JOBS:-4}"
if [[ "$soc" != "ascend950" ]]; then
    echo "[experimental][ERROR] Runtime validation is currently defined only for ascend950." >&2
    exit 2
fi
if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[experimental][ERROR] Source the CANN toolkit set_env.sh first." >&2
    exit 2
fi

python3 -m unittest ci.tests.test_experimental_build_contract -q
bash build.sh --pkg --experimental --soc="$soc" --ops="$op" -j"$jobs"

run_package="${repo_dir}/build_out/fla_npu_linux-$(uname -m).run"
if [[ ! -f "$run_package" ]]; then
    echo "[experimental][ERROR] Run package not found: $run_package" >&2
    exit 1
fi

tmp_root="${CI_TMPDIR:-/tmp}"
if [[ ! -d "$tmp_root" ]]; then
    echo "[experimental][ERROR] CI_TMPDIR does not exist: $tmp_root" >&2
    exit 2
fi
install_root="$(mktemp -d "${tmp_root%/}/fla-experimental-opp.XXXXXX")"
"$run_package" --install-path="$install_root" --quiet
vendor="${install_root}/vendors/fla_npu_transformer"
kernel_root="${vendor}/op_impl/ai_core/tbe/kernel/${soc}"
mapfile -t kernel_dirs < <(find "$kernel_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
if [[ ${#kernel_dirs[@]} -ne 1 || "${kernel_dirs[0]}" != "$op" ]]; then
    echo "[experimental][ERROR] Kernel directory set is not exactly {$op}: ${kernel_dirs[*]}" >&2
    exit 1
fi
if [[ -d "${vendor}/op_impl/ai_core/tbe/fla_npu_transformer_impl/ascendc/recompute_w_u_fwd" ]]; then
    echo "[experimental][ERROR] Stable recompute source entered the experimental OPP." >&2
    exit 1
fi

impl="${vendor}/op_impl/ai_core/tbe/fla_npu_transformer_impl"
for required in \
    "${impl}/dynamic/${op}.py" \
    "${impl}/ascendc/${op}/recompute_w_u_fwd_experimental.cpp" \
    "${vendor}/op_impl/ai_core/tbe/kernel/config/${soc}/${op}.json" \
    "${vendor}/op_proto/inc/recompute_wu_fwd_experimental_proto.h" \
    "${vendor}/op_api/include/aclnnop/aclnn_${op}.h"; do
    if [[ ! -f "$required" ]]; then
        echo "[experimental][ERROR] Missing package file: $required" >&2
        exit 1
    fi
done
while IFS= read -r header || [[ -n "$header" ]]; do
    header="${header%$'\r'}"
    [[ -z "$header" || "$header" == \#* ]] && continue
    if [[ ! -f "${impl}/ascendc/common/${header}" ]]; then
        echo "[experimental][ERROR] Missing experimental common header: $header" >&2
        exit 1
    fi
done < experimental/_common/manifest.txt

opapi="${vendor}/op_api/lib/libcust_opapi.so"
for symbol in aclnnRecomputeWUFwdExperimental aclnnRecomputeWUFwdExperimentalGetWorkspaceSize; do
    if ! nm -D "$opapi" | awk '{print $3}' | grep -Fxq "$symbol"; then
        echo "[experimental][ERROR] Missing ACLNN symbol: $symbol" >&2
        exit 1
    fi
done
if nm -D "$opapi" | awk '{print $3}' | grep -Eq '^aclnnRecomputeWUFwd(GetWorkspaceSize)?$'; then
    echo "[experimental][ERROR] Stable ACLNN symbol entered the experimental OPP." >&2
    exit 1
fi

bash experimental/recompute_w_u_fwd_experimental/tests/run_aclnn_smoke.sh \
    "$install_root" "$device_id" 128 64
bash experimental/recompute_w_u_fwd_experimental/tests/run_aclnn_smoke.sh \
    "$install_root" "$device_id" 256 128
echo "[experimental] PASS: build, isolated install, package audit, and ACLNN execution"
echo "[experimental] Isolated OPP: $install_root"
