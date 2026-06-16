#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

select_ci_tmpdir() {
    local min_free_kb="${CI_TMPDIR_MIN_KB:-65536}"
    local candidates=()

    if [[ -n "${CI_TMPDIR:-}" ]]; then
        candidates=("$CI_TMPDIR")
    elif [[ -n "${CI_TMPDIR_CANDIDATES:-}" ]]; then
        local old_ifs="$IFS"
        IFS=':'
        read -r -a candidates <<< "$CI_TMPDIR_CANDIDATES"
        IFS="$old_ifs"
    else
        [[ -n "${TMPDIR:-}" ]] && candidates+=("$TMPDIR")
        candidates+=("$repo_dir/.ci-tmp")

        local root
        for root in /workspace /mnt /home /tmp /var/tmp; do
            [[ -d "$root" ]] && candidates+=("$root/fla-npu-ci-tmp")
        done

        local mount
        for mount in /mnt/*; do
            [[ -d "$mount" ]] && candidates+=("$mount/fla-npu-ci-tmp")
        done
    fi

    local best_dir=""
    local best_free_kb=-1
    local candidate
    local -A seen=()
    local checked=()

    for candidate in "${candidates[@]}"; do
        [[ -n "$candidate" ]] || continue
        if [[ -n "${seen[$candidate]:-}" ]]; then
            continue
        fi
        seen["$candidate"]=1

        if ! mkdir -p "$candidate" 2>/dev/null; then
            checked+=("$candidate=create-failed")
            continue
        fi
        if [[ ! -w "$candidate" ]]; then
            checked+=("$candidate=not-writable")
            continue
        fi

        local free_kb
        free_kb="$(df -Pk "$candidate" 2>/dev/null | awk 'NR == 2 { print $4 }')"
        if [[ ! "$free_kb" =~ ^[0-9]+$ ]]; then
            checked+=("$candidate=df-failed")
            continue
        fi
        checked+=("$candidate=${free_kb}KB")
        if (( free_kb > best_free_kb )); then
            best_dir="$candidate"
            best_free_kb="$free_kb"
        fi
    done

    if [[ -z "$best_dir" ]]; then
        echo "[CI][ERROR] No writable TMPDIR candidate found. Checked: ${checked[*]:-<none>}" >&2
        return 1
    fi
    if (( best_free_kb < min_free_kb )); then
        echo "[CI][ERROR] No TMPDIR candidate has at least ${min_free_kb} KB free. Checked: ${checked[*]}" >&2
        return 1
    fi

    echo "$best_dir"
}

ci_tmpdir="$(select_ci_tmpdir)"
export TMPDIR="$ci_tmpdir"
echo "[CI] TMPDIR=$TMPDIR ($(df -h "$TMPDIR" | awk 'NR == 2 { print $4 " free" }'))"

bash ci/cleanup_ci_logs.sh

if [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
    # shellcheck disable=SC1091
    set +u
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
    set -u
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    set +u
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi

if command -v npu-smi >/dev/null 2>&1; then
    if [[ -n "${NPU_SELECTED_DEVICE:-}" ]]; then
        if ! summary="$(bash ci/detect_npu.sh --summary 2>&1)"; then
            echo "[CI][WARN] Container npu-smi did not report devices; using host-selected NPU ${NPU_SELECTED_DEVICE}."
        else
            echo "$summary"
        fi
    else
        eval "$(bash ci/detect_npu.sh --env)"
        bash ci/detect_npu.sh --summary
    fi
fi

ci_mode="${CI_MODE:-quick}"
ci_soc="${CI_SOC:-${NPU_SOC:-ascend910b}}"
ci_ops="${CI_OPS:-}"
ci_jobs="${CI_JOBS:-$(nproc)}"
ci_cpack_jobs="${CI_CPACK_JOBS:-$ci_jobs}"
ci_test_device="${CI_CONTAINER_DEVICE:-0}"

if [[ "$ci_soc" == "unknown" ]]; then
    ci_soc="ascend910b"
fi

export CMAKE_BUILD_PARALLEL_LEVEL="$ci_cpack_jobs"
export MAKEFLAGS="${MAKEFLAGS:+$MAKEFLAGS }-j${ci_cpack_jobs}"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
if [[ -z "${PYTORCH_VERSION:-}" ]]; then
    PYTORCH_VERSION="$(python3 - <<'PY'
import torch
print(torch.__version__.split("+", 1)[0])
PY
)"
fi
export PYTORCH_VERSION

bash ci/prepare_ci_cache.sh

torch_custom_built=false

build_torch_custom() {
    if [[ "$torch_custom_built" == "true" ]]; then
        return
    fi
    (cd torch_custom/fla_npu && bash build.sh)
    torch_custom_built=true
}

install_custom_opp_package() {
    shopt -s nullglob
    local run_files=(build_out/fla-npu-*.run build/fla-npu-*.run)
    shopt -u nullglob

    if (( ${#run_files[@]} == 0 )); then
        echo "[CI][ERROR] No fla-npu .run package found in build_out/ or build/." >&2
        exit 1
    fi
    if (( ${#run_files[@]} > 1 )); then
        echo "[CI][WARN] Multiple .run packages found; installing ${run_files[0]}."
    fi

    echo "[CI] Installing custom OPP package: ${run_files[0]}"
    chmod +x "${run_files[0]}"
    "${run_files[0]}" --quiet

    local vendor_name="${CI_VENDOR_NAME:-fla_npu}"
    local vendor_dir="${vendor_name}_transformer"
    local op_api_lib=""
    for candidate in \
        "${ASCEND_OPP_PATH:-}/vendors/${vendor_dir}/op_api/lib" \
        "/usr/local/Ascend/vendors/${vendor_dir}/op_api/lib"; do
        if [[ -d "$candidate" ]]; then
            op_api_lib="$candidate"
            break
        fi
    done
    if [[ -z "$op_api_lib" ]]; then
        echo "[CI][ERROR] Custom OPP op_api lib path not found for vendor ${vendor_name}." >&2
        exit 1
    fi
    export LD_LIBRARY_PATH="${op_api_lib}:${LD_LIBRARY_PATH:-}"
    echo "[CI] Custom OPP op_api lib: ${op_api_lib}"
}

check_example_python_deps() {
python3 - <<'PY'
import importlib
from importlib import metadata

missing = []
for name in (
    "torch",
    "torch_npu",
    "triton",
    "pybind11",
    "torchnpugen.gen_op_plugin_functions",
    "torchnpugen.gen_backend_stubs",
    "torchnpugen.struct.gen_struct_opapi",
):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
try:
    metadata.version("triton-ascend")
except Exception as exc:
    missing.append(f"triton-ascend: {exc}")

if missing:
    raise SystemExit("Missing Python dependencies for Example ST:\n" + "\n".join(missing))
PY
}

ops_arg=()
if [[ -n "$ci_ops" ]]; then
    ops_arg=(--ops="$ci_ops")
fi

echo "[CI] mode=$ci_mode soc=$ci_soc ops=${ci_ops:-<all>} jobs=$ci_jobs cpack_jobs=$ci_cpack_jobs"

case "$ci_mode" in
    quick)
        bash build.sh --pkg --soc="$ci_soc" --vendor_name=fla_npu "${ops_arg[@]}" -j"$ci_jobs"
        ;;
    full)
        extra=()
        if [[ -n "$ci_ops" ]]; then
            extra=(--mode single --op "$ci_ops")
        fi
        bash gdn-verify.sh --device "$ci_test_device" "${extra[@]}"
        ;;
    *)
        echo "[CI][ERROR] Unsupported CI_MODE: $ci_mode" >&2
        exit 2
        ;;
esac

if [[ "${CI_BUILD_TORCH_CUSTOM:-false}" == "true" ]]; then
    build_torch_custom
fi

if [[ "${CI_RUN_TORCH_TESTS:-false}" == "true" ]]; then
    test_args=(--device "$ci_test_device")
    if [[ -n "${CI_TEST_OP:-}" ]]; then
        test_args+=(--op "$CI_TEST_OP")
    fi
    (cd torch_custom/fla_npu/test && bash test.sh "${test_args[@]}")
fi

if [[ "${CI_RUN_EXAMPLE_ST:-true}" == "true" ]]; then
    install_custom_opp_package
    check_example_python_deps
    build_torch_custom
    example_st_args=(--device "$ci_test_device" --cases-file "${CI_EXAMPLE_CASES_FILE:-ci/example_st_cases.json}")
    if [[ -n "${CI_EXAMPLE_CASE_FILTER:-}" ]]; then
        example_st_args+=(--case-filter "$CI_EXAMPLE_CASE_FILTER")
    fi
    python3 ci/run_example_st_cases.py "${example_st_args[@]}"
fi
