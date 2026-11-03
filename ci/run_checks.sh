#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"

ci_mode="${CI_MODE:-quick}"
ci_soc="${CI_SOC:-${NPU_SOC:-ascend910b}}"
ci_build_kind="${CI_BUILD_KIND:-full}"
ci_build_ops="${CI_BUILD_OPS:-}"
if [[ "$ci_mode" == "full" ]]; then
    ci_build_kind=full
    ci_build_ops=""
fi
export CI_BUILD_KIND="$ci_build_kind" CI_BUILD_OPS="$ci_build_ops"
if [[ "$ci_soc" == "unknown" ]]; then
    ci_soc="ascend910b"
fi
export CI_SOC="$ci_soc"

stage_report_file="${CI_STAGE_REPORT_FILE:-.ci-tmp/npu-ci-stages-${CI_ACCURACY_PLATFORM:-local}.json}"
ci_bootstrap_complete=false

print_ci_reproduction() {
    local stage="$1"
    local image="fla-npu-ci:9.1.0-910b"
    local dockerfile="ci/Dockerfile"
    local require_preloaded_image="false"
    if [[ "$ci_soc" == "ascend950" ]]; then
        image="fla-npu-ci:9.1.0-950"
        dockerfile="ci/Dockerfile.ascend950"
        require_preloaded_image="true"
    fi
    printf '[CI][REPRO] CI_STAGE=%q CI_MODE=%q CI_SOC=%q FLA_NPU_SOC=%q' \
        "$stage" "$ci_mode" "$ci_soc" "$ci_soc" >&2
    printf ' CI_IMAGE=%q CI_DOCKERFILE=%q CI_REQUIRE_PRELOADED_IMAGE=%q' \
        "$image" "$dockerfile" "$require_preloaded_image" >&2
    if [[ "$ci_soc" == "ascend950" ]]; then
        printf ' CI_TMPDIR=%q' "/tmp/fla-npu-ci" >&2
    fi
    printf ' CI_BUILD_KIND=%q CI_BUILD_OPS=%q CI_ATK_OPS=%q CI_BASE_REF=%q CI_DEVICE_ID=%q' \
        "$ci_build_kind" "$ci_build_ops" "${CI_ATK_OPS:-}" "${CI_BASE_REF:-}" \
        "${CI_DEVICE_ID:-${NPU_SELECTED_DEVICE:-auto}}" >&2
    if [[ -n "${CI_BASE_WHEEL_VERSION:-}" ]]; then
        printf ' CI_BASE_WHEEL_VERSION=%q' "$CI_BASE_WHEEL_VERSION" >&2
    fi
    if [[ -n "${CI_BASE_WHEEL_INDEX_URL:-}" ]]; then
        printf ' CI_BASE_WHEEL_INDEX_URL=%q' "$CI_BASE_WHEEL_INDEX_URL" >&2
    fi
    printf ' bash ci/run_ci_container.sh\n' >&2
}

finalize_bootstrap_failure() {
    local exit_code=$?
    trap - EXIT
    if (( exit_code != 0 )) && [[ "$ci_bootstrap_complete" != "true" ]]; then
        set +e
        if [[ -f "$stage_report_file" ]]; then
            python3 ci/manage_npu_ci_stage_report.py \
                --output "$stage_report_file" \
                update \
                --stage environment-contracts \
                --status failure \
                --exit-code "$exit_code" \
                --reason "CI environment initialization exited with code ${exit_code}"
            local stage
            for stage in opp-package standalone-layout torch-adapter gdr-example-st atk-accuracy; do
                python3 ci/manage_npu_ci_stage_report.py \
                    --output "$stage_report_file" \
                    update \
                    --stage "$stage" \
                    --status skipped \
                    --reason "prerequisite stage environment-contracts failed"
            done
            python3 ci/manage_npu_ci_stage_report.py --output "$stage_report_file" finalize >/dev/null 2>&1
        fi
        echo "[CI][STAGE] FAIL environment-contracts: CI environment initialization (exit ${exit_code})" >&2
        print_ci_reproduction environment-contracts
    fi
    exit "$exit_code"
}

trap finalize_bootstrap_failure EXIT
python3 ci/manage_npu_ci_stage_report.py --output "$stage_report_file" init
python3 ci/manage_npu_ci_stage_report.py \
    --output "$stage_report_file" \
    update \
    --stage environment-contracts \
    --status running

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

# AscendNPU-IR (Bisheng compiler) environment
if [[ -f /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh ]]; then
    # shellcheck disable=SC1091
    set +u
    source /usr/local/Ascend/cann-9.1.0/share/info/ascendnpu-ir/bin/set_env.sh
    set -u
elif [[ -f /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh ]]; then
    # shellcheck disable=SC1091
    set +u
    source /usr/local/Ascend/cann-9.0.0/share/info/ascendnpu-ir/bin/set_env.sh
    set -u
fi

# NNAL (ATB) environment
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    # shellcheck disable=SC1091
    set +u
    source /usr/local/Ascend/nnal/atb/set_env.sh
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

ci_jobs="${CI_JOBS:-16}"
ci_cpack_jobs="${CI_CPACK_JOBS:-$ci_jobs}"
ci_test_device="${CI_CONTAINER_DEVICE:-${CI_DEVICE_ID:-0}}"

export CMAKE_BUILD_PARALLEL_LEVEL="$ci_cpack_jobs"
export MAKEFLAGS="${MAKEFLAGS:+$MAKEFLAGS }-j${ci_cpack_jobs}"
export MAX_JOBS="$ci_jobs"
export FLA_NPU_BUILD_ARGS="${FLA_NPU_BUILD_ARGS:+$FLA_NPU_BUILD_ARGS }-j${ci_jobs}"
export TORCH_DEVICE_BACKEND_AUTOLOAD="${TORCH_DEVICE_BACKEND_AUTOLOAD:-0}"
if [[ -z "${PYTORCH_VERSION:-}" ]]; then
    PYTORCH_VERSION="$(python3 - <<'PY'
import torch
print(torch.__version__.split("+", 1)[0])
PY
)"
fi
export PYTORCH_VERSION

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
    raise SystemExit("[CI][ERROR] Missing Python dependencies for Example ST: " + "; ".join(missing))
PY
}

configure_installed_fla_npu_opp() {
    local package_env="${CI_PACKAGE_ENV_FILE:-.ci-tmp/npu-ci-package-env.sh}"
    if [[ ! -f "$package_env" ]]; then
        echo "[CI][ERROR] Missing package environment: $package_env; rerun the opp-package stage." >&2
        return 1
    fi
    # Every stage runs in its own subshell. Load the completed installation and
    # select overlay/base paths before starting a fresh Python or ATK process.
    source "$package_env"
    if [[ ! -f "${FLA_NPU_ENV:-}" ]]; then
        echo "[CI][ERROR] Installed package environment is unavailable: ${FLA_NPU_ENV:-<unset>}" >&2
        return 1
    fi
    set +u
    source "$FLA_NPU_ENV"
    set -u
    local lib
    local -a libraries=()
    local -a lib_dirs=()
    IFS=':' read -r -a libraries <<< "${FLA_NPU_OP_API_LIBS:-}"
    for lib in "${libraries[@]}"; do
        [[ -n "$lib" ]] && lib_dirs+=("$(dirname "$lib")")
    done
    if (( ${#lib_dirs[@]} )); then
        export LD_LIBRARY_PATH="$(IFS=:; echo "${lib_dirs[*]}")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
}

check_environment_contracts() {
    python3 tests/test_wheel_environment.py -b
    python3 torch_custom/fla_npu/tools/stable_coverage.py --strict
    python3 torch_custom/fla_npu/tools/op_abi_parity.py
    python3 tests/test_opp_overlay.py -b
    python3 tests/test_opp_abi_contract.py -b
    python3 tests/test_stable_gates.py -b
    python3 tests/test_stable_only_runtime.py -b
    python3 tests/test_stable_build_policy.py -b
    python3 torch_custom/fla_npu/test/test_aclnn_ctypes_abi.py -b
    python3 torch_custom/fla_npu/test/test_runtime_device_guard.py -b
    python3 torch_custom/fla_npu/test/test_ascendc_mutation_contract.py -b
}

prepare_ci_package() {
    case "$ci_mode" in
        quick|full) bash ci/prepare_ci_package.sh ;;
        *) echo "[CI][ERROR] Unsupported CI_MODE: $ci_mode" >&2; return 2 ;;
    esac
}

check_pytorch_adapter() {
    configure_installed_fla_npu_opp
    check_example_python_deps
    if [[ "${CI_RUN_TORCH_TESTS:-false}" == "true" || "$ci_mode" == "full" ]]; then
        local test_args=(--device "$ci_test_device")
        if [[ -n "${CI_TEST_OP:-}" ]]; then
            test_args+=(--op "$CI_TEST_OP")
        fi
        (cd torch_custom/fla_npu/test && bash test.sh "${test_args[@]}")
    fi
}

run_gdr_example_st() {
    configure_installed_fla_npu_opp
    local example_st_args=(
        --device "$ci_test_device"
        --cases-file "${CI_EXAMPLE_CASES_FILE:-ci/example_st_cases.json}"
    )
    local accuracy_report_file="${CI_ACCURACY_REPORT_FILE:-output/gdr_accuracy_report.json}"
    mkdir -p "$(dirname "$accuracy_report_file")"
    rm -f "$accuracy_report_file" "$accuracy_report_file.tmp"
    export CI_ACCURACY_HEAD_SHA="${CI_ACCURACY_HEAD_SHA:-${NPU_CI_TARGET_SHA:-}}"
    example_st_args+=(--accuracy-report-file "$accuracy_report_file")
    if [[ -n "${CI_EXAMPLE_CASE_FILTER:-}" ]]; then
        example_st_args+=(--case-filter "$CI_EXAMPLE_CASE_FILTER")
    fi
    python3 ci/run_example_st_cases.py "${example_st_args[@]}"
    if [[ -f "$accuracy_report_file" ]]; then
        echo "[CI] Accuracy report generated: $accuracy_report_file"
    else
        echo "[CI][ERROR] Accuracy report was not generated: $accuracy_report_file" >&2
        return 1
    fi
}

check_wheel_install_layouts() {
    configure_installed_fla_npu_opp
    # Check the actual installation; never compile another standalone wheel here.
    python3 - <<'PY_LAYOUT'
import os
from pathlib import Path

opp = Path(os.environ["FLA_NPU_OPP_PATH"])
base = opp / "vendors/fla_npu_transformer"
required = [opp / "vendors/config.ini", base / "op_api/lib/libcust_opapi.so",
            base / "bin/set_env.bash"]
overlay = os.environ.get("FLA_NPU_OVERLAY_VENDOR", "")
if overlay:
    required.append(opp / "vendors" / overlay / "op_api/lib/libcust_opapi.so")
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise SystemExit("[CI][ERROR] Installed wheel/OPP layout is incomplete: " + ", ".join(missing))
print("[CI] Installed wheel/OPP layout passed (base and selected overlay preserved).")
PY_LAYOUT
    if [[ "${CI_RUN_WHEEL_API_CHECK:-false}" == "true" ]]; then
        local api_args=()
        [[ "${CI_CHECK_TRITON_API:-false}" == "true" ]] && api_args+=(--check-triton)
        python3 scripts/check_packaged_wheel_api.py "${api_args[@]}"
    fi
}

run_atk_accuracy() {
    configure_installed_fla_npu_opp
    CI_ATK_OPS="$CI_ATK_OPS" NPU_DEVICE_ID="$ci_test_device" SOC="$ci_soc" \
        ATK_OUTPUT_ROOT="${ATK_OUTPUT_ROOT:-$repo_dir/.ci-tmp/atk-results}" \
        bash ci/run_atk_checks.sh
}

declare -A ci_stage_status=()

update_ci_stage() {
    local stage="$1"
    local status="$2"
    shift 2
    python3 ci/manage_npu_ci_stage_report.py \
        --output "$stage_report_file" \
        update \
        --stage "$stage" \
        --status "$status" \
        "$@"
    ci_stage_status["$stage"]="$status"
}

run_ci_stage() {
    local stage="$1"
    local label="$2"
    shift 2
    local exit_code

    update_ci_stage "$stage" running
    echo "[CI][STAGE] START ${stage}: ${label}"
    set +e
    (
        set -euo pipefail
        "$@"
    )
    exit_code=$?
    set -e
    if (( exit_code == 0 )); then
        update_ci_stage "$stage" success --exit-code 0
        echo "[CI][STAGE] PASS ${stage}: ${label}"
    else
        update_ci_stage \
            "$stage" \
            failure \
            --exit-code "$exit_code" \
            --reason "stage command exited with code ${exit_code}"
        echo "[CI][STAGE] FAIL ${stage}: ${label} (exit ${exit_code})" >&2
        print_ci_reproduction "$stage"
    fi
}

skip_ci_stage() {
    local stage="$1"
    local reason="$2"
    update_ci_stage "$stage" skipped --reason "$reason"
    echo "[CI][STAGE] SKIP ${stage}: ${reason}"
}

stage_is_selected() {
    local stage="$1"
    (( stage_position[$stage] <= target_stage_position ))
}

run_pipeline_stage() {
    local stage="$1"
    local label="$2"
    local enabled="$3"
    shift 3

    if ! stage_is_selected "$stage"; then
        skip_ci_stage "$stage" "not selected by CI_STAGE=${requested_stage}"
        return
    fi
    if [[ -n "$failed_stage" ]]; then
        skip_ci_stage "$stage" "prerequisite stage ${failed_stage} failed"
        return
    fi
    if [[ "$enabled" != "true" ]]; then
        if [[ "$stage" == "atk-accuracy" ]]; then
            skip_ci_stage "$stage" "no ATK operators selected"
        else
            skip_ci_stage "$stage" "stage is disabled by the current CI configuration"
        fi
        return
    fi

    run_ci_stage "$stage" "$label" "$@"
    if [[ "${ci_stage_status[$stage]}" != "success" ]]; then
        failed_stage="$stage"
    fi
}

echo "[CI] mode=$ci_mode soc=$ci_soc build=$ci_build_kind build_ops=${ci_build_ops:-<none>} atk_ops=${CI_ATK_OPS:-<none>} jobs=$ci_jobs cpack_jobs=$ci_cpack_jobs"
requested_stage="${CI_STAGE:-all}"
declare -A stage_position=(
    [environment-contracts]=1
    [opp-package]=2
    [standalone-layout]=3
    [torch-adapter]=4
    [gdr-example-st]=5
    [atk-accuracy]=6
)
if [[ "$requested_stage" == "all" ]]; then
    target_stage_position=6
elif [[ -n "${stage_position[$requested_stage]:-}" ]]; then
    target_stage_position="${stage_position[$requested_stage]}"
else
    echo "[CI][ERROR] Unsupported CI_STAGE: ${requested_stage}" >&2
    exit 2
fi

ci_bootstrap_complete=true
trap - EXIT
failed_stage=""

run_pipeline_stage \
    environment-contracts \
    "environment, wheel configuration, and runtime contracts" \
    true \
    check_environment_contracts

run_pipeline_stage \
    opp-package \
    "prepare the target-SOC package ($ci_build_kind)" \
    true \
    prepare_ci_package

run_pipeline_stage \
    standalone-layout \
    "check the installed wheel and OPP layout" \
    true \
    check_wheel_install_layouts

run_pipeline_stage \
    torch-adapter \
    "check the installed PyTorch adapter and optional torch tests" \
    true \
    check_pytorch_adapter

gdr_example_enabled="${CI_RUN_EXAMPLE_ST:-true}"
if [[ "$requested_stage" != "all" ]]; then
    gdr_example_enabled=true
fi
run_pipeline_stage \
    gdr-example-st \
    "run all enabled GDR Example/ST accuracy cases" \
    "$gdr_example_enabled" \
    run_gdr_example_st

atk_enabled=false
[[ -n "${CI_ATK_OPS:-}" ]] && atk_enabled=true
run_pipeline_stage \
    atk-accuracy \
    "run selected operator ATK accuracy checks" \
    "$atk_enabled" \
    run_atk_accuracy

stage_report_status=0
python3 ci/manage_npu_ci_stage_report.py --output "$stage_report_file" finalize || stage_report_status=$?
exit "$stage_report_status"
