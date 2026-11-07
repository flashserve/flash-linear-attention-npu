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
ci_ops="${CI_OPS:-}"
change_scope="${CI_CHANGE_SCOPE:-}"
change_ops="${CI_CHANGE_OPS:-}"
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
    if [[ -n "$ci_ops" ]]; then
        printf ' CI_OPS=%q' "$ci_ops" >&2
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
            for stage in opp-package standalone-layout torch-adapter gdr-example-st; do
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

classify_changed_files() {
    local classification_output=""
    local base_ref="" head_ref=""

    if [[ -n "${CI_CHANGED_FILES:-}" ]]; then
        if ! classification_output="$(printf '%s\n' "$CI_CHANGED_FILES" | python3 ci/classify_changes.py --paths-from-stdin)"; then
            classification_output=""
        fi
    else
        # The workflow creates a no-ff merge commit. Its first parent is the
        # base actually tested and the merge commit contains the PR tree.
        if git rev-parse --verify HEAD^2 >/dev/null 2>&1; then
            base_ref="HEAD^1"
            head_ref="HEAD"
        elif git rev-parse --verify HEAD~1 >/dev/null 2>&1; then
            base_ref="HEAD~1"
            head_ref="HEAD"
        fi
        if [[ -n "$base_ref" ]]; then
            if ! classification_output="$(python3 ci/classify_changes.py --git-range "$base_ref" "$head_ref")"; then
                classification_output=""
            fi
        fi
    fi

    if [[ -n "$classification_output" ]]; then
        change_scope="$(sed -n '1p' <<<"$classification_output")"
        change_ops="$(sed -n '2p' <<<"$classification_output")"
    fi
    case "$change_scope" in
        docs-only|tests-only|partial|full) ;;
        *)
            echo "[CI][WARN] Could not classify changed files; using full compilation." >&2
            change_scope="full"
            change_ops=""
            ;;
    esac

    # Preserve the old /run-npu-ci ops=... contract as an explicit override;
    # automatic file classification is used when no filter was requested.
    if [[ -n "${CI_OPS:-}" ]]; then
        change_scope="partial"
        change_ops="$CI_OPS"
        ci_ops="$CI_OPS"
    else
        case "$change_scope" in
            docs-only|tests-only|full) ci_ops="" ;;
            partial) ci_ops="$change_ops" ;;
        esac
    fi
    export CI_CHANGE_SCOPE="$change_scope"
    export CI_CHANGE_OPS="$change_ops"
    echo "[CI] changed-file scope=$change_scope ops=${change_ops:-<none>}"
}

write_docs_only_reports() {
    local stage
    local accuracy_report_file="${CI_ACCURACY_REPORT_FILE:-.ci-tmp/gdr_accuracy_report-${CI_ACCURACY_PLATFORM:-local}.json}"
    mkdir -p "$(dirname "$accuracy_report_file")"
    for stage in environment-contracts opp-package standalone-layout torch-adapter gdr-example-st; do
        python3 ci/manage_npu_ci_stage_report.py \
            --output "$stage_report_file" \
            update \
            --stage "$stage" \
            --status success \
            --exit-code 0 \
            --reason "documentation-only change; compilation and tests are not required"
    done
    python3 ci/manage_npu_ci_stage_report.py --output "$stage_report_file" finalize
    ACCURACY_REPORT_FILE="$accuracy_report_file" python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["ACCURACY_REPORT_FILE"])
payload = {
    "schema": "gdr-accuracy-report-v1",
    "complete": True,
    "status": "skipped",
    "classification": "docs-only",
    "metadata": {},
    "summary": {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "not_run": 0,
        "accuracy_total": 0,
        "accuracy_passed": 0,
        "accuracy_failed": 0,
        "accuracy_not_run": 0,
    },
    "cases": [],
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
    echo "[CI] Documentation-only change: reports written; compilation and tests skipped."
}

classify_changed_files
if [[ "$change_scope" == "docs-only" ]]; then
    write_docs_only_reports
    exit 0
fi
if [[ "$change_scope" == "partial" ]]; then
    echo "[CI] Partial operator scope selected: ${change_ops}"
fi

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
ci_test_device="${CI_CONTAINER_DEVICE:-0}"

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

cleanup_installed_fla_npu_python_packages() {
    echo "[CI] Cleaning stale installed flash-linear-attention-npu Python artifacts"
    python3 - <<'PY'
from __future__ import annotations

import shutil
import site
import sysconfig
from pathlib import Path


def site_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("purelib", "platlib"):
        path = sysconfig.get_paths().get(key)
        if path:
            roots.append(Path(path))
    try:
        roots.extend(Path(path) for path in site.getsitepackages())
    except Exception:
        pass
    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass
    result: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        if resolved not in seen:
            result.append(resolved)
            seen.add(resolved)
    return result


def remove_path(root: Path, path: Path) -> None:
    try:
        resolved = path.resolve()
    except OSError:
        return
    if resolved == root or root not in resolved.parents:
        return
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    print(f"[CI] Removed stale Python artifact: {path}")


def scrub_pth(root: Path) -> None:
    markers = (
        "flash-linear-attention-npu",
        "flash_linear_attention_npu",
        "torch_custom/fla_npu",
        "torch_custom\\fla_npu",
    )
    for pth in root.glob("*.pth"):
        try:
            lines = pth.read_text(encoding="utf-8", errors="ignore").splitlines(keepends=True)
        except OSError:
            continue
        kept = [line for line in lines if not any(marker in line for marker in markers)]
        if kept == lines:
            continue
        if kept:
            pth.write_text("".join(kept), encoding="utf-8")
            print(f"[CI] Scrubbed stale fla_npu entry from: {pth}")
        else:
            remove_path(root, pth)


patterns = (
    "fla",
    "fla_npu",
    "fla_npu.egg-info",
    "fla_npu.egg-link",
    "fla_npu-*.dist-info",
    "flash_linear_attention_npu*.egg-info",
    "flash_linear_attention_npu*.egg-link",
    "flash_linear_attention_npu*.dist-info",
    "__editable__*fla_npu*.*",
    "__editable__*flash_linear_attention_npu*.*",
    "__editable__*flash_linear_attention_npu*",
)

for root in site_roots():
    if not root.exists() or not root.is_dir():
        continue
    for pattern in patterns:
        for path in root.glob(pattern):
            remove_path(root, path)
    scrub_pth(root)
PY
}

torch_custom_built=false

build_torch_custom() {
    if [[ "$torch_custom_built" == "true" ]]; then
        return
    fi
    cleanup_installed_fla_npu_python_packages
    (cd torch_custom/fla_npu && bash build.sh)
    torch_custom_built=true
}

build_and_check_wheel_api() {
    rm -rf dist
    python3 -m pip wheel --no-build-isolation --no-deps . -w dist
    shopt -s nullglob
    local wheels=(dist/flash_linear_attention_npu*.whl)
    shopt -u nullglob
    if (( ${#wheels[@]} != 1 )); then
        echo "[CI][ERROR] Expected exactly one flash_linear_attention_npu wheel, found ${#wheels[@]}." >&2
        exit 1
    fi
    cleanup_installed_fla_npu_python_packages
    python3 -m pip install --force-reinstall --no-deps --no-cache-dir "${wheels[0]}"
    wheel_api_args=()
    if [[ "${CI_CHECK_TRITON_API:-false}" == "true" ]]; then
        wheel_api_args+=(--check-triton)
    fi
    python3 scripts/check_packaged_wheel_api.py "${wheel_api_args[@]}"
}

check_standalone_torch_custom_wheel_layout() {
    local check_dir="$TMPDIR/fla-npu-standalone-wheel-check"
    local dist_dir="$check_dir/dist"
    local target_dir="$check_dir/site"
    local run_file

    echo "[CI] Checking standalone torch_custom wheel plus run package OPP layout"
    rm -rf "$check_dir"
    mkdir -p "$dist_dir" "$target_dir"

    (cd torch_custom/fla_npu && python3 setup.py bdist_wheel --dist-dir "$dist_dir")

    shopt -s nullglob
    local wheels=("$dist_dir"/flash_linear_attention_npu*.whl)
    shopt -u nullglob
    if (( ${#wheels[@]} != 1 )); then
        echo "[CI][ERROR] Expected exactly one flash-linear-attention-npu standalone wheel, found ${#wheels[@]}." >&2
        exit 1
    fi

    python3 -m pip install --force-reinstall --no-deps --target "$target_dir" "${wheels[0]}"
    PYTHONPATH="$target_dir" python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("fla_npu")
if spec is None or spec.origin is None:
    raise SystemExit("[CI][ERROR] Standalone fla_npu wheel is not discoverable")
package_dir = Path(spec.origin).resolve().parent
required = [
    package_dir / "opp" / "vendors" / "config.ini",
    package_dir / "opp" / "vendors" / "fla_npu_transformer" / "README.txt",
]
missing = [str(path.relative_to(package_dir)) for path in required if not path.exists()]
if missing:
    raise SystemExit("[CI][ERROR] Standalone fla_npu wheel is missing OPP skeleton files: " + ", ".join(missing))
print("[CI] Standalone torch_custom wheel OPP skeleton check passed.")
PY

    run_file="$(find_single_run_package)"
    python3 scripts/check_install_workflows.py \
        --wheel "${wheels[0]}" \
        --base-mode skeleton \
        --run-package "$run_file" \
        --work-cwd "$repo_dir"
}

find_single_run_package() {
    shopt -s nullglob
    local run_files=(build_out/fla_npu_linux-*.run build/fla_npu_linux-*.run)
    shopt -u nullglob

    if (( ${#run_files[@]} == 0 )); then
        echo "[CI][ERROR] No fla_npu_linux-*.run package found in build_out/ or build/." >&2
        exit 1
    fi
    if (( ${#run_files[@]} > 1 )); then
        echo "[CI][WARN] Multiple .run packages found; using ${run_files[0]}." >&2
    fi
    printf '%s\n' "${run_files[0]}"
}

install_custom_opp_package() {
    local run_file
    run_file="$(find_single_run_package)"

    echo "[CI] Installing custom OPP package: ${run_file}"
    chmod +x "${run_file}"
    if [[ "$change_scope" == "partial" ]]; then
        # A partial run package contains only selected operators. Install it
        # into a temporary root, then register that vendor as a package-local
        # overlay so the complete architecture wheel remains the fallback for
        # every unmodified operator.
        local staging_dir="$TMPDIR/fla-npu-partial-opp"
        rm -rf -- "$staging_dir"
        mkdir -p "$staging_dir"
        "${run_file}" --install-path="$staging_dir" --quiet
        python3 -m fla_npu_opp_manage \
            --overlay \
            --source "$staging_dir/vendors/fla_npu_transformer" \
            --force
    else
        "${run_file}" --quiet
    fi

    local vendor_name="fla_npu"
    local vendor_dir="fla_npu_transformer"
    local op_api_lib=""
    local package_dir=""
    package_dir="$(python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("fla_npu")
if spec is not None and spec.origin is not None:
    print(Path(spec.origin).resolve().parent)
PY
)"
    for candidate in \
        "${package_dir}/opp/vendors/${vendor_dir}/op_api/lib" \
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
    local package_dir
    local vendor_dir
    local env_script
    package_dir="$(python3 - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("fla_npu")
if spec is None or spec.origin is None:
    raise SystemExit("[CI][ERROR] Installed fla_npu package is not discoverable")
print(Path(spec.origin).resolve().parent)
PY
)"
    vendor_dir="${package_dir}/opp/vendors/fla_npu_transformer"
    if [[ ! -f "${vendor_dir}/op_api/lib/libcust_opapi.so" ]]; then
        echo "[CI][ERROR] Installed fla_npu OPP is incomplete." >&2
        return 1
    fi
    env_script="${vendor_dir}/bin/set_env.bash"
    if [[ -f "$env_script" ]]; then
        # Preserve the generated overlay-first order and multi-library Stable
        # lookup, falling back to the base vendor for unmodified operators.
        set +u
        source "$env_script"
        set -u
    else
        # Keep compatibility with older wheels that predate the environment
        # script and only expose a single package-local vendor.
        export ASCEND_CUSTOM_OPP_PATH="${vendor_dir}${ASCEND_CUSTOM_OPP_PATH:+:${ASCEND_CUSTOM_OPP_PATH}}"
        export FLA_NPU_OP_API_LIB="${vendor_dir}/op_api/lib/libcust_opapi.so"
    fi
    export FLA_NPU_OPP_PATH="${package_dir}/opp"
    export LD_LIBRARY_PATH="${vendor_dir}/op_api/lib:${LD_LIBRARY_PATH:-}"
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

install_latest_arch_wheel() {
    local package_tier
    case "$ci_soc" in
        ascend910b) package_tier="a2" ;;
        ascend910_93) package_tier="a3" ;;
        ascend950) package_tier="a5" ;;
        *)
            echo "[CI][ERROR] No prebuilt flash-linear-attention-npu tier for CI_SOC=$ci_soc" >&2
            return 2
            ;;
    esac
    local wheel_index="https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly/flash-linear-attention-npu-${package_tier}/latest/"
    local package_name="flash-linear-attention-npu-${package_tier}"
    echo "[CI] Installing latest ${package_name} wheel from ${wheel_index}"
    PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install \
        --pre \
        --force-reinstall \
        --no-cache-dir \
        --no-deps \
        --no-index \
        --find-links "$wheel_index" \
        "${package_name}==0.*"
}

build_opp_run_package() {
    if [[ "$change_scope" == "partial" ]]; then
        install_latest_arch_wheel
    fi
    bash ci/prepare_ci_cache.sh
    case "$ci_mode" in
        quick|full)
            if [[ "$change_scope" == "partial" ]]; then
                bash build.sh --pkg --soc="$ci_soc" --vendor_name=fla_npu \
                    --ops="$ci_ops" -j"$ci_jobs"
            else
                FLA_NPU_SOC="$ci_soc" python3 scripts/build_wheel.py
            fi
            ;;
        *)
            echo "[CI][ERROR] Unsupported CI_MODE: $ci_mode" >&2
            return 2
            ;;
    esac
}

build_pytorch_adapter() {
    install_custom_opp_package
    check_example_python_deps
    if [[ "$change_scope" == "partial" ]]; then
        echo "[CI] Keeping the architecture base wheel and overlaying only the selected operators"
        return 0
    fi
    build_torch_custom

    if [[ "${CI_RUN_TORCH_TESTS:-false}" == "true" || \
          ( "$ci_mode" == "full" && -z "$ci_ops" ) ]]; then
        local test_args=(--device "$ci_test_device")
        if [[ -n "${CI_TEST_OP:-}" ]]; then
            test_args+=(--op "$CI_TEST_OP")
        fi
        (cd torch_custom/fla_npu/test && bash test.sh "${test_args[@]}")
    fi
}

run_tests_only() {
    # tests-only changes use the latest architecture wheel as the runtime
    # base. The existing GDR/Example-ST stage below remains the test entry
    # point; this preparation stage must not add a new torch_custom test set.
    install_latest_arch_wheel
    configure_installed_fla_npu_opp
    check_example_python_deps
    echo "[CI] Tests-only scope prepared the base wheel; existing Example-ST cases provide test execution."
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
    if [[ "${CI_RUN_STANDALONE_WHEEL_LAYOUT_CHECK:-false}" == "true" || \
          "$requested_stage" != "all" || "$ci_mode" == "full" ]]; then
        check_standalone_torch_custom_wheel_layout
    fi
    if [[ "${CI_RUN_WHEEL_API_CHECK:-false}" == "true" ]]; then
        build_and_check_wheel_api
    fi
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

pass_disabled_ci_stage() {
    local stage="$1"
    local reason="$2"
    # A stage disabled by the selected CI scope is a successful no-op. Keep
    # skipped for stages that were not selected or were blocked by a previous
    # failure, because those states still carry pipeline-control semantics.
    update_ci_stage "$stage" success --exit-code 0 --reason "$reason"
    echo "[CI][STAGE] PASS ${stage}: ${reason}"
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
        pass_disabled_ci_stage "$stage" "stage is disabled by the current CI configuration"
        return
    fi

    run_ci_stage "$stage" "$label" "$@"
    if [[ "${ci_stage_status[$stage]}" != "success" ]]; then
        failed_stage="$stage"
    fi
}

echo "[CI] mode=$ci_mode soc=$ci_soc ops=${ci_ops:-<all>} jobs=$ci_jobs cpack_jobs=$ci_cpack_jobs"
requested_stage="${CI_STAGE:-all}"
declare -A stage_position=(
    [environment-contracts]=1
    [opp-package]=2
    [standalone-layout]=3
    [torch-adapter]=4
    [gdr-example-st]=5
)
if [[ "$requested_stage" == "all" ]]; then
    target_stage_position=5
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

opp_package_enabled=true
if [[ "$change_scope" == "tests-only" ]]; then
    opp_package_enabled=false
fi
run_pipeline_stage \
    opp-package \
    "build the target-SOC OPP run package" \
    "$opp_package_enabled" \
    build_opp_run_package

standalone_layout_enabled=false
if [[ "$change_scope" != "tests-only" && ( "$requested_stage" != "all" || \
      "${CI_RUN_STANDALONE_WHEEL_LAYOUT_CHECK:-false}" == "true" || \
      "${CI_RUN_WHEEL_API_CHECK:-false}" == "true" || \
      "$ci_mode" == "full" ) ]]; then
    standalone_layout_enabled=true
fi
run_pipeline_stage \
    standalone-layout \
    "standalone torch_custom wheel and OPP install layout" \
    "$standalone_layout_enabled" \
    check_wheel_install_layouts

torch_adapter_enabled=false
if [[ "$requested_stage" != "all" || \
      "${CI_RUN_EXAMPLE_ST:-true}" == "true" || \
      "${CI_BUILD_TORCH_CUSTOM:-false}" == "true" || \
      "${CI_RUN_TORCH_TESTS:-false}" == "true" || \
      "$ci_mode" == "full" ]]; then
    torch_adapter_enabled=true
fi
if [[ "$change_scope" == "tests-only" ]]; then
    torch_adapter_enabled=true
fi
torch_adapter_label="install the OPP and build the PyTorch adapter"
torch_adapter_command=build_pytorch_adapter
if [[ "$change_scope" == "tests-only" ]]; then
    torch_adapter_label="install the prebuilt OPP and run tests without compiling"
    torch_adapter_command=run_tests_only
fi
run_pipeline_stage \
    torch-adapter \
    "$torch_adapter_label" \
    "$torch_adapter_enabled" \
    "$torch_adapter_command"

gdr_example_enabled="${CI_RUN_EXAMPLE_ST:-true}"
if [[ "$requested_stage" != "all" || "$change_scope" == "tests-only" ]]; then
    gdr_example_enabled=true
fi
run_pipeline_stage \
    gdr-example-st \
    "run all enabled GDR Example/ST accuracy cases" \
    "$gdr_example_enabled" \
    run_gdr_example_st

stage_report_status=0
python3 ci/manage_npu_ci_stage_report.py --output "$stage_report_file" finalize || stage_report_status=$?
exit "$stage_report_status"
