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
    "flash_linear_attention_npu.egg-info",
    "flash_linear_attention_npu.egg-link",
    "flash_linear_attention_npu-*.dist-info",
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
    local wheels=(dist/flash_linear_attention_npu-*.whl)
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
    local external_opp_root="$check_dir/external_opp"
    local run_file

    echo "[CI] Checking standalone torch_custom wheel plus run package OPP layout"
    rm -rf "$check_dir"
    mkdir -p "$dist_dir" "$target_dir"

    (cd torch_custom/fla_npu && python3 setup.py bdist_wheel --dist-dir "$dist_dir")

    shopt -s nullglob
    local wheels=("$dist_dir"/flash_linear_attention_npu-*.whl)
    shopt -u nullglob
    if (( ${#wheels[@]} != 1 )); then
        echo "[CI][ERROR] Expected exactly one flash-linear-attention-npu standalone wheel, found ${#wheels[@]}." >&2
        exit 1
    fi

    python3 -m pip install --force-reinstall --no-deps --target "$target_dir" "${wheels[0]}"
    PYTHONPATH="$target_dir" python3 - <<'PY'
from pathlib import Path

import fla_npu

package_dir = Path(fla_npu.__file__).resolve().parent
required = [
    package_dir / "opp" / "vendors" / "config.ini",
    package_dir / "opp" / "vendors" / "fla_npu_transformer" / "README.txt",
]
missing = [str(path.relative_to(package_dir)) for path in required if not path.exists()]
if missing:
    raise SystemExit("Standalone fla_npu wheel is missing OPP skeleton files: " + ", ".join(missing))
print("[CI] Standalone torch_custom wheel OPP skeleton check passed.")
PY

    mkdir -p "$external_opp_root/vendors/fla_npu_transformer/op_api/lib"
    touch "$external_opp_root/vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so"
    PYTHONPATH="$target_dir" ASCEND_OPP_PATH="$external_opp_root" ASCEND_CUSTOM_OPP_PATH= FLA_NPU_OPP_PATH= python3 - <<'PY'
import os
from pathlib import Path

import fla_npu

expected = (Path(os.environ["ASCEND_OPP_PATH"]) / "vendors" / "fla_npu_transformer").resolve()
actual = fla_npu._resolve_vendor_dir()
if actual != expected:
    raise SystemExit(f"Standalone OPP skeleton shadows external OPP: actual={actual}, expected={expected}")
print("[CI] Standalone torch_custom wheel does not shadow external OPP check passed.")
PY

    run_file="$(find_single_run_package)"
    chmod +x "$run_file"
    PYTHONPATH="$target_dir${PYTHONPATH:+:$PYTHONPATH}" "$run_file" --full --quiet

    PYTHONPATH="$target_dir" python3 - <<'PY'
from pathlib import Path

import fla_npu

package_dir = Path(fla_npu.__file__).resolve().parent
vendor_dir = package_dir / "opp" / "vendors" / "fla_npu_transformer"
required = [
    vendor_dir / "op_api" / "lib" / "libcust_opapi.so",
    vendor_dir / "op_api" / "lib" / "libopapi.so",
]
missing = [str(path.relative_to(package_dir)) for path in required if not path.exists()]
if missing:
    raise SystemExit("Standalone fla_npu wheel run-package install is missing OPP files: " + ", ".join(missing))
print("[CI] Standalone torch_custom wheel plus run package OPP layout check passed.")
PY
}

find_single_run_package() {
    shopt -s nullglob
    local run_files=(build_out/fla-npu-*.run build/fla-npu-*.run)
    shopt -u nullglob

    if (( ${#run_files[@]} == 0 )); then
        echo "[CI][ERROR] No fla-npu .run package found in build_out/ or build/." >&2
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
    "${run_file}" --quiet

    local vendor_name="fla_npu"
    local vendor_dir="fla_npu_transformer"
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

check_scoped_wheel_opp_install() {
    local scoped_op="${CI_SCOPED_WHEEL_INSTALL_OP:-chunk_fwd_o}"
    local run_file
    local install_log

    echo "[CI] Checking scoped run package replacement of installed wheel OPP: ${scoped_op}"
    build_and_check_wheel_api

    rm -rf build build_out
    bash build.sh --pkg --soc="$ci_soc" --vendor_name=fla_npu --ops="$scoped_op" -j"$ci_jobs"
    run_file="$(find_single_run_package)"
    chmod +x "$run_file"

    install_log="$(mktemp)"
    "$run_file" --full --quiet 2>&1 | tee "$install_log"

    if ! grep -q "Operator support status after installing this run package" "$install_log"; then
        echo "[CI][ERROR] Scoped run package install did not print the operator support status table." >&2
        exit 1
    fi
    if ! grep -q "$scoped_op" "$install_log"; then
        echo "[CI][ERROR] Scoped run package install output did not mention ${scoped_op}." >&2
        exit 1
    fi
    if ! grep -q "WARNING" "$install_log"; then
        echo "[CI][ERROR] Scoped run package install did not warn about operators outside the scoped build." >&2
        exit 1
    fi
    if grep -q "aclnn ABI header changes detected before installing this run package" "$install_log"; then
        echo "[CI][ERROR] Scoped run package install printed the removed duplicate ABI warning block." >&2
        exit 1
    fi
    if grep -q "Continue to overwrite the installed wheel OPP" "$install_log"; then
        echo "[CI][ERROR] Scoped run package install printed the removed duplicate confirmation prompt." >&2
        exit 1
    fi

    python3 - <<'PY'
import pathlib
import fla_npu

package_dir = pathlib.Path(fla_npu.__file__).resolve().parent
vendor_dir = package_dir / "opp" / "vendors" / "fla_npu_transformer"
required = [
    vendor_dir / "op_api" / "lib" / "libcust_opapi.so",
    vendor_dir / "op_api" / "lib" / "libopapi.so",
]
missing = [path.name for path in required if not path.exists()]
if missing:
    raise SystemExit("Missing scoped wheel OPP files: " + ", ".join(missing))
print("[CI] Scoped wheel OPP install check passed.")
PY
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

if [[ "${CI_RUN_WHEEL_API_CHECK:-false}" == "true" ]]; then
    build_and_check_wheel_api
fi

if [[ "${CI_RUN_STANDALONE_WHEEL_LAYOUT_CHECK:-false}" == "true" ]]; then
    check_standalone_torch_custom_wheel_layout
fi

if [[ "${CI_RUN_EXAMPLE_ST:-true}" == "true" ]]; then
    install_custom_opp_package
    check_example_python_deps
    build_torch_custom
    example_st_args=(--device "$ci_test_device" --cases-file "${CI_EXAMPLE_CASES_FILE:-ci/example_st_cases.json}")
    accuracy_report_file="${CI_ACCURACY_REPORT_FILE:-output/gdr_accuracy_report.json}"
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
        echo "[CI][WARN] Accuracy report was not generated: $accuracy_report_file" >&2
    fi
fi

if [[ "${CI_RUN_SCOPED_WHEEL_INSTALL_CHECK:-false}" == "true" ]]; then
    check_scoped_wheel_opp_install
fi
