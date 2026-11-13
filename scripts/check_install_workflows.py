#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Exercise repeatable wheel and scoped run-package installation workflows."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path
from typing import Iterable, Optional


# The wheel is published per product tier (flash-linear-attention-npu-a2/a3/a5)
# and a local build carries the same name, so match the family instead of the
# old base name.  The concrete name is read back from the installed dist-info.
DIST_INFO_GLOB = "flash_linear_attention_npu*.dist-info"
VENDOR_DIR = "fla_npu_transformer"
# PEP 427 lets a wheel keep its payload under ``<dist>-<version>.data/<scheme>/``
# instead of the archive root.  ``bdist_wheel`` builds exactly that layout when a
# distribution reports ``root_is_pure == False`` without extension modules, so the
# installed tree and the archive no longer share entry names while pip still maps
# both ``purelib`` and ``platlib`` onto the environment site root.
WHEEL_PAYLOAD_SCHEMES = ("purelib", "platlib")


def _dist_name(dist_info: Path) -> str:
    """Project name of an installed wheel.

    The directory is ``<name>-<version>.dist-info``, so dropping the suffix
    yields a name that still carries the version -- and ``pip uninstall`` rejects
    ``name-1.2.3`` as an invalid requirement.  The wheel's own METADATA owns the
    project name, so read it there and only fall back to trimming the trailing
    ``-<version>`` component when the metadata cannot be read.
    """

    metadata = dist_info / "METADATA"
    try:
        lines = metadata.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        lines = []
    for line in lines:
        if line.lower().startswith("name:"):
            name = line.split(":", 1)[1].strip()
            if name:
                return name
    stem = dist_info.name[: -len(".dist-info")]
    return stem.rsplit("-", 1)[0] if "-" in stem else stem


def _run(command: list[str], *, env: dict[str, str], cwd: Path) -> None:
    display_command = list(command)
    for index, part in enumerate(display_command):
        if index > 0 and display_command[index - 1] == "-c" and "\n" in part:
            display_command[index] = "<inline-code>"
    printable = " ".join(shlex.quote(part) for part in display_command)
    print("[install-workflows] RUN", printable, flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def _manifest_from_wheel(wheel: Path) -> dict[str, str]:
    result = {}
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            relative = _installed_relative_name(info.filename)
            if not relative.startswith("fla_npu/opp/"):
                continue
            result[relative] = hashlib.sha256(archive.read(info)).hexdigest()
    return result


def _installed_relative_name(entry: str) -> str:
    """Map a wheel entry name onto its installed, site-root-relative path.

    ``fla_npu/opp/...`` stays as-is.  ``<dist>-<version>.data/purelib/fla_npu/...``
    and its ``platlib`` twin resolve to the same installed path, because the
    environment under test keeps both schemes in one site root.
    """

    name = entry.replace("\\", "/")
    while name.startswith("./"):
        name = name[2:]
    parts = name.split("/")
    if len(parts) > 2 and parts[0].endswith(".data") and parts[1] in WHEEL_PAYLOAD_SCHEMES:
        return "/".join(parts[2:])
    return name


def _is_generated_bytecode(path: Path) -> bool:
    return path.suffix == ".pyc" and "__pycache__" in path.parts


def _is_manager_file(path: Path, package_dir: Path) -> bool:
    relative = path.relative_to(package_dir).parts
    return relative == ("opp", ".overlay.lock") or relative[:2] == ("opp", "wheel_layers")


def _opp_files(package_dir: Path, *, managed: bool = False) -> list[Path]:
    return [
        path
        for path in sorted((package_dir / "opp").rglob("*"))
        if (path.is_file() or path.is_symlink())
        and not (managed and _is_manager_file(path, package_dir))
    ]


def _manifest_from_directory(package_dir: Path, *, managed: bool = False) -> dict[str, str]:
    site_root = package_dir.parent
    result = {}
    for path in _opp_files(package_dir, managed=managed):
        if _is_generated_bytecode(path):
            continue
        relative = path.relative_to(site_root).as_posix()
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _assert_manifest_matches_wheel(package_dir: Path, wheel: Path, *, managed: bool = False) -> None:
    expected = _manifest_from_wheel(wheel)
    actual = _manifest_from_directory(package_dir, managed=managed)
    if actual == expected:
        return

    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys())
    changed = sorted(
        path for path in expected.keys() & actual.keys() if expected[path] != actual[path]
    )
    raise AssertionError(
        "Installed OPP differs from the selected wheel: "
        f"missing={missing}, unexpected={unexpected}, changed={changed}"
    )


def _find_package_dir(python: Path, env: dict[str, str], cwd: Path) -> Path:
    output = subprocess.check_output(
        [
            str(python),
            "-c",
            (
                "import importlib.util; from pathlib import Path; "
                "spec = importlib.util.find_spec('fla_npu'); "
                "assert spec is not None and spec.origin is not None; "
                "print(Path(spec.origin).resolve().parent)"
            ),
        ],
        cwd=str(cwd),
        env=env,
        encoding="utf-8",
    ).strip()
    return Path(output)


def _assert_record_covers_opp(package_dir: Path, *, managed: bool = False) -> None:
    dist_infos = sorted(package_dir.parent.glob(DIST_INFO_GLOB))
    if len(dist_infos) != 1:
        raise AssertionError(
            f"Expected exactly one {DIST_INFO_GLOB}, found {len(dist_infos)}"
        )
    record = dist_infos[0] / "RECORD"
    with record.open("r", encoding="utf-8", newline="") as handle:
        recorded = {row[0]: row[1:] for row in csv.reader(handle) if row}
    site_root = package_dir.parent
    opp_files = {
        path.relative_to(site_root).as_posix(): path
        for path in _opp_files(package_dir, managed=managed)
    }
    missing = sorted(opp_files.keys() - recorded.keys())
    if missing:
        raise AssertionError(f"Wheel RECORD does not cover installed OPP files: {missing}")

    missing_payload = [relative for relative in recorded
                       if relative.startswith("fla_npu/opp/") and not (site_root / relative).is_file()]
    if missing_payload:
        raise AssertionError(f"Wheel RECORD references missing OPP files: {missing_payload}")
    if managed and any(_is_manager_file(site_root / relative, package_dir)
                       for relative in recorded if relative.startswith("fla_npu/")):
        raise AssertionError("Overlay state/payload must not be added to the base wheel RECORD")

    invalid = []
    for relative, path in sorted(opp_files.items()):
        if _is_generated_bytecode(path) and recorded[relative] == ["", ""]:
            continue
        content = path.read_bytes()
        digest = base64.urlsafe_b64encode(
            hashlib.sha256(content).digest()
        ).rstrip(b"=")
        expected = [f"sha256={digest.decode('ascii')}", str(len(content))]
        if recorded[relative] != expected:
            invalid.append(relative)
    if invalid:
        raise AssertionError(f"Wheel RECORD has stale OPP hashes or sizes: {invalid}")


def _assert_opp_layout(package_dir: Path, *, require_runtime: bool,
                       vendors: Optional[list[Path]] = None) -> None:
    if vendors is not None:
        if require_runtime and not vendors:
            raise AssertionError("No active OPP vendor")
        for vendor in vendors:
            _assert_vendor_layout(vendor, require_runtime=require_runtime)
        return
    _assert_vendor_layout(package_dir / "opp" / "vendors" / VENDOR_DIR,
                          require_runtime=require_runtime)


def _assert_vendor_layout(vendor_dir: Path, *, require_runtime: bool) -> None:
    packaged_opapi = vendor_dir / "op_api" / "lib" / "libcust_opapi.so"
    conflicting_alias = packaged_opapi.with_name("libopapi.so")
    if conflicting_alias.exists() or conflicting_alias.is_symlink():
        raise AssertionError(f"Installed wheel contains conflicting alias: {conflicting_alias}")
    if require_runtime and not packaged_opapi.is_file():
        raise AssertionError(f"Installed wheel is missing packaged op_api: {packaged_opapi}")
    if require_runtime:
        dynamic = subprocess.check_output(
            ["readelf", "-d", str(packaged_opapi)],
            encoding="utf-8",
            errors="replace",
        )
        if "(SONAME)" in dynamic:
            raise AssertionError(
                "Packaged FLA op_api must not claim the generic libcust_opapi.so SONAME"
            )


def _assert_set_env_idempotent(package_dir: Path, env: dict[str, str], cwd: Path,
                               *, vendors: Optional[list[Path]] = None) -> None:
    vendor_dir = package_dir / "opp" / "vendors" / VENDOR_DIR
    set_env = vendor_dir / "bin" / "set_env.bash"
    if not set_env.is_file():
        if vendors is not None and not (vendor_dir / "op_api/lib/libcust_opapi.so").is_file():
            print("[install-workflows] skeleton has no set_env; validating .pth in a fresh process", flush=True)
            return
        raise AssertionError(f"Packaged set_env.bash is missing: {set_env}")

    check = r'''
set -euo pipefail
unset ASCEND_CUSTOM_OPP_PATH LD_LIBRARY_PATH FLA_NPU_OPP_PATH FLA_NPU_OP_API_LIB FLA_NPU_OP_API_LIBS
source "$1"
printf '%s\n' "$ASCEND_CUSTOM_OPP_PATH" "${LD_LIBRARY_PATH-}" "$FLA_NPU_OPP_PATH" "$FLA_NPU_OP_API_LIB" "${FLA_NPU_OP_API_LIBS-}"
source "$1"
printf '%s\n' "$ASCEND_CUSTOM_OPP_PATH" "${LD_LIBRARY_PATH-}" "$FLA_NPU_OPP_PATH" "$FLA_NPU_OP_API_LIB" "${FLA_NPU_OP_API_LIBS-}"
'''
    output = subprocess.check_output(
        ["bash", "-c", check, "check-set-env", str(set_env)],
        cwd=str(cwd),
        env=env,
        encoding="utf-8",
    ).splitlines()
    op_api_lib = str(vendor_dir / "op_api" / "lib" / "libcust_opapi.so")
    root_based_snapshot = [
        f"{vendor_dir.parent.parent}:{vendor_dir}",
        "",
        str(vendor_dir.parent.parent),
        op_api_lib,
        "",
    ]
    vendor_order_snapshot = [
        str(vendor_dir),
        "",
        str(vendor_dir.parent.parent),
        op_api_lib,
        op_api_lib,
    ]
    snapshot_size = len(root_based_snapshot)
    first = output[:snapshot_size]
    second = output[snapshot_size:]
    expected_snapshots = [root_based_snapshot, vendor_order_snapshot]
    if vendors is not None:
        expected_snapshots = [[os.pathsep.join(map(str, vendors)), "",
                               str(vendor_dir.parent.parent), op_api_lib,
                               os.pathsep.join(str(p / "op_api/lib/libcust_opapi.so") for p in vendors)]]
    if first not in expected_snapshots:
        raise AssertionError(
            "set_env.bash produced an unexpected environment after the first source: "
            f"actual={first}, expected one of {expected_snapshots}"
        )
    if second != first:
        raise AssertionError(
            f"set_env.bash is not idempotent: first={first}, second={second}"
        )


def _assert_runtime(
    python: Path,
    env: dict[str, str],
    cwd: Path,
    expected_ops: Iterable[str],
    *,
    vendors: Optional[list[Path]] = None,
) -> None:
    child_env = env.copy()
    child_env["FLA_NPU_EXPECT_OPS"] = ",".join(expected_ops)
    if vendors is None:
        vendors = [_find_package_dir(python, env, cwd) / "opp/vendors" / VENDOR_DIR]
    child_env["FLA_NPU_EXPECT_VENDORS"] = json.dumps([str(p.resolve()) for p in vendors])
    code = r'''
import ctypes
import json
import os
from pathlib import Path

import fla_npu
from fla_npu.ops import ascendc

for name in filter(None, os.environ.get("FLA_NPU_EXPECT_OPS", "").split(",")):
    if not hasattr(ascendc, name) or not hasattr(ascendc, f"npu_{name}"):
        raise AssertionError(f"missing installed Ascend C API: {name}")

first = fla_npu.load_ascendc_opapi_libraries()
second = fla_npu.load_ascendc_opapi_libraries()
if first is not second or not first:
    raise AssertionError("Ascend C op_api loading is not idempotent")
vendors = [Path(p) for p in json.loads(os.environ["FLA_NPU_EXPECT_VENDORS"])]
expected_libraries = [p / "op_api/lib/libcust_opapi.so" for p in vendors]
if len(first) != len(vendors) + 1:
    raise AssertionError(f"expected {len(vendors)} custom and one CANN handle, got {len(first)}")
if [Path(handle._name).resolve() for handle in first[:-1]] != expected_libraries:
    raise AssertionError("runtime did not load the selected overlay/base libraries in order")
configured_libraries = os.environ["FLA_NPU_OP_API_LIBS"].split(os.pathsep)
if configured_libraries != list(map(str, expected_libraries)):
    raise AssertionError(f"wrong configured library order: {configured_libraries}")

package_dir = Path(fla_npu.__file__).resolve().parent
expected = (
    package_dir
    / "opp"
    / "vendors"
    / "fla_npu_transformer"
    / "op_api"
    / "lib"
    / "libcust_opapi.so"
).resolve()
if not expected.is_file():
    expected = expected_libraries[0]
configured = Path(os.environ["FLA_NPU_OP_API_LIB"]).resolve()
if configured != expected:
    raise AssertionError(f"runtime selected {configured}, expected {expected}")

maps = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
for library in expected_libraries:
    if str(library) not in maps:
        raise AssertionError(f"selected libcust_opapi.so is not mapped: {library}")
if "/libopapi.so" not in maps:
    raise AssertionError("CANN libopapi.so is not mapped into the process")
if "custom_aclnn_extension_lib" in maps:
    raise AssertionError("default import unexpectedly loaded the legacy C++ extension")

# Regression for issue #310: importing fla_npu and then opening CANN opapi must
# leave the subprocess able to finish normal interpreter teardown.
ctypes.CDLL("libopapi.so")
'''
    _run([str(python), "-c", code], env=child_env, cwd=cwd)


def _check_stale_alias_recovery(
    *,
    python: Path,
    env: dict[str, str],
    cwd: Path,
    expected_wheel: Path,
    expected_ops: Iterable[str],
    managed: bool = False,
) -> None:
    package_dir = _find_package_dir(python, env, cwd)
    packaged_opapi = (
        package_dir
        / "opp"
        / "vendors"
        / VENDOR_DIR
        / "op_api"
        / "lib"
        / "libcust_opapi.so"
    )
    stale_alias = (
        package_dir
        / "opp"
        / "vendors"
        / VENDOR_DIR
        / "op_api"
        / "lib"
        / "libopapi.so"
    )
    stale_alias.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(packaged_opapi, stale_alias)

    _assert_runtime(
        python,
        env,
        cwd,
        expected_ops,
    )
    _assert_opp_layout(package_dir, require_runtime=True)
    _assert_manifest_matches_wheel(package_dir, expected_wheel, managed=managed)
    print("[install-workflows] PASS stale-libopapi-alias-recovery", flush=True)


def _check_stage(
    name: str,
    *,
    python: Path,
    env: dict[str, str],
    cwd: Path,
    expected_wheel: Optional[Path],
    require_runtime: bool,
    load_runtime: bool,
    expected_ops: Iterable[str],
    managed: bool = False,
    vendors: Optional[list[Path]] = None,
) -> dict[str, str]:
    package_dir = _find_package_dir(python, env, cwd)
    if expected_wheel is not None:
        _assert_manifest_matches_wheel(package_dir, expected_wheel, managed=managed)
    _assert_record_covers_opp(package_dir, managed=managed)
    _assert_opp_layout(package_dir, require_runtime=require_runtime, vendors=vendors)
    if require_runtime:
        _assert_set_env_idempotent(package_dir, env, cwd, vendors=vendors)
    if load_runtime and require_runtime:
        _assert_runtime(python, env, cwd, expected_ops, vendors=vendors)
    elif expected_ops:
        raise AssertionError(
            f"Cannot validate APIs without loading the runtime: {list(expected_ops)}"
        )
    print(f"[install-workflows] PASS {name}", flush=True)
    return _manifest_from_directory(package_dir, managed=managed)


def _payload_manifest(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and not _is_generated_bytecode(p)}


def _base_snapshot(package_dir: Path) -> dict[str, str]:
    result = {p.relative_to(package_dir.parent).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in package_dir.rglob("*") if p.is_file()
              and not _is_generated_bytecode(p) and not _is_manager_file(p, package_dir)}
    for info in package_dir.parent.glob(DIST_INFO_GLOB):
        result[info.name + "/RECORD"] = hashlib.sha256((info / "RECORD").read_bytes()).hexdigest()
    return result


def _assert_base_unchanged(package_dir: Path, expected: dict[str, str]) -> None:
    if _base_snapshot(package_dir) != expected:
        raise AssertionError("Run installation/removal changed the base package or wheel RECORD")


def _active_run_vendors(python: Path, env: dict[str, str], cwd: Path) -> list[Path]:
    code = r'''
import json
import fla_npu_wheel_overlay as layers
package = layers.package_dir()
state = layers.state(package)
if not state.get("active") or state.get("pending"):
    raise AssertionError(f"Run installation must activate, not just return success: {state}")
snapshot = layers.active(package)
selected = layers.components(package, snapshot)
if selected["adapter"] is not None or not selected["opp"]:
    raise AssertionError(f"Run-only installation changed the adapter selection: {selected}")
plan = layers.runtime_plan(package)
if not plan or plan["errors"]:
    raise AssertionError(f"Invalid active run composition: {plan}")
expected = [layers.layer_path(package, selected["opp"]) / "fla_npu/opp/vendors" / layers.BASE]
base = package / "opp/vendors" / layers.BASE
if (base / layers.LIBRARY).is_file():
    expected.append(base)
if plan["vendors"] != list(map(str, expected)):
    raise AssertionError(f"Expected overlay then available base: {plan['vendors']}")
print(json.dumps(plan["vendors"]))
'''
    output = subprocess.check_output([str(python), "-c", code], env=env, cwd=cwd, encoding="utf-8")
    return [Path(p) for p in json.loads(output)]


def _assert_removed_overlay(package_dir: Path) -> set[Path]:
    """Removal retains only a lock inode and an inactive state, outside pip ownership."""
    root = package_dir / "opp/wheel_layers"
    state_file = root / "state.json"
    state = json.loads(state_file.read_text(encoding="utf-8"))
    if state != {"version": 2, "active": None, "pending": None}:
        raise AssertionError(f"Overlay is still selected after removal: {state}")
    if set(root.iterdir()) != {state_file}:
        raise AssertionError("Overlay removal left component payloads or unexpected state files")
    lock = package_dir / "opp/.overlay.lock"
    if lock.is_symlink() or lock.read_bytes() != b"":
        raise AssertionError("Unexpected overlay lock file")
    return {package_dir, package_dir / "opp", root, state_file, lock}


def _install_wheel(python: Path, wheel: Path, env: dict[str, str], cwd: Path) -> None:
    _run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--force-reinstall",
            "--no-cache-dir",
            "--no-deps",
            str(wheel),
        ],
        env=env,
        cwd=cwd,
    )


def _uninstall_and_assert_clean(
    name: str,
    *,
    python: Path,
    env: dict[str, str],
    cwd: Path,
    managed: bool = False,
) -> None:
    package_dir = _find_package_dir(python, env, cwd)
    site_root = package_dir.parent
    dist_infos = sorted(site_root.glob(DIST_INFO_GLOB))
    if len(dist_infos) != 1:
        raise AssertionError(
            f"Expected exactly one {DIST_INFO_GLOB} before uninstall, found {len(dist_infos)}"
        )

    record = dist_infos[0] / "RECORD"
    allowed = _assert_removed_overlay(package_dir) if managed else set()
    with record.open("r", encoding="utf-8", newline="") as handle:
        recorded_paths = [site_root / row[0] for row in csv.reader(handle) if row]

    _run(
        [str(python), "-m", "pip", "uninstall", "-y", _dist_name(dist_infos[0])],
        env=env,
        cwd=cwd,
    )

    residual_record_paths = [
        str(path) for path in recorded_paths if path.exists() or path.is_symlink()
    ]
    residual_dist_infos = [str(path) for path in site_root.glob(DIST_INFO_GLOB)]
    residual_package_entries = (
        [str(path) for path in package_dir.rglob("*")] if package_dir.exists() else []
    )
    if package_dir.exists():
        residual_package_entries.insert(0, str(package_dir))
    residual_package_entries = [p for p in residual_package_entries if Path(p) not in allowed]
    if residual_record_paths or residual_dist_infos or residual_package_entries:
        raise AssertionError(
            f"pip uninstall left {_dist_name(dist_infos[0])} files behind: "
            f"record={residual_record_paths}, dist_info={residual_dist_infos}, "
            f"package={residual_package_entries}"
        )

    child_env = env.copy()
    child_env["FLA_NPU_UNINSTALL_SITE_ROOT"] = str(site_root)
    code = r'''
import importlib.metadata
import importlib.util
import os
from pathlib import Path

site_root = Path(os.environ["FLA_NPU_UNINSTALL_SITE_ROOT"]).resolve()
spec = importlib.util.find_spec("fla_npu")
if spec is not None and spec.origin is not None:
    origin = Path(spec.origin).resolve()
    if origin == site_root or site_root in origin.parents:
        raise AssertionError(f"fla_npu remains importable from the test venv: {origin}")
for distribution in importlib.metadata.distributions():
    name = distribution.metadata["Name"] or ""
    if not name.replace("_", "-").startswith("flash-linear-attention-npu"):
        continue
    metadata_path = Path(distribution._path).resolve()
    if metadata_path == site_root or site_root in metadata_path.parents:
        raise AssertionError(
            f"distribution metadata remains in the test venv: {metadata_path}"
        )
'''
    _run([str(python), "-c", code], env=child_env, cwd=cwd)
    print(f"[install-workflows] PASS {name}", flush=True)


def _apply_run_package(run_package: Path, env: dict[str, str], cwd: Path,
                       options: Iterable[str] = ()) -> None:
    run_package.chmod(run_package.stat().st_mode | 0o100)
    _run([str(run_package), *options], env=env, cwd=cwd)


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run_workflows(args: argparse.Namespace, work_dir: Path) -> None:
    work_cwd = Path(args.work_cwd).expanduser().resolve()
    wheel = Path(args.wheel).expanduser().resolve()
    updated_wheel = (
        Path(args.updated_wheel).expanduser().resolve() if args.updated_wheel else None
    )
    run_package = (
        Path(args.run_package).expanduser().resolve() if args.run_package else None
    )
    for artifact in (wheel, updated_wheel, run_package):
        if artifact is not None and not artifact.is_file():
            raise FileNotFoundError(artifact)

    venv_dir = work_dir / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_dir)
    python = _venv_python(venv_dir)
    temp_dir = work_dir / "tmp"
    temp_dir.mkdir()

    env = os.environ.copy()
    env["PATH"] = os.pathsep.join([str(python.parent), env.get("PATH", "")])
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["TMPDIR"] = str(temp_dir)
    for name in ("PYTHONPATH", "FLA_NPU_OPP_PATH", "FLA_NPU_OP_API_LIB", "FLA_NPU_OP_API_LIBS",
                 "ASCEND_CUSTOM_OPP_PATH", "FLA_NPU_OVERLAY_VENDOR", "FLA_NPU_DISABLE_PTH",
                 "FLA_NPU_STABLE_LIB", "FLA_NPU_OVERLAY_OPS"):
        env.pop(name, None)

    base_requires_runtime = args.base_mode == "full"
    _install_wheel(python, wheel, env, work_cwd)
    _check_stage(
        "base-wheel-install-1",
        python=python,
        env=env,
        cwd=work_cwd,
        expected_wheel=wheel,
        require_runtime=base_requires_runtime,
        load_runtime=not args.skip_runtime_load,
        expected_ops=args.wheel_op,
    )
    _install_wheel(python, wheel, env, work_cwd)
    _check_stage(
        "base-wheel-install-2",
        python=python,
        env=env,
        cwd=work_cwd,
        expected_wheel=wheel,
        require_runtime=base_requires_runtime,
        load_runtime=not args.skip_runtime_load,
        expected_ops=args.wheel_op,
    )

    if run_package is not None:
        copied_run = work_dir / run_package.name
        shutil.copy2(run_package, copied_run)
        extracted = work_dir / "run-payload"
        _run(["bash", str(copied_run), "--noexec", "--extract=" + str(extracted)], env=env, cwd=work_cwd)
        source_vendor = extracted / "packages/vendors" / VENDOR_DIR
        if not (source_vendor / "op_api/lib/libcust_opapi.so").is_file():
            raise AssertionError(f"Run payload has no expected vendor: {source_vendor}")
        expected_payload = _payload_manifest(source_vendor)
        package_dir = _find_package_dir(python, env, work_cwd)
        base_snapshot = _base_snapshot(package_dir)
        # Each replacement must select identical payload bytes, regardless of
        # snapshot UUIDs or retained snapshots used by older running processes.
        for label, options in (("default", ()), ("repeat", ()),
                               ("install", ("--install",)), ("full", ("--full",))):
            _apply_run_package(copied_run, env, work_cwd, options)
            vendors = _active_run_vendors(python, env, work_cwd)
            if not vendors or _payload_manifest(vendors[0]) != expected_payload:
                raise AssertionError("Active OPP payload differs from the selected run package")
            _assert_base_unchanged(package_dir, base_snapshot)
            _check_stage(
                "run-package-" + label, python=python, env=env, cwd=work_cwd,
                expected_wheel=wheel, require_runtime=True,
                load_runtime=not args.skip_runtime_load, expected_ops=args.run_op,
                managed=True, vendors=vendors,
            )
            _assert_base_unchanged(package_dir, base_snapshot)
        _run([str(python), "-m", "fla_npu_opp_manage", "--remove-overlay"], env=env, cwd=work_cwd)
        _assert_removed_overlay(package_dir)
        _assert_base_unchanged(package_dir, base_snapshot)
        _check_stage(
            "run-package-removed", python=python, env=env, cwd=work_cwd,
            expected_wheel=wheel, require_runtime=base_requires_runtime,
            load_runtime=not args.skip_runtime_load, expected_ops=args.wheel_op, managed=True,
        )
        _uninstall_and_assert_clean(
            "run-package-uninstall",
            python=python,
            env=env,
            cwd=work_cwd,
            managed=True,
        )

    if updated_wheel is not None or run_package is not None:
        final_wheel = updated_wheel or wheel
        final_mode = args.updated_mode if updated_wheel is not None else args.base_mode
        final_ops = args.updated_wheel_op if updated_wheel is not None else args.wheel_op
        for attempt in (1, 2):
            _install_wheel(python, final_wheel, env, work_cwd)
            _check_stage(
                f"final-wheel-install-{attempt}",
                python=python,
                env=env,
                cwd=work_cwd,
                expected_wheel=final_wheel,
                require_runtime=final_mode == "full",
                load_runtime=not args.skip_runtime_load,
                expected_ops=final_ops,
                managed=run_package is not None,
            )

    final_wheel = updated_wheel or wheel
    final_mode = args.updated_mode if updated_wheel is not None else args.base_mode
    final_ops = args.updated_wheel_op if updated_wheel is not None else args.wheel_op
    if final_mode == "full" and not args.skip_runtime_load:
        _check_stale_alias_recovery(
            python=python,
            env=env,
            cwd=work_cwd,
            expected_wheel=final_wheel,
            expected_ops=final_ops,
            managed=run_package is not None,
        )

    _uninstall_and_assert_clean(
        "final-wheel-uninstall",
        python=python,
        env=env,
        cwd=work_cwd,
        managed=run_package is not None,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, help="Base wheel built by the one-command workflow")
    parser.add_argument("--updated-wheel", help="Optional wheel rebuilt after source or adapter changes")
    parser.add_argument("--run-package", help="Scoped .run to check default/repeated/--install/--full installation")
    parser.add_argument("--base-mode", choices=("full", "skeleton"), default="full")
    parser.add_argument("--updated-mode", choices=("full", "skeleton"), default="full")
    parser.add_argument("--wheel-op", action="append", default=[], help="Ascend C API expected in the base wheel")
    parser.add_argument("--run-op", action="append", default=[], help="Ascend C API expected after run-package install")
    parser.add_argument(
        "--updated-wheel-op",
        action="append",
        default=[],
        help="Ascend C API expected in the updated wheel",
    )
    parser.add_argument("--work-cwd", default=Path.cwd(), help="Working directory for checks")
    parser.add_argument(
        "--skip-runtime-load",
        action="store_true",
        help="Skip import and dlopen checks for source-only hosts",
    )
    parser.add_argument("--keep-work-dir", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    temp_root = os.environ.get("TMPDIR") or None
    if args.keep_work_dir:
        work_dir = Path(tempfile.mkdtemp(prefix="fla-npu-install-workflows-", dir=temp_root))
        print(f"[install-workflows] work_dir={work_dir}", flush=True)
        _run_workflows(args, work_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="fla-npu-install-workflows-", dir=temp_root) as temp_dir:
            _run_workflows(args, Path(temp_dir))
    print("[install-workflows] ALL PASS", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
