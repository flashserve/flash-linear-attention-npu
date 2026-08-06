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


DIST_INFO_GLOB = "flash_linear_attention_npu-*.dist-info"
DIST_NAME = "flash-linear-attention-npu"
VENDOR_DIR = "fla_npu_transformer"


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
            if info.is_dir() or not info.filename.startswith("fla_npu/opp/"):
                continue
            result[info.filename] = hashlib.sha256(archive.read(info)).hexdigest()
    return result


def _is_generated_bytecode(path: Path) -> bool:
    return path.suffix == ".pyc" and "__pycache__" in path.parts


def _opp_files(package_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted((package_dir / "opp").rglob("*"))
        if path.is_file() or path.is_symlink()
    ]


def _manifest_from_directory(package_dir: Path) -> dict[str, str]:
    site_root = package_dir.parent
    result = {}
    for path in _opp_files(package_dir):
        if _is_generated_bytecode(path):
            continue
        relative = path.relative_to(site_root).as_posix()
        result[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _assert_manifest_matches_wheel(package_dir: Path, wheel: Path) -> None:
    expected = _manifest_from_wheel(wheel)
    actual = _manifest_from_directory(package_dir)
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


def _assert_record_covers_opp(package_dir: Path) -> None:
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
        for path in _opp_files(package_dir)
    }
    missing = sorted(opp_files.keys() - recorded.keys())
    if missing:
        raise AssertionError(f"Wheel RECORD does not cover installed OPP files: {missing}")

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


def _assert_opp_layout(package_dir: Path, *, require_runtime: bool) -> None:
    vendor_dir = package_dir / "opp" / "vendors" / VENDOR_DIR
    custom_opapi = vendor_dir / "op_api" / "lib" / "libcust_opapi.so"
    conflicting_alias = custom_opapi.with_name("libopapi.so")
    if conflicting_alias.exists() or conflicting_alias.is_symlink():
        raise AssertionError(f"Installed wheel contains conflicting alias: {conflicting_alias}")
    if require_runtime and not custom_opapi.is_file():
        raise AssertionError(f"Installed wheel is missing custom op_api: {custom_opapi}")


def _assert_set_env_idempotent(package_dir: Path, env: dict[str, str], cwd: Path) -> None:
    vendor_dir = package_dir / "opp" / "vendors" / VENDOR_DIR
    set_env = vendor_dir / "bin" / "set_env.bash"
    if not set_env.is_file():
        raise AssertionError(f"Packaged set_env.bash is missing: {set_env}")

    check = r'''
set -euo pipefail
unset ASCEND_CUSTOM_OPP_PATH LD_LIBRARY_PATH FLA_NPU_OPP_PATH FLA_NPU_OP_API_LIB
source "$1"
source "$1"
printf '%s\n' "$ASCEND_CUSTOM_OPP_PATH" "$LD_LIBRARY_PATH" "$FLA_NPU_OPP_PATH" "$FLA_NPU_OP_API_LIB"
'''
    output = subprocess.check_output(
        ["bash", "-c", check, "check-set-env", str(set_env)],
        cwd=str(cwd),
        env=env,
        encoding="utf-8",
    ).splitlines()
    expected = [
        f"{vendor_dir.parent.parent}:{vendor_dir}",
        str(vendor_dir / "op_api" / "lib"),
        str(vendor_dir.parent.parent),
        str(vendor_dir / "op_api" / "lib" / "libcust_opapi.so"),
    ]
    if output != expected:
        raise AssertionError(f"set_env.bash is not idempotent: actual={output}, expected={expected}")


def _assert_runtime(
    python: Path,
    env: dict[str, str],
    cwd: Path,
    expected_ops: Iterable[str],
) -> None:
    child_env = env.copy()
    child_env["FLA_NPU_EXPECT_OPS"] = ",".join(expected_ops)
    code = r'''
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
configured = Path(os.environ["FLA_NPU_OP_API_LIB"]).resolve()
if configured != expected:
    raise AssertionError(f"runtime selected {configured}, expected {expected}")

maps = Path("/proc/self/maps").read_text(encoding="utf-8", errors="replace")
if str(expected) not in maps:
    raise AssertionError("packaged libcust_opapi.so is not mapped into the process")
'''
    _run([str(python), "-c", code], env=child_env, cwd=cwd)


def _check_stale_alias_recovery(
    *,
    python: Path,
    env: dict[str, str],
    cwd: Path,
    expected_wheel: Path,
    expected_ops: Iterable[str],
) -> None:
    package_dir = _find_package_dir(python, env, cwd)
    custom_opapi = (
        package_dir
        / "opp"
        / "vendors"
        / VENDOR_DIR
        / "op_api"
        / "lib"
        / "libcust_opapi.so"
    )
    stale_alias = custom_opapi.with_name("libopapi.so")
    shutil.copy2(custom_opapi, stale_alias)

    _assert_runtime(
        python,
        env,
        cwd,
        expected_ops,
    )
    _assert_opp_layout(package_dir, require_runtime=True)
    _assert_manifest_matches_wheel(package_dir, expected_wheel)
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
) -> dict[str, str]:
    package_dir = _find_package_dir(python, env, cwd)
    if expected_wheel is not None:
        _assert_manifest_matches_wheel(package_dir, expected_wheel)
    _assert_record_covers_opp(package_dir)
    _assert_opp_layout(package_dir, require_runtime=require_runtime)
    if require_runtime:
        _assert_set_env_idempotent(package_dir, env, cwd)
    if load_runtime and require_runtime:
        _assert_runtime(python, env, cwd, expected_ops)
    elif expected_ops:
        raise AssertionError(
            f"Cannot validate APIs without loading the runtime: {list(expected_ops)}"
        )
    print(f"[install-workflows] PASS {name}", flush=True)
    return _manifest_from_directory(package_dir)


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
) -> None:
    package_dir = _find_package_dir(python, env, cwd)
    site_root = package_dir.parent
    dist_infos = sorted(site_root.glob(DIST_INFO_GLOB))
    if len(dist_infos) != 1:
        raise AssertionError(
            f"Expected exactly one {DIST_INFO_GLOB} before uninstall, found {len(dist_infos)}"
        )

    record = dist_infos[0] / "RECORD"
    with record.open("r", encoding="utf-8", newline="") as handle:
        recorded_paths = [site_root / row[0] for row in csv.reader(handle) if row]

    _run(
        [str(python), "-m", "pip", "uninstall", "-y", DIST_NAME],
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
    if residual_record_paths or residual_dist_infos or residual_package_entries:
        raise AssertionError(
            "pip uninstall left flash-linear-attention-npu files behind: "
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
try:
    distribution = importlib.metadata.distribution("flash-linear-attention-npu")
except importlib.metadata.PackageNotFoundError:
    pass
else:
    metadata_path = Path(distribution._path).resolve()
    if metadata_path == site_root or site_root in metadata_path.parents:
        raise AssertionError(
            f"distribution metadata remains in the test venv: {metadata_path}"
        )
'''
    _run([str(python), "-c", code], env=child_env, cwd=cwd)
    print(f"[install-workflows] PASS {name}", flush=True)


def _apply_run_package(run_package: Path, env: dict[str, str], cwd: Path) -> None:
    run_package.chmod(run_package.stat().st_mode | 0o100)
    _run([str(run_package), "--install", "--quiet"], env=env, cwd=cwd)


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
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    env["TMPDIR"] = str(temp_dir)
    for name in ("PYTHONPATH", "FLA_NPU_OPP_PATH", "FLA_NPU_OP_API_LIB", "ASCEND_CUSTOM_OPP_PATH"):
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
        _apply_run_package(copied_run, env, work_cwd)
        first_run_manifest = _check_stage(
            "run-package-install-1",
            python=python,
            env=env,
            cwd=work_cwd,
            expected_wheel=None,
            require_runtime=True,
            load_runtime=not args.skip_runtime_load,
            expected_ops=args.run_op,
        )
        _apply_run_package(copied_run, env, work_cwd)
        second_run_manifest = _check_stage(
            "run-package-install-2",
            python=python,
            env=env,
            cwd=work_cwd,
            expected_wheel=None,
            require_runtime=True,
            load_runtime=not args.skip_runtime_load,
            expected_ops=args.run_op,
        )
        if first_run_manifest != second_run_manifest:
            raise AssertionError("Applying the same run package twice changed the installed OPP")
        _uninstall_and_assert_clean(
            "run-package-uninstall",
            python=python,
            env=env,
            cwd=work_cwd,
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
        )

    _uninstall_and_assert_clean(
        "final-wheel-uninstall",
        python=python,
        env=env,
        cwd=work_cwd,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", required=True, help="Base wheel built by the one-command workflow")
    parser.add_argument("--updated-wheel", help="Optional wheel rebuilt after source or adapter changes")
    parser.add_argument("--run-package", help="Optional scoped .run package to apply twice")
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
