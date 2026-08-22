"""Preflight checks for building and running flash-linear-attention-npu."""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from importlib import metadata
from typing import Optional

from packaging.version import InvalidVersion, Version

from fla_npu_build_capabilities import (
    probe_legacy_build_capabilities,
    torch_npu_gdn_stream_fix_error,
)


MIN_PYTHON = (3, 9)
MIN_TORCH = "2.6.0"
MIN_TRITON_ASCEND = "3.2.0"
MIN_TRITON_ASCEND_A5 = "3.2.1"


def _ok(message: str) -> None:
    print(f"[OK] {message}")


def _warn(message: str) -> None:
    print(f"[WARN] {message}")


def _fail(failures: list[str], message: str) -> None:
    failures.append(message)
    print(f"[FAIL] {message}")


def _import_module(failures: list[str], module_name: str):
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        _fail(failures, f"{module_name}: {exc}")
        return None
    origin = getattr(module, "__file__", None) or "built-in"
    _ok(f"{module_name}: {origin}")
    return module


def _distribution_version(name: str) -> str:
    try:
        return metadata.version(name)
    except Exception:
        return ""


def _check_min_version(failures: list[str], name: str, actual: str, minimum: str) -> None:
    if not actual:
        _fail(failures, f"{name} version is unknown")
        return
    actual_version = _version_obj(actual)
    minimum_version = _version_obj(minimum)
    if actual_version is None:
        _fail(failures, f"{name} has unsupported version string: {actual}")
    elif minimum_version is not None and actual_version < minimum_version:
        _fail(failures, f"{name}>={minimum} is required, got {actual}")


def _version_obj(value: str) -> Optional[Version]:
    try:
        return Version(value.split("+", 1)[0])
    except InvalidVersion:
        return None


def _check_triton_ascend_a5_compat(failures: list[str], actual: str) -> None:
    soc = os.getenv("FLA_NPU_SOC", "ascend910b")
    if soc != "ascend950":
        return
    actual_version = _version_obj(actual)
    if actual_version is None or actual_version < Version(MIN_TRITON_ASCEND_A5):
        _fail(
            failures,
            f"triton-ascend>={MIN_TRITON_ASCEND_A5} is required for FLA_NPU_SOC={soc}; got {actual}. "
            "triton-ascend 3.2.0 can crash on the A5 Triton runtime.",
        )


def _detect_cann_version() -> str:
    candidates = []
    for env_name in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH"):
        value = os.getenv(env_name)
        if not value:
            continue
        path = os.path.abspath(value)
        candidates.extend(
            [
                os.path.join(path, "version.info"),
                os.path.join(path, "ascend_toolkit_install.info"),
                os.path.join(os.path.dirname(path), "version.info"),
                os.path.join(os.path.dirname(path), "ascend_toolkit_install.info"),
            ]
        )

    for candidate in candidates:
        if not os.path.exists(candidate):
            continue
        try:
            with open(candidate, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    stripped = line.strip()
                    lower = stripped.lower()
                    if "version" in lower and "=" in stripped:
                        return stripped
                    if lower.startswith("version"):
                        return stripped
        except OSError:
            continue
    return "<unknown>"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only check dependencies required to build the Python-only wheel.",
    )
    parser.add_argument(
        "--legacy-extension",
        action="store_true",
        help="Also check torch, torch_npu, torchnpugen and triton-ascend for legacy extension builds.",
    )
    parser.add_argument(
        "--skip-torchnpugen",
        action="store_true",
        help=(
            "Skip torchnpugen import checks for internal maintenance builds. "
            "The PyTorch C++ extension capability check still runs."
        ),
    )
    args = parser.parse_args()

    failures: list[str] = []

    if sys.version_info >= MIN_PYTHON:
        _ok(f"python version: {sys.version.split()[0]}")
    else:
        _fail(
            failures,
            f"python>={MIN_PYTHON[0]}.{MIN_PYTHON[1]} is required, "
            f"got {sys.version_info.major}.{sys.version_info.minor}",
        )

    if shutil.which("bash"):
        _ok(f"bash: {shutil.which('bash')}")
    else:
        _fail(failures, "bash is required")

    ascend_home = os.getenv("ASCEND_HOME_PATH")
    ascend_opp = os.getenv("ASCEND_OPP_PATH")
    if ascend_home or ascend_opp:
        _ok(f"ASCEND_HOME_PATH={ascend_home or '<unset>'}")
        _ok(f"ASCEND_OPP_PATH={ascend_opp or '<unset>'}")
        _ok(f"CANN version: {_detect_cann_version()}")
    else:
        _fail(failures, "ASCEND_HOME_PATH or ASCEND_OPP_PATH must be set")

    check_runtime = not args.build_only or args.legacy_extension
    if not check_runtime:
        _ok("skipping torch/torch_npu/triton checks for Python-only wheel build")
        torch = None
        torch_npu = None
    else:
        torch = _import_module(failures, "torch")
        torch_npu = _import_module(failures, "torch_npu")

    if torch is not None:
        torch_version = getattr(torch, "__version__", "<unknown>")
        _ok(f"torch version: {torch_version}")
        _check_min_version(failures, "torch", torch_version, MIN_TORCH)
        if hasattr(torch, "npu"):
            try:
                npu_available = bool(torch.npu.is_available())
            except Exception as exc:
                npu_available = False
                _warn(f"torch.npu.is_available() raised: {exc}")
            if npu_available:
                _ok("torch.npu.is_available(): True")
            elif args.build_only:
                _warn("torch.npu.is_available(): False")
            else:
                _fail(failures, "torch.npu.is_available() is False")
        else:
            _fail(failures, "torch.npu is missing")

    if torch_npu is not None:
        torch_npu_version = getattr(torch_npu, "__version__", "<unknown>")
        _ok(f"torch_npu version: {torch_npu_version}")
        if args.legacy_extension:
            stream_fix_error = torch_npu_gdn_stream_fix_error(torch_npu_version)
            if stream_fix_error:
                _fail(failures, stream_fix_error)
            else:
                _ok(
                    "torch_npu legacy GDN stream safety: "
                    f"{torch_npu_version}"
                )

    if args.legacy_extension:
        if args.skip_torchnpugen:
            _warn("skipping torchnpugen import checks")
        for probe in probe_legacy_build_capabilities(
            include_torchnpugen=not args.skip_torchnpugen
        ):
            if probe.available:
                _ok(f"{probe.requirement}: {probe.detail}")
            else:
                _fail(failures, f"{probe.requirement}: {probe.detail}")

    if check_runtime:
        triton = _import_module(failures, "triton")
        triton_ascend_version = _distribution_version("triton-ascend")
        if triton_ascend_version:
            _ok(f"triton-ascend version: {triton_ascend_version}")
            _check_min_version(failures, "triton-ascend", triton_ascend_version, MIN_TRITON_ASCEND)
            _check_triton_ascend_a5_compat(failures, triton_ascend_version)
        elif triton is not None:
            _fail(failures, "triton is importable, but triton-ascend distribution was not found")
        else:
            _fail(failures, "triton-ascend distribution was not found")

    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
