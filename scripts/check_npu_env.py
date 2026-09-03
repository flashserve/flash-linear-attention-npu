"""Preflight checks for building and running flash-linear-attention-npu."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from importlib import metadata
from typing import Optional, Tuple

from packaging.version import InvalidVersion, Version


MIN_PYTHON = (3, 9)
# Runtime/version tables are shared with the PyPI import guard via
# scripts/npu_compat.py (single source of truth).
from npu_compat import (  # noqa: E402
    MIN_CANN,
    MIN_TORCH,
    MIN_TRITON_ASCEND,
    MIN_TRITON_ASCEND_A5,
    MIN_TRITON_ASCEND_CANN9,
    MIN_TORCH_NPU_FUTURE_FIX_FAMILY,
    TORCH_NPU_GDN_FIX_MINIMUMS,
    VALIDATED_COMBOS,
)

# Toolchain and build dependencies checked in addition to the torch-related
# checks. Values mirror CMakeLists.txt (cmake_minimum_required),
# install_deps.sh (gcc) and pyproject.toml ([build-system] requires).
MIN_CMAKE = "3.16"
MIN_GCC = "7.3"
MIN_SETUPTOOLS = "70.1"
TORCH_NPU_GDN_FIX_RELEASE_URL = (
    "https://gitcode.com/Ascend/pytorch/releases?"
    "presetConfig={%22tags%22:229,%22release%22:122}"
)

TORCHNPUGEN_MODULES = (
    "torchnpugen.gen_op_plugin_functions",
    "torchnpugen.gen_derivatives",
    "torchnpugen.gen_op_backend",
    "torchnpugen.gen_backend_stubs",
    "torchnpugen.struct.gen_struct_opapi",
)


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


def _find_module(failures: list[str], module_name: str) -> None:
    try:
        spec = importlib.util.find_spec(module_name)
    except Exception as exc:
        _fail(failures, f"{module_name}: {exc}")
        return
    if spec is None:
        _fail(failures, f"{module_name}: not found")
        return
    _ok(f"{module_name}: {spec.origin or 'namespace package'}")


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


def _version_key(value: str) -> Optional[Tuple[int, int, int]]:
    parts = re.findall(r"\d+", value.split("+", 1)[0])
    if not parts:
        return None
    nums = [int(part) for part in parts[:3]]
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def _check_torch_npu_gdn_fix(failures: list[str], actual: str) -> None:
    actual_version = _version_obj(actual)
    if actual_version is None:
        _fail(failures, f"torch_npu has unsupported version string: {actual}")
        return

    minimum = TORCH_NPU_GDN_FIX_MINIMUMS.get(actual_version.base_version)
    if minimum and actual_version >= Version(minimum):
        return

    if actual_version >= Version(MIN_TORCH_NPU_FUTURE_FIX_FAMILY):
        return

    if minimum is None:
        return

    requirements = ", ".join(
        f"{family}>={minimum}" for family, minimum in TORCH_NPU_GDN_FIX_MINIMUMS.items()
    )
    _fail(
        failures,
        "torch_npu must come from an Ascend PyTorch release that contains the "
        "GDN aclnn_extension stream fix. Packages from releases before "
        "v26.1.0-beta.1, such as v26.0.0-pytorch2.x, are rejected. "
        f"Expected one of: {requirements}, or torch_npu>={MIN_TORCH_NPU_FUTURE_FIX_FAMILY} "
        f"from {TORCH_NPU_GDN_FIX_RELEASE_URL}; got {actual}.",
    )


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


def _check_triton_ascend_cann_compat(failures: list[str], actual: str) -> None:
    """CANN 9.x needs triton-ascend >= 3.2.1.

    3.2.0 is the generic lower bound, but on CANN 9.0.0+ the Ascend Triton
    backend JIT-compiles npu_utils.cpp against newer rt.h headers and fails
    (e.g. RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE), so raise the floor for CANN 9.x.
    """
    cann = _detect_cann_version()
    cann_match = re.search(r"(\d+\.\d+)", cann)
    if not cann_match:
        return
    if int(cann_match.group(1).split(".")[0]) < 9:
        return
    actual_version = _version_obj(actual)
    if actual_version is None or actual_version < Version(MIN_TRITON_ASCEND_CANN9):
        _fail(
            failures,
            f"triton-ascend>={MIN_TRITON_ASCEND_CANN9} is required for CANN {cann_match.group(1)} "
            f"(detected); got {actual}. triton-ascend 3.2.0 fails to JIT-compile "
            "npu_utils.cpp on CANN 9.x.",
        )


def _check_cmake_version(failures: list[str]) -> None:
    """Check the CMake version against the project's minimum requirement.

    The lower bound comes from ``cmake_minimum_required(VERSION 3.16)`` in
    ``CMakeLists.txt`` (mirrored by ``install_deps.sh``), so any version
    >= MIN_CMAKE is supported.
    """
    cmake = shutil.which("cmake")
    if cmake is None:
        _fail(failures, f"cmake not found (cmake>={MIN_CMAKE} is required)")
        return
    try:
        output = subprocess.run(
            [cmake, "--version"], capture_output=True, text=True, timeout=10
        ).stdout
    except OSError as exc:
        _fail(failures, f"cmake: cannot run {cmake}: {exc}")
        return
    match = re.search(r"cmake version ([\d.]+)", output)
    if not match:
        _fail(failures, f"cmake: cannot parse version from: {output.strip()!r}")
        return
    actual = match.group(1)
    _ok(f"cmake: {cmake} (version {actual})")
    _check_min_version(failures, "cmake", actual, MIN_CMAKE)


def _check_setuptools_version(failures: list[str]) -> None:
    """Check the setuptools version against the build-system requirement.

    ``pyproject.toml`` declares ``setuptools>=70.1`` because setuptools only
    ships ``setuptools.command.bdist_wheel`` (used by setup.py as a fallback)
    from 70.x onward, so any version >= MIN_SETUPTOOLS is supported.
    """
    try:
        actual = metadata.version("setuptools")
    except Exception as exc:
        _fail(failures, f"setuptools: cannot determine version: {exc}")
        return
    _ok(f"setuptools version: {actual}")
    _check_min_version(failures, "setuptools", actual, MIN_SETUPTOOLS)


def _check_build_system_deps(failures: list[str]) -> None:
    """Check the other pyproject build-system dependencies are importable.

    The README build flow runs ``pip wheel --no-build-isolation``, so the
    build-system packages declared in ``pyproject.toml`` (wheel, packaging,
    psutil) must already be installed in the current interpreter. setuptools is
    checked separately with a version bound; the remaining three have no
    minimum version.
    """
    for module_name in ("wheel", "packaging", "psutil"):
        try:
            module = importlib.import_module(module_name)
            _ok(
                f"{module_name} version: "
                f"{getattr(module, '__version__', '<unknown>')}"
            )
        except Exception as exc:
            _fail(
                failures,
                f"{module_name} not importable (pyproject build-system requires "
                f"it when using --no-build-isolation): {exc}",
            )


def _check_gcc_version(failures: list[str]) -> None:
    """Check the host C/C++ compiler version against install_deps.sh.

    ``install_deps.sh`` requires gcc/g++ >= 7.3.0; host-side C++ code in
    build.sh and the Ascend C host wrapper are compiled with it.
    """
    gcc = shutil.which("gcc")
    if gcc is None:
        _fail(failures, f"gcc not found (gcc>={MIN_GCC} is required)")
    else:
        _ok(f"gcc: {gcc}")
        actual = _tool_version(gcc, r"(\d+\.\d+\.\d+)")
        if actual:
            _check_min_version(failures, "gcc", actual, MIN_GCC)
        else:
            _fail(failures, "gcc: cannot parse version")

    gxx = shutil.which("g++")
    if gxx is None:
        _fail(failures, f"g++ not found (g++>={MIN_GCC} is required)")
    else:
        _ok(f"g++: {gxx}")
        actual = _tool_version(gxx, r"(\d+\.\d+\.\d+)")
        if actual:
            _check_min_version(failures, "g++", actual, MIN_GCC)
        else:
            _fail(failures, "g++: cannot parse version")


def _check_make_exists(failures: list[str]) -> None:
    """Check that a CMake build backend is available.

    The build uses CMake's default ``Unix Makefiles`` generator (ninja is an
    alternative, so either make or ninja is sufficient).
    """
    make = shutil.which("make")
    ninja = shutil.which("ninja")
    if make:
        _ok(f"make: {make}")
    elif ninja:
        _ok(f"make not found; ninja: {ninja}")
    else:
        _fail(failures, "make not found (nor ninja as a CMake build backend)")


def _check_patch_exists(failures: list[str]) -> None:
    """Check that the system ``patch`` command is available.

    The CMake third-party build applies source patches via
    ``PATCH_COMMAND patch -p1 < ...`` (abseil-cpp and ascend protobuf), so a
    missing ``patch`` only surfaces halfway through the build. It has no
    version requirement, so we only check presence.
    """
    patch = shutil.which("patch")
    if patch:
        _ok(f"patch: {patch}")
    else:
        _fail(
            failures,
            "patch not found. It is required by the CMake third-party build "
            "(abseil-cpp / ascend protobuf PATCH_COMMAND); install it with the "
            "system package manager (e.g. apt-get install -y patch).",
        )


def _check_bisheng_exists(failures: list[str]) -> None:
    """Check that the Ascend C kernel compiler (bisheng) is available.

    bisheng is shipped with the CANN toolkit and exported to PATH by the CANN
    ``setenv.bash``; build.sh fails if ``which bisheng`` is empty. Its
    ``--version`` reports a clang version, not a CANN component version, so we
    only check presence, not a minimum version.
    """
    bisheng = shutil.which("bisheng")
    if bisheng:
        _ok(f"bisheng: {bisheng}")
    else:
        _fail(
            failures,
            "bisheng not found. It is shipped with the CANN toolkit and should be "
            "on PATH after sourcing the CANN set_env.sh/setenv.bash.",
        )


def _tool_version(tool: str, pattern: str) -> Optional[str]:
    """Run ``tool --version`` and return the first regex match group, or None."""
    try:
        output = subprocess.run(
            [tool, "--version"], capture_output=True, text=True, timeout=10
        ).stdout
    except OSError:
        return None
    match = re.search(pattern, output)
    return match.group(1) if match else None


def _detect_cann_version() -> str:
    # The OPP install dir's version.info is the authoritative CANN version
    # (e.g. "Version=8.3.0.1.200" or "Version=9.1.0-beta.1"). ASCEND_HOME_PATH
    # can resolve to driver install files whose version is not the CANN
    # version (e.g. "version=25.5.1"), so it is only consulted after the OPP
    # dir, and driver-version-like values are skipped.
    opp = os.getenv("ASCEND_OPP_PATH")
    candidates: list[str] = []
    if opp:
        path = os.path.abspath(opp)
        candidates.extend(
            [
                os.path.join(path, "version.info"),
                os.path.join(os.path.dirname(path), "version.info"),
            ]
        )
    home = os.getenv("ASCEND_HOME_PATH")
    if home:
        path = os.path.abspath(home)
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
                lines = [line.strip() for line in file]
        except OSError:
            continue
        for line in lines:
            if "driver" in line.lower():
                continue
            key, _, value = line.partition("=")
            if key.strip().lower() == "version" and value.strip():
                return line
        for line in lines:
            if "driver" in line.lower():
                continue
            key, _, value = line.partition("=")
            if key.strip().lower() == "version_dir" and value.strip():
                return line
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
        help="Skip torchnpugen checks. Intended only for internal maintenance builds.",
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

    _check_cmake_version(failures)
    _check_gcc_version(failures)
    _check_make_exists(failures)
    _check_patch_exists(failures)
    _check_bisheng_exists(failures)
    _check_setuptools_version(failures)
    _check_build_system_deps(failures)

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
        _check_torch_npu_gdn_fix(failures, torch_npu_version)

    if args.legacy_extension and not args.skip_torchnpugen:
        for module_name in TORCHNPUGEN_MODULES:
            _find_module(failures, module_name)
    elif args.skip_torchnpugen:
        _warn("skipping torchnpugen checks")

    if check_runtime:
        triton = _import_module(failures, "triton")
        triton_ascend_version = _distribution_version("triton-ascend")
        if triton_ascend_version:
            _ok(f"triton-ascend version: {triton_ascend_version}")
            _check_min_version(failures, "triton-ascend", triton_ascend_version, MIN_TRITON_ASCEND)
            _check_triton_ascend_cann_compat(failures, triton_ascend_version)
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
