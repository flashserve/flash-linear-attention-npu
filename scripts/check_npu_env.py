"""Preflight checks for building and running flash-linear-attention-npu."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import re
import shutil
import sys
from importlib import metadata

from packaging.version import InvalidVersion, Version


MIN_PYTHON = (3, 9)
MIN_TORCH = "2.7.0"
MIN_TRITON_ASCEND = "3.2.0"
TORCH_NPU_GDN_FIX_MINIMUMS = {
    "2.7.1": "2.7.1.post5",
    "2.8.0": "2.8.0.post5",
    "2.9.0": "2.9.0.post3",
    "2.10.0": "2.10.0.post2",
    "2.11.0": "2.11.0rc3",
    "2.12.0": "2.12.0rc1",
}
MIN_TORCH_NPU_FUTURE_FIX_FAMILY = "2.13.0"
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


def _version_obj(value: str) -> Version | None:
    try:
        return Version(value.split("+", 1)[0])
    except InvalidVersion:
        return None


def _version_key(value: str) -> tuple[int, int, int] | None:
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
        help="Allow torch.npu.is_available() to be false, while still checking build dependencies.",
    )
    parser.add_argument(
        "--skip-torchnpugen",
        action="store_true",
        help="Skip torchnpugen checks. Intended only for FLA_NPU_SKIP_TORCH_GEN=TRUE.",
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

    if not args.skip_torchnpugen:
        for module_name in TORCHNPUGEN_MODULES:
            _find_module(failures, module_name)

    triton = _import_module(failures, "triton")
    triton_ascend_version = _distribution_version("triton-ascend")
    if triton is not None and triton_ascend_version:
        _ok(f"triton-ascend version: {triton_ascend_version}")
        _check_min_version(failures, "triton-ascend", triton_ascend_version, MIN_TRITON_ASCEND)
    elif triton is not None:
        _fail(failures, "triton is importable, but triton-ascend distribution was not found")

    if failures:
        print("\nEnvironment check failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
