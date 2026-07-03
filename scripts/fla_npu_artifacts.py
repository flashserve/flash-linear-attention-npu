"""Helpers for flash-linear-attention-npu package version names."""

from __future__ import annotations

import argparse
import os
import platform
import re
import sys
from pathlib import Path


PACKAGE_NAME = "flash-linear-attention-npu"
WHEEL_DIST_NAME = PACKAGE_NAME.replace("-", "_")
DEFAULT_SOC = "ascend910b"
DEFAULT_VENDOR_NAME = "fla_npu"


def env_flag(name: str) -> bool:
    return os.getenv(name, "FALSE").upper() in {"1", "TRUE", "YES", "ON"}


def read_public_version(repo_root: Path) -> str:
    init_py = repo_root / "fla" / "__init__.py"
    match = re.search(
        r'^__version__\s*=\s*[\'"]([^\'"]+)[\'"]',
        init_py.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not match:
        raise RuntimeError("Unable to find __version__ in fla/__init__.py")
    return match.group(1)


def get_soc() -> str:
    return os.getenv("FLA_NPU_SOC", DEFAULT_SOC)


def get_vendor_name() -> str:
    return DEFAULT_VENDOR_NAME


def _compact_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value).lower()


def _normalize_local_version(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value.lower())
    return ".".join(parts)


def _get_torch_version() -> str:
    override = os.getenv("FLA_NPU_TORCH_VERSION", "").strip()
    if override:
        return override
    try:
        import torch
    except Exception:
        return ""
    return torch.__version__.split("+", 1)[0]


def _get_cxx11_abi() -> str:
    override = os.getenv("FLA_NPU_CXX11_ABI", "").strip()
    if override:
        return override.lower()
    try:
        import torch
    except Exception:
        return ""
    return str(bool(torch._C._GLIBCXX_USE_CXX11_ABI)).lower()


def get_local_version() -> str:
    explicit = os.getenv("FLA_NPU_LOCAL_VERSION", "").strip()
    if explicit:
        return _normalize_local_version(explicit)
    if env_flag("FLA_NPU_DISABLE_LOCAL_VERSION"):
        return ""

    torch_version = _get_torch_version()
    if not torch_version:
        return ""

    soc_tag = _compact_tag(get_soc())
    cxx11_abi = _get_cxx11_abi() or "unknown"
    return f"soc{soc_tag}torch{torch_version}cxx11abi{cxx11_abi}"


def get_package_version(repo_root: Path) -> str:
    public_version = read_public_version(repo_root)
    local_version = get_local_version()
    if not local_version:
        return public_version
    return f"{public_version}+{local_version}"


def get_python_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def get_platform_name() -> str:
    override = os.getenv("FLA_NPU_PLATFORM_NAME", "").strip()
    if override:
        return override
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"
    raise RuntimeError(f"Unsupported platform for wheel build: {sys.platform}")


def get_wheel_filename(repo_root: Path) -> str:
    python_tag = get_python_tag()
    platform_name = get_platform_name()
    return (
        f"{WHEEL_DIST_NAME}-{get_package_version(repo_root)}-"
        f"{python_tag}-{python_tag}-{platform_name}.whl"
    )


def get_run_filename(repo_root: Path) -> str:
    public_version = read_public_version(repo_root)
    soc_tag = _compact_tag(get_soc())
    vendor_name = re.sub(r"[^A-Za-z0-9_.]+", "_", get_vendor_name())
    platform_name = get_platform_name()
    return f"fla-npu-{public_version}+soc{soc_tag}-{vendor_name}-{platform_name}.run"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "field",
        choices=[
            "public-version",
            "local-version",
            "package-version",
            "wheel-filename",
            "run-filename",
        ],
    )
    parser.add_argument("--repo-root", default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    if args.field == "public-version":
        value = read_public_version(repo_root)
    elif args.field == "local-version":
        value = get_local_version()
    elif args.field == "package-version":
        value = get_package_version(repo_root)
    elif args.field == "wheel-filename":
        value = get_wheel_filename(repo_root)
    else:
        value = get_run_filename(repo_root)
    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
