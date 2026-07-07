"""Helpers for flash-linear-attention-npu package version names."""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
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


def _run_git(repo_root: Path, args: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            encoding="utf-8",
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def _normalize_branch_name(value: str) -> str:
    branch = value.strip()
    if branch in {"main", "origin/main", "master", "origin/master"}:
        return "main"
    if branch.startswith("refs/heads/"):
        branch = branch[len("refs/heads/") :]
    if branch.startswith("origin/"):
        branch = branch[len("origin/") :]
    return branch


def get_branch_name(repo_root: Path) -> str:
    for env_name in ("FLA_NPU_BRANCH_NAME", "GITHUB_BASE_REF", "CI_MERGE_REQUEST_TARGET_BRANCH_NAME"):
        branch = os.getenv(env_name, "").strip()
        if branch:
            return _normalize_branch_name(branch)

    current = _run_git(repo_root, ["rev-parse", "--abbrev-ref", "HEAD"])
    current = "" if current == "HEAD" else current
    normalized_current = _normalize_branch_name(current)
    if normalized_current == "main" or re.match(r"^v\d", normalized_current):
        return normalized_current

    remotes = _run_git(repo_root, ["branch", "-r", "--points-at", "HEAD", "--format", "%(refname:short)"]).splitlines()
    normalized_remotes = [_normalize_branch_name(branch) for branch in remotes]
    for branch in normalized_remotes:
        if branch == "main":
            return branch
    for branch in normalized_remotes:
        if re.match(r"^v\d", branch):
            return branch

    for env_name in ("GITHUB_REF_NAME", "CI_COMMIT_REF_NAME"):
        branch = os.getenv(env_name, "").strip()
        if branch:
            return _normalize_branch_name(branch)

    return normalized_current or "unknown"


def get_arch() -> str:
    arch = os.getenv("FLA_NPU_ARCH", "").strip().lower() or platform.machine().lower()
    if arch in {"amd64", "x86", "x86_64", "x64"}:
        return "x86_64"
    if arch in {"aarch64", "arm", "arm64"}:
        return "aarch64"
    return _compact_tag(arch) or "unknown"


def get_vendor_name() -> str:
    return DEFAULT_VENDOR_NAME


def _compact_tag(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", value).lower()


def _wheel_tag_part(value: str) -> str:
    return ".".join(re.findall(r"[A-Za-z0-9_]+", value))


def _normalize_local_version(value: str) -> str:
    parts = re.findall(r"[A-Za-z0-9]+", value.lower())
    return ".".join(parts)


def get_commit_id(repo_root: Path) -> str:
    for env_name in ("FLA_NPU_COMMIT_ID", "GITHUB_HEAD_SHA", "GITHUB_SHA", "CI_COMMIT_SHA"):
        commit = os.getenv(env_name, "").strip()
        if commit:
            normalized = _normalize_local_version(commit)
            return normalized[:7] if normalized else ""

    commit = _run_git(repo_root, ["rev-parse", "--short=7", "HEAD"])
    return _normalize_local_version(commit)


def get_product_tag() -> str:
    soc_tag = _compact_tag(get_soc())
    if soc_tag in {"910b", "ascend910b"}:
        return "910b"
    if soc_tag in {"a3", "91093", "ascend91093"}:
        return "910_93"
    if soc_tag in {"950", "ascend950"}:
        return "950"
    return soc_tag or "unknown"


def get_wheel_build_tag(repo_root: Path, public_version: str | None = None) -> str:
    explicit = os.getenv("FLA_NPU_WHEEL_BUILD_TAG", "").strip()
    if explicit:
        build_tag = _wheel_tag_part(explicit)
        if build_tag and not build_tag[0].isdigit():
            return f"1{build_tag}"
        return build_tag
    if env_flag("FLA_NPU_DISABLE_LOCAL_VERSION"):
        return ""

    public_version = public_version or read_public_version(repo_root)
    product_tag = get_product_tag()
    arch_tag = get_arch()
    if not product_tag or not arch_tag:
        return ""

    build_tag = ".".join([product_tag, arch_tag])
    if build_tag and not build_tag[0].isdigit():
        build_tag = f"1{build_tag}"
    return build_tag


def get_local_version(repo_root: Path, public_version: str | None = None) -> str:
    explicit = os.getenv("FLA_NPU_LOCAL_VERSION", "").strip()
    if explicit:
        return _normalize_local_version(explicit)
    if env_flag("FLA_NPU_DISABLE_LOCAL_VERSION"):
        return ""

    if get_branch_name(repo_root) != "main":
        return ""

    commit_id = get_commit_id(repo_root)
    return f"main.{commit_id}" if commit_id else "main"


def get_package_version(repo_root: Path) -> str:
    public_version = read_public_version(repo_root)
    local_version = get_local_version(repo_root, public_version)
    if not local_version:
        return public_version
    return f"{public_version}+{local_version}"


def get_wheel_filename(repo_root: Path) -> str:
    public_version = read_public_version(repo_root)
    package_version = get_package_version(repo_root)
    build_tag = get_wheel_build_tag(repo_root, public_version)
    if build_tag:
        return f"{WHEEL_DIST_NAME}-{package_version}-{build_tag}-py3-none-any.whl"
    return f"{WHEEL_DIST_NAME}-{package_version}-py3-none-any.whl"


def get_platform_name() -> str:
    override = os.getenv("FLA_NPU_PLATFORM_NAME", "").strip()
    if override:
        return override
    if sys.platform.startswith("linux"):
        return f"linux_{platform.uname().machine}"
    raise RuntimeError(f"Unsupported platform for run package build: {sys.platform}")


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
            "commit-id",
            "wheel-build-tag",
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
        value = get_local_version(repo_root)
    elif args.field == "package-version":
        value = get_package_version(repo_root)
    elif args.field == "commit-id":
        value = get_commit_id(repo_root)
    elif args.field == "wheel-build-tag":
        value = get_wheel_build_tag(repo_root)
    elif args.field == "wheel-filename":
        value = get_wheel_filename(repo_root)
    else:
        value = get_run_filename(repo_root)
    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
