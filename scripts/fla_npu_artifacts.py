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


# PEP 600 watermark every published wheel claims.  It is the highest GLIBC_x.y
# any shared object in the payload needs, measured on the release build image;
# scripts/check_pypi_wheel.py fails the release if a wheel asks for more.
WHEEL_PLATFORM_TAG = "manylinux_2_34"


def get_wheel_platform_tag() -> str:
    """PEP 600 wheel platform tag for the current arch.

    Every wheel says ``manylinux_<glibc>_<arch>``; the plain ``linux_<arch>``
    tag bdist_wheel derives from sysconfig is rejected by PyPI and says nothing
    about the host a payload can load on.  A local build uses the same tag as a
    release, so the artifact under test is the artifact that ships.

    The glibc watermark is the measured one, not a wish.  The pinned build image
    (``ci/Dockerfile`` -> ``cann:9.1.0-*-ubuntu22.04``, glibc 2.35) stamps the
    launcher's ``dlopen``/``dlsym``/``dlerror`` at GLIBC_2.34, and the OPP host
    libraries land on 2.34 as well, so a ``manylinux_2_28`` label would promise
    hosts the payload cannot load on.  Lower it only together with a build image
    whose glibc is that old (and a re-measured gate run).
    """
    return f"{WHEEL_PLATFORM_TAG}_{get_arch()}"


def get_tier(soc: str | None = None) -> str:
    """Map FLA_NPU_SOC to the published product tier (a2/a3/a5)."""
    soc_tag = _compact_tag(soc or get_soc())
    if soc_tag in {"910b", "ascend910b"}:
        return "a2"
    if soc_tag in {"a3", "91093", "ascend91093"}:
        return "a3"
    if soc_tag in {"950", "ascend950"}:
        return "a5"
    raise ValueError(
        f"FLA_NPU_SOC={soc or get_soc()!r} has no published product tier; "
        "expected ascend910b / ascend910_93 / ascend950"
    )


def get_distribution_name() -> str:
    """Distribution (PyPI project) name for the current build.

    A wheel carries a prebuilt OPP for exactly one SoC, and pip cannot pick a
    chip-specific payload out of one project name, so each product tier is its
    own project (flash-linear-attention-npu-a2/a3/a5, derived from FLA_NPU_SOC).
    Local, GitHub Release and PyPI artifacts all carry that name.

    One name everywhere is the point: a locally built wheel and the published
    one are the same distribution, so ``pip install`` upgrades one with the
    other.  Two names for one payload would let both stay installed, each
    owning ``fla_npu/``, and uninstalling either would leave the other behind.
    ``FLA_NPU_SOC`` must therefore map to a published tier -- an unknown SoC
    fails the build instead of silently producing an unnameable artifact.
    """
    return f"{PACKAGE_NAME}-{get_tier()}"


def get_wheel_dist_name() -> str:
    return get_distribution_name().replace("-", "_")


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


def get_wheel_build_tag() -> str:
    """Optional filename build tag; empty for every ordinary build.

    The tier is in the distribution name and the arch in the platform tag, so
    there is nothing left for a build tag to say: a local build now produces the
    same file name a release does.  ``FLA_NPU_WHEEL_BUILD_TAG`` stays as the
    explicit escape hatch for a deliberately labelled artifact (publishing
    rejects it -- see scripts/check_pypi_wheel.py).
    """
    explicit = os.getenv("FLA_NPU_WHEEL_BUILD_TAG", "").strip()
    if not explicit:
        return ""
    build_tag = _wheel_tag_part(explicit)
    if build_tag and not build_tag[0].isdigit():
        return f"1{build_tag}"
    return build_tag


def get_daily_version_label(repo_root: Path) -> str:
    """Branch-derived prefix of a daily build's local version.

    A daily wheel is "this branch as of this commit", so the label names the
    branch: ``main`` on the development line, the released version on a release
    line (``v26.9.1`` -> ``26.9.1``), which is the one that could otherwise be
    confused with the release built from the same tree.
    """

    branch = get_branch_name(repo_root)
    label = re.sub(r"^v(?=\d)", "", branch)
    return _normalize_local_version(label) or "unknown"


# A development line writes ``<next>.dev0`` in fla/__init__.py, and a daily build
# of that line names the release it leads to: the local part is the only thing
# that says "not released yet", so ``26.10.0+main_dev0a1b2c3`` rather than
# ``26.10.0.dev0+main_dev0a1b2c3``.
_DEV_NUMBER_SUFFIX = re.compile(r"\.dev\d+$")


def daily_base_version(public_version: str) -> str:
    """Public version a daily build of *public_version* is labelled with."""

    return _DEV_NUMBER_SUFFIX.sub("", public_version)


def get_local_version(repo_root: Path, public_version: str | None = None) -> str:
    """Local version that marks a build as a daily (non-release) build.

    ``FLA_NPU_DISABLE_LOCAL_VERSION=TRUE`` is the release switch: the wheel then
    carries the bare released version from ``fla/__init__.py`` and nothing else,
    which is what ``check_pypi_wheel.py`` demands of an upload to the real index
    (its default is the exact released version).  Every other build is labelled with where it came
    from -- ``main_dev0a1b2c3`` on the development line, a bare ``dev0a1b2c3`` on
    a release line -- so a daily wheel is never mistaken for the release of the
    same version, and it still sorts above that release (a local version outranks
    the same version without one).

    A release line already names itself in the public part, so repeating it in
    the local part would only add noise: ``26.9.1+dev0a1b2c3``, not
    ``26.9.1+26.9.1.dev0a1b2c3``.
    The public part of a daily build is ``daily_base_version``, so the
    development line reads ``26.10.0+main_dev0a1b2c3`` too.
    """

    explicit = os.getenv("FLA_NPU_LOCAL_VERSION", "").strip()
    if explicit:
        return _normalize_local_version(explicit)
    if env_flag("FLA_NPU_DISABLE_LOCAL_VERSION"):
        return ""

    version = public_version or read_public_version(repo_root)
    label = get_daily_version_label(repo_root)
    if label == version:
        label = ""
    commit_id = get_commit_id(repo_root)
    if not commit_id:
        return label or "dev"
    # PEP 440 local versions are dot-separated alphanumerics: ``main_dev0a1b2c3``
    # is normalised to ``main.dev0a1b2c3``, and pip compares the normalised form.
    return f"{label}.dev{commit_id}" if label else f"dev{commit_id}"


def get_package_version(repo_root: Path) -> str:
    public_version = read_public_version(repo_root)
    local_version = get_local_version(repo_root, public_version)
    if not local_version:
        return public_version
    # A daily build of a development line names the release it leads to, not the
    # ``<next>.dev0`` the tree carries while that release is still unwritten.
    return f"{daily_base_version(public_version)}+{local_version}"


def get_wheel_filename(repo_root: Path) -> str:
    public_version = read_public_version(repo_root)
    package_version = get_package_version(repo_root)
    build_tag = get_wheel_build_tag()
    platform_tag = get_wheel_platform_tag()
    dist_name = get_wheel_dist_name()
    # The wheel is not pure Python (it carries a host launcher and the OPP) but
    # it is not CPython-versioned either, so only the platform tag is filled in.
    if build_tag:
        return (f"{dist_name}-{package_version}-{build_tag}-"
                f"py3-none-{platform_tag}.whl")
    return f"{dist_name}-{package_version}-py3-none-{platform_tag}.whl"


def get_platform_name() -> str:
    override = os.getenv("FLA_NPU_PLATFORM_NAME", "").strip()
    if override:
        return override
    if sys.platform.startswith("linux"):
        # Normalize through get_arch(): platform.machine() reports arm64/AMD64
        # on some hosts, which would not match the wheel tags pip looks for.
        return f"linux_{get_arch()}"
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
            "tier",
            "distribution-name",
            "wheel-dist-name",
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
        value = get_wheel_build_tag()
    elif args.field == "tier":
        value = get_tier()
    elif args.field == "distribution-name":
        value = get_distribution_name()
    elif args.field == "wheel-dist-name":
        value = get_wheel_dist_name()
    elif args.field == "wheel-filename":
        value = get_wheel_filename(repo_root)
    else:
        value = get_run_filename(repo_root)
    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
