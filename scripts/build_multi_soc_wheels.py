"""Build multiple FLA NPU wheels while reusing the torch_custom extension.

The AscendC OPP payload is rebuilt for every SoC. Only the torch_custom Python
extension is reused, and only after the first full wheel build produces it for
the current Python ABI, torch, torch_npu, source tree, and CPU architecture.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fla_npu_artifacts import get_arch, get_branch_name, get_commit_id  # noqa: E402


DEFAULT_SOCS = ("ascend910b", "ascend910_93", "ascend950")
IGNORED_SOURCE_PATTERNS = (
    ".git",
    ".ci-cache",
    ".history",
    "__pycache__",
    "*.pyc",
    "cmake-build-*",
)
IGNORED_SOURCE_PATHS = (
    "build",
    "build_out",
    "dist",
    "output",
    "torch_custom/fla_npu/build",
    "torch_custom/fla_npu/dist",
    "torch_custom/fla_npu/fla_npu.egg-info",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--soc",
        action="append",
        default=[],
        help=(
            "Target SoC. May be passed multiple times or as a comma-separated list. "
            "Default: ascend910b,ascend910_93,ascend950."
        ),
    )
    parser.add_argument("-w", "--wheel-dir", default="dist", help="Output wheel directory.")
    parser.add_argument(
        "--work-dir",
        default="build/multi_soc_wheels",
        help="Temporary isolated source/cache directory under build/. It is recreated by default.",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the temporary isolated source directories after the build.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Forward FLA_NPU_BUILD_JOBS to setup.py, which calls build.sh -jN.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used for pip wheel. Default: current interpreter.",
    )
    parser.add_argument(
        "--no-reuse-torch-custom",
        action="store_true",
        help="Build torch_custom from scratch for every SoC.",
    )
    parser.add_argument(
        "--reuse-torch-custom-so",
        default="",
        help="Use an existing custom_aclnn_extension_lib*.so instead of building the first one.",
    )
    return parser.parse_args()


def split_socs(values: list[str]) -> list[str]:
    socs: list[str] = []
    for value in values:
        socs.extend(part.strip() for part in value.split(",") if part.strip())
    return socs or list(DEFAULT_SOCS)


def canonical_soc_dir(soc: str) -> str:
    compact = re.sub(r"[^A-Za-z0-9]+", "", soc).lower()
    if compact in {"910b", "ascend910b"}:
        return "ascend910b"
    if compact in {"a3", "91093", "ascend91093"}:
        return "ascend910_93"
    if compact in {"950", "ascend950"}:
        return "ascend950"
    return soc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    printable = " ".join(cmd)
    print(f"[multi-soc wheel] START {printable}", flush=True)
    start = time.monotonic()
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)
    print(f"[multi-soc wheel] DONE in {time.monotonic() - start:.1f}s: {printable}", flush=True)


def output(cmd: list[str], cwd: Path, env: dict[str, str]) -> str:
    return subprocess.check_output(cmd, cwd=str(cwd), env=env, encoding="utf-8").strip()


def python_extension_suffix(python: str, cwd: Path, env: dict[str, str]) -> str:
    code = "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX') or '.so')"
    return output([python, "-c", code], cwd, env)


def validate_reuse_so(path: Path, python: str, cwd: Path, env: dict[str, str]) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f"--reuse-torch-custom-so does not point to a file: {resolved}")
    if not resolved.name.startswith("custom_aclnn_extension_lib"):
        raise RuntimeError(
            "--reuse-torch-custom-so must point to custom_aclnn_extension_lib*.so, "
            f"got {resolved.name}"
        )
    expected_suffix = python_extension_suffix(python, cwd, env)
    if not resolved.name.endswith(expected_suffix):
        raise RuntimeError(
            "--reuse-torch-custom-so Python ABI or architecture does not match this build. "
            f"Expected suffix {expected_suffix!r}, got {resolved.name!r}."
        )
    if resolved.stat().st_size <= 0:
        raise RuntimeError(f"--reuse-torch-custom-so is empty: {resolved}")
    return resolved


def safe_rmtree(path: Path, repo_root: Path) -> None:
    resolved = path.resolve()
    build_root = (repo_root / "build").resolve()
    if resolved == build_root or build_root not in resolved.parents:
        raise RuntimeError(f"Refusing to remove work dir outside {build_root}: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def copy_source(repo_root: Path, dst: Path) -> None:
    root = repo_root.resolve()
    ignored_paths = set(IGNORED_SOURCE_PATHS)

    def ignore(src: str, names: list[str]) -> set[str]:
        src_path = Path(src).resolve()
        try:
            rel_dir = src_path.relative_to(root)
        except ValueError:
            rel_dir = Path()

        ignored: set[str] = set()
        for name in names:
            rel_path = name if rel_dir == Path(".") else (rel_dir / name).as_posix()
            if rel_path in ignored_paths:
                ignored.add(name)
                continue
            if any(fnmatch.fnmatch(name, pattern) for pattern in IGNORED_SOURCE_PATTERNS):
                ignored.add(name)
        return ignored

    shutil.copytree(repo_root, dst, ignore=ignore)


def build_env(repo_root: Path, soc: str, args: argparse.Namespace, reuse_so: Path | None) -> dict[str, str]:
    env = os.environ.copy()
    env["FLA_NPU_SOC"] = soc
    env.setdefault("FLA_NPU_ARCH", get_arch())
    env.setdefault("FLA_NPU_BRANCH_NAME", get_branch_name(repo_root))
    env.setdefault("FLA_NPU_COMMIT_ID", get_commit_id(repo_root))
    if args.jobs is not None:
        if args.jobs <= 0:
            raise RuntimeError(f"--jobs must be positive, got {args.jobs}")
        env["FLA_NPU_BUILD_JOBS"] = str(args.jobs)
    if reuse_so:
        # The reused extension is a host-side torch_custom adapter. It is only
        # safe within one Python ABI / torch / torch_npu / C++ ABI / CPU arch
        # combination; each wheel still rebuilds and embeds its own SoC OPP.
        env["FLA_NPU_REUSE_TORCH_CUSTOM_SO"] = str(reuse_so.resolve())
    else:
        env.pop("FLA_NPU_REUSE_TORCH_CUSTOM_SO", None)
    return env


def expected_wheel_name(src: Path, python: str, env: dict[str, str]) -> str:
    return output([python, "scripts/fla_npu_artifacts.py", "wheel-filename"], src, env)


def find_single_torch_custom_so(src: Path) -> Path:
    package_dir = src / "torch_custom" / "fla_npu" / "fla_npu"
    so_files = sorted(package_dir.glob("custom_aclnn_extension_lib*.so"))
    if len(so_files) != 1:
        raise RuntimeError(f"Expected one torch_custom extension under {package_dir}, got {len(so_files)}")
    return so_files[0]


def inspect_wheel(wheel: Path, expected_soc_dir: str, expected_so_hash: str | None) -> str:
    with zipfile.ZipFile(wheel) as zf:
        names = zf.namelist()
        so_names = [
            name
            for name in names
            if name.endswith(".so") and "custom_aclnn_extension_lib" in name
        ]
        if len(so_names) != 1:
            raise RuntimeError(f"{wheel.name}: expected one torch_custom extension, got {len(so_names)}")
        so_hash = hashlib.sha256(zf.read(so_names[0])).hexdigest()
        if expected_so_hash and so_hash != expected_so_hash:
            raise RuntimeError(
                f"{wheel.name}: reused torch_custom hash mismatch, "
                f"expected {expected_so_hash}, got {so_hash}"
            )

        joined_names = "\n".join(names)
        soc_dirs = sorted(set(re.findall(r"op_impl/ai_core/tbe/kernel/config/([^/]+)/", joined_names)))
        if soc_dirs != [expected_soc_dir]:
            raise RuntimeError(
                f"{wheel.name}: expected only SoC config dir {expected_soc_dir}, got {soc_dirs}"
            )

        opapi_libs = [name for name in names if name.endswith("op_api/lib/libcust_opapi.so")]
        if len(opapi_libs) != 1:
            raise RuntimeError(f"{wheel.name}: expected one embedded libcust_opapi.so, got {len(opapi_libs)}")

    print(
        f"[multi-soc wheel][OK] {wheel.name}: "
        f"torch_custom_sha256={so_hash}, soc_config={expected_soc_dir}, libcust_opapi=1"
    )
    return so_hash


def build_one(
    repo_root: Path,
    work_dir: Path,
    wheel_dir: Path,
    cache_dir: Path,
    soc: str,
    args: argparse.Namespace,
    reuse_so: Path | None,
    expected_so_hash: str | None,
) -> tuple[Path, Path, str, int]:
    src = work_dir / f"src_{canonical_soc_dir(soc)}"
    if src.exists():
        shutil.rmtree(src)
    copy_source(repo_root, src)

    env = build_env(repo_root, soc, args, reuse_so)
    expected_name = expected_wheel_name(src, args.python, env)
    wheel_path = wheel_dir / expected_name
    if wheel_path.exists():
        wheel_path.unlink()

    start = time.monotonic()
    run(
        [args.python, "-m", "pip", "wheel", ".", "-w", str(wheel_dir), "--no-build-isolation", "--no-deps"],
        src,
        env,
    )
    elapsed = int(time.monotonic() - start)
    if not wheel_path.is_file():
        raise RuntimeError(f"Expected wheel was not produced: {wheel_path}")

    built_so = find_single_torch_custom_so(src)
    if reuse_so is None and not args.no_reuse_torch_custom:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_so = cache_dir / built_so.name
        shutil.copy2(built_so, cached_so)
    else:
        cached_so = reuse_so or built_so

    expected_hash_for_wheel = None if args.no_reuse_torch_custom else expected_so_hash
    so_hash = inspect_wheel(wheel_path, canonical_soc_dir(soc), expected_hash_for_wheel)
    if expected_so_hash is None and not args.no_reuse_torch_custom:
        expected_so_hash = so_hash

    return wheel_path, cached_so, expected_so_hash, elapsed


def main() -> int:
    args = parse_args()
    if args.no_reuse_torch_custom and args.reuse_torch_custom_so:
        raise RuntimeError("--no-reuse-torch-custom cannot be used with --reuse-torch-custom-so")

    repo_root = Path(__file__).resolve().parents[1]
    wheel_dir = (repo_root / args.wheel_dir).resolve()
    work_dir = (repo_root / args.work_dir).resolve()
    cache_dir = work_dir / "torch_custom_cache"
    socs = split_socs(args.soc)

    wheel_dir.mkdir(parents=True, exist_ok=True)
    if not args.keep_work_dir:
        safe_rmtree(work_dir, repo_root)
    work_dir.mkdir(parents=True, exist_ok=True)

    reuse_so = (
        validate_reuse_so(Path(args.reuse_torch_custom_so), args.python, repo_root, os.environ.copy())
        if args.reuse_torch_custom_so
        else None
    )
    expected_so_hash = sha256_file(reuse_so) if reuse_so else None
    produced: list[tuple[str, Path, int]] = []

    try:
        for index, soc in enumerate(socs):
            # First build produces the torch_custom extension from source; later
            # SoCs reuse that exact .so while setup.py still rebuilds the SoC OPP
            # run package. Use --no-reuse-torch-custom for compatibility probes.
            current_reuse = None if (args.no_reuse_torch_custom or (index == 0 and reuse_so is None)) else reuse_so
            wheel_path, cached_so, expected_so_hash, elapsed = build_one(
                repo_root,
                work_dir,
                wheel_dir,
                cache_dir,
                soc,
                args,
                current_reuse,
                expected_so_hash,
            )
            produced.append((soc, wheel_path, elapsed))
            if reuse_so is None and not args.no_reuse_torch_custom:
                reuse_so = cached_so
                expected_so_hash = sha256_file(reuse_so)
                print(f"[multi-soc wheel] Cached torch_custom extension at {reuse_so}")
    finally:
        if not args.keep_work_dir:
            safe_rmtree(work_dir, repo_root)

    print("[multi-soc wheel] Produced wheels:")
    for soc, wheel_path, elapsed in produced:
        print(f"  {soc}: {wheel_path.name} ({elapsed}s, sha256={sha256_file(wheel_path)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
