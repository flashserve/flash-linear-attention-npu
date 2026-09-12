#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""为打包进 wheel 准备一份自包含的离线 third-party bundle。

wheel 内已经携带预编译 OPP，用户 ``pip install`` 后即可离线使用。需要从匹配源码
(重) 编译 wheel 的开发者，可以先把这份离线 bundle 还原到源码树的 ``third_party/``
（即 ``CANN_3RD_LIB_PATH``），从而在完全离线（不访问 gitcode/gitee）的情况下编译。

bundle 只保留 (重) 编译所需的最小源码子集：

- 纯头文件库（json / eigen / catlass）只带 include 目录；
- makeself 只带其少量必需文件（由 ``makeself.sh`` / ``makeself-header.sh`` 探测）；
- abseil / protobuf 通过 ``ExternalProject`` 从 ``pkg/`` 源码归档构建（而不是预置
  的源码目录），所以 bundle 只保留它们的 ``pkg/*.tar.gz`` 归档，由 CMake 在
  configure/build 时解压并 patch 进源码树；
- opbase 携带完整源码（直接参与编译），并裁剪掉非构建目录。

目录布局与 CANN_3RD_LIB_PATH 探测一致：

    pkg/abseil-cpp-20230802.1.tar.gz   pkg/protobuf-25.1.tar.gz   pkg/include.zip
    pkg/eigen-5.0.0.tar.gz            pkg/googletest-1.14.0.tar.gz
    pkg/makeself-release-2.5.0-patch1.tar.gz
    json/include/    eigen/Eigen/    makeself/    opbase/    catlass/include/

bundle 被定义为“自包含”：``_COMPONENTS`` / ``_ARCHIVES`` 中列出的每个组件和归档
都是离线 (重) 编译所必需的。任一缺失都会在产出不完整 bundle 之前以非零退出，并且
生成完成后会按清单复核 bundle，成功前不会放行不完整产物，避免把缺件的离线 bundle
打进 wheel。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


# (name, source sub-path in cache, target sub-path inside bundle, prune list)
#   prune: top-level entries removed from the copied target dir to keep it minimal.
# abseil / protobuf sources are NOT copied here: ExternalProject extracts them from
#   pkg/*.tar.gz at build time, so only their archives are staged (see _ARCHIVES).
_COMPONENTS = [
    ("json", "json", "json", ["single_include"]),
    ("eigen", "eigen", "eigen", ["bench", "blas", "ci", "cmake", "debug", "demos", "doc", "failtest", "test"]),
    ("makeself", "makeself", "makeself", []),
    ("opbase", "opbase", "opbase", ["docs"]),
    ("catlass", "catlass/include", "catlass/include", []),
]

# name -> archive bytes to copy into bundle/pkg (source is cache root or cache/pkg)
_ARCHIVES = [
    "abseil-cpp-20230802.1.tar.gz",
    "protobuf-25.1.tar.gz",
    "include.zip",
    "eigen-5.0.0.tar.gz",
    "googletest-1.14.0.tar.gz",
    "makeself-release-2.5.0-patch1.tar.gz",
]


def _copy_dir(src: Path, dst_root: Path, target: str, prune: list[str]) -> None:
    target_dir = dst_root / target
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, target_dir)
    git_dir = target_dir / ".git"
    if git_dir.exists():
        shutil.rmtree(git_dir, ignore_errors=True)
    for name in prune:
        item = target_dir / name
        if item.exists():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()


def _missing_components(cache: Path) -> list[str]:
    """Return human-readable list of required components missing from the cache."""
    missing: list[str] = []
    for name, src_path, _, _ in _COMPONENTS:
        if not (cache / src_path).is_dir():
            missing.append(f"source dir {name!r} ({cache / src_path})")
    for archive in _ARCHIVES:
        if not (cache / archive).is_file() and not (cache / "pkg" / archive).is_file():
            missing.append(f"archive {archive!r} (cache root or cache/pkg)")
    return missing


def _verify_bundle(out: Path, pkg: Path) -> list[str]:
    """Re-check the produced bundle against the manifest before claiming success."""
    problems: list[str] = []
    for name, _, target, _ in _COMPONENTS:
        if not (out / target).is_dir():
            problems.append(f"missing built component dir {target!r} ({name})")
    for archive in _ARCHIVES:
        if not (pkg / archive).is_file():
            problems.append(f"missing built archive {archive!r}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        default="third_party",
        help="source third-party cache directory (default: repository third_party/)",
    )
    parser.add_argument(
        "--out",
        default="offline_bundle",
        help="output bundle directory (default: offline_bundle/)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    cache = Path(args.cache)

    if not cache.is_absolute():
        cache = repo_root / cache
    out = Path(args.out)
    if not out.is_absolute():
        out = repo_root / out

    if not cache.is_dir():
        raise SystemExit(f"cache directory not found: {cache}")

    # Hard-fail on any missing required component so an incomplete bundle is
    # never treated as a valid self-contained offline release artifact.
    missing = _missing_components(cache)
    if missing:
        raise SystemExit(
            "cannot build a self-contained offline bundle; missing required "
            "third-party components:\n  - " + "\n  - ".join(missing)
        )

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    pkg = out / "pkg"
    pkg.mkdir(parents=True, exist_ok=True)

    for name, src_path, target, prune in _COMPONENTS:
        _copy_dir(cache / src_path, out, target, prune)
        print(f"[ok] {name}: {cache / src_path} -> {out / target}", flush=True)

    for archive in _ARCHIVES:
        src = next(c for c in (cache / archive, cache / "pkg" / archive) if c.is_file())
        shutil.copy2(src, pkg / archive)
        print(f"[ok] archive {archive}", flush=True)

    # Post-generation verification: refuse to report success / ship a bundle
    # that does not match the manifest (defends against copy/rmtree failures).
    problems = _verify_bundle(out, pkg)
    if problems:
        raise SystemExit(
            "offline bundle is incomplete after generation:\n  - "
            + "\n  - ".join(problems)
        )

    print(f"\n[fla-npu] offline bundle ready: {out}")
    size_mb = sum(
        f.stat().st_size for f in out.rglob("*") if f.is_file()
    ) / (1024 * 1024)
    print(f"[fla-npu] bundle size (uncompressed): {size_mb:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
