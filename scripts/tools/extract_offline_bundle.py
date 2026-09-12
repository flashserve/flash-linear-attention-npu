#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""将 wheel 内嵌的离线 third-party bundle 还原到源码树。

wheel 携带两部分：预编译的 OPP（用于离线使用），以及一份内嵌的离线 third-party
bundle（位于 ``fla_npu/offline/third_party``）。本脚本把该 bundle 复制回源码树的
``third_party/``（即 ``CANN_3RD_LIB_PATH``），从而能在完全离线（不访问
gitcode/gitee）的情况下，从该源码树重新编译项目。

用法：

    python scripts/tools/extract_offline_bundle.py --src <仓库/克隆路径>

bundle 的目录布局与 ``cmake/third_party/*.cmake`` 离线探测的期望一致
（abseil-cpp/、protobuf/、json/、eigen/、makeself/、opbase/、
catlass/include/、pkg/*.tar.gz），因此还原后再 configure 全程保持离线。

安全约束：``--src`` 必须指向源码树（即包含 ``CMakeLists.txt`` 与 ``build.sh``
的目录）。该目录下已有的 ``third_party/`` 内容会被有意覆盖，以保证每个组件都匹配
wheel 固化的版本；混用本地缓存与 wheel 版本可能破坏构建。本脚本绝不改动系统路径
或已安装的软件包。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def _installed_bundle() -> Path | None:
    """Locate the offline bundle inside the installed fla_npu package."""
    try:
        import fla_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        print(f"[warn] unable to import fla_npu: {exc}", flush=True)
        return None
    pkg_dir = Path(fla_npu.__file__).resolve().parent  # type: ignore[attr-defined]
    bundle = pkg_dir / "offline" / "third_party"
    if not bundle.is_dir():
        print(f"[warn] no embedded offline bundle at {bundle}", flush=True)
        return None
    return bundle


def _looks_like_source_tree(root: Path) -> bool:
    return (root / "CMakeLists.txt").is_file() and (root / "build.sh").is_file()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        required=True,
        help="path to the source tree whose third_party/ should be (over)written",
    )
    args = parser.parse_args()

    src_root = Path(args.src).expanduser().resolve()
    if not src_root.is_dir():
        raise SystemExit(f"source tree not found: {src_root}")
    if not _looks_like_source_tree(src_root):
        raise SystemExit(
            f"{src_root} does not look like a flash-linear-attention-npu source "
            "tree (missing CMakeLists.txt and/or build.sh). Refusing to write."
        )

    bundle = _installed_bundle()
    if bundle is None:
        raise SystemExit(
            "could not locate the embedded offline bundle; is fla-npu installed "
            "with the offline bundle (see prepare_offline_bundle.py)?"
        )

    dest = src_root / "third_party"
    dest.mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in bundle.iterdir():
        target = dest / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
        copied += 1
        print(f"[ok] {item.name}", flush=True)

    print(
        f"\n[fla-npu] extracted and overwrote {copied} components under {dest} "
        "(matched wheel-pinned versions).",
        flush=True,
    )
    print(
        "[fla-npu] source-tree third_party is ready for offline (re)build.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
