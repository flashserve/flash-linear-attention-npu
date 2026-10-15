#!/usr/bin/env python3
"""Refresh or verify the vendored torch Stable-ABI header closure.

The launcher is compiled against ``csrc/include/torch_stable_abi`` rather
than the installed torch, because the header that decides the runtime floor is
also the one that moves: from 2.10 on ``library.h`` registers through
``torch_library_impl``, which 2.7.1 -- 2.9 do not export.  See the tree's
README for the full reasoning.

The closure is taken from the *compiler*, not from walking ``#include`` lines:
half of the headers torch pulls in sit behind capability guards
(``CPU_CAPABILITY_AVX2`` and friends) that never fire here, and a textual walk
copies them anyway.  ``-MD`` reports the headers a translation unit actually
opened, which is the set worth pinning.

Modes:

  --check                 the tree against ``MANIFEST.sha256``.  Offline; this
                          is what the gate runs.
  --check --torch DIR     and the headers the launcher's TUs reach: with the
                          vendored tree first on the include path, every
                          torch/c10 header that comes from *outside* it is
                          reported.  Needs a compiler and torch >= 2.9.
  --write --torch DIR     re-derive the closure from DIR alone, copy it in,
                          drop what is no longer reached, rewrite the manifest.

usage::

    python tools/vendor_stable_headers.py --check
    python tools/vendor_stable_headers.py --check   --torch /path/site-packages/torch
    python tools/vendor_stable_headers.py --write   --torch /path/site-packages/torch
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]        # torch_custom/fla_npu
CSRC = ROOT / "csrc"
VENDOR = CSRC / "include" / "torch_stable_abi"
MANIFEST_NAME = "MANIFEST.sha256"


def translation_units() -> tuple[Path, ...]:
    """What build_stable.py actually compiles, not what looks like a source file."""

    builder = CSRC / "build_stable.py"
    spec = importlib.util.spec_from_file_location("fla_build_stable", builder)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return tuple(module.TRANSLATION_UNITS)


def _include_root(torch: Path) -> Path:
    return torch / "include" if (torch / "include").is_dir() else torch


def _relative_to(path: str, root: Path) -> str | None:
    try:
        resolved = Path(path).resolve()
    except OSError:
        return None
    try:
        rel = resolved.relative_to(root.resolve())
    except ValueError:
        return None
    return rel.as_posix()


def _is_ours(name: str) -> bool:
    return name.startswith(("torch/", "c10/", "ATen/"))


def compile_deps(compiler: str, include_dirs: list[Path], source: Path) -> list[str]:
    """Headers *source* really opens, as gcc's ``-MD`` reports them."""

    with tempfile.NamedTemporaryFile(suffix=".d", delete=False) as handle:
        depfile = Path(handle.name)
    try:
        cmd = [compiler, "-std=c++17", "-fsyntax-only", "-MD", "-MF",
               str(depfile), *[f"-I{path}" for path in include_dirs],
               str(source)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        text = depfile.read_text(encoding="utf-8", errors="ignore")
    finally:
        depfile.unlink(missing_ok=True)
    # Makefile syntax: continuations, then whitespace-separated prerequisites.
    text = text.replace("\\\n", " ")
    _, _, prerequisites = text.partition(":")
    return [token for token in prerequisites.split() if token]


def reached_outside_vendor(compiler: str, torch: Path) -> list[str]:
    """torch/c10 headers the TUs take from the build torch, not from VENDOR.

    The vendored tree goes first on the include path, so anything reported here
    is a header the copy does not carry -- either a new one to vendor or a
    capability-guarded one that the preprocessor never opens.
    """

    include_root = _include_root(torch)
    offenders = set()
    for unit in translation_units():
        deps = compile_deps(compiler, [CSRC / "include", VENDOR, include_root],
                            unit)
        for dep in deps:
            if _relative_to(dep, VENDOR) is not None:
                continue
            name = _relative_to(dep, include_root)
            if name is not None and _is_ours(name):
                offenders.add(name)
    return sorted(offenders)


def compiled_closure(compiler: str, torch: Path) -> set[str]:
    """The torch/c10 headers the TUs would take from *torch* directly."""

    include_root = _include_root(torch)
    names = set()
    for unit in translation_units():
        for dep in compile_deps(compiler, [CSRC / "include", include_root], unit):
            name = _relative_to(dep, include_root)
            if name is not None and _is_ours(name):
                names.add(name)
    return names


def read_manifest(root: Path = VENDOR) -> dict[str, str]:
    manifest = root / MANIFEST_NAME
    if not manifest.is_file():
        return {}
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        entries[name.strip()] = digest.strip()
    return entries


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_tree(root: Path = VENDOR) -> list[str]:
    """The tree against its manifest -- no torch, no compiler."""

    expected = read_manifest(root)
    actual = {path.relative_to(root).as_posix(): _digest(path)
              for path in sorted(root.rglob("*.h"))}
    problems = []
    for name in sorted(set(expected) - set(actual)):
        problems.append(f"missing from the tree: {name}")
    for name in sorted(set(actual) - set(expected)):
        problems.append(f"not listed in {MANIFEST_NAME}: {name}")
    for name in sorted(set(actual) & set(expected)):
        if actual[name] != expected[name]:
            problems.append(
                f"edited: {name} (tree {actual[name][:12]}, "
                f"{MANIFEST_NAME} {expected[name][:12]}) -- these headers are a "
                f"verbatim copy, refresh them instead of patching")
    return problems


def write_tree(compiler: str, torch: Path, root: Path = VENDOR) -> int:
    """Re-derive the closure from *torch* alone, copy it in, rewrite the manifest."""

    include_root = _include_root(torch)
    found = compiled_closure(compiler, torch)
    if not found:
        raise SystemExit(f"no torch headers reached under {include_root}; "
                         f"is that a torch install?")
    for name in sorted(found):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(include_root / name, target)
    dropped = []
    for path in sorted(root.rglob("*.h")):
        name = path.relative_to(root).as_posix()
        if name not in found:
            path.unlink()
            dropped.append(name)
            parent = path.parent
            while parent != root and not any(parent.iterdir()):
                parent.rmdir()
                parent = parent.parent
    (root / MANIFEST_NAME).write_text(
        "".join(f"{_digest(root / name)}  {name}\n" for name in sorted(found)),
        encoding="utf-8")
    print(f"copied {len(found)} headers from {include_root}")
    for name in dropped:
        print(f"  dropped (no longer reached): {name}")
    print(f"rewrote {MANIFEST_NAME}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch", type=Path,
                        help="torch package dir or its include dir")
    parser.add_argument("--check", action="store_true",
                        help="verify the tree (and the closure, with --torch)")
    parser.add_argument("--write", action="store_true",
                        help="re-copy the closure from --torch")
    parser.add_argument("--compiler", default=os.environ.get("CXX", "g++"))
    args = parser.parse_args()

    if args.write and args.check:
        raise SystemExit("--check and --write are mutually exclusive")
    if args.write:
        if args.torch is None:
            raise SystemExit("--write needs --torch")
        return write_tree(args.compiler, args.torch)

    problems = verify_tree()
    if args.torch is not None:
        offenders = reached_outside_vendor(args.compiler, args.torch)
        problems.extend(
            f"{name}: the launcher includes this from the build torch, not from "
            f"the vendored tree (vendor it, or it changes with the build torch)"
            for name in offenders)
    if problems:
        print("FAIL vendored stable-abi headers")
        for line in problems:
            print("  -", line)
        return 1
    print(f"OK vendored stable-abi headers ({len(read_manifest())} headers match "
          f"{MANIFEST_NAME})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
