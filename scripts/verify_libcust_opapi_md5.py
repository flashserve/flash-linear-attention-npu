# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Verify that the running fla_npu really loads freshly built artifacts.

Three kinds of artifacts are compared between what ``import fla_npu`` actually
loads at runtime and the freshly built/staged copies:

1. OPP library: ``libcust_opapi.so``
   - runtime: resolved by mirroring fla_npu's own search order (see
     ``_resolve_runtime_lib``), without importing/loading fla_npu, so no CANN
     environment is required to run this script;
   - built:   ``build/libcust_opapi.so`` by default, or ``--built-lib`` /
     ``--run-package`` (extracted from a Makeself run package).
2. Kernel binaries: the ``*.o`` NPU kernels under
   ``op_impl/ai_core/tbe/kernel/``.  These are what actually runs on the NPU;
   a kernel-only source change does not alter ``libcust_opapi.so``, so without
   this comparison the script would not notice a stale kernel.  Compared by
   default against ``build/lib/fla_npu/opp/vendors/.../kernel``; disable with
   ``--no-kernel``.
3. Python wrapper: the core ``.py`` files under the installed package
   (``fla_npu/__init__.py``, ``ops/ascendc/*.py``), compared against the
   sources under ``torch_custom/fla_npu/fla_npu/``.  Opt-in via ``--python``.

If both sides are found, it reports whether their md5 match.  ``[OK]`` means
the running wheel is exactly the one freshly built; ``[FAIL]`` means a stale
copy is still in use and the new wheel / run package must be reinstalled.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path


# Python wrapper files that are pure copies of the sources (no build-time
# rewriting), as paths relative to the fla_npu package root, so the runtime
# side can be located through the installed package directory.
PYTHON_WRAPPER_MODULES = (
    "__init__.py",
    "ops/ascendc/__init__.py",
    "ops/ascendc/_aclnn_ctypes.py",
    "ops/ascendc/_runtime.py",
)


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_run_package_lib(run_package: Path) -> Path | None:
    """Locate libcust_opapi.so inside a .run package (Makeself self-extracting archive)."""
    import subprocess
    import tempfile

    if not run_package.exists():
        print(f"[WARN] run package not found: {run_package}", file=sys.stderr)
        return None
    with tempfile.TemporaryDirectory(prefix="fla-npu-run-extract-") as tmp:
        try:
            proc = subprocess.run(
                [str(run_package), f"--extract={tmp}", "--noexec"],
                check=True,
                capture_output=True,
                timeout=300,
            )
        except subprocess.CalledProcessError as exc:
            print(
                f"[WARN] failed to extract run package {run_package} "
                f"(exit {exc.returncode}):\n{exc.stderr.decode(errors='replace')[-500:]}",
                file=sys.stderr,
            )
            return None
        except (subprocess.TimeoutExpired, OSError) as exc:
            print(f"[WARN] failed to extract run package {run_package}: {exc}", file=sys.stderr)
            return None
        extracted = Path(tmp) / "packages" / "vendors" / "fla_npu_transformer" / "op_api" / "lib" / "libcust_opapi.so"
        if not extracted.exists():
            return None
        # Copy to a stable location so the md5 survives tempdir cleanup.
        stable = run_package.parent / f".{run_package.name}.libcust_opapi.so"
        try:
            import shutil

            shutil.copy2(extracted, stable)
            return stable
        except OSError:
            return extracted


def _find_built_lib(repo_root: Path, built_lib: str | None, run_package: str | None) -> Path | None:
    if run_package:
        return _resolve_run_package_lib(Path(run_package))
    if built_lib:
        candidate = Path(built_lib)
        if candidate.exists():
            return candidate
        return None

    candidates = [
        repo_root / "build" / "libcust_opapi.so",
        repo_root / "build" / "lib" / "fla_npu" / "opp" / "vendors" / "fla_npu_transformer" / "op_api" / "lib" / "libcust_opapi.so",
    ]
    return next((path for path in candidates if path.exists()), None)


def _installed_fla_npu_root() -> Path | None:
    """Locate the installed fla_npu package directory without importing it.

    ``importlib.util.find_spec`` returns the module spec without executing
    ``fla_npu/__init__.py``, so no CANN library is loaded and no CANN
    environment is required.
    """
    spec = importlib.util.find_spec("fla_npu")
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def _resolve_runtime_lib() -> Path | None:
    """Resolve the libcust_opapi.so that ``import fla_npu`` would load.

    Mirrors ``fla_npu/__init__.py`` (``_candidate_opp_roots`` +
    ``_resolve_vendor_dir``) without actually loading anything:

    1. ``FLA_NPU_OPP_PATH`` (external-OPP debug override), if set;
    2. the OPP embedded inside the installed package (site-packages/fla_npu/opp);
    3. ``ASCEND_CUSTOM_OPP_PATH`` / ``ASCEND_OPP_PATH`` from the CANN env.

    The first root containing ``op_api/lib/libcust_opapi.so`` (directly, under
    ``vendors/fla_npu_transformer``, or as a single vendor) wins.
    """
    candidates: list[Path] = []
    override = os.environ.get("FLA_NPU_OPP_PATH", "").strip()
    if override:
        candidates.append(Path(override).expanduser())

    installed_root = _installed_fla_npu_root()
    if installed_root is not None:
        candidates.append(installed_root / "opp")

    for env_name in ("ASCEND_CUSTOM_OPP_PATH", "ASCEND_OPP_PATH"):
        for part in os.environ.get(env_name, "").split(os.pathsep):
            if part:
                candidates.append(Path(part).expanduser())

    seen: set[Path] = set()
    for root in candidates:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)

        direct = root / "op_api" / "lib" / "libcust_opapi.so"
        if direct.exists():
            return direct

        vendors_root = root / "vendors"
        vendor_dir = vendors_root / "fla_npu_transformer"
        lib = vendor_dir / "op_api" / "lib" / "libcust_opapi.so"
        if lib.exists():
            return lib

        if vendors_root.exists():
            vendor_dirs = [
                path
                for path in vendors_root.iterdir()
                if path.is_dir() and (path / "op_api" / "lib" / "libcust_opapi.so").exists()
            ]
            if len(vendor_dirs) == 1:
                return vendor_dirs[0] / "op_api" / "lib" / "libcust_opapi.so"

    return None


def _check_opp_lib(repo_root: Path, built_lib: str | None, run_package: str | None) -> bool:
    loaded_lib = _resolve_runtime_lib()
    if loaded_lib is None:
        print(
            "[FAIL] runtime libcust_opapi.so not found; "
            "fla_npu is not installed or no OPP root is resolvable."
        )
        return False

    loaded_md5 = _md5(loaded_lib)
    print(f"loaded libcust_opapi.so: {loaded_lib}")
    print(f"loaded md5:              {loaded_md5}")

    built_lib_path = _find_built_lib(repo_root, built_lib, run_package)
    if built_lib_path is None:
        print(
            "[WARN] no freshly built libcust_opapi.so found; pass --built-lib or --run-package "
            "to compare against a build artifact."
        )
        return True

    built_md5 = _md5(built_lib_path)
    print(f"built  libcust_opapi.so: {built_lib_path}")
    print(f"built  md5:              {built_md5}")

    if loaded_md5 == built_md5:
        print("[OK] loaded libcust_opapi.so matches the freshly built one.")
        return True
    print(
        "[FAIL] loaded libcust_opapi.so differs from the freshly built one; "
        "the running wheel still uses an older OPP. Reinstall the new wheel/run package."
    )
    return False


def _kernel_dir_from_lib(lib_path: Path) -> Path | None:
    """Derive the kernel ``.o`` root from a libcust_opapi.so path.

    Inside a vendor OPP the layout is::

        .../vendors/fla_npu_transformer/op_api/lib/libcust_opapi.so
        .../vendors/fla_npu_transformer/op_impl/ai_core/tbe/kernel/<soc>/<op>/*.o
    """
    kernel = lib_path.parents[2] / "op_impl" / "ai_core" / "tbe" / "kernel"
    return kernel if kernel.is_dir() else None


def _collect_kernel_objects(root: Path) -> dict[str, str]:
    """Map ``soc/op/name.o`` -> md5 for every ``*.o`` under a kernel root."""
    objs: dict[str, str] = {}
    for path in root.rglob("*.o"):
        objs[str(path.relative_to(root))] = _md5(path)
    return objs


MAX_KERNEL_DIFF_PRINT = 20


def _check_kernel_o(repo_root: Path, built_kernel: str | None = None) -> bool:
    """Compare the runtime embedded OPP kernel ``.o`` files with the freshly built ones.

    Kernel ``.o`` files are what actually run on the NPU and are not part of
    ``libcust_opapi.so``, so a kernel-only source change can only be detected
    through them.
    """
    loaded_lib = _resolve_runtime_lib()
    runtime_root = _kernel_dir_from_lib(loaded_lib) if loaded_lib else None
    if runtime_root is None:
        print("[WARN] runtime OPP kernel dir not found; kernel .o comparison skipped.")
        return True

    built_root = (
        Path(built_kernel).resolve()
        if built_kernel
        else repo_root
        / "build"
        / "lib"
        / "fla_npu"
        / "opp"
        / "vendors"
        / "fla_npu_transformer"
        / "op_impl"
        / "ai_core"
        / "tbe"
        / "kernel"
    )
    if not built_root.is_dir():
        print(f"[WARN] freshly built kernel dir not found: {built_root}; kernel .o comparison skipped.")
        return True

    runtime_objs = _collect_kernel_objects(runtime_root)
    built_objs = _collect_kernel_objects(built_root)
    print(f"runtime kernel: {runtime_root} ({len(runtime_objs)} .o)")
    print(f"built   kernel: {built_root} ({len(built_objs)} .o)")

    problems: list[str] = []
    n_diff = 0
    n_missing = 0
    for rel in sorted(set(runtime_objs) | set(built_objs)):
        in_runtime = rel in runtime_objs
        in_built = rel in built_objs
        if in_runtime and in_built:
            if runtime_objs[rel] != built_objs[rel]:
                n_diff += 1
                if n_diff <= MAX_KERNEL_DIFF_PRINT:
                    problems.append(
                        f"[DIFF] {rel}\n"
                        f"  runtime md5: {runtime_objs[rel]}\n"
                        f"  built   md5: {built_objs[rel]}"
                    )
        else:
            n_missing += 1
            side = "runtime side" if in_runtime else "built side"
            if n_missing <= MAX_KERNEL_DIFF_PRINT:
                problems.append(f"[MISSING] {rel} (only on {side})")

    for line in problems:
        print(line)
    tail = len(problems) - MAX_KERNEL_DIFF_PRINT
    if tail > 0:
        print(f"  ... and {tail} more")

    if n_diff == 0 and n_missing == 0:
        print(f"[OK] all {len(runtime_objs)} kernel .o files match the freshly built OPP.")
        return True
    print(
        f"[FAIL] {n_diff} kernel .o differ, {n_missing} present on one side only; "
        "the running wheel still uses an older kernel. "
        "Reinstall the new wheel/run package."
    )
    return False


def _check_python_wrapper(repo_root: Path) -> bool:
    """Compare installed wrapper .py files against their sources."""
    installed_root = _installed_fla_npu_root()
    if installed_root is None:
        print("[WARN] installed fla_npu package not found; wrapper comparison skipped.", file=sys.stderr)
        return True
    source_root = repo_root / "torch_custom" / "fla_npu" / "fla_npu"
    if not source_root.exists():
        print(f"[WARN] wrapper source dir not found: {source_root}", file=sys.stderr)
        return True

    ok = True
    checked = 0
    for rel in PYTHON_WRAPPER_MODULES:
        installed = installed_root / rel
        source = source_root / rel
        if not installed.exists():
            print(f"[FAIL] installed wrapper file not found: {installed}")
            ok = False
            continue
        if not source.exists():
            print(f"[WARN] wrapper source file not found: {source}", file=sys.stderr)
            continue
        imd5 = _md5(installed)
        smd5 = _md5(source)
        status = "OK" if imd5 == smd5 else "FAIL"
        if imd5 != smd5:
            ok = False
        print(f"[{status}] {rel}")
        print(f"  loaded: {installed}  md5={imd5}")
        print(f"  source: {source}      md5={smd5}")
        checked += 1

    if checked == 0:
        print("[WARN] no Python wrapper files compared", file=sys.stderr)
        return True
    if ok:
        print("[OK] installed Python wrapper files match the current sources.")
    else:
        print(
            "[FAIL] some installed Python wrapper files differ from the current sources; "
            "reinstall the newly built wheel."
        )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--built-lib",
        default=None,
        help="Path to the freshly built libcust_opapi.so. Defaults to the one staged under build/.",
    )
    parser.add_argument(
        "--run-package",
        default=None,
        help="Path to a freshly built fla_npu_linux-*.run package; its libcust_opapi.so is extracted for comparison.",
    )
    parser.add_argument(
        "--python",
        action="store_true",
        help="Also compare the installed Python wrapper files against the current sources.",
    )
    parser.add_argument(
        "--no-kernel",
        action="store_true",
        help="Skip the kernel .o comparison (kernel .o are compared by default).",
    )
    parser.add_argument(
        "--built-kernel",
        default=None,
        help="Path to the freshly built kernel .o root. Defaults to the one staged under build/.",
    )
    args = parser.parse_args()

    # No fla_npu import here: the script is purely file-based md5 comparison,
    # so it works without a sourced CANN environment or a loadable fla_npu.
    repo_root = Path(__file__).resolve().parents[1]

    results = [_check_opp_lib(repo_root, args.built_lib, args.run_package)]
    if not args.no_kernel:
        results.append(_check_kernel_o(repo_root, args.built_kernel))
    if args.python:
        results.append(_check_python_wrapper(repo_root))

    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
