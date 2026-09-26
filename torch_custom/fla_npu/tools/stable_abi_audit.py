#!/usr/bin/env python3
"""Audit a Stable-ABI launcher against the surface it is allowed to address.

Three independent checks, all of them about the same promise -- the artifact
loads on every torch from ``min_torch`` up, and it keeps loading:

1. Source level.  The stable translation units may only include
   ``torch/csrc/stable/*`` (plus the vendored tree); any ATen / c10 / pybind11 /
   ``torch/extension.h`` include re-introduces the unstable ABI.
2. ELF level.  No C++ ATen/c10/pybind symbol may stay undefined (those mangled
   names are exactly what breaks when torch is upgraded while the ``aoti_torch_*``
   shims keep their promise), the registration entry point has to be the one
   every supported torch exports, no symbol or NEEDED entry outside
   ``stable_abi_symbols.json`` may appear, and the C++ standard library the
   artifact demands may not exceed the floor the wheel claims.
3. Compile level.  ``--vendor-syntax-only`` compiles the TUs with the vendored
   headers and *no* torch include path: if that stops working, the build has a
   compile-time torch dependency again, which is what raised the runtime floor
   before the headers were pinned.

The header closure itself is checked by ``vendor_stable_headers.py --check
--torch``; this file checks the artifact those headers produce.

usage::

  python stable_abi_audit.py --source ../csrc/src/*.cpp
  python stable_abi_audit.py --lib ../libfla_npu_stable.so
  python stable_abi_audit.py --vendor-syntax-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]        # torch_custom/fla_npu
CSRC = ROOT / "csrc"
SYMBOLS = Path(__file__).with_name("stable_abi_symbols.json")


def translation_units() -> tuple[Path, ...]:
    """What build_stable.py actually compiles, not what looks like a source file."""

    builder = CSRC / "build_stable.py"
    spec = importlib.util.spec_from_file_location("fla_build_stable", builder)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return tuple(module.TRANSLATION_UNITS)

FORBIDDEN_INCLUDES = (
    "ATen/",
    "c10/",
    "torch/extension.h",
    "torch/csrc/utils/",
    "torch/csrc/autograd/",
    "torch/csrc/api/",
    "pybind11/",
)
ALLOWED_PREFIXES = ("torch/csrc/stable/", "torch/headeronly/")

MANGLED_UNSTABLE = (re.compile(r"^_ZN2at"), re.compile(r"^_ZN3c10"),
                    re.compile(r"^_ZN8pybind11"))

_INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)
_GLIBCXX_RE = re.compile(r"GLIBCXX_([0-9][0-9.]*)")


def load_spec(path: Path = SYMBOLS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _version_key(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def audit_source(path: Path) -> list[str]:
    problems = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for include in _INCLUDE_RE.findall(text):
        if include.startswith(ALLOWED_PREFIXES):
            continue
        if any(bad in include for bad in FORBIDDEN_INCLUDES):
            problems.append(f"{path.name}: forbidden include <{include}>")
    return problems


def _undefined(path: Path) -> list[str]:
    out = subprocess.run(["nm", "-D", "--undefined-only", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return [line.split()[-1] for line in out.splitlines() if line.strip()]


def _dynamic(path: Path) -> tuple[list[str], list[str]]:
    out = subprocess.run(["readelf", "-d", str(path)],
                         check=True, capture_output=True, text=True).stdout
    needed, rpath = [], []
    for line in out.splitlines():
        if "(NEEDED)" in line or "(RPATH)" in line or "(RUNPATH)" in line:
            entry = line.split("[", 1)[1].rstrip("]") if "[" in line else ""
            if "(NEEDED)" in line:
                needed.append(entry)
            else:
                rpath.append(entry)
    return needed, rpath


def _glibcxx(path: Path) -> list[str]:
    out = subprocess.run(["readelf", "-V", str(path)],
                         check=True, capture_output=True, text=True).stdout
    return sorted(set(_GLIBCXX_RE.findall(out)), key=_version_key)


def audit_lib(path: Path, spec: dict) -> list[str]:
    problems = []
    undefined = _undefined(path)
    unstable = [name for name in undefined
                if any(pattern.match(name.split("@")[0])
                       for pattern in MANGLED_UNSTABLE)]
    if unstable:
        problems.append(
            f"{path.name}: {len(unstable)} unstable C++ symbol(s) referenced, "
            f"e.g. {unstable[:3]}")

    needed, rpath = _dynamic(path)
    if "libtorch_python.so" in needed:
        problems.append(f"{path.name}: links libtorch_python.so (Python ABI)")

    # The registration entry point is the whole reason the headers are pinned:
    # 2.10 moved it to torch_library_impl, which 2.7.1 -- 2.9 do not export.
    entry = spec["entry_symbol"]
    names = {name.split("@")[0] for name in undefined}
    if entry not in names:
        problems.append(f"{path.name}: does not reference {entry}; the artifact "
                        f"would not register on the runtimes this wheel supports")
    for forbidden in spec["forbidden_entry_symbols"]:
        if forbidden in names:
            problems.append(
                f"{path.name}: references {forbidden}, which torch "
                f"{spec['min_torch']} does not export -- this artifact was "
                f"built against a newer torch than the wheel claims")

    known = set(spec["symbols"])
    referenced = {name.split("@")[0] for name in undefined
                  if name.startswith("aoti_torch_")}
    for name in sorted(referenced - known):
        problems.append(
            f"{path.name}: new runtime symbol {name}; confirm it exists in torch "
            f"{spec['min_torch']} and then list it in {SYMBOLS.name}")

    allowed_torch = set(spec["allowed_torch_libraries"])
    patterns = [re.compile(p) for p in spec["system_library_patterns"]]
    for library in needed:
        if library in allowed_torch or any(p.match(library) for p in patterns):
            continue
        problems.append(f"{path.name}: unexpected NEEDED entry {library} "
                        f"(the launcher links torch and the C++ runtime only)")

    if rpath:
        problems.append(f"{path.name}: has RPATH/RUNPATH {rpath}; the loader has "
                        f"to resolve libtorch from the running environment")

    demanded = _glibcxx(path)
    limit = spec["max_glibcxx"]
    if demanded and _version_key(demanded[-1]) > _version_key(limit):
        problems.append(
            f"{path.name}: demands GLIBCXX_{demanded[-1]}, above the "
            f"{limit} floor -- it was built against a newer libstdc++ than the "
            f"target systems ship (build in the pinned container)")

    print(f"  ELF: {len(undefined)} undefined symbols, "
          f"{len(referenced)} aoti_torch_*, NEEDED {needed}, "
          f"GLIBCXX up to {demanded[-1] if demanded else 'none'}")
    return problems


def audit_vendor_syntax(compiler: str) -> list[str]:
    """The TUs must compile with the vendored headers and no torch at all."""

    include = [CSRC / "include", CSRC / "include" / "torch_stable_abi"]
    problems = []
    for unit in translation_units():
        if not unit.is_file():
            problems.append(f"{unit} is missing")
            continue
        cmd = [compiler, "-std=c++17", "-fsyntax-only",
               *[f"-I{path}" for path in include], str(unit)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            problems.append(
                f"{unit.name}: does not compile against the vendored headers "
                f"alone -- the build has a compile-time torch dependency again: "
                f"{done.stderr.strip()[-300:]}")
        else:
            print(f"  {unit.name}: compiles without a torch include path")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", nargs="*", type=Path, default=[])
    parser.add_argument("--lib", type=Path)
    parser.add_argument("--vendor-syntax-only", action="store_true",
                        help="compile the TUs with the vendored headers only")
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--symbols", type=Path, default=SYMBOLS)
    args = parser.parse_args()

    spec = load_spec(args.symbols)
    problems: list[str] = []
    for source in args.source:
        problems.extend(audit_source(source))
    if args.lib:
        problems.extend(audit_lib(args.lib, spec))
    if args.vendor_syntax_only:
        problems.extend(audit_vendor_syntax(args.compiler))

    if not (args.source or args.lib or args.vendor_syntax_only):
        parser.error("nothing to check: pass --source, --lib or "
                     "--vendor-syntax-only")

    if problems:
        print("FAIL stable-abi audit")
        for line in problems:
            print("  -", line)
        return 1
    print("OK stable-abi audit")
    return 0


if __name__ == "__main__":
    sys.exit(main())
