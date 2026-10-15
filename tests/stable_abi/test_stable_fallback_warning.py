#!/usr/bin/env python3
"""A launcher that cannot load must say so, not silently serve from ctypes.

The failure this guards is quiet by construction: the wheel installs, every
accuracy test passes, the kernels are the same ones the reference drives -- only
the host-side speedup is missing.  It showed up in the torch-version matrix as
"2.7.1 loads fine but the ratios look like the reference", which is exactly the
kind of report a warning has to pre-empt.

Three cases, each in a fresh interpreter because the warning fires once per
process and the backend record is global:

  missing    FLA_NPU_STABLE_LIB points at a file that is not there
  foreign    ... at a real shared object that is not the launcher
  real       ... at the built launcher (skipped when none is available)

The first two must warn and record the reason as "the stable launcher is
unusable"; the third must stay quiet and resolve to the stable backend.  The
reason is read back through ``FLA_NPU_STABLE_TRACE=1``, which is also the only
place the two fallback causes can be told apart.

usage::

    FLA_NPU_STABLE_LIB=/path/libfla_npu_stable.so PYTHONPATH=<env> \
        python tests/stable_abi/test_stable_fallback_warning.py [--lib PATH]
"""

from __future__ import annotations

import argparse
import ctypes.util
import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OP = "npu_fast_gelu_custom"
WARNING_MARKER = "Stable-ABI launcher is unusable"
REASON = "the stable launcher is unusable"

# Runs in a fresh interpreter: resolve one operator, then report what happened.
# ``fla_npu`` comes from PYTHONPATH like every other test here, so point that at
# the package layout under test (the wheel payload, or the source tree with the
# OPP overlaid into it).
CHILD = r'''
import json, sys, warnings
result = {"warnings": [], "backends": {}, "fallbacks": {}, "error": None}
try:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import fla_npu.ops.ascendc as ascendc
        try:
            ascendc._get_direct_op(sys.argv[1])
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
        result["warnings"] = [f"{w.category.__name__}: {w.message}" for w in caught]
        result["backends"] = dict(ascendc.BACKENDS)
        result["fallbacks"] = dict(ascendc.FALLBACKS)
except Exception as exc:
    result["skip"] = f"{type(exc).__name__}: {exc}"
print("RESULT " + json.dumps(result))
'''


def run_case(lib: str | None) -> dict:
    env = os.environ.copy()
    env["FLA_NPU_STABLE_TRACE"] = "1"
    if lib is None:
        env.pop("FLA_NPU_STABLE_LIB", None)
    else:
        env["FLA_NPU_STABLE_LIB"] = lib
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, OP],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True)
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            payload = json.loads(line[len("RESULT "):])
    if payload is None:
        payload = {"skip": f"child produced no result (rc={proc.returncode}): "
                           f"{proc.stderr.strip()[-400:]}"}
    payload["stderr"] = proc.stderr
    return payload


def check_warns(case: str, result: dict, problems: list[str]) -> None:
    if "skip" in result:
        problems.append(f"{case}: child could not run: {result['skip']}")
        return
    hits = [w for w in result["warnings"] if WARNING_MARKER in w]
    if len(hits) != 1:
        problems.append(
            f"{case}: expected exactly one launcher warning, got {len(hits)} "
            f"({result['warnings'][:3]})")
    if result["backends"].get(OP) != "ctypes":
        problems.append(f"{case}: backend is {result['backends'].get(OP)!r}, "
                        f"expected 'ctypes'")
    if OP not in result["fallbacks"]:
        problems.append(f"{case}: no fallback recorded for {OP}")
    if f"({REASON})" not in result["stderr"]:
        problems.append(f"{case}: the trace line does not blame the launcher "
                        f"({REASON!r} missing from stderr)")


def check_quiet(result: dict, problems: list[str]) -> None:
    if "skip" in result:
        problems.append(f"real: child could not run: {result['skip']}")
        return
    hits = [w for w in result["warnings"] if WARNING_MARKER in w]
    if hits:
        problems.append(f"real: the launcher warned while it worked: {hits}")
    if result["backends"].get(OP) != "stable":
        problems.append(f"real: backend is {result['backends'].get(OP)!r}, "
                        f"expected 'stable'")
    if OP in result["fallbacks"]:
        problems.append(f"real: {OP} fell back although the launcher answered")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", default=os.environ.get("FLA_NPU_STABLE_LIB"),
                        help="path to a built libfla_npu_stable.so")
    args = parser.parse_args()

    problems: list[str] = []

    missing = Path("/nonexistent/fla_npu/libfla_npu_stable.so")
    check_warns("missing", run_case(str(missing)), problems)

    foreign = ctypes.util.find_library("m") or "/lib/x86_64-linux-gnu/libm.so.6"
    check_warns("foreign", run_case(foreign), problems)

    if args.lib and Path(args.lib).is_file():
        check_quiet(run_case(str(Path(args.lib).resolve())), problems)
    else:
        print(f"SKIP real: no launcher at {args.lib!r}")

    if problems:
        print("FAIL stable fallback warning")
        for line in problems:
            print("  -", line)
        return 1
    print("OK stable fallback warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
