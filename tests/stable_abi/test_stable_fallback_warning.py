#!/usr/bin/env python3
"""Fail closed for missing/foreign launchers, and resolve the real launcher.

The historical filename is retained for CI invocations. No case may resolve a
ctypes backend. Run with --lib pointing at the built Stable launcher.
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
result = {"warnings": [], "backends": {}, "error": None}
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
    if not result["error"] or "Stable" not in result["error"]:
        problems.append(f"{case}: expected actionable Stable launcher error, got {result}")
    if result["backends"].get(OP):
        problems.append(f"{case}: failed launcher selected a backend: {result['backends']}")


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
    if result["error"]:
        problems.append(f"real: {result['error']}")


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
        print("FAIL Stable-only launcher failure handling")
        for line in problems:
            print("  -", line)
        return 1
    print("OK Stable-only launcher failure handling")
    return 0


if __name__ == "__main__":
    sys.exit(main())
