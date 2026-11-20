#!/usr/bin/env python3
"""Fail when a stable adapter delegates a call to the ctypes reference.

The ctypes layer is the reference implementation, not a transport: reaching it
from the stable path costs the descriptor forest the migration exists to remove
(measured: conv1d update 0.72 ms per call against 0.22 ms through the launcher,
and chunk_kda_bwd_intra 0.85 ms against 0.20 ms).  A guard that sends a legal
call there is therefore a performance bug, and one that grows silently as more
operators are ported.

So every delegation has to be listed below with a reason.  The list is meant to
shrink: an entry is either a kernel limitation (nothing the wrapper can do) or
work that has not been done yet, and the check reports which.

Usage:
  python stable_ctypes_fallbacks.py            # exit 1 on an undeclared delegation
  python stable_ctypes_fallbacks.py --json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SETUP_DIR = Path(__file__).resolve().parents[1]
ASCENDC_DIR = SETUP_DIR / "fla_npu" / "ops" / "ascendc"
MODULES = ("_stable.py",)

# op -> (kind, reason).  "kernel" means no wrapper-side fix exists yet because
# the operator itself is broken; "pending" means the port has not happened.
# Empty on purpose: no adapter reaches the ctypes reference any more.  The
# operator-level gaps that remain are not delegations -- they are recorded as
# recorded kernel limitations in the regression baseline (see
# npu_solve_tri's refused 'tnd' spelling, and the conv1d FN initial-state
# rows), which is why this list has nothing left to declare.
DECLARED: dict[str, tuple[str, str]] = {}


def passthrough_names(stable_text: str) -> list[str]:
    """Names a module-level ``PASSTHROUGH_OPS`` re-export makes ctypes-backed."""

    match = re.search(r"PASSTHROUGH_OPS\s*=\s*\((.*?)\)", stable_text, re.S)
    if match is None:
        return []
    return [name.strip().strip('"') for name in match.group(1).split(",")
            if name.strip()]


def delegating_ops(text: str) -> list[str]:
    """Operators whose body reaches the ctypes module."""

    found: list[str] = []
    current = None
    for line in text.splitlines():
        if line and not line[0].isspace():
            # A module-level line ends whatever function body preceded it, so a
            # module-level import is not attributed to the last def above it.
            current = None
        match = re.match(r"def (\w+)\(", line)
        if match:
            current = match.group(1)
        if "_aclnn_ctypes" in line and current and current not in found:
            found.append(current)
    # A module-level `from ._aclnn_ctypes import (...)` re-export is a
    # delegation too, but it happens outside any function body.
    if re.search(r"^from \._aclnn_ctypes import", text, re.M):
        for name in passthrough_names(text):
            if name not in found:
                found.append(name)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    rows = []
    undeclared = []
    declared_here = set()
    for module in MODULES:
        path = ASCENDC_DIR / module
        if not path.is_file():
            continue
        for op in delegating_ops(path.read_text(encoding="utf-8")):
            # The module-level import of PASSTHROUGH_OPS re-exports three names;
            # each of them is still a delegation and is listed above.
            kind_reason = DECLARED.get(op)
            row = {
                "op": op,
                "module": module,
                "declared": kind_reason is not None,
                "kind": kind_reason[0] if kind_reason else None,
                "reason": kind_reason[1] if kind_reason else None,
            }
            rows.append(row)
            if kind_reason is None:
                undeclared.append(op)
            else:
                declared_here.add(op)

    stale = sorted(set(DECLARED) - declared_here)

    if args.json:
        print(json.dumps({"rows": rows, "undeclared": undeclared,
                          "stale": stale}, indent=2))
    else:
        for row in rows:
            mark = "declared(%s)" % row["kind"] if row["declared"] else "UNDECLARED"
            print("%-32s %-16s %s" % (row["op"], mark, row["module"]))
        if stale:
            print("\ndeclared but no longer delegating (drop the entry): %s"
                  % ", ".join(stale))

    if undeclared:
        print("\nFAIL: %d operator(s) delegate to the ctypes reference without "
              "a declared reason: %s" % (len(undeclared), ", ".join(undeclared)))
        return 1
    print("\nOK: %d delegation(s), all declared" % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
