#!/usr/bin/env python3
"""Check a built CI overlay; exit 10 when the base wheel needs a full rebuild."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay", required=True, type=Path)
    args = parser.parse_args()
    # Load the shared ABI checker without importing fla_npu or initializing CANN.
    spec = importlib.util.spec_from_file_location(
        "ci_op_abi", REPO_ROOT / "torch_custom/fla_npu/fla_npu/_op_abi.py"
    )
    abi = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(abi)
    try:
        # A malformed build is a CI failure, not a reason to silently rebuild.
        abi.validated_contracts(args.overlay, require_manifest=True)
        package = importlib.util.find_spec("fla_npu")
        if package is None or package.origin is None:
            raise RuntimeError("Installed architecture base wheel is not discoverable")
        base = Path(package.origin).parent / "opp/vendors/fla_npu_transformer"
        problems = abi.compare_overlay_to_base(args.overlay, base)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[CI][ERROR] Cannot validate the built partial OPP: {exc}", file=sys.stderr)
        return 2
    if problems:
        print("[CI][WARN] The base wheel cannot safely run this partial OPP:", file=sys.stderr)
        for problem in problems:
            print(f"[CI][WARN]   {problem}", file=sys.stderr)
        print("[CI] A complete wheel from the merged PR source is required.")
        return 10
    print("[CI] Partial OPP ABI matches the architecture base wheel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
