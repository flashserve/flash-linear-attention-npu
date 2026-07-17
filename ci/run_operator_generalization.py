#!/usr/bin/env python3
"""Run JSON-driven operator generalization cases on one NPU SOC."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "tests" / "op_cases"
REQUIRED_SOCS = ("ascend910b", "ascend910_93", "ascend950")


def discover_operators() -> list[str]:
    operators = []
    for path in sorted(CASE_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("implementation") == "ascendc":
            operators.append(manifest["op"])
    return operators


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soc", required=True, choices=REQUIRED_SOCS)
    parser.add_argument("--ops", help="comma-separated operators (default: all registered Ascend C operators)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registered = discover_operators()
    requested = parse_csv(args.ops) if args.ops else registered
    unknown = sorted(set(requested) - set(registered))
    if unknown:
        parser.error(f"operators are not registered in tests/op_cases: {unknown}")

    env = os.environ.copy()
    env.update(
        FLA_NPU_RUN_OPERATOR_TESTS="1",
        FLA_NPU_SOC=args.soc,
        TEST_DEVICE_ID=str(args.device),
    )
    failures = []
    for op in requested:
        test_path = ROOT / "tests" / "operators" / op / "accuracy" / f"test_{op}.py"
        if not test_path.is_file():
            failures.append((op, "missing canonical accuracy entry"))
            continue
        command = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            str(test_path.relative_to(ROOT)),
            "-k",
            "json_generalization_cases",
        ]
        print(f"[GENERALIZATION] soc={args.soc} op={op}")
        print("[GENERALIZATION] " + " ".join(shlex.quote(item) for item in command))
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
        if completed.returncode != 0:
            failures.append((op, f"pytest returned {completed.returncode}"))
    if failures:
        for op, reason in failures:
            print(f"[GENERALIZATION][ERROR] {op}: {reason}", file=sys.stderr)
        return 1
    action = "planned" if args.dry_run else "passed"
    print(f"[GENERALIZATION] {action} {len(requested)} operators on {args.soc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
