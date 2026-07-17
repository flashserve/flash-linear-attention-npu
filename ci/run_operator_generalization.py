#!/usr/bin/env python3
"""Run JSON-driven operator generalization cases on one NPU SOC."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CASE_ROOT = ROOT / "tests" / "op_cases"
REQUIRED_SOCS = ("ascend910b", "ascend910_93", "ascend950")


def discover_cases() -> dict[str, list[str]]:
    operators = {}
    for path in sorted(CASE_ROOT.glob("*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("implementation") == "ascendc":
            case_ids = [
                case["id"]
                for case in manifest.get("cases", [])
                if "generalization" in case.get("tags", [])
                and case.get("expect", {}).get("return_code") == "ACLNN_SUCCESS"
                and "ascendc" in case.get("run_on", [])
            ]
            operators[manifest["op"]] = case_ids
    return operators


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--soc", required=True, choices=REQUIRED_SOCS)
    parser.add_argument("--ops", help="comma-separated operators (default: all registered Ascend C operators)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=300, help="per-case timeout in seconds")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    registered = discover_cases()
    requested = parse_csv(args.ops) if args.ops else list(registered)
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
    case_count = 0
    for op in requested:
        test_path = ROOT / "tests" / "operators" / op / "accuracy" / f"test_{op}.py"
        if not test_path.is_file():
            failures.append((op, "<all>", "missing canonical accuracy entry"))
            continue
        if not registered[op]:
            failures.append((op, "<all>", "no executable generalization cases"))
            continue
        for case_id in registered[op]:
            case_count += 1
            command = [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(test_path.relative_to(ROOT)),
                "-k",
                "json_generalization_cases",
            ]
            case_env = env.copy()
            case_env["FLA_NPU_CASE_IDS"] = case_id
            print(f"[GENERALIZATION] soc={args.soc} op={op} case={case_id}", flush=True)
            print(
                "[GENERALIZATION] " + " ".join(shlex.quote(item) for item in command),
                flush=True,
            )
            if args.dry_run:
                continue
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=case_env,
                start_new_session=True,
            )
            try:
                return_code = process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                failures.append((op, case_id, f"timed out after {args.timeout}s"))
                continue
            if return_code != 0:
                failures.append((op, case_id, f"pytest returned {return_code}"))
    if failures:
        for op, case_id, reason in failures:
            print(f"[GENERALIZATION][ERROR] {op}/{case_id}: {reason}", file=sys.stderr, flush=True)
        return 1
    action = "planned" if args.dry_run else "passed"
    print(
        f"[GENERALIZATION] {action} {case_count} cases for "
        f"{len(requested)} operators on {args.soc}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
