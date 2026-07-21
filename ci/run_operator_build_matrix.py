#!/usr/bin/env python3
"""Build the registered Ascend C operator set for the required SOC matrix."""

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
    if not operators:
        raise RuntimeError("no Ascend C operator manifests found")
    return operators


def parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def package_candidates() -> list[Path]:
    return sorted((ROOT / "build_out").glob("fla-npu-*.run")) + sorted(
        (ROOT / "build").glob("fla-npu-*.run")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--soc",
        action="append",
        choices=(*REQUIRED_SOCS, "all"),
        help="SOC to build; repeat for multiple SOCs (default: all)",
    )
    parser.add_argument("--ops", help="comma-separated operators (default: all registered Ascend C operators)")
    parser.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    parser.add_argument("--vendor-name", default="fla_npu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", type=Path, help="optional JSON result path")
    args = parser.parse_args()

    registered = discover_operators()
    requested = parse_csv(args.ops) if args.ops else registered
    unknown = sorted(set(requested) - set(registered))
    if unknown:
        parser.error(f"operators are not registered in tests/op_cases: {unknown}")
    socs = list(args.soc or REQUIRED_SOCS)
    if "all" in socs:
        socs = list(REQUIRED_SOCS)
    socs = list(dict.fromkeys(socs))

    results = []
    for soc in socs:
        command = [
            "bash",
            "build.sh",
            "--pkg",
            f"--soc={soc}",
            f"--vendor_name={args.vendor_name}",
            f"--ops={','.join(requested)}",
            f"-j{args.jobs}",
        ]
        print(f"[BUILD-MATRIX] soc={soc} ops={len(requested)}")
        print("[BUILD-MATRIX] " + " ".join(shlex.quote(item) for item in command))
        result = {"soc": soc, "operators": requested, "command": command, "status": "dry-run"}
        if not args.dry_run:
            previous_packages = {path: path.stat().st_mtime_ns for path in package_candidates()}
            env = os.environ.copy()
            env["FLA_NPU_SOC"] = soc
            completed = subprocess.run(command, cwd=ROOT, env=env, check=False)
            if completed.returncode != 0:
                result.update(status="failed", return_code=completed.returncode)
                results.append(result)
                break
            fresh_packages = [
                path
                for path in package_candidates()
                if path not in previous_packages or path.stat().st_mtime_ns > previous_packages[path]
            ]
            if not fresh_packages:
                result.update(status="failed", reason="build produced no fresh run package")
                results.append(result)
                break
            result.update(
                status="passed",
                return_code=0,
                packages=[str(path.relative_to(ROOT)) for path in fresh_packages],
            )
        results.append(result)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({"results": results}, indent=2) + "\n", encoding="utf-8")
    failed = [result for result in results if result["status"] == "failed"]
    if failed:
        print(f"[BUILD-MATRIX][ERROR] failed SOCs: {[item['soc'] for item in failed]}", file=sys.stderr)
        return 1
    action = "planned" if args.dry_run else "passed"
    print(f"[BUILD-MATRIX] {action} SOCs: {socs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
