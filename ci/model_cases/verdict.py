"""Validate the two suites from both platforms before marking the optional job green."""
import argparse
from datetime import date
import json
from pathlib import Path
import sys

from run import plan


def verify(root, day, head, run_id):
    catalog = plan(day)
    expected = {c["id"] for c in catalog}
    passed = True
    for platform in ("a2", "a5"):
        path = root / f"model-cases-{platform}-{run_id}" / "summary.json"
        if not path.is_file():
            print(f"{platform.upper()}: missing model report")
            passed = False
            continue
        report = json.loads(path.read_bytes())
        if (report["platform"], report["date"], report["head_sha"], report["run_id"]) != (
                platform, day.isoformat(), head, run_id):
            print(f"{platform.upper()}: report identity mismatch")
            passed = False
            continue
        outcomes = report["cases"]
        failures = []
        for case in catalog:
            name = case["id"]
            detail = path.parent / (name + ".json")
            if outcomes.get(name) is not True or not detail.is_file():
                failures.append(name)
                continue
            row = json.loads(detail.read_bytes())
            if row.get("id") != name or row.get("suite") != case["suite"] or row.get("passed") is not True:
                failures.append(name)
                continue
            if case["suite"] == "kda":
                metrics = row.get("metrics", {})
                checks = [metrics.get(k, {}) for k in ("o", "final_state", "dq", "dk", "dv", "dbeta", "dg")]
            else:
                checks = [row.get("output", {})]
                if case["entry"] == "decode":
                    checks.append(row.get("state", {}))
                elif platform == "a5":
                    checks.append(row.get("route", {}))
            if any(metric.get("passed") is not True for metric in checks):
                failures.append(name)
        print(f"{platform.upper()}: {len(expected) - len(failures)}/{len(expected)} passed")
        if failures or set(outcomes) != expected:
            print("Failed or missing:", ", ".join(failures), "Unexpected:", ", ".join(sorted(outcomes.keys() - expected)))
            passed = False
    return passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("day", type=date.fromisoformat)
    parser.add_argument("head")
    parser.add_argument("run_id")
    args = parser.parse_args()
    sys.exit(0 if verify(args.root, args.day, args.head, args.run_id) else 1)
