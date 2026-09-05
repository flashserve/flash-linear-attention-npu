#!/usr/bin/env python3
"""聚合分片 ATK summary，并对完整 case ID 集合做闭环校验。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def aggregate(root: Path, expected_cases: int) -> dict:
    shard_files = sorted(root.glob("shard_*_*/summary.json"))
    case_ids: list[int] = []
    execution_failed: list[int] = []
    accuracy_failed: list[int] = []
    shards = []
    for path in shard_files:
        summary = json.loads(path.read_text(encoding="utf-8"))
        statistic = summary.get("statistic") or {}
        ids = [int(value) for value in statistic.get("case_ids", [])]
        execution = [
            int(value) for value in statistic.get("execution_failed_case_ids", [])
        ]
        accuracy = [
            int(value) for value in statistic.get("accuracy_failed_case_ids", [])
        ]
        case_ids.extend(ids)
        execution_failed.extend(execution)
        accuracy_failed.extend(accuracy)
        shards.append(
            {
                "path": str(path.parent.relative_to(root)),
                "expected_cases": int(summary.get("expected_cases", 0)),
                "case_ids": ids,
                "execution_failed_case_ids": execution,
                "accuracy_failed_case_ids": accuracy,
            }
        )

    expected_ids = list(range(expected_cases))
    unique_ids = sorted(set(case_ids))
    duplicate_ids = sorted(
        case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
    )
    missing_ids = sorted(set(expected_ids) - set(unique_ids))
    extra_ids = sorted(set(unique_ids) - set(expected_ids))
    execution_failed = sorted(set(execution_failed))
    accuracy_failed = sorted(set(accuracy_failed))
    complete = (
        unique_ids == expected_ids
        and not duplicate_ids
        and sum(shard["expected_cases"] for shard in shards) == expected_cases
    )
    passed = complete and not execution_failed and not accuracy_failed
    return {
        "schema": "gdn-core-double-benchmark-matrix-aggregate/v1",
        "expected_cases": expected_cases,
        "shard_count": len(shards),
        "observed_cases": len(unique_ids),
        "case_ids": unique_ids,
        "duplicate_case_ids": duplicate_ids,
        "missing_case_ids": missing_ids,
        "extra_case_ids": extra_ids,
        "execution_failed_case_ids": execution_failed,
        "accuracy_failed_case_ids": accuracy_failed,
        "execution_success": expected_cases - len(execution_failed) if complete else None,
        "accuracy_passed": (
            expected_cases - len(set(execution_failed) | set(accuracy_failed))
            if complete
            else None
        ),
        "complete": complete,
        "passed": passed,
        "shards": shards,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-cases", type=int, default=500)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    result = aggregate(args.root, args.expected_cases)
    text = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.json_out:
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    if not result["complete"]:
        return 97
    return 0 if result["passed"] else 99


if __name__ == "__main__":
    raise SystemExit(main())
