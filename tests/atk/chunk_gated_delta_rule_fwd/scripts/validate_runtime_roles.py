#!/usr/bin/env python3
"""从 ATK 日志核验每个 case 的三个命名执行角色。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atk_role_contract import RUNTIME_ROLE_ORDER


ROLE_PATTERN = re.compile(
    r"\[gdn-double-atk\]\s+case=(\d+)\s+role=(dut|benchmark|golden)\b"
)


def audit_roles(log_text: str, expected_cases: int) -> dict:
    if expected_cases <= 0:
        raise ValueError("expected_cases 必须是正整数")
    observations = ROLE_PATTERN.findall(log_text)
    expected_count = expected_cases * len(RUNTIME_ROLE_ORDER)
    failures = []
    if len(observations) != expected_count:
        failures.append(f"role_count={len(observations)} != {expected_count}")
    roles_by_case: dict[int, list[str]] = {}
    for case_id_text, role in observations:
        roles_by_case.setdefault(int(case_id_text), []).append(role)
    if len(roles_by_case) != expected_cases:
        failures.append(f"case_count={len(roles_by_case)} != {expected_cases}")
    expected_roles = sorted(RUNTIME_ROLE_ORDER)
    for case_id, roles in sorted(roles_by_case.items()):
        if sorted(roles) != expected_roles:
            failures.append(
                f"case={case_id} roles={roles} "
                f"expected_once_each={list(RUNTIME_ROLE_ORDER)}"
            )
    return {
        "schema": "gdn-atk-runtime-role-contract/v1",
        "expected_cases": expected_cases,
        "expected_role_order": list(RUNTIME_ROLE_ORDER),
        "observed_role_count": len(observations),
        "observed_case_groups": len(roles_by_case),
        "observed_case_ids": sorted(roles_by_case),
        "failures": failures,
        "passed": not failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_log", type=Path)
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    result = audit_roles(
        args.task_log.read_text(encoding="utf-8", errors="replace"),
        args.expected_cases,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not result["passed"]:
        for failure in result["failures"]:
            print(f"[gdn-double-atk] 角色合同失败：{failure}")
        return 1
    print(
        "[gdn-double-atk] 角色合同通过："
        f"{result['observed_case_groups']} case，"
        f"角色={result['expected_role_order']}（顺序不敏感）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
