#!/usr/bin/env python3
"""Read migrated operator-local cases from the canonical JSON manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[3]
CASE_ROOT = ROOT / "tests" / "op_cases"


def load_legacy_cases(op: str, suite: Optional[str] = None) -> List[Dict]:
    manifest = json.loads((CASE_ROOT / f"{op}.json").read_text(encoding="utf-8"))
    cases = [case for case in manifest["cases"] if "legacy" in case]
    if suite is not None:
        cases = [case for case in cases if case["legacy"].get("suite") == suite]
    return cases


def find_legacy_case(op: str, original_id: str, suite: Optional[str] = None) -> Dict:
    matches = [
        case
        for case in load_legacy_cases(op, suite)
        if str(case["legacy"].get("original_id")) == str(original_id)
    ]
    if len(matches) != 1:
        raise KeyError(f"expected one migrated case for {op}:{suite}:{original_id}, got {len(matches)}")
    return matches[0]


def legacy_param_values(
    op: str,
    suite: str,
    group: Optional[str] = None,
    param_names: Optional[List[str]] = None,
    dtype_module=None,
):
    """Rebuild pytest parameter tuples without keeping a second case table."""
    result = []
    for case in load_legacy_cases(op, suite):
        raw = case["legacy"]["raw"]
        if group is not None and raw.get("group") != group:
            continue
        if "values" in raw:
            values = list(raw["values"])
        elif param_names is not None:
            values = [raw[name] for name in param_names]
        else:
            raise KeyError(f"{case['id']} has neither raw.values nor requested param_names")
        if dtype_module is not None:
            values = [
                getattr(dtype_module, value)
                if isinstance(value, str) and value in {"float16", "bfloat16", "float32", "float64"}
                else value
                for value in values
            ]
        result.append((str(case["legacy"]["original_id"]), tuple(values)))
    return result


def materialize(op: str, suite: str, output: Path) -> int:
    cases = load_legacy_cases(op, suite)
    raw_cases = [case["legacy"]["raw"] for case in cases]
    if not raw_cases:
        raise ValueError(f"no migrated cases found for {op}:{suite}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(raw_cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(raw_cases)


def dqkwg_case_info(original_id: str) -> Dict:
    raw = find_legacy_case("chunk_bwd_dqkwg", original_id, "direct_regression")["legacy"]["raw"]
    return {
        "B": raw["B"],
        "HV": raw["H_v"],
        "HK": raw["H_k"],
        "T": raw["T"],
        "chunk_size": raw["C"],
        "dtype": raw["dtype"],
        "Gtype": raw["g_dtype"],
        "scale": raw["scale"],
        "cu_seqlens": raw["cu_seqlens"],
        "K": raw["K"],
        "V": raw["V"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--op", required=True)
    list_parser.add_argument("--suite")

    write_parser = subparsers.add_parser("materialize")
    write_parser.add_argument("--op", required=True)
    write_parser.add_argument("--suite", required=True)
    write_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "list":
        for case in load_legacy_cases(args.op, args.suite):
            print(case["legacy"]["original_id"])
        return 0
    count = materialize(args.op, args.suite, args.output)
    print(f"materialized {count} cases for {args.op}:{args.suite}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
