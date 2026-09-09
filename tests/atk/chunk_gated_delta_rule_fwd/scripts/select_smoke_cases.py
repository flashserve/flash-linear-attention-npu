#!/usr/bin/env python3
"""从实际 ATK case JSON 中选择并校验五场景冒烟用例。"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


SCENARIOS = (
    "dense",
    "varlen",
    "state_initial_final",
    "state_initial_only",
    "state_zero_final",
)


def _input_values(case: dict) -> dict:
    return {item["name"]: item.get("range_values") for item in case["inputs"]}


def _case_values(case: dict) -> dict:
    items = {item["name"]: item for item in case["inputs"]}
    values = {name: item.get("range_values") for name, item in items.items()}
    q_shape = items.get("q", {}).get("shape")
    v_shape = items.get("v", {}).get("shape")
    values["q_shape"] = tuple(q_shape or ())
    values["q_dtype"] = items.get("q", {}).get("dtype")
    values["value_dim"] = v_shape[-1] if v_shape else None
    values["gva"] = bool(q_shape and v_shape and q_shape[1] != v_shape[1])
    return values


def _features(case: dict) -> set[str]:
    values = _case_values(case)
    return {
        f"dtype:{values.get('qkv_dtype', values.get('q_dtype'))}",
        f"chunk:{values.get('chunk_size')}",
        f"value_dim:{values.get('value_dim')}",
        f"gva:{values.get('gva')}",
        f"varlen:{values.get('is_varlen')}",
    }


def select_smoke_cases(cases: list[dict]) -> tuple[tuple[int, str], ...]:
    by_scenario = {scenario: [] for scenario in SCENARIOS}
    for case in cases:
        scenario = str(_input_values(case).get("scenario"))
        if scenario in by_scenario:
            by_scenario[scenario].append(case)

    selected = []
    covered: set[str] = set()
    for scenario in SCENARIOS:
        candidates = by_scenario[scenario]
        if not candidates:
            raise ValueError(f"五场景冒烟缺少场景：{scenario}")
        case = max(
            candidates,
            key=lambda item: (
                len(_features(item) - covered),
                -int(item["id"]),
            ),
        )
        covered.update(_features(case))
        selected.append((int(case["id"]), scenario))
    return tuple(selected)


def make_smoke_subset(cases: list[dict]) -> tuple[tuple[tuple[int, str], ...], list[dict]]:
    selected = select_smoke_cases(cases)
    by_id = {int(case["id"]): case for case in cases}
    subset = []
    for new_id, (source_id, _scenario) in enumerate(selected):
        case = copy.deepcopy(by_id[source_id])
        case["id"] = new_id
        subset.append(case)
    return selected, subset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_json", type=Path)
    parser.add_argument("--subset-out", type=Path)
    args = parser.parse_args()
    with args.case_json.open(encoding="utf-8") as handle:
        cases = json.load(handle)
    selected, subset = make_smoke_subset(cases)
    if args.subset_out:
        args.subset_out.parent.mkdir(parents=True, exist_ok=True)
        args.subset_out.write_text(
            json.dumps(subset, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    for case_id, scenario in selected:
        print(f"{case_id}\t{scenario}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
