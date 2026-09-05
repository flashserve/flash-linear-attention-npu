#!/usr/bin/env python3
"""校验 GDN 双标杆 ATK 的五场景用例矩阵。"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from case_matrix_contract import case_contract


SCENARIOS = (
    "dense",
    "varlen",
    "state_initial_final",
    "state_initial_only",
    "state_zero_final",
)


def _inputs(case: dict) -> dict[str, dict]:
    return {item["name"]: item for item in case["inputs"]}


def _dtype_text(value) -> str:
    text = str(value).lower()
    if "bf16" in text or "bfloat16" in text:
        return "bf16"
    if "fp16" in text or "float16" in text:
        return "fp16"
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_json", type=Path)
    parser.add_argument("--cases-per-scenario", type=int, default=100)
    parser.add_argument(
        "--contract-mode",
        choices=("generated", "semantic"),
        default="generated",
        help="generated 校验本工程确定性合同；semantic 只校验通用执行合同",
    )
    parser.add_argument(
        "--expected-dtypes",
        default="bf16,fp16",
        help="每个场景要求覆盖的 dtype 集合，逗号分隔",
    )
    parser.add_argument(
        "--allow-no-gva",
        action="store_true",
        help="允许矩阵只有 MHA；仅用于适配历史矩阵",
    )
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--split-dir", type=Path)
    args = parser.parse_args()

    cases = json.loads(args.case_json.read_text(encoding="utf-8"))
    errors: list[str] = []
    expected_dtypes = {
        _dtype_text(value)
        for value in args.expected_dtypes.split(",")
        if value.strip()
    }
    if not expected_dtypes or not expected_dtypes <= {"bf16", "fp16"}:
        raise SystemExit(
            f"--expected-dtypes 仅支持 bf16/fp16，实际为 {sorted(expected_dtypes)}"
        )
    expected_total = len(SCENARIOS) * args.cases_per_scenario
    counts = collections.Counter()
    expected_contracts = collections.Counter()
    if args.contract_mode == "generated":
        expected_contracts = collections.Counter(
            (scenario, *case_contract(scenario, local_index))
            for scenario in SCENARIOS
            for local_index in range(args.cases_per_scenario)
        )
    actual_contracts = collections.Counter()
    selected = {scenario: [] for scenario in SCENARIOS}
    coverage = {
        scenario: {
            "batch": set(),
            "k_heads": set(),
            "v_heads": set(),
            "tokens": set(),
            "value_dim": set(),
            "chunk_size": set(),
            "dtype": set(),
            "varlen": 0,
            "gva": 0,
        }
        for scenario in SCENARIOS
    }

    if len(cases) != expected_total:
        errors.append(f"用例数 {len(cases)} != {expected_total}")
    if [case.get("id") for case in cases] != list(range(len(cases))):
        errors.append("case id 必须从 0 连续递增")

    for position, case in enumerate(cases):
        values = _inputs(case)
        required = {
            "q",
            "k",
            "v",
            "g",
            "beta",
            "scale",
            "chunk_size",
            "is_varlen",
            "scenario",
            "cu_seqlens_spec",
            "qkv_dtype",
        }
        missing = required - values.keys()
        if missing:
            errors.append(f"case {position}: 缺少输入 {sorted(missing)}")
            continue
        scenario = values["scenario"]["range_values"]
        if scenario not in selected:
            errors.append(f"case {position}: 未知场景 {scenario}")
            continue
        counts[scenario] += 1
        selected[scenario].append(case)

        q_shape = values["q"]["shape"]
        k_shape = values["k"]["shape"]
        v_shape = values["v"]["shape"]
        g_shape = values["g"]["shape"]
        beta_shape = values["beta"]["shape"]
        batch, k_heads, tokens, key_dim = q_shape
        v_heads = v_shape[1]
        value_dim = v_shape[-1]
        chunk_size = int(values["chunk_size"]["range_values"])
        is_varlen = bool(values["is_varlen"]["range_values"])
        cu_spec = str(values["cu_seqlens_spec"]["range_values"])
        dtype = _dtype_text(values["qkv_dtype"]["range_values"])

        actual_contracts[(
            scenario,
            batch,
            k_heads,
            v_heads,
            tokens,
            value_dim,
            chunk_size,
            is_varlen,
            cu_spec,
        )] += 1

        if q_shape != k_shape or key_dim != 128:
            errors.append(f"case {position}: q/k/K 契约不匹配")
        if v_shape != [batch, v_heads, tokens, value_dim]:
            errors.append(f"case {position}: v shape 不匹配")
        if g_shape != [batch, tokens, v_heads] or beta_shape != g_shape:
            errors.append(f"case {position}: g/beta shape 不匹配")
        if v_heads % k_heads != 0:
            errors.append(f"case {position}: Hv={v_heads} 不能整除 Hk={k_heads}")
        if value_dim not in (128, 256) or chunk_size not in (64, 128):
            errors.append(f"case {position}: V/chunk 不受支持")
        if dtype not in {"bf16", "fp16"}:
            errors.append(f"case {position}: dtype={dtype} 不受支持")
        for name in ("q", "k", "v", "beta"):
            actual = _dtype_text(values[name]["dtype"])
            if actual != dtype:
                errors.append(
                    f"case {position}: {name} dtype={actual} 与 qkv_dtype={dtype} 不一致"
                )

        if is_varlen:
            try:
                offsets = [int(value) for value in cu_spec.split(",")]
            except ValueError:
                offsets = []
            if (
                batch != 1
                or len(offsets) < 2
                or offsets[0] != 0
                or offsets[-1] != tokens
                or any(left >= right for left, right in zip(offsets, offsets[1:]))
            ):
                errors.append(f"case {position}: 变长元数据无效")
            coverage[scenario]["varlen"] += 1
        elif cu_spec != "none":
            errors.append(f"case {position}: 定长用例不应携带 cu_seqlens")
        if scenario == "dense" and is_varlen:
            errors.append(f"case {position}: dense 场景不能为 varlen")
        if scenario == "varlen" and not is_varlen:
            errors.append(f"case {position}: varlen 场景必须为 varlen")
        if v_heads != k_heads:
            coverage[scenario]["gva"] += 1
        for key, value in (
            ("batch", batch),
            ("k_heads", k_heads),
            ("v_heads", v_heads),
            ("tokens", tokens),
            ("value_dim", value_dim),
            ("chunk_size", chunk_size),
            ("dtype", dtype),
        ):
            coverage[scenario][key].add(value)

    for scenario in SCENARIOS:
        if counts[scenario] != args.cases_per_scenario:
            errors.append(
                f"{scenario}: 用例数 {counts[scenario]} != {args.cases_per_scenario}"
            )
        for key, expected in (
            ("value_dim", {128, 256}),
            ("chunk_size", {64, 128}),
            ("dtype", expected_dtypes),
        ):
            actual = coverage[scenario][key]
            if actual != expected:
                errors.append(f"{scenario}: {key} 覆盖为 {sorted(actual)}")
        if not args.allow_no_gva and coverage[scenario]["gva"] == 0:
            errors.append(f"{scenario}: 未覆盖 GVA")
        expected_varlen = (
            args.cases_per_scenario
            if scenario == "varlen"
            else (args.cases_per_scenario // 2 if scenario.startswith("state_") else 0)
        )
        if coverage[scenario]["varlen"] != expected_varlen:
            errors.append(
                f"{scenario}: varlen 数 {coverage[scenario]['varlen']} != {expected_varlen}"
            )

    if args.contract_mode == "generated":
        missing_contracts = list((expected_contracts - actual_contracts).elements())
        extra_contracts = list((actual_contracts - expected_contracts).elements())
        if missing_contracts:
            errors.append(
                f"缺少确定性契约 {len(missing_contracts)} 条：{missing_contracts[:10]}"
            )
        if extra_contracts:
            errors.append(
                f"多出确定性契约 {len(extra_contracts)} 条：{extra_contracts[:10]}"
            )

    serializable_coverage = {
        scenario: {
            key: sorted(value) if isinstance(value, set) else value
            for key, value in fields.items()
        }
        for scenario, fields in coverage.items()
    }
    summary = {
        "schema": "chunk-gated-delta-rule-fwd-case-matrix/v1",
        "case_json": str(args.case_json),
        "total_cases": len(cases),
        "contract_mode": args.contract_mode,
        "expected_dtypes": sorted(expected_dtypes),
        "require_gva": not args.allow_no_gva,
        "cases_per_scenario": dict(counts),
        "coverage": serializable_coverage,
        "valid": not errors,
        "errors": errors,
    }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if args.split_dir and not errors:
        args.split_dir.mkdir(parents=True, exist_ok=True)
        for scenario, scenario_cases in selected.items():
            (args.split_dir / f"{scenario}.json").write_text(
                json.dumps(scenario_cases, indent=2), encoding="utf-8"
            )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
