"""从正式 JSON 选择 A2/A3 资产，保留原 case id 与预期 key 映射。"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

OP_DIR = Path(__file__).resolve().parents[1]
OP = "chunk_gated_delta_rule_bwd_finalize"
API = "executor_" + OP + "_arch22"


def expected_key(case):
    values = {value["name"]: value for value in case["inputs"]}
    attr = lambda name: values[name]["range_values"]
    q, v = values["q"]["shape"], values["v"]["shape"]
    owner = q[1] == v[1] and v[-1] == 128 and attr("chunk_size") == 64
    return (10 | ((10 if values["g"]["dtype"] == "bf16" else 30) << 8)
            | (int(attr("use_qk_l2_norm_in_kernel")) << 16)
            | (int(attr("use_beta_sigmoid_in_kernel")) << 17)
            | (int(attr("use_exp2")) << 18) | (int(owner) << 19))


def parse_source_case_ids(value):
    try:
        ids = [int(part.strip()) for part in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError("源 case ID 必须为非空的逗号分隔整数") from error
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError("源 case ID 不能重复")
    return ids


def select(scope, output, soc, source_case_ids=None):
    suffix = "" if scope == "accuracy" else "_mss"
    source = OP_DIR / f"atk_{OP}_arch22{suffix}.json"
    cases = [case for case in json.loads(source.read_text(encoding="utf-8")) if case["api_type"] == API]
    if source_case_ids is not None:
        ids = list(source_case_ids)
        if not ids or any(type(value) is not int for value in ids):
            raise ValueError("源 case ID 必须为非空整数列表")
        if len(set(ids)) != len(ids):
            raise ValueError("源 case ID 不能重复")
        missing = set(ids) - {case["id"] for case in cases}
        if missing:
            raise ValueError(f"当前 scope 不存在源 case ID: {sorted(missing)}")
        requested = set(ids)
        cases = [case for case in cases if case["id"] in requested]
    selected, mapping = [], []
    for case in cases:
        mapping.append(dict(case_id=len(selected), source_case_id=case["id"], name=case["name"],
                            seed=case["default_seed"], expected_key=expected_key(case),
                            actual_key=None, selection_evidence="pending Host UT/runtime evidence"))
        case["id"] = len(selected)
        selected.append(case)
    if not selected:
        raise RuntimeError("正式 JSON 中没有 A2/A3 资产")
    output.mkdir(parents=True, exist_ok=True)
    for name, value in (("cases.json", selected), ("selection.json", dict(
            schema="finalize-arch22-selection/v1", scope=scope, soc=soc,
            source=str(source), count=len(selected), expected_key_count=len({x['expected_key'] for x in mapping}),
            cases=mapping))):
        target = output / name
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return len(selected)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("accuracy", "mssanitizer", "determinism", "keys"), required=True)
    parser.add_argument("--soc", choices=("A2", "A3"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-case-ids", type=parse_source_case_ids,
                        help="可选的逗号分隔源 case ID；省略时选择该 scope 全部用例")
    args = parser.parse_args()
    try:
        print(select(args.scope, args.output, args.soc, args.source_case_ids))
    except ValueError as error:
        parser.error(str(error))
