#!/usr/bin/env python3
"""从冻结 ATK JSON 生成 ChunkKdaFwdPrepare 的逐 TilingKey 覆盖附录。"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


OP_DIR = Path(__file__).resolve().parents[1]
ACCURACY_JSON = OP_DIR / "atk_chunk_kda_fwd_prepare.json"
MSS_JSON = OP_DIR / "atk_chunk_kda_fwd_prepare_mss.json"
DEFAULT_OUTPUT = OP_DIR / "tiling_key_matrix.md"


def _metadata(case: dict) -> dict:
    for item in case.get("inputs", ()):
        if item.get("name") == "case_spec":
            value = item.get("range_values")
            if isinstance(value, str):
                return json.loads(value)
            if isinstance(value, dict):
                return value
    raise ValueError(f"case {case.get('id')} 缺少 case_spec")


def _load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} 必须是 JSON 数组")
    return [_metadata(case) for case in value]


def _beta_mode(meta: dict) -> str:
    if not meta["use_beta_sigmoid_in_kernel"]:
        return "raw"
    return "two_sigmoid" if meta["allow_neg_eigval"] else "sigmoid"


def _gate_mode(meta: dict) -> str:
    if not meta["use_gate_in_kernel"]:
        return "precomputed"
    return "safe" if meta["safe_gate"] else "softplus"


def _condition(meta: dict) -> str:
    norm = "l2" if meta["use_qk_l2norm_in_kernel"] else "identity"
    exp_domain = "exp2" if meta["use_exp2"] else "exp"
    shape = "/".join(
        str(meta[name]) for name in ("B", "HK", "HV", "T")
    )
    profile = meta.get("mss_profile", "")
    return (
        f"gate_dtype={meta['gate_dtype']},beta_dtype={meta['beta_dtype']},"
        f"norm={norm},beta={_beta_mode(meta)},gate={_gate_mode(meta)},"
        f"exp={exp_domain},output={meta['backward_mode']},"
        f"dt_bias={str(bool(meta['dt_bias'])).lower()},"
        f"layout={meta['layout']},B/HK/HV/T={shape},profile={profile}"
    )


def _ids(values: list[int]) -> str:
    return ",".join(str(value) for value in values) if values else "未覆盖"


def render() -> str:
    accuracy = _load(ACCURACY_JSON)
    mss = _load(MSS_JSON)
    accuracy_by_key: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: {"普通": [], "边界": []}
    )
    for meta in accuracy:
        tags = set(str(meta.get("tags", "")).split(","))
        bucket = "边界" if "boundary" in tags else "普通"
        accuracy_by_key[int(meta["expected_tiling_key"])][bucket].append(
            int(meta["case_id"])
        )

    rows = [
        "# ChunkKdaFwdPrepare TilingKey 覆盖附录",
        "",
        "本文件由 `scripts/generate_tiling_key_matrix.py` 从冻结的 `atk_chunk_kda_fwd_prepare.json` "
        "和 `atk_chunk_kda_fwd_prepare_mss.json` 机械生成，禁止手工修改。",
        "",
        "- `普通/边界` 是精度 JSON 中对应 TilingKey 的 case ID；边界只认统一清单里的 `boundary` 标签。",
        "- `确定性` 与四类 sanitizer 共用同一条 `_mss.json` case；四列分别表示该工具运行时应执行的 case ID。",
        "- `A2/A3/A5（规格 all）` 仅表示 JSON 规格允许的平台，不代表三个平台已运行。实际选择证据和测试结果未提供时统一标为未提供。",
        "- `未覆盖`/`未提供` 是交付缺口的明确记录，不能解释为通过。",
        "",
        "| case_id | tiling_key | 选择条件与测试形状 | 精度普通 case | 精度边界 case | 确定性 case | memcheck case | racecheck case | initcheck case | synccheck case | 适用 SoC | 实际选择证据 |",
        "| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for meta in mss:
        key = int(meta["expected_tiling_key"])
        case_id = int(meta["case_id"])
        acc = accuracy_by_key[key]
        rows.append(
            "| "
            + " | ".join(
                (
                    str(case_id),
                    str(key),
                    _condition(meta),
                    _ids(acc["普通"]),
                    _ids(acc["边界"]),
                    str(case_id),
                    str(case_id),
                    str(case_id),
                    str(case_id),
                    str(case_id),
                    "A2/A3/A5（规格 all）",
                    "未提供（待各 SoC 运行）",
                )
            )
            + " |"
        )
    return "\n".join(rows) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="只校验目标文件与冻结 JSON 机械生成结果一致",
    )
    args = parser.parse_args()
    expected = render()
    if args.check:
        if not args.output.is_file():
            raise SystemExit(f"覆盖附录不存在：{args.output}")
        actual = args.output.read_text(encoding="utf-8")
        if actual != expected:
            raise SystemExit(
                f"覆盖附录与冻结 JSON 不一致，请重新生成：{args.output}"
            )
        print(f"tiling key matrix ok: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8", newline="\n")
    print(f"generated tiling key matrix: {args.output} ({len(_load(MSS_JSON))} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
