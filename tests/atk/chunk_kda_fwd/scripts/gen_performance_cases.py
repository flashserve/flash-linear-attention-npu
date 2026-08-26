#!/usr/bin/env python3
"""Generate the dense chunk_kda_fwd performance subset."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


# Existing matrix entries provide the canonical input schema.  The three long
# lengths are dense performance-only cases and are derived from the same schema.
CASE_SPECS = (
    (266, 266, 2048, False, "recompute"),
    (267, 267, 2048, True, "export"),
    (282, 282, 8192, False, "recompute"),
    (283, 283, 8192, True, "export"),
    (290, 290, 16384, False, "recompute"),
    (291, 291, 16384, True, "export"),
    (298, 266, 32768, False, "recompute"),
    (299, 267, 32768, True, "export"),
    (300, 266, 65536, False, "recompute"),
    (301, 267, 65536, True, "export"),
)


def _input_by_name(case: dict, name: str) -> dict:
    matches = [item for item in case["inputs"] if item["name"] == name]
    if len(matches) != 1:
        raise ValueError(
            f"case {case['id']} must contain exactly one {name!r} input"
        )
    return matches[0]


def _make_dense_case(
    source_case: dict,
    output_id: int,
    total_tokens: int,
    disable_recompute: bool,
    suffix: str,
) -> dict:
    case = copy.deepcopy(source_case)
    case["id"] = output_id
    spec_input = _input_by_name(case, "case_spec")
    spec = json.loads(spec_input["range_values"])
    expected = {
        "B": 1,
        "H": 96,
        "HV": 96,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "layout": "BSND",
        "q_dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "bf16",
        "soc": "ascend950",
    }
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(
                f"case {source_case['id']} has unexpected {key}: "
                f"{spec.get(key)!r}, expected {value!r}"
            )

    spec.update(
        T=total_tokens,
        case_key=f"ascend950_h96_t{total_tokens}_c64_dense_bsnd_{suffix}",
        cu_seqlens="",
        distribution="dense",
        disable_recompute=disable_recompute,
        explicit_chunk_indices=False,
        optional_spec=(
            "initial=False,final=False,varlen=False,"
            f"disable_recompute={disable_recompute}"
        ),
        profile="a5_performance",
        shape_spec=(
            f"B=1,H=96,HV=96,T={total_tokens},K=128,V=128"
        ),
        tags="performance,model_target,regression",
    )
    spec_input["range_values"] = json.dumps(
        spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    _input_by_name(case, "total_tokens")["range_values"] = total_tokens
    _input_by_name(case, "varlen")["range_values"] = False
    _input_by_name(case, "disable_recompute")["range_values"] = disable_recompute
    return case


def generate(source_path: Path) -> list[dict]:
    source_cases = json.loads(source_path.read_text(encoding="utf-8"))
    by_id = {case["id"]: case for case in source_cases}
    missing = sorted({source_id for _, source_id, *_ in CASE_SPECS} - set(by_id))
    if missing:
        raise ValueError(f"source matrix is missing case ids: {missing}")
    return [
        _make_dense_case(
            by_id[source_id], output_id, total_tokens, disable_recompute, suffix
        )
        for output_id, source_id, total_tokens, disable_recompute, suffix in CASE_SPECS
    ]


def main() -> None:
    op_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=op_dir / "atk_chunk_kda_fwd.json")
    parser.add_argument(
        "--output", type=Path, default=op_dir / "atk_chunk_kda_fwd_performance.json"
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    rendered = json.dumps(generate(args.source), ensure_ascii=False, indent=2) + "\n"
    if args.check:
        if not args.output.is_file() or args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"performance cases are stale: {args.output}")
        print("performance cases are up to date: 10 dense BSND cases")
        return
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote 10 dense BSND cases to {args.output}")


if __name__ == "__main__":
    main()
