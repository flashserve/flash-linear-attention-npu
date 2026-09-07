#!/usr/bin/env python3
"""生成 200 条中小 shape 双标杆精度用例。

覆盖：
  chunk 16/32/64/128、fp16/bf16、bsnd/bnsd/tnd/ntd、
  定长/变长、对齐/非对齐。
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


OP_DIR = Path(__file__).resolve().parent
CHUNK_SIZES = (16, 32, 64, 128)
DTYPES = ("bf16", "fp16")
DENSE_LAYOUTS = ("bsnd", "bnsd")
PACKED_LAYOUTS = ("tnd", "ntd")
LAYOUTS = DENSE_LAYOUTS + PACKED_LAYOUTS
DOUBLE_STANDARD = {
    "acc": {
        "cv_fused_double_benchmark": {
            "max_re_ratio": 5,
            "avg_re_ratio": 1.5,
            "root_mean_squared_ratio": 1.5,
        }
    },
    "perf": "not_key",
    "mem": 1.1,
}


def _tail(chunk: int, kind: str) -> int:
    if kind == "half":
        return max(1, chunk // 2)
    if kind == "quarter":
        return max(1, chunk // 4)
    if kind == "almost":
        return max(1, chunk - 1)
    return 1


def _dense(chunk: int, dtype: str, layout: str, aligned: bool, size_id: int) -> dict:
    heads = 1 if size_id == 0 else 4
    batch = 1 if size_id == 0 else 2
    if aligned:
        tokens = 2 * chunk
    else:
        tokens = 2 * chunk + _tail(chunk, "half" if size_id == 0 else "quarter")
    return {
        "dtype": dtype,
        "B": batch,
        "H": heads,
        "T": tokens,
        "chunk_size": chunk,
        "layout": layout,
        "num_seqs": 1,
        "aligned": aligned,
        "varlen": False,
    }


def _packed(
    chunk: int,
    dtype: str,
    layout: str,
    aligned: bool,
    varlen: bool,
    size_id: int,
) -> dict:
    heads = 1 if size_id == 0 else 4
    if not varlen:
        seqlens = [2 * chunk] if aligned else [2 * chunk + _tail(chunk, "half" if size_id == 0 else "quarter")]
    elif aligned:
        seqlens = [chunk, chunk] if size_id == 0 else [2 * chunk, chunk]
    else:
        seqlens = (
            [chunk, chunk + _tail(chunk, "half")]
            if size_id == 0
            else [chunk + 1, 2 * chunk]
        )
    return {
        "dtype": dtype,
        "B": 1,
        "H": heads,
        "T": sum(seqlens),
        "chunk_size": chunk,
        "layout": layout,
        "num_seqs": len(seqlens),
        "aligned": aligned,
        "varlen": varlen,
        "seqlens": seqlens,
    }


def _specials() -> list[dict]:
    return [
        {
            "dtype": "fp16",
            "B": 1,
            "H": 2,
            "T": 17,
            "chunk_size": 16,
            "layout": "bsnd",
            "num_seqs": 1,
            "aligned": False,
            "varlen": False,
        },
        {
            "dtype": "bf16",
            "B": 1,
            "H": 2,
            "T": 33,
            "chunk_size": 32,
            "layout": "bnsd",
            "num_seqs": 1,
            "aligned": False,
            "varlen": False,
        },
        {
            "dtype": "fp16",
            "B": 1,
            "H": 2,
            "T": 65,
            "chunk_size": 32,
            "layout": "tnd",
            "num_seqs": 2,
            "aligned": False,
            "varlen": True,
            "seqlens": [32, 33],
        },
        {
            "dtype": "bf16",
            "B": 1,
            "H": 2,
            "T": 160,
            "chunk_size": 64,
            "layout": "ntd",
            "num_seqs": 2,
            "aligned": False,
            "varlen": True,
            "seqlens": [64, 96],
        },
        {
            "dtype": "fp16",
            "B": 4,
            "H": 4,
            "T": 128,
            "chunk_size": 64,
            "layout": "bsnd",
            "num_seqs": 1,
            "aligned": True,
            "varlen": False,
        },
        {
            "dtype": "bf16",
            "B": 2,
            "H": 4,
            "T": 256,
            "chunk_size": 128,
            "layout": "bnsd",
            "num_seqs": 1,
            "aligned": True,
            "varlen": False,
        },
        {
            "dtype": "fp16",
            "B": 1,
            "H": 2,
            "T": 65,
            "chunk_size": 16,
            "layout": "tnd",
            "num_seqs": 3,
            "aligned": False,
            "varlen": True,
            "seqlens": [16, 32, 17],
        },
        {
            "dtype": "bf16",
            "B": 1,
            "H": 1,
            "T": 321,
            "chunk_size": 128,
            "layout": "ntd",
            "num_seqs": 3,
            "aligned": False,
            "varlen": True,
            "seqlens": [128, 64, 129],
        },
    ]


def iter_profiles() -> list[dict]:
    profiles = []
    for size_id in (0, 1):
        for chunk in CHUNK_SIZES:
            for dtype in DTYPES:
                for layout in DENSE_LAYOUTS:
                    for aligned in (True, False):
                        profiles.append(_dense(chunk, dtype, layout, aligned, size_id))
    for size_id in (0, 1):
        for chunk in CHUNK_SIZES:
            for dtype in DTYPES:
                for layout in PACKED_LAYOUTS:
                    for aligned in (True, False):
                        for varlen in (False, True):
                            profiles.append(
                                _packed(chunk, dtype, layout, aligned, varlen, size_id)
                            )
    profiles.extend(_specials())
    if len(profiles) != 200:
        raise RuntimeError(f"期望 200 条，实际 {len(profiles)}")
    return profiles


def _name(spec: dict) -> str:
    align = "align" if spec["aligned"] else "unalign"
    length = "varlen" if spec.get("varlen") else "fixed"
    if spec["layout"] in PACKED_LAYOUTS:
        return (
            f"{spec['dtype']}_{spec['layout']}_{length}_{align}"
            f"_NS{spec['num_seqs']}_H{spec['H']}_T{spec['T']}_C{spec['chunk_size']}"
        )
    return (
        f"{spec['dtype']}_{spec['layout']}_{length}_{align}"
        f"_B{spec['B']}_H{spec['H']}_T{spec['T']}_C{spec['chunk_size']}"
    )


def _nbytes(spec: dict) -> int:
    if spec["layout"] in PACKED_LAYOUTS:
        return spec["T"] * spec["H"] * spec["chunk_size"] * 2
    return spec["B"] * spec["T"] * spec["H"] * spec["chunk_size"] * 2


def _case(index: int, profile: dict) -> dict:
    spec = deepcopy(profile)
    spec.pop("varlen", None)
    spec.update(
        {
            "name": _name(profile),
            "op": "solve_tri",
            "case_id": index,
            "seed": 20260817 + index,
            "route": "ascendc",
            "soc": "ascend950",
        }
    )
    dtype = spec["dtype"]
    inputs = [
        {"name": "low_precision_marker", "type": "tensor", "required": True, "dtype": dtype, "shape": [1], "range_values": [0, 0], "backward": True, "align_32B": None, "outlier_values": None},
        {"name": "fp32_marker", "type": "tensor", "required": True, "dtype": "fp32", "shape": [1], "range_values": [0, 0], "backward": True, "align_32B": None, "outlier_values": None},
        {"name": "case_spec", "type": "attr", "required": True, "dtype": "non_param", "shape": None, "range_values": json.dumps(spec, ensure_ascii=False, separators=(",", ":")), "backward": False, "align_32B": None, "outlier_values": None},
    ]
    extra_attrs = {
        "aligned": int(bool(spec["aligned"])),
        "seqlens": json.dumps(spec["seqlens"], separators=(",", ":")) if spec.get("seqlens") else "",
    }
    for key in ("dtype", "B", "H", "T", "chunk_size", "layout", "num_seqs", "case_id", "seed", "soc", "route"):
        value = spec[key]
        atype = "string" if isinstance(value, str) else "int"
        inputs.append(
            {
                "name": key,
                "type": "attr",
                "required": True,
                "dtype": atype,
                "shape": None,
                "range_values": value,
                "backward": False,
                "align_32B": None,
                "outlier_values": None,
            }
        )
    for key, value in extra_attrs.items():
        inputs.append(
            {
                "name": key,
                "type": "attr",
                "required": True,
                "dtype": "int" if isinstance(value, int) else "string",
                "shape": None,
                "range_values": value,
                "backward": False,
                "align_32B": None,
                "outlier_values": None,
            }
        )
    return {
        "id": index,
        "default_seed": spec["seed"],
        "name": f"solve_tri_{index:04d}_{spec['name']}",
        "aclnn_name": None,
        "triton_name": None,
        "kernel_name": None,
        "version": "v2.1",
        "expected_error_msg": None,
        "api": "pytorch",
        "api_type": "executor_solve_tri",
        "aclnn_api_type": "aclnn_function",
        "triton_api_type": "triton_function",
        "fusion_api_type": "fusion_function",
        "fusion_mode": None,
        "dist_api_type": "dist_function",
        "kernel_api_type": "kernel_function",
        "backward": False,
        "standard": deepcopy(DOUBLE_STANDARD),
        "outputs": None,
        "inputs": inputs,
        "acl_json": "",
        "method_inputs": None,
        "tensor_input": None,
        "compute_times": None,
        "save_name": None,
        "uuid": None,
        "downloaded": False,
        "is_boundary": not spec["aligned"],
        "xrun_cs_name": None,
        "xrun_data": None,
        "strategy": None,
    }


def main() -> None:
    profiles = iter_profiles()
    cases = [_case(index, profile) for index, profile in enumerate(profiles)]
    out = OP_DIR / "atk_solve_tri.json"
    out.write_text(json.dumps(cases, ensure_ascii=False), encoding="utf-8")
    sizes = [_nbytes(profile) for profile in profiles]
    counts = {
        "chunk": {str(chunk): 0 for chunk in CHUNK_SIZES},
        "dtype": {name: 0 for name in DTYPES},
        "layout": {name: 0 for name in LAYOUTS},
        "aligned": {"align": 0, "unalign": 0},
        "length": {"fixed": 0, "varlen": 0},
    }
    for profile in profiles:
        counts["chunk"][str(profile["chunk_size"])] += 1
        counts["dtype"][profile["dtype"]] += 1
        counts["layout"][profile["layout"]] += 1
        counts["aligned"]["align" if profile["aligned"] else "unalign"] += 1
        counts["length"]["varlen" if profile.get("varlen") else "fixed"] += 1
    print(f"wrote {len(cases)} cases -> {out}")
    print("coverage", json.dumps(counts, ensure_ascii=False))
    print(f"tensor_bytes min={min(sizes)} max={max(sizes)} sum={sum(sizes)}")


if __name__ == "__main__":
    main()
