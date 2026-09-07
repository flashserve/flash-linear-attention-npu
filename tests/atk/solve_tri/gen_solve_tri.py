"""solve_tri 混合容差用例生成器。

生成 200 条中小 shape 双标杆精度用例（与 GPU 双标杆套同一套 profile），标准为
``mixed_tolerance_bm``。同时写出 perf / mss 精简集。

覆盖维度：
  chunk 16/32/64/128、fp16/bf16、bsnd/bnsd/tnd/ntd、
  定长/变长、对齐/非对齐。
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

OP_DIR = Path(__file__).resolve().parent
OP_NAME = "solve_tri"
KERNEL_OP = "solve_tri"
MIXED_STANDARD = {"acc": "mixed_tolerance_bm", "perf": "not_key", "mem": 1.1}

# ===========================================================================
# 用例 profile 定义（原 scripts/generate_solve_tri_case.py 的数据源部分）
# ===========================================================================

CHUNK_SIZES = (16, 32, 64, 128)
DTYPES = ("bf16", "fp16")
DENSE_LAYOUTS = ("bsnd", "bnsd")
PACKED_LAYOUTS = ("tnd", "ntd")
LAYOUTS = DENSE_LAYOUTS + PACKED_LAYOUTS


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


# ===========================================================================
# ATK 用例生成（mixed_tolerance_bm 标准）
# ===========================================================================

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None


def _spec(index: int, profile: dict) -> dict:
    spec = deepcopy(profile)
    spec.pop("varlen", None)
    spec.update(
        {
            "name": _name(profile),
            "op": KERNEL_OP,
            "case_id": index,
            "seed": 20260817 + index,
            "route": "ascendc",
            "soc": "ascend950",
        }
    )
    return spec


def _case_json(index: int, profile: dict) -> dict:
    spec = _spec(index, profile)
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
        inputs.append(
            {
                "name": key,
                "type": "attr",
                "required": True,
                "dtype": "string" if isinstance(value, str) else "int",
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
        "name": f"{OP_NAME}_{index:04d}_{spec['name']}",
        "aclnn_name": None,
        "triton_name": None,
        "kernel_name": None,
        "version": "v2.1",
        "expected_error_msg": None,
        "api": "pytorch",
        "api_type": f"executor_{OP_NAME}",
        "aclnn_api_type": "aclnn_function",
        "triton_api_type": "triton_function",
        "fusion_api_type": "fusion_function",
        "fusion_mode": None,
        "dist_api_type": "dist_function",
        "kernel_api_type": "kernel_function",
        "backward": False,
        "standard": deepcopy(MIXED_STANDARD),
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


def dump_json_files(out_dir: Path | None = None) -> None:
    out_dir = out_dir or OP_DIR
    profiles = iter_profiles()
    all_cases = [_case_json(index, profile) for index, profile in enumerate(profiles)]
    (out_dir / f"atk_{OP_NAME}.json").write_text(
        json.dumps(all_cases, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    def _first(pred) -> int:
        for index, profile in enumerate(profiles):
            if pred(profile):
                return index
        raise RuntimeError("no matching profile")

    mss_idx = [
        _first(lambda p: p["chunk_size"] == 16 and p["layout"] == "bsnd" and p["aligned"] and not p.get("varlen")),
        _first(lambda p: p["chunk_size"] == 32 and p["layout"] == "bnsd" and not p["aligned"] and not p.get("varlen")),
        _first(lambda p: p["chunk_size"] == 64 and p["layout"] == "tnd" and p.get("varlen")),
        _first(lambda p: p["chunk_size"] == 128 and p["layout"] == "ntd" and p.get("varlen")),
        _first(lambda p: p["layout"] == "bsnd" and p["T"] == 17),
        _first(lambda p: p["layout"] == "ntd" and p.get("num_seqs") == 3),
    ]
    mss = [all_cases[i] for i in mss_idx]
    (out_dir / f"atk_{OP_NAME}_mss.json").write_text(
        json.dumps(mss, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    perf_idx = [
        i for i, profile in enumerate(profiles)
        if profile["chunk_size"] in {64, 128} and profile["aligned"] and not profile.get("varlen")
    ][:6]
    if not perf_idx:
        perf_idx = [0, 1]
    perf = [all_cases[i] for i in perf_idx]
    (out_dir / f"atk_{OP_NAME}_perf.json").write_text(
        json.dumps(perf, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    sizes = [_nbytes(profile) for profile in profiles]
    print(f"wrote {len(all_cases)} acc / {len(perf)} perf / {len(mss)} mss -> {out_dir}")
    print(f"tensor_bytes min={min(sizes)} max={max(sizes)}")


if GENERATOR_REGISTRY is not None:
    PROFILES = iter_profiles()

    @GENERATOR_REGISTRY.register(f"generator_{OP_NAME}")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            spec = _spec(index, PROFILES[index % len(PROFILES)])
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = f"{OP_NAME}_{index:04d}_{spec.get('name', 'case')}"
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    cfg.dtype = spec.get("dtype", "bf16")
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name == "seqlens":
                    cfg.range_values = json.dumps(spec["seqlens"], separators=(",", ":")) if spec.get("seqlens") else ""
                elif cfg.name == "aligned":
                    cfg.range_values = int(bool(spec.get("aligned")))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config


if __name__ == "__main__":
    dump_json_files()
