"""chunk_gdn_fwd_prepare 的 ATK 泛化用例生成器。

100 个 shape × 2 条（bf16 双 seed）= 200。不支持的 fp16 入口会改回 bf16。
约束：chunk_size=64，K=128，V∈{128,256}，HV/HK∈{1,2,3,4}，use_exp2=True，
use_qk_l2norm=True，use_gate=False。
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

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

OP_NAME = "chunk_gdn_fwd_prepare"


def _ok(hk: int, hv: int) -> bool:
    return hk > 0 and hv % hk == 0 and 1 <= (hv // hk) <= 4


def _shape_table() -> list[dict]:
    rows: list[dict] = []

    def add(name: str, B: int, HK: int, HV: int, T: int, V: int) -> None:
        if not _ok(HK, HV):
            return
        rows.append(
            dict(name=name, B=B, HK=HK, HV=HV, T=T, K=128, V=V, chunk_size=64)
        )

    # Six golden-like shapes (case2 T is scaled down for ATK CPU inv cost).
    add("gva_B2_T256", 2, 16, 32, 256, 128)
    add("noGVA_wide_T128", 4, 8, 8, 128, 128)
    add("tail_T160", 1, 8, 8, 160, 128)
    add("mid_HK6_T108", 2, 6, 6, 108, 128)
    add("mid_HK12_T108", 1, 12, 12, 108, 128)
    add("partial_pack_HV6", 1, 6, 6, 64, 128)

    for V in (128, 256):
        for ratio in (1, 2, 3, 4):
            hk = 4
            add(f"r{ratio}_T64_V{V}", 1, hk, hk * ratio, 64, V)
            add(f"r{ratio}_T96_V{V}", 1, hk, hk * ratio, 96, V)
            add(f"r{ratio}_T160_V{V}", 1, hk, hk * ratio, 160, V)
            add(f"r{ratio}_T256_V{V}", 1, hk, hk * ratio, 256, V)

    for B, HK, HV, T, V in [
        (1, 8, 8, 192, 128),
        (1, 8, 16, 192, 128),
        (1, 8, 16, 192, 256),
        (2, 4, 4, 128, 128),
        (2, 4, 8, 128, 256),
        (2, 8, 8, 96, 128),
        (3, 4, 4, 80, 128),
        (1, 16, 16, 128, 128),
        (1, 16, 32, 128, 256),
        (1, 16, 48, 128, 128),
        (1, 16, 64, 128, 256),
        (1, 5, 5, 64, 128),
        (1, 5, 10, 96, 256),
        (1, 5, 15, 160, 128),
        (1, 7, 7, 70, 128),
        (1, 7, 14, 140, 256),
        (1, 9, 9, 100, 128),
        (1, 9, 18, 200, 128),
        (1, 3, 12, 64, 256),
        (4, 4, 4, 64, 128),
        (4, 4, 8, 96, 256),
        (8, 2, 2, 64, 128),
        (8, 2, 8, 64, 256),
        (1, 32, 32, 64, 128),
        (1, 32, 32, 96, 256),
        (1, 12, 24, 128, 128),
        (1, 12, 36, 128, 256),
        (1, 10, 40, 80, 128),
        (2, 6, 12, 108, 128),
        (2, 6, 18, 108, 256),
        (1, 4, 4, 320, 128),
        (1, 4, 8, 320, 256),
        (1, 8, 8, 384, 128),
        (1, 8, 24, 192, 256),
        (1, 1, 1, 64, 128),
        (1, 1, 4, 96, 256),
        (16, 2, 2, 64, 128),
        (1, 24, 24, 128, 128),
        (1, 24, 48, 64, 256),
        (1, 4, 4, 512, 128),
        (1, 8, 16, 512, 128),
        (1, 4, 16, 256, 256),
        (2, 16, 16, 160, 128),
        (1, 20, 20, 80, 128),
        (1, 20, 40, 80, 256),
        (1, 11, 11, 75, 128),
        (1, 11, 22, 130, 256),
        (1, 13, 13, 64, 128),
        (1, 13, 26, 96, 128),
        (1, 15, 15, 120, 256),
        (1, 15, 30, 64, 128),
        (1, 18, 18, 72, 128),
        (1, 18, 36, 72, 256),
        (1, 4, 12, 200, 128),
        (1, 8, 32, 128, 256),
        (3, 8, 8, 100, 128),
        (1, 6, 24, 96, 256),
        (1, 4, 4, 1024, 128),
        (1, 4, 8, 768, 256),
        (1, 16, 16, 256, 128),
        (1, 8, 8, 640, 128),
        (2, 8, 16, 192, 128),
        (1, 2, 6, 160, 256),
        (1, 2, 8, 128, 128),
        (1, 3, 3, 90, 128),
        (1, 3, 9, 90, 256),
        (5, 4, 4, 64, 128),
        (1, 28, 28, 64, 128),
        (1, 28, 56, 64, 256),
        (1, 4, 4, 65, 128),
        (1, 4, 8, 127, 256),
        (1, 8, 8, 63, 128),
        (1, 8, 16, 1, 128),
        (1, 4, 4, 32, 128),
        (1, 4, 8, 48, 256),
    ]:
        add(f"B{B}_HK{HK}_HV{HV}_T{T}_V{V}", B, HK, HV, T, V)

    # Dedup by shape tuple, keep first 100.
    seen = set()
    uniq = []
    for row in rows:
        key = (row["B"], row["HK"], row["HV"], row["T"], row["V"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(row)
        if len(uniq) >= 100:
            break
    if len(uniq) < 100:
        raise RuntimeError(f"need 100 shapes, got {len(uniq)}")
    return uniq


def _make_profiles() -> list[dict]:
    profiles = []
    case_id = 0
    for shape in _shape_table():
        for slot, dtype_name in enumerate(("bf16", "fp16")):
            spec = dict(shape)
            spec.update(
                dtype="bf16",  # fp16 unsupported; remap
                op=OP_NAME,
                case_id=case_id,
                seed=20260817 + case_id,
                route="ascendc",
                soc="ascend950",
                use_qk_l2norm_in_kernel=True,
                use_gate_in_kernel=False,
                use_beta_sigmoid_in_kernel=True,
                allow_neg_eigval=True,
                use_exp2=True,
            )
            spec["name"] = f"{shape['name']}_{dtype_name}"
            profiles.append(spec)
            case_id += 1
    return profiles


PROFILES = _make_profiles()


def _dtype(name: str) -> str:
    return "bf16"


def _spec(index: int) -> dict:
    return deepcopy(PROFILES[index % len(PROFILES)])


def _case_json(spec: dict, case_id: int) -> dict:
    inputs = [
        {
            "name": "low_precision_marker",
            "type": "tensor",
            "required": True,
            "dtype": "bf16",
            "shape": [1],
            "range_values": [0, 0],
            "backward": True,
            "align_32B": None,
            "outlier_values": None,
        },
        {
            "name": "fp32_marker",
            "type": "tensor",
            "required": True,
            "dtype": "fp32",
            "shape": [1],
            "range_values": [0, 0],
            "backward": True,
            "align_32B": None,
            "outlier_values": None,
        },
        {
            "name": "case_spec",
            "type": "attr",
            "required": True,
            "dtype": "non_param",
            "shape": None,
            "range_values": json.dumps(spec, ensure_ascii=False, separators=(",", ":")),
            "backward": False,
            "align_32B": None,
            "outlier_values": None,
        },
    ]
    for key in (
        "dtype", "B", "HK", "HV", "T", "K", "V", "chunk_size", "case_id", "seed", "soc", "route",
    ):
        val = spec[key]
        dtype = "string" if isinstance(val, str) else "int"
        inputs.append(
            {
                "name": key,
                "type": "attr",
                "required": True,
                "dtype": dtype,
                "shape": None,
                "range_values": val,
                "backward": False,
                "align_32B": None,
                "outlier_values": None,
            }
        )
    return {
        "id": case_id,
        "default_seed": spec["seed"],
        "name": f"{OP_NAME}_{case_id:04d}_{spec.get('name', 'case')}",
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
        "standard": {"acc": "mixed_tolerance_bm", "perf": "not_key", "mem": 1.1},
        "outputs": None,
        "inputs": inputs,
    }


def dump_json_files(out_dir: Path | None = None) -> None:
    out_dir = out_dir or Path(__file__).resolve().parent
    all_cases = [_case_json(spec, i) for i, spec in enumerate(PROFILES)]
    (out_dir / f"atk_{OP_NAME}.json").write_text(
        json.dumps(all_cases, indent=1, ensure_ascii=False) + "\n"
    )
    # mss: GVA, tail, partial pack, V=256, V=256+GVA+tail
    mss_idx = [0, 4, 10, 44, 46, 52, 6, 100]
    mss = [all_cases[i] for i in mss_idx if i < len(all_cases)]
    (out_dir / f"atk_{OP_NAME}_mss.json").write_text(
        json.dumps(mss, indent=1, ensure_ascii=False) + "\n"
    )
    perf_idx = [i for i, s in enumerate(PROFILES) if s["T"] >= 256][:6]
    if not perf_idx:
        perf_idx = [0, 1]
    perf = [all_cases[i] for i in perf_idx]
    (out_dir / f"atk_{OP_NAME}_perf.json").write_text(
        json.dumps(perf, indent=1, ensure_ascii=False) + "\n"
    )


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register(f"generator_{OP_NAME}")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            spec = _spec(index)
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = f"{OP_NAME}_{index:04d}_{spec.get('name', 'case')}"
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    cfg.dtype = _dtype(spec.get("dtype", "bf16"))
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config


if __name__ == "__main__":
    dump_json_files()
    print(f"wrote {len(PROFILES)} profiles")
