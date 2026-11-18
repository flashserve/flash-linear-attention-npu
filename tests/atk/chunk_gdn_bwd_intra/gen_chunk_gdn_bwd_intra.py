"""chunk_gdn_bwd_intra 的泛化用例生成器。"""

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


OP_NAME = "chunk_gdn_bwd_intra"
PROFILES = [
    {
        "name": "fixed_bf16_g1_batch2_partial_cg",
        "dtype": "bf16",
        "g_dtype": "bf16",
        "beta_dtype": "bf16",
        "B": 2,
        "HK": 5,
        "HV": 5,
        "T": 64,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": True,
    },
    {
        "name": "fixed_bf16_g2_tail1_exp",
        "dtype": "bf16",
        "g_dtype": "bf16",
        "beta_dtype": "fp32",
        "B": 1,
        "HK": 3,
        "HV": 6,
        "T": 65,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": False,
    },
    {
        "name": "fixed_bf16_g3_tail32",
        "dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "bf16",
        "B": 1,
        "HK": 2,
        "HV": 6,
        "T": 96,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": True,
    },
    {
        "name": "fixed_bf16_g4_tail33_exp",
        "dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "fp32",
        "B": 1,
        "HK": 2,
        "HV": 8,
        "T": 97,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "scale": 0.03125,
        "use_exp2": False,
    },
    {
        "name": "fixed_fp16_g4_multichunk",
        "dtype": "fp16",
        "g_dtype": "bf16",
        "beta_dtype": "bf16",
        "B": 1,
        "HK": 1,
        "HV": 4,
        "T": 128,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": True,
    },
    {
        "name": "fixed_fp16_g3_tail1_exp",
        "dtype": "fp16",
        "g_dtype": "bf16",
        "beta_dtype": "fp32",
        "B": 1,
        "HK": 1,
        "HV": 3,
        "T": 65,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": False,
    },
    {
        "name": "fixed_fp16_g2_tail32",
        "dtype": "fp16",
        "g_dtype": "fp32",
        "beta_dtype": "bf16",
        "B": 1,
        "HK": 2,
        "HV": 4,
        "T": 96,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": True,
    },
    {
        "name": "fixed_fp16_g1_tail33_exp",
        "dtype": "fp16",
        "g_dtype": "fp32",
        "beta_dtype": "fp32",
        "B": 1,
        "HK": 4,
        "HV": 4,
        "T": 97,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": False,
    },
    {
        "name": "varlen_bf16_g4_empty_tail1_33",
        "dtype": "bf16",
        "g_dtype": "bf16",
        "beta_dtype": "bf16",
        "B": 1,
        "HK": 1,
        "HV": 4,
        "T": 98,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "cu_seqlens": [0, 0, 1, 65, 98],
        "use_exp2": True,
    },
    {
        "name": "varlen_bf16_g3_tail32_exp",
        "dtype": "bf16",
        "g_dtype": "bf16",
        "beta_dtype": "fp32",
        "B": 1,
        "HK": 1,
        "HV": 3,
        "T": 96,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "cu_seqlens": [0, 32, 96],
        "use_exp2": False,
    },
    {
        "name": "varlen_bf16_g2_multichunk_tail1",
        "dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "bf16",
        "B": 1,
        "HK": 2,
        "HV": 4,
        "T": 129,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "cu_seqlens": [0, 65, 129],
        "use_exp2": True,
    },
    {
        "name": "varlen_bf16_g1_tail33_exp",
        "dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "fp32",
        "B": 1,
        "HK": 5,
        "HV": 5,
        "T": 161,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "cu_seqlens": [0, 33, 97, 161],
        "use_exp2": False,
    },
    {
        "name": "varlen_fp16_g1_full_tail1",
        "dtype": "fp16",
        "g_dtype": "bf16",
        "beta_dtype": "bf16",
        "B": 1,
        "HK": 4,
        "HV": 4,
        "T": 65,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "cu_seqlens": [0, 64, 65],
        "use_exp2": True,
    },
    {
        "name": "varlen_fp16_g2_tail32_exp",
        "dtype": "fp16",
        "g_dtype": "bf16",
        "beta_dtype": "fp32",
        "B": 1,
        "HK": 2,
        "HV": 4,
        "T": 160,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "cu_seqlens": [0, 32, 96, 160],
        "use_exp2": False,
    },
    {
        "name": "varlen_fp16_g3_tail33",
        "dtype": "fp16",
        "g_dtype": "fp32",
        "beta_dtype": "bf16",
        "B": 1,
        "HK": 2,
        "HV": 6,
        "T": 97,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "cu_seqlens": [0, 33, 97],
        "use_exp2": True,
    },
    {
        "name": "varlen_fp16_g4_multichunk_tail32_exp",
        "dtype": "fp16",
        "g_dtype": "fp32",
        "beta_dtype": "fp32",
        "B": 1,
        "HK": 2,
        "HV": 8,
        "T": 224,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "cu_seqlens": [0, 128, 224],
        "scale": 0.125,
        "use_exp2": False,
    },
    {
        "name": "fixed_bf16_g1_cross_work_tail1",
        "dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "fp32",
        "B": 1,
        "HK": 1,
        "HV": 1,
        "T": 1793,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": True,
    },
    {
        "name": "varlen_fp16_g1_cross_work_tail1_exp",
        "dtype": "fp16",
        "g_dtype": "bf16",
        "beta_dtype": "fp32",
        "B": 1,
        "HK": 1,
        "HV": 1,
        "T": 1793,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "cu_seqlens": [0, 1792, 1793],
        "use_exp2": False,
    },
]




# A2/A3（DAV_2201）运行时 G 桶（G_KEY=0）专用 profile。
# 与主 PROFILES 分离：主 yaml 只消费 950 profile，
# chunk_gdn_bwd_intra_a2.yaml 经 A2 生成器消费本列表，
# 二进制快照由 main() 写入 A2 用例集共用的 atk_chunk_gdn_bwd_intra_arch22.json。
# A3（ascend910_93）与 A2 同为 DAV_2201，共用同一构建产物，不再单独维护
# ascend910_93 用例；如需 910_93 复核，可原样在对应机器执行同一快照。
ARCH22_PROFILES = [
    {
        "name": "arch22_fixed_bf16_g5_runtime",
        "dtype": "bf16",
        "g_dtype": "bf16",
        "beta_dtype": "bf16",
        "B": 1,
        "HK": 2,
        "HV": 10,
        "T": 64,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": True,
        "soc": "ascend910b",
    },
    {
        "name": "arch22_fixed_fp16_g6_cross_slice_runtime",
        "dtype": "fp16",
        "g_dtype": "fp32",
        "beta_dtype": "bf16",
        "B": 1,
        "HK": 1,
        "HV": 6,
        "T": 65,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": False,
        "soc": "ascend910b",
    },
    {
        "name": "arch22_fixed_bf16_g8_partial_slice_runtime",
        "dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "fp32",
        "B": 2,
        "HK": 2,
        "HV": 16,
        "T": 96,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "use_exp2": True,
        "soc": "ascend910b",
    },
    {
        "name": "arch22_varlen_fp16_g16_runtime_tail33",
        "dtype": "fp16",
        "g_dtype": "bf16",
        "beta_dtype": "fp32",
        "B": 1,
        "HK": 1,
        "HV": 16,
        "T": 97,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "cu_seqlens": [0, 33, 97],
        "use_exp2": True,
        "soc": "ascend910b",
    },
    {
        "name": "arch22_varlen_bf16_g12_runtime_multichunk",
        "dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "bf16",
        "B": 1,
        "HK": 1,
        "HV": 12,
        "T": 129,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "cu_seqlens": [0, 65, 129],
        "use_exp2": False,
        "soc": "ascend910b",
    }]


def _spec(index: int) -> dict:
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.update(
        {
            "op": OP_NAME,
            "case_id": index,
            "seed": 20260904 + index,
            "route": "ascendc",
            "soc": "ascend950",
        }
    )
    return profile


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_chunk_gdn_bwd_intra")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            spec = _spec(index)
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = f"{OP_NAME}_{index:04d}_{spec['name']}"
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    cfg.dtype = spec["dtype"]
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config


def _arch22_spec(index: int, soc: str = "ascend910b") -> dict:
    profile = deepcopy(ARCH22_PROFILES[index % len(ARCH22_PROFILES)])
    profile.pop("soc", None)
    profile.update(
        {
            "op": OP_NAME,
            "case_id": index,
            "seed": 20260904 + index,
            "route": "ascendc",
            "soc": soc,
        }
    )
    return profile


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_chunk_gdn_bwd_intra_a2")
    class GeneratorA2(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            spec = _arch22_spec(index)
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = f"{OP_NAME}_arch22_{index:04d}_{spec['name']}"
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    cfg.dtype = spec["dtype"]
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config


def _case_payload(index: int, spec: dict, template: dict, save_name: str) -> dict:
    """基于现有 950 快照的 case 结构派生 A2/A3 case，保持字段布局一致。"""
    case = deepcopy(template)
    case["id"] = index
    case["default_seed"] = spec["seed"]
    case["name"] = f"aclnn.{OP_NAME.replace('_', '.')}"
    case["api_type"] = "executor_chunk_gdn_bwd_intra"
    case["save_name"] = save_name
    for item in case["inputs"]:
        if isinstance(item, list):
            continue
        if item["name"] == "low_precision_marker":
            item["dtype"] = spec["dtype"]
            item["range_values"] = [0, 0]
        elif item["name"] == "case_spec":
            item["range_values"] = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
        elif item["name"] in spec:
            item["range_values"] = spec[item["name"]]
    return case


def main():
    """写 A2 运行时 G 桶快照，供 atk node 直接消费。

    A2/A3（DAV_2201）共用同一构建产物，快照
    `atk_chunk_gdn_bwd_intra_arch22.json` 只维护 5 条 ascend910b 用例
    （G=5/6/8/12/16）；A3 不再单独设置 ascend910_93 用例，需要复核时
    在对应 SoC 机器上执行同一快照即可。
    """
    folder = Path(__file__).resolve().parent
    base = folder / "atk_chunk_gdn_bwd_intra.json"
    if not base.exists():
        raise SystemExit(f"missing template snapshot: {base}")
    template = json.loads(base.read_text(encoding="utf-8"))[0]
    # 精度标准沿用主 json 的 mixed_tolerance_bm，与仓上本算子的验收口径
    # （主 yaml 声明）保持一致。
    cases = [
        _case_payload(
            index,
            _arch22_spec(index),
            template,
            f"{OP_NAME}_arch22_{index:04d}_{_arch22_spec(index)['name']}",
        )
        for index in range(len(ARCH22_PROFILES))
    ]
    out = folder / "atk_chunk_gdn_bwd_intra_arch22.json"
    out.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(cases)} arch22 runtime-G cases -> {out.name}")


if __name__ == "__main__":
    main()
