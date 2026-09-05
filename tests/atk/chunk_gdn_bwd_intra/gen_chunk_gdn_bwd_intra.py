"""chunk_gdn_bwd_intra 的逐 Stage 泛化用例生成器。"""

from __future__ import annotations

import json
from copy import deepcopy

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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
        "stage": 2,
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
]


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
