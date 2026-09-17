"""ChunkGatedDeltaRuleBwd composite ATK case generator."""

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


OP_NAME = "chunk_gated_delta_rule_bwd"
PROFILES = [
    {
        "name": "bnsd_g1_full",
        "B": 1, "HK": 2, "HV": 2, "T": 64, "layout": "BNSD",
    },
    {
        "name": "bsnd_g2_tail_state_dht",
        "B": 2, "HK": 2, "HV": 4, "T": 65, "layout": "BSND",
        "with_state": True, "with_dht": True,
    },
    {
        "name": "bnsd_g3_tail32_scalar_bf16",
        "B": 1, "HK": 1, "HV": 3, "T": 96, "layout": "BNSD",
        "scalar_dtype": "bf16",
    },
    {
        "name": "bsnd_g4_tail2_norm_beta",
        "B": 1, "HK": 1, "HV": 4, "T": 130, "layout": "BSND",
        "use_qk_l2norm": True, "use_beta_sigmoid": True,
    },
    {
        "name": "ntd_g2_varlen_state",
        "B": 1, "HK": 2, "HV": 4, "T": 129, "layout": "NTD",
        "seqlens": [65, 64], "with_state": True,
    },
    {
        "name": "tnd_g3_varlen_dht_vfirst",
        "B": 1, "HK": 1, "HV": 3, "T": 97, "layout": "TND",
        "seqlens": [33, 64], "with_state": True, "with_dht": True,
        "state_v_first": True,
    },
    {
        "name": "bnsd_g2_two_chunks_reserved",
        "B": 1, "HK": 2, "HV": 4, "T": 128, "layout": "BNSD",
        "reserved_inputs": True,
    },
    {
        "name": "bsnd_g1_short_norm",
        "B": 1, "HK": 4, "HV": 4, "T": 16, "layout": "BSND",
        "use_qk_l2norm": True,
    },
    {
        "name": "bnsd_g4_multichunk_dht",
        "B": 1, "HK": 2, "HV": 8, "T": 192, "layout": "BNSD",
        "with_dht": True,
    },
    {
        "name": "bsnd_g3_batch2_beta_bf16",
        "B": 2, "HK": 2, "HV": 6, "T": 64, "layout": "BSND",
        "scalar_dtype": "bf16", "use_beta_sigmoid": True,
    },
    {
        "name": "ntd_g1_varlen_norm_beta",
        "B": 1, "HK": 2, "HV": 2, "T": 160, "layout": "NTD",
        "seqlens": [32, 64, 64], "use_qk_l2norm": True,
        "use_beta_sigmoid": True,
    },
    {
        "name": "tnd_g4_varlen_tail_vfirst_reserved",
        "B": 1, "HK": 1, "HV": 4, "T": 131, "layout": "TND",
        "seqlens": [1, 65, 65], "with_state": True,
        "state_v_first": True, "reserved_inputs": True,
    },
]


def _spec(index: int) -> dict:
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.update(
        {
            "op": OP_NAME,
            "case_id": index,
            "seed": 20260911 + index,
            "route": "ascendc",
            "soc": "ascend950",
            "dtype": "bf16",
            "scalar_dtype": profile.get("scalar_dtype", "fp32"),
            "K": 128,
            "V": 128,
            "chunk_size": 64,
            "use_exp2": index % 2 == 0,
        }
    )
    return profile


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_chunk_gated_delta_rule_bwd")
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
                if cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config
