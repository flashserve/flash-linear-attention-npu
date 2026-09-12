"""causal_conv1d_bwd 的 ATK 泛化用例生成器。

生成确定性的泛化用例矩阵，覆盖 executor 当前支持的执行域：
- layout 固定为 BSND（逻辑 [B, T, D]），activation=0，不携带 initial_state / dht；
- dtype: bf16 / fp16；
- B: 1 / 2 / 4 / 8；
- T: 16 / 32 / 64 / 128 / 256，且恒有 T >= W；
- D: 16 / 32 / 64 / 128 / 256，均满足 BSND 下 D % 16 == 0；
- W: 2 / 3 / 4。

shape 组合经固定种子打乱后按 (shape, dtype) 展开，保证任意用例数量
（由 YAML dtype_numbers 或 -dt 控制）下 dtype 均衡、shape 空间均匀采样。
"""

from __future__ import annotations

import json
import random
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

OP_NAME = "causal_conv1d_bwd"
SEED_BASE = 20260817

DTYPES = ("bf16", "fp16")
B_VALUES = (1, 2, 4, 8)
T_VALUES = (16, 32, 64, 128, 256)
D_VALUES = (16, 32, 64, 128, 256)
W_VALUES = (2, 3, 4)


def _build_profiles():
    shape_combos = [
        (B, T, D, W)
        for B in B_VALUES
        for T in T_VALUES
        for D in D_VALUES
        for W in W_VALUES
        if T >= W
    ]
    random.Random(SEED_BASE).shuffle(shape_combos)
    profiles = []
    for B, T, D, W in shape_combos:
        for dtype in DTYPES:
            profiles.append(
                {
                    "name": f"{dtype}_bsnd_B{B}_T{T}_D{D}_W{W}",
                    "dtype": dtype,
                    "B": B,
                    "T": T,
                    "D": D,
                    "W": W,
                }
            )
    return profiles


PROFILES = _build_profiles()


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _spec(index):
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.update(
        {
            "op": OP_NAME,
            "case_id": index,
            "seed": SEED_BASE + index,
            "route": "ascendc",
            "soc": "ascend910b",
        }
    )
    return profile


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_causal_conv1d_bwd")
    class CausalConv1dBwdGenerator(CaseGenerator):
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
