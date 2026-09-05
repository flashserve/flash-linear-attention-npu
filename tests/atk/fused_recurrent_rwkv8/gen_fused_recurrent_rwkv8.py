"""fused_recurrent_rwkv8 的 ATK 泛化用例生成器。"""

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

OP_NAME = "fused_recurrent_rwkv8"
PROFILES = [
  {
    "name": "fp32_main",
    "dtype": "fp32",
    "B": 2, "H": 4, "T": 64, "K": 64, "V": 64,
    "scale": 1.0, "chunk_len": 16,
    "initial_state": False,
    "output_chunk_state": False, "output_sa": False,
    "seed": 42
  },
  {
    "name": "fp16_chunk8",
    "dtype": "fp16",
    "B": 2, "H": 4, "T": 64, "K": 64, "V": 32,
    "scale": 1.0, "chunk_len": 8,
    "initial_state": True,
    "output_chunk_state": True, "output_sa": True,
    "seed": 51
  },
  {
    "name": "bf16_init_scale",
    "dtype": "bf16",
    "B": 1, "H": 2, "T": 16, "K": 64, "V": 64,
    "scale": 0.125, "chunk_len": 16,
    "initial_state": True,
    "output_chunk_state": False, "output_sa": False,
    "seed": 43
  }
]

def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")

# 泛化变体池：以 PROFILES 为模板，按 index 确定性派生参数。
# 前 48 个 index 穷举完整的标志位网格（3 dtype × 2 初态 × 2 s × 2 sa × 2 chunk_len），
# 保证 TilingData 特征空间全覆盖（本算子 tilingKey 恒 0，差异全走 TilingData 标志位）；
# 48 之后网格再循环，但 T/scale/seed 继续变化，保证每题输入数据唯一。
SEED_BASE = 20260825
T_POOL = (16, 33, 64, 128)
SCALE_POOL = (1.0, 0.125)
COMBO_GRID = 3 * 2 * 2 * 2 * 2  # 48

def _spec(index):
    combo = index % COMBO_GRID
    profile = deepcopy(PROFILES[combo % len(PROFILES)])  # dtype/B/H/K/V 骨架
    flags = combo // len(PROFILES)                       # 0..15
    profile["initial_state"] = bool(flags & 1)
    profile["output_chunk_state"] = bool(flags & 2)
    profile["output_sa"] = bool(flags & 4)
    profile["chunk_len"] = 16 if (flags & 8) else 8
    # 形状/缩放随总 index 变化，与标志位网格解耦
    profile["T"] = T_POOL[(index // COMBO_GRID + index) % len(T_POOL)]
    profile["scale"] = SCALE_POOL[(index // COMBO_GRID + index // 2) % len(SCALE_POOL)]
    if profile["T"] < profile["chunk_len"]:
        profile["chunk_len"] = 8
    profile["name"] = f"{profile['name']}_{index:04d}"
    profile.update({"op": OP_NAME, "case_id": index, "route": "ascendc", "soc": "ascend910b"})
    profile["seed"] = SEED_BASE + index
    return profile

if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_fused_recurrent_rwkv8")
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
