"""causal_conv1d 的 ATK 泛化用例生成器。"""

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

OP_NAME = "causal_conv1d"
PROFILES = [
  {
    "name": "bf16_small",
    "dtype": "bf16",
    "B": 1,
    "T": 8,
    "D": 16,
    "W": 4
  },
  {
    "name": "fp16_small",
    "dtype": "fp16",
    "B": 1,
    "T": 8,
    "D": 16,
    "W": 4
  }
]

def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")

def _spec(index):
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.update({"op": OP_NAME, "case_id": index, "seed": 20260817 + index, "route": "ascendc", "soc": "ascend910b"})
    return profile

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
