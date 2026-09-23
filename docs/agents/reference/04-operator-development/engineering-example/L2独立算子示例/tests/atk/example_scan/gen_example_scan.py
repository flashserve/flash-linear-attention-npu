"""example_scan 的 ATK 泛化用例生成器（示例）。

注意事项：
  1. 本文件只生成**精度候选用例**（atk_example_scan.json 的候选版）；`_perf.json` 按用户模型 case
     手工建立，`_mss.json` 按全部可达 TilingKey 手工建立，都不由本文件产出。
  2. PROFILES 是"结构参数"集合：dtype/layout/shape/chunk_size/档位固定，只有随机种子随 case 变化；
     收窄 range、删 case 或用随机数据分布规避精度失败都是禁止的。
  3. 注册名 `generator_example_scan` 必须与 example_scan.yaml 的 `generate:` 字段一致。
  4. 没有 ATK 环境时（本地静态检查）允许 import 失败但不得吞掉其它模块的 ImportError。
  5. 生成后的用例要筛选/补充：每个可达逻辑分支（layout × dtype × 档位 × fixed/varlen × tail）
     至少一条最小用例，并把映射写进算子 ATK README。
"""

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

OP_NAME = "example_scan"
SEED_BASE = 20260923

# 结构参数组合：dtype × layout × 档位 × 是否带初始状态。
PROFILES = [
    {"name": "bf16_bsnd_none", "dtype": "bf16", "B": 1, "H": 2, "T": 64, "D": 128,
     "chunk_size": 64, "layout": "BSND", "keep_saved": False, "output_mode": 0, "has_initial_state": False},
    {"name": "fp16_bnsd_save", "dtype": "fp16", "B": 1, "H": 2, "T": 96, "D": 128,
     "chunk_size": 64, "layout": "BNSD", "keep_saved": True, "output_mode": 1, "has_initial_state": True},
    {"name": "bf16_bsnd_save_tail", "dtype": "bf16", "B": 1, "H": 1, "T": 65, "D": 128,
     "chunk_size": 64, "layout": "BSND", "keep_saved": True, "output_mode": 1, "has_initial_state": False},
    {"name": "fp16_bsnd_none_chunk128", "dtype": "fp16", "B": 1, "H": 2, "T": 128, "D": 128,
     "chunk_size": 128, "layout": "BSND", "keep_saved": False, "output_mode": 0, "has_initial_state": False},
]


def _dtype(dtype: str) -> str:
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _spec(index: int) -> dict:
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.setdefault("scale", 1.0)
    profile.setdefault("epsilon", 1e-06)
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

    @GENERATOR_REGISTRY.register("generator_example_scan")
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
