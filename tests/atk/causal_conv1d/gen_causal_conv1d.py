"""causal_conv1d 的 ATK 泛化用例生成器。"""

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

OP_NAME = "causal_conv1d"

def build_profiles():
    """在内存中动态构建所有的测试组合 Profile。"""
    profiles = []
    dtypes = ["fp16", "bf16"]
    kernels = [2, 3, 4]
    dims = [16, 64, 256, 512, 1024, 2048]
    activations = ["none", "silu", "swish"]
    batch_sizes = [1, 2, 4, 7]
    seq_lens_fn = [8, 64, 256, 1024]
    head_nums = [0, 2, 4, 16]

    case_count = 0

    # === 场景 A: runMode = 0 (FN / prefill) 组合 ===
    for dt in dtypes:
        for w in kernels:
            for d in dims:
                for act in activations:
                    for b in batch_sizes:
                        for t in seq_lens_fn:
                            for h in head_nums:
                                if h > 0 and (d % h != 0 or (d // h) % 16 != 0):
                                    continue
                                
                                has_init = (case_count % 2 == 1)
                                has_bias = (case_count % 3 != 0)

                                profiles.append({
                                    "name": f"fn_{dt}_w{w}_d{d}_{act}_b{b}_t{t}_h{h}",
                                    "run_mode": 0,
                                    "dtype": dt,
                                    "B": b,
                                    "T": t,
                                    "D": d,
                                    "W": w,
                                    "activation": act,
                                    "has_bias": has_bias,
                                    "head_num": h,
                                    "is_varlen": False,
                                    "max_query_len": -1,
                                    "has_initial_state": has_init,
                                    "op": OP_NAME,
                                    "route": "ascendc",
                                    "soc": "ascend960dt",
                                })
                                case_count += 1

    # === 场景 B: runMode = 1 (UPDATE / decode) 组合 ===
    for dt in dtypes:
        for w in kernels:
            for d in dims:
                for act in activations:
                    for b in batch_sizes:
                        has_bias = (case_count % 2 == 0)

                        profiles.append({
                            "name": f"update_{dt}_w{w}_d{d}_{act}_b{b}_step1",
                            "run_mode": 1,
                            "dtype": dt,
                            "B": b,
                            "T": 1,
                            "D": d,
                            "W": w,
                            "activation": act,
                            "has_bias": has_bias,
                            "head_num": 0,
                            "is_varlen": False,
                            "max_query_len": -1,
                            "has_initial_state": False,
                            "op": OP_NAME,
                            "route": "ascendc",
                            "soc": "ascend960dt",
                        })
                        case_count += 1

                        for max_q in [3, 8]:
                            profiles.append({
                                "name": f"update_varlen_{dt}_w{w}_d{d}_{act}_b{b}_maxq{max_q}",
                                "run_mode": 1,
                                "dtype": dt,
                                "B": b,
                                "T": max_q * 2,
                                "D": d,
                                "W": w,
                                "activation": act,
                                "has_bias": True,
                                "head_num": 0,
                                "is_varlen": True,
                                "max_query_len": max_q,
                                "has_initial_state": False,
                                "op": OP_NAME,
                                "route": "ascendc",
                                "soc": "ascend960dt",
                            })
                            case_count += 1

    random.seed(20261008)
    random.shuffle(profiles)
    return profiles[:200]  # 只取前 200 个，你可以根据需要调整

# 在模块加载时就构建好 Profile 列表
PROFILES = build_profiles()

def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")

def _spec(index):
    # 动态获取对应的 profile 并补充框架需要的字段
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.update({"case_id": index, "seed": 20260817 + index})
    return profile

if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_causal_conv1d")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)
            # 告诉 ATK 引擎我们一共有多少个用例要生成
            self.total_case_num = len(PROFILES)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            # self.index 由 ATK 引擎在迭代调用时自动提供（通常从 1 开始）
            index = max(int(self.index) - 1, 0)
            
            # 安全检查，防止越界（虽然 ATK 正常情况下会根据 total_case_num 停止）
            if index >= len(PROFILES):
                return case_config
                
            spec = _spec(index)
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = f"{OP_NAME}_{index:04d}_{spec.get('name', 'case')}"
            
            # 动态填充 JSON 中原本定义的 inputs
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
    print(f"Generating ATK JSON for {len(PROFILES)} cases...")
    all_cases = []
    for i in range(len(PROFILES)):
        spec = _spec(i)
        case = {
            "id": i,
            "default_seed": spec["seed"],
            "name": f"causal_conv1d",
            "aclnn_name": "",
            "version": "v2.1",
            "api": "pytorch",
            "api_type": "executor_causal_conv1d",
            "expected_error_msg": None,
            "backward": False,
            "standard": {
                "acc": "mixed_tolerance_bm",
                "perf": "not_key"
            },
            "outputs": None,
            "inputs": [
                { "name": "low_precision_marker", "type": "tensor", "required": True, "dtype": _dtype(spec.get("dtype", "bf16")), "shape": [1], "range_values": [0,0], "backward": False },
                { "name": "fp32_marker", "type": "tensor", "required": True, "dtype": "fp32", "shape": [1], "range_values": [0,0], "backward": False },
                { "name": "case_spec", "type": "attr", "required": True, "dtype": "non_param", "shape": None, "range_values": json.dumps(spec, ensure_ascii=False, separators=(",", ":")), "backward": False },
                { "name": "dtype", "type": "attr", "required": True, "dtype": "string", "shape": None, "range_values": spec.get("dtype", "bf16"), "backward": False },
                { "name": "B", "type": "attr", "required": True, "dtype": "int", "shape": None, "range_values": spec.get("B", 1), "backward": False },
                { "name": "T", "type": "attr", "required": True, "dtype": "int", "shape": None, "range_values": spec.get("T", 8), "backward": False },
                { "name": "D", "type": "attr", "required": True, "dtype": "int", "shape": None, "range_values": spec.get("D", 16), "backward": False },
                { "name": "W", "type": "attr", "required": True, "dtype": "int", "shape": None, "range_values": spec.get("W", 4), "backward": False },
                { "name": "case_id", "type": "attr", "required": True, "dtype": "int", "shape": None, "range_values": i, "backward": False },
                { "name": "seed", "type": "attr", "required": True, "dtype": "int", "shape": None, "range_values": spec["seed"], "backward": False },
                { "name": "route", "type": "attr", "required": True, "dtype": "string", "shape": None, "range_values": "ascendc", "backward": False },
                { "name": "soc", "type": "attr", "required": True, "dtype": "string", "shape": None, "range_values": spec.get("soc", "ascend910b"), "backward": False }
            ]
        }
        all_cases.append(case)
        
    with open("atk_causal_conv1d.json", "w", encoding="utf-8") as f:
        json.dump(all_cases, f, indent=2)
    print(f"Successfully generated {len(all_cases)} cases and updated atk_causal_conv1d.json!")
