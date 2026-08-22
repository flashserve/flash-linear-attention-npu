# Copyright (c) Tianjin University, Ltd. 2026. All rights reserved.
"""solve_tri 的 ATK 泛化用例生成器。

生成 400 个泛化用例：200 个 chunk_size=64 (FP32 路径) + 200 个 chunk_size=128 (Cube 路径)。
"""

import json
import random
from copy import deepcopy

from atk.case_generator.generator.base_generator import CaseGenerator
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.configs.case_config import CaseConfig

OP_NAME = "solve_tri"


def _generate_profiles(seed: int = 20260817):
    """生成 400 个泛化用例配置。
    
    - 前 200 个: chunk_size=64，走 FP32 路径
    - 后 200 个: chunk_size=128，走 Cube 低精度路径
    
    覆盖：
    - layout: bsnd, tnd
    - dtype: bf16, fp16
    - B: 1-16
    - H: 1-32 (受硬件约束 H * chunk_size * 16 + 16 <= 65536)
    - T: 64-4096
    - num_seqs: 1-8 (仅 tnd 模式)
    """
    random.seed(seed)
    profiles = []
    
    layouts = ["bsnd", "tnd"]
    dtypes = ["bf16", "fp16"]
    B_values = [1, 2, 4, 8, 16]
    H_values = [1, 2, 4, 8, 16, 32]
    T_values = [64, 128, 256, 512, 1024, 2048, 4096]
    num_seqs_values = [1, 2, 4, 8]
    
    for chunk_size in [64, 128]:
        for _ in range(200):
            layout = random.choice(layouts)
            dtype = random.choice(dtypes)
            H = random.choice(H_values)
            
            # 硬件约束: H * chunk_size * 16 + 16 <= 65536
            while H * chunk_size * 16 + 16 > 65536:
                H = random.choice(H_values)
            
            if layout == "bsnd":
                B = random.choice(B_values)
                T = random.choice(T_values)
                if T < chunk_size:
                    T = chunk_size
                num_seqs = 1
                name = f"{dtype}_{layout}_B{B}_H{H}_T{T}_C{chunk_size}"
            else:
                num_seqs = random.choice(num_seqs_values)
                min_seq_len = chunk_size
                max_seq_len = random.choice([256, 512, 1024, 2048])
                T = sum(random.randint(min_seq_len, max_seq_len) for _ in range(num_seqs))
                B = 1
                name = f"{dtype}_{layout}_NS{num_seqs}_H{H}_T{T}_C{chunk_size}"
            
            profile = {
                "name": name,
                "dtype": dtype,
                "B": B,
                "H": H,
                "T": T,
                "chunk_size": chunk_size,
                "layout": layout,
                "num_seqs": num_seqs,
            }
            profiles.append(profile)
    
    # 追加边界 case
    boundary_cases = [
        # chunk_size=64 边界: H=32 (32*64*16+16=32784, 合法)
        {"dtype": "fp16", "B": 1, "H": 32, "T": 256, "chunk_size": 64, "layout": "bsnd", "num_seqs": 1,
         "name": "fp16_bsnd_boundary_H32_C64_T256"},
        {"dtype": "bf16", "B": 1, "H": 32, "T": 512, "chunk_size": 64, "layout": "tnd", "num_seqs": 2,
         "name": "bf16_tnd_boundary_H32_C64_NS2_T512"},
        # chunk_size=128 边界: H=16 (16*128*16+16=32784, 合法最大 H)
        {"dtype": "fp16", "B": 1, "H": 16, "T": 512, "chunk_size": 128, "layout": "bsnd", "num_seqs": 1,
         "name": "fp16_bsnd_boundary_H16_C128_T512"},
        {"dtype": "bf16", "B": 1, "H": 16, "T": 1024, "chunk_size": 128, "layout": "tnd", "num_seqs": 4,
         "name": "bf16_tnd_boundary_H16_C128_NS4_T1024"},
    ]
    profiles.extend(boundary_cases)
    
    return profiles

PROFILES = _generate_profiles(seed=20260817)


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _spec(index):
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.update({
        "op": OP_NAME,
        "case_id": index,
        "seed": 20260817 + index,
        "route": "ascendc",
        "soc": "ascend910b"
    })
    return profile


@GENERATOR_REGISTRY.register("generator_solve_tri")
class SolveTriGenerator(CaseGenerator):
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
