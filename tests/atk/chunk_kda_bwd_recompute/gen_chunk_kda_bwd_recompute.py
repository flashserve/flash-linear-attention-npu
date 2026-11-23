"""chunk_kda_bwd_recompute 的 ATK 泛化用例生成器。"""

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
    from types import SimpleNamespace
    CaseGenerator = object
    GENERATOR_REGISTRY = SimpleNamespace(register=lambda name: lambda cls: cls)
    CaseConfig = None

OP_NAME = "chunk_kda_bwd_recompute"
def _profiles():
    result = []
    lengths = [1, 15, 16, 17, 63, 64, 65, 127, 129]
    for ratio in range(1, 17):
        for variable in (False, True):
            for combo in range(4):
                i = len(result)
                t = lengths[(ratio + combo) % len(lengths)]
                hk = (1, 2, 3)[(ratio + combo) % 3]
                spec = dict(name=f'g{ratio}_{"var" if variable else "fixed"}_{combo}',
                    dtype='bf16', B=1 if variable else 1 + (i % 5 == 0),
                    HK=hk, HV=hk*ratio, T=t, K=128, V=128, chunk_size=64,
                    g_dtype=('bf16','fp32')[combo//2], beta_dtype=('bf16','fp32')[combo%2],
                    use_gate=True, use_exp2=True, has_dt_bias=True, lower_bound=-5.0,
                    variable=variable)
                if variable:
                    spec['lengths']=[0,t//3,0,t-t//3]
                result.append(spec)
    for extra in range(72):
        spec = deepcopy(result[(extra*17)%128])
        spec['name'] += f'_edge{extra}'
        spec['use_gate'] = extra % 3 != 0
        spec['use_exp2'] = extra % 2 == 0
        spec['has_dt_bias'] = extra % 4 != 0
        spec['lower_bound'] = (-5.0,-1.0,-0.01)[extra%3]
        # Force multi-page reuse without large head counts or model-size tensors.
        if extra >= 64:
            spec.update(B=1,HK=3,HV=3*(extra-63),T=1025,variable=False)
            spec.pop('lengths',None)
        result.append(spec)
    return result

PROFILES = _profiles()


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _spec(index):
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    if str(profile.get("dtype", "bf16")).lower() == "fp16":
        profile["dtype"] = "bf16"
        profile["name"] = str(profile.get("name", "fp16_small")).replace("fp16", "bf16")
    profile.update(
        {
            "op": OP_NAME,
            "case_id": index,
            "seed": 20260817 + index,
            "route": "ascendc",
            "soc": "ascend910b",
        }
    )
    return profile



@GENERATOR_REGISTRY.register("generator_chunk_kda_bwd_recompute")
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
