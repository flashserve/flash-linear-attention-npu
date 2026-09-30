"""fused_recurrent_rwkv8 的 ATK 泛化用例生成器。

用例集为 200 条定长清单，由 `_cases()` 确定性构造，分三个来源：

1. **模板对齐代表用例**（38 条）：与转测文档 7.1~7.6 各表逐行对应，用于跨轮对比。
   覆盖 B1H4T4096 三 dtype × 四组 (K,V)、B64/B128 多核、512-block、chunk_len 对照、
   scale 对照、8 种 flags 组合。
2. **边界与语义用例**（22 条）：T ∈ {1,4,8,15,16,17} 的快照边角（含 T < chunk_len 的
   零快照）、K/V ∈ {8,96,120} 的非 64 倍数取值、K≠V 极端组合、单 block 与 >40 核排队。
3. **系统性网格**（140 条）：dtype × (K,V) × 8 flags × chunk_len {8,16} × scale {1.0,0.125}
   × (B,T) 组合的确定性轮转，保证 TilingData 特征空间（本算子 tilingKey 恒 0，
   差异全走 TilingData 标志位与形状字段）全覆盖。

seed 逐 case 唯一（SEED_BASE + index），满足"每个精度用例固定种子"的要求。
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

OP_NAME = "fused_recurrent_rwkv8"
N_CASES = 200
SEED_BASE = 20260918

DTYPES = ("fp32", "fp16", "bf16")
KVS = ((64, 64), (64, 128), (128, 64), (128, 128))
FLAGS = tuple((bool(f & 1), bool(f & 2), bool(f & 4)) for f in range(8))  # init / s / sa


def _c(dtype, B, H, T, K, V, scale=1.0, chunk_len=16, init=False, s=False, sa=False, tag=""):
    return {
        "dtype": dtype, "B": B, "H": H, "T": T, "K": K, "V": V,
        "scale": scale, "chunk_len": chunk_len,
        "initial_state": init, "output_chunk_state": s, "output_sa": sa,
        "tag": tag,
    }


def _template_cases():
    """转测文档 7.1~7.6 的代表 shape，逐行对应。"""
    out = []
    # 7.2 dtype × (K,V)，固定 B1 H4 T4096，flags 全关，chunk_len 16
    for dt, K, V, sc in (
        ("fp32", 64, 64, 1.0), ("fp32", 64, 128, 1.0),
        ("fp32", 128, 64, 1.0), ("fp32", 128, 128, 0.125),
        ("fp16", 64, 64, 1.0), ("fp16", 64, 128, 1.0), ("fp16", 128, 128, 1.0),
        ("bf16", 128, 64, 1.0), ("bf16", 128, 128, 1.0),
    ):
        out.append(_c(dt, 1, 4, 4096, K, V, scale=sc, tag="t_7_2"))
    # 7.1 典型场景（长序列 K128V128 / 典型训练 / decode / 512-block）
    for dt, B, H, T, K, V, tag in (
        ("fp16", 4, 4, 1024, 128, 128, "t_7_1_train"),
        ("fp16", 64, 2, 1024, 128, 128, "t_7_1_bigbatch"),
        ("fp16", 1, 4, 64, 64, 64, "t_7_1_decode"),
        ("fp16", 128, 4, 64, 64, 64, "t_7_1_512block"),
    ):
        out.append(_c(dt, B, H, T, K, V, tag=tag))
    # 7.3 flags 8 组合，固定 fp16 B4 H4 T64 K128 V64
    for init, s, sa in FLAGS:
        out.append(_c("fp16", 4, 4, 64, 128, 64, init=init, s=s, sa=sa, tag="t_7_3_flags"))
    # 7.4 chunk_len 8 vs 16 对照
    for dt, B, H, T, K, V in (
        ("fp32", 1, 4, 2048, 128, 128),
        ("fp16", 4, 4, 1024, 128, 128),
        ("fp32", 64, 2, 1024, 64, 64),
        ("fp16", 4, 4, 4096, 64, 128),
    ):
        for cl in (8, 16):
            out.append(_c(dt, B, H, T, K, V, chunk_len=cl, init=True, s=True, sa=True,
                          tag="t_7_4_chunk"))
    # 7.5 scale 1.0 vs 0.125 对照
    for dt, B, H, T, K, V in (
        ("fp32", 4, 4, 2048, 128, 128),
        ("bf16", 64, 2, 2048, 64, 64),
        ("bf16", 1, 4, 64, 128, 128),
    ):
        for sc in (1.0, 0.125):
            out.append(_c(dt, B, H, T, K, V, scale=sc, tag="t_7_5_scale"))
    # 7.6 多核扩展，固定 fp32 T1024 K64 V64
    for B, H in ((1, 4), (64, 2), (128, 2)):
        out.append(_c("fp32", B, H, 1024, 64, 64, tag="t_7_6_multicore"))
    return out


def _boundary_cases():
    """快照边角、非 64 倍数 K/V、极端 K≠V、多核排队边界。"""
    return [
        _c("fp32", 1, 2, 4, 64, 64, s=True, sa=True, tag="b_t4_zerosnap"),
        _c("fp32", 1, 2, 8, 64, 64, s=True, sa=True, tag="b_t_lt_chunk_zerosnap"),
        _c("fp32", 1, 2, 15, 64, 64, s=True, sa=True, tag="b_t_chunk_minus1"),
        _c("fp32", 1, 2, 16, 64, 64, s=True, sa=True, tag="b_t_eq_chunk_1snap"),
        _c("fp32", 1, 2, 17, 64, 64, s=True, sa=True, tag="b_t_chunk_plus1"),
        _c("fp32", 2, 4, 33, 64, 64, init=True, tag="b_t33_tail"),
        _c("fp32", 1, 1, 1, 64, 64, tag="b_decode_t1"),
        _c("fp32", 1, 2, 16, 8, 8, tag="b_kv8_min"),
        _c("fp32", 1, 2, 16, 8, 128, tag="b_k8_v128"),
        _c("fp32", 1, 2, 16, 128, 8, tag="b_k128_v8"),
        _c("fp32", 1, 2, 16, 96, 96, tag="b_kv96"),
        _c("fp32", 1, 2, 16, 64, 96, tag="b_k64_v96"),
        _c("fp32", 1, 2, 16, 96, 64, tag="b_k96_v64"),
        _c("fp32", 1, 2, 16, 120, 120, tag="b_kv120"),
        _c("fp32", 1, 2, 16, 32, 128, tag="b_k32_v128"),
        _c("fp32", 1, 2, 16, 128, 32, tag="b_k128_v32"),
        _c("fp32", 1, 1, 64, 64, 64, tag="b_1block"),
        _c("fp32", 1, 8, 64, 64, 64, tag="b_h8"),
        _c("fp32", 1, 32, 64, 64, 64, tag="b_h32_overqueue"),
        _c("fp16", 128, 4, 4, 64, 64, tag="b_512block_min_t"),
        _c("bf16", 64, 1, 64, 64, 64, tag="b_b64_h1"),
        _c("bf16", 4, 1, 2048, 64, 128, tag="b_b4_h1_t2048"),
    ]


BT_POOL = (
    (1, 2, 4), (1, 2, 64), (1, 2, 1024), (1, 4, 4096),
    (4, 2, 64), (4, 4, 1024), (4, 4, 2048), (4, 2, 4096),
    (64, 1, 4), (64, 2, 64), (64, 2, 1024), (64, 1, 2048),
    (128, 2, 4), (128, 1, 64), (128, 2, 1024), (128, 4, 4),
    (16, 2, 256),
)


def _grid_cases():
    """系统性网格：flags × dtype、dtype × (K,V) 各正交换二维展开。"""
    out = []
    # flags × dtype（8 × 3）@ B1 H2 T64 K64 V64
    for init, s, sa in FLAGS:
        for dt in DTYPES:
            out.append(_c(dt, 1, 2, 64, 64, 64, chunk_len=8 if sa else 16,
                          init=init, s=s, sa=sa, tag="g_flags_dtype"))
    # dtype × (K,V) 四组合，四种配置轮转
    for dt in DTYPES:
        for K, V in KVS:
            out.append(_c(dt, 2, 4, 64, K, V, tag="g_kv_default"))
    for dt in DTYPES:
        for K, V in KVS:
            out.append(_c(dt, 2, 4, 64, K, V, chunk_len=8, init=True, s=True, sa=True,
                          tag="g_kv_chunk8_allflags"))
    for dt in DTYPES:
        for K, V in KVS:
            out.append(_c(dt, 4, 2, 128, K, V, scale=0.125, tag="g_kv_scale"))
    # chunk_len × flags @ fp16 B2 H4 T64 K64 V128
    for cl in (8, 16):
        for init, s, sa in FLAGS:
            out.append(_c("fp16", 2, 4, 64, 64, 128, chunk_len=cl,
                          init=init, s=s, sa=sa, tag="g_chunk_flags"))
    # (B, T) 17 组，dtype 与 (K,V) 轮转
    for i, (B, H, T) in enumerate(BT_POOL):
        dt = DTYPES[i % 3]
        K, V = KVS[i % 4]
        out.append(_c(dt, B, H, T, K, V, chunk_len=8 if i % 2 else 16,
                      init=bool(i & 1), s=bool(i & 2), sa=bool(i & 4), tag="g_bt_pool"))
    return out


def _fill_cases(n):
    """补足到 n 条：dtype × (K,V) × flags × chunk_len × scale 的确定性轮转。"""
    out = []
    i = 0
    while len(out) < n:
        dt = DTYPES[i % 3]
        K, V = KVS[(i // 3) % 4]
        init, s, sa = FLAGS[(i // 12) % 8]
        chunk_len = 8 if ((i // 96) % 2) else 16
        scale = 0.125 if ((i // 192) % 2) else 1.0
        B, H, T = (2, 4, 128) if (i % 2) else (1, 2, 64)
        out.append(_c(dt, B, H, T, K, V, scale=scale, chunk_len=chunk_len,
                      init=init, s=s, sa=sa, tag="g_fill"))
        i += 1
    return out


def _dedup_key(c):
    return (c["dtype"], c["B"], c["H"], c["T"], c["K"], c["V"], c["scale"],
            c["chunk_len"], c["initial_state"], c["output_chunk_state"], c["output_sa"])


_CACHE = None


def _cases():
    """200 条定长用例清单（确定性）。"""
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    cases = []
    seen = set()
    for c in _template_cases() + _boundary_cases() + _grid_cases():
        k = _dedup_key(c)
        if k in seen:
            continue
        seen.add(k)
        cases.append(c)
    for c in _fill_cases(N_CASES):
        if len(cases) >= N_CASES:
            break
        k = _dedup_key(c)
        if k in seen:
            continue
        seen.add(k)
        cases.append(c)
    if len(cases) != N_CASES:
        raise AssertionError(f"用例数 {len(cases)} != {N_CASES}")
    _CACHE = cases
    return cases


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _spec(index):
    """按 index 取用例并派生 ATK case 字段。"""
    case = deepcopy(_cases()[index])
    tag = case.pop("tag", "case")
    spec = {
        "name": f"{tag}_{index:04d}",
        "dtype": case["dtype"],
        "B": case["B"], "H": case["H"], "T": case["T"],
        "K": case["K"], "V": case["V"],
        "scale": case["scale"], "chunk_len": case["chunk_len"],
        "initial_state": case["initial_state"],
        "output_chunk_state": case["output_chunk_state"],
        "output_sa": case["output_sa"],
        "seed": SEED_BASE + index,
    }
    spec.update({"op": OP_NAME, "case_id": index, "route": "ascendc", "soc": "ascend910b"})
    return spec


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
