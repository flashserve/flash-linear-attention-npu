"""chunk_local_cumsum 的 ATK 泛化用例生成器。"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

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

OP_NAME = "chunk_local_cumsum"
DTYPES = ("bf16", "fp16")
SCALES = (1.0, 0.5, 0.25)
_SEED_BASE = 20260817
_MAX_BHT = 512 * 1024


# (name, B, HV, HK, T, V, K, chunk_size)
_GDN_TABLE_33: list[tuple[Any, ...]] = [
    ("BSND_noGVA_V128_01", 64, 8, 8, 1024, 128, 128, 64),
    ("BSND_noGVA_V128_02", 32, 16, 16, 2048, 128, 128, 64),
    ("BSND_noGVA_V128_03", 16, 32, 32, 4096, 128, 128, 64),
    ("BSND_noGVA_V128_04", 8, 32, 32, 8192, 128, 128, 64),
    ("BSND_noGVA_V128_05", 128, 4, 4, 1024, 128, 128, 64),
    ("BSND_noGVA_V128_06", 64, 8, 8, 4096, 128, 128, 64),
    ("BSND_noGVA_V128_07", 32, 16, 16, 8192, 128, 128, 64),
    ("BSND_noGVA_V128_08", 16, 32, 32, 16384, 128, 128, 64),
    ("BSND_noGVA_V128_09", 64, 8, 8, 2048, 128, 128, 128),
    ("BSND_noGVA_V128_10", 32, 16, 16, 4096, 128, 128, 128),
    ("BSND_noGVA_V128_11", 16, 32, 32, 8192, 128, 128, 128),
    ("BSND_noGVA_V128_12", 8, 32, 32, 16384, 128, 128, 128),
    ("BSND_noGVA_V128_13", 1, 4, 4, 1024, 128, 128, 64),
    ("BSND_noGVA_V128_14", 48, 8, 8, 2048, 128, 128, 64),
    ("BSND_noGVA_V128_15", 24, 16, 16, 4096, 128, 128, 64),
    ("BSND_noGVA_V128_16", 12, 32, 32, 8192, 128, 128, 64),
    ("BSND_noGVA_V128_17", 1, 16, 16, 32768, 128, 128, 64),
    ("BSND_noGVA_V128_18", 1, 8, 8, 65536, 128, 128, 64),
    ("BSND_noGVA_V128_19", 1, 32, 32, 65536, 128, 128, 64),
    ("BSND_noGVA_V128_20", 1, 32, 32, 16384, 128, 128, 64),
    ("BSND_GVA_V256_21", 1, 32, 16, 16384, 256, 128, 64),
    ("BSND_GVA_V256_22", 1, 32, 16, 16384, 256, 128, 64),
    ("BSND_GVA_V256_23", 1, 63, 21, 16384, 256, 128, 64),
    ("BSND_GVA_V256_24", 1, 32, 8, 65536, 256, 128, 128),
    ("BSND_GVA_V128_25", 1, 32, 16, 36621, 128, 128, 64),
    ("BSND_GVA_V128_26", 1, 32, 4, 7178, 128, 128, 128),
    ("BSND_GVA_V256_27", 1, 64, 2, 11202, 256, 128, 64),
    ("BSND_GVA_V256_28", 1, 32, 16, 4096, 256, 128, 64),
    ("BSND_GVA_V256_29", 16, 63, 21, 2048, 256, 128, 64),
    ("BSND_GVA_V128_30", 711, 32, 4, 196, 128, 128, 128),
    ("BSND_GVA_V256_31", 176, 64, 2, 24, 256, 128, 64),
    ("BSND_GVA_V256_32", 1, 48, 16, 16387, 256, 128, 64),
    ("BSND_GVA_V256_33", 1, 48, 16, 8999, 256, 128, 128),
]


def _align_down(value: int, align: int) -> int:
    if align <= 0:
        return max(1, value)
    return max(align, (value // align) * align)


def _normalize(name: str, B: int, HV: int, HK: int, T: int, V: int, K: int, chunk_size: int) -> dict[str, Any]:
    if HK <= 0 or HV % HK != 0:
        raise ValueError(f"invalid heads HV={HV} HK={HK} for {name}")
    if K != 128:
        raise ValueError(f"K must be 128, got {K}")
    if V not in (128, 256):
        raise ValueError(f"V must be 128/256, got {V}")
    if chunk_size not in (64, 128):
        raise ValueError(f"chunk_size must be 64/128, got {chunk_size}")

    scaled = False
    bht = B * HV * T
    if bht > _MAX_BHT:
        target_t = max(chunk_size, _MAX_BHT // max(B * HV, 1))
        T = _align_down(target_t, chunk_size)
        scaled = True
        bht = B * HV * T
    if bht > _MAX_BHT and B > 1:
        target_b = max(1, _MAX_BHT // max(HV * T, 1))
        B = max(1, target_b)
        scaled = True

    return {
        "name": f"{name}_scaled" if scaled else name,
        "B": int(B),
        "HK": int(HK),
        "HV": int(HV),
        "T": int(T),
        "K": int(K),
        "V": int(V),
        "chunk_size": int(chunk_size),
        "source": name,
        "scaled": scaled,
    }


def _key(shape: dict[str, Any]) -> tuple:
    return (
        shape["B"],
        shape["HK"],
        shape["HV"],
        shape["T"],
        shape["K"],
        shape["V"],
        shape["chunk_size"],
    )


def _base_shapes() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for row in _GDN_TABLE_33:
        shape = _normalize(*row)
        key = _key(shape)
        if key in seen:
            continue
        seen.add(key)
        out.append(shape)
    return out


def _expand_shapes(base: list[dict[str, Any]], target: int = 100) -> list[dict[str, Any]]:
    shapes = list(base)
    seen = {_key(shape) for shape in shapes}

    def _try_add(raw: dict[str, Any]) -> None:
        if len(shapes) >= target:
            return
        shape = _normalize(
            raw["name"],
            raw["B"],
            raw["HV"],
            raw["HK"],
            raw["T"],
            raw["V"],
            raw["K"],
            raw["chunk_size"],
        )
        key = _key(shape)
        if key in seen:
            return
        seen.add(key)
        shapes.append(shape)

    for src in list(shapes):
        other_chunk_size = 128 if src["chunk_size"] == 64 else 64
        _try_add({**src, "name": f"{src['source']}_cs{other_chunk_size}", "chunk_size": other_chunk_size})

    for src in list(shapes):
        t2 = _align_down(max(src["chunk_size"], src["T"] // 2), src["chunk_size"])
        if t2 != src["T"]:
            _try_add({**src, "name": f"{src['source']}_T{t2}", "T": t2})

    for src in list(shapes):
        if src["B"] >= 2:
            _try_add({**src, "name": f"{src['source']}_B{src['B'] // 2}", "B": src["B"] // 2})

    head_sets = [
        (1, 1),
        (2, 2),
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (16, 8),
        (32, 16),
        (32, 8),
        (32, 4),
        (64, 2),
        (48, 16),
        (63, 21),
    ]
    t_cands = [64, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096]
    b_cands = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    for i, (hv, hk) in enumerate(head_sets):
        for j, t in enumerate(t_cands):
            for k, b in enumerate(b_cands):
                if len(shapes) >= target:
                    return shapes
                chunk_size = 64 if (i + j + k) % 2 == 0 else 128
                v = 256 if hv > hk else 128
                t_aligned = _align_down(t, chunk_size) if t % chunk_size == 0 else t
                if (i + j) % 3 == 0:
                    t_aligned = t
                _try_add(
                    {
                        "name": f"synth_B{b}_HV{hv}_HK{hk}_T{t_aligned}_V{v}_cs{chunk_size}",
                        "B": b,
                        "HV": hv,
                        "HK": hk,
                        "T": t_aligned,
                        "V": v,
                        "K": 128,
                        "chunk_size": chunk_size,
                    }
                )

    step = 0
    while len(shapes) < target:
        src = base[step % len(base)]
        chunk_size = src["chunk_size"]
        t = max(chunk_size, src["T"] - (step + 1) * chunk_size)
        _try_add({**src, "name": f"{src['source']}_tuneT{t}", "T": t})
        step += 1
        if step > target * 20:
            break
    return shapes[:target]


def _build_unique_shape_profiles(count: int = 100, key_fields: tuple[str, ...] | None = None) -> list[dict[str, Any]]:
    key_fields = key_fields or ("B", "HK", "HV", "T", "K", "V", "chunk_size")
    candidate_count = max(count, 120)
    while candidate_count <= count * 10:
        out: list[dict[str, Any]] = []
        seen: set[tuple] = set()
        for shape in _expand_shapes(_base_shapes(), target=candidate_count):
            key = tuple(shape[field] for field in key_fields)
            if key in seen:
                continue
            seen.add(key)
            out.append(shape)
            if len(out) >= count:
                return out
        candidate_count *= 2
    raise RuntimeError(f"failed to build {count} unique GDN shape profiles for fields={key_fields!r}")


def _build_profiles(shape_count: int = 100, soc: str = "ascend910b"):
    profiles = []
    shapes = _build_unique_shape_profiles(shape_count, key_fields=("B", "HV", "T", "chunk_size"))
    for shape_id, shape in enumerate(shapes):
        base = {
            "name": shape["name"],
            "B": shape["B"],
            "H": shape["HV"],
            "T": shape["T"],
            "chunk_size": shape["chunk_size"],
            "reverse": bool(shape_id % 2),
            "scale": SCALES[shape_id % len(SCALES)],
            "source": shape["source"],
            "scaled": shape["scaled"],
        }
        for dtype_id, dtype in enumerate(DTYPES):
            case_id = shape_id * len(DTYPES) + dtype_id
            profile = deepcopy(base)
            profile.update(
                {
                    "dtype": dtype,
                    "op": OP_NAME,
                    "case_id": case_id,
                    "shape_id": shape_id,
                    "seed": _SEED_BASE + case_id,
                    "route": "ascendc",
                    "soc": soc,
                }
            )
            profiles.append(profile)
    return profiles


PROFILES = _build_profiles()

def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")

def _spec(index):
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    profile.update({"op": OP_NAME, "case_id": index, "seed": _SEED_BASE + index, "route": "ascendc"})
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
