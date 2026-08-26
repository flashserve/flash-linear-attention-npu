"""prepare_wy_repr_bwd_da 的 ATK 泛化用例生成器。"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path

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

OP_NAME = "prepare_wy_repr_bwd_da"
CASE_COUNT = 200
SMALL_CASE_COUNT = 180
LARGE_CASE_COUNT = 20
SMALL_ELEMENT_LIMIT = 20_000_000
LARGE_ELEMENT_CAP = 80_000_000
REFERENCE_LARGE_ELEMENT_CAP = 800_000_000
REFERENCE_CASES_PATH = (
    Path(__file__).resolve().parents[3]
    / "fla/ops/ascendc/gdn/chunk_gdn_bwd/prepare_wy_repr_bwd_da/test/test_da_cases.json"
)
DTYPE_GTYPE_PAIRS = (
    ("fp16", "fp16"),
    ("fp16", "fp32"),
    ("bf16", "bf16"),
    ("bf16", "fp32"),
)
CHUNK_SIZES = (64, 128)
SMALL_BATCHES = (1, 2, 4)
SMALL_HEAD_PAIRS = (
    (2, 2),
    (4, 4),
    (8, 8),
    (16, 16),
    (2, 4),
    (4, 8),
    (4, 16),
    (8, 16),
    (4, 32),
    (8, 32),
)
SMALL_T_VALUES = (24, 128, 196, 256, 512)
SMALL_V_VALUES = (128, 256)
SMALL_MEAN_LENGTHS = (2, 3, 4, 5, 9, 16)

SAFE_PROFILE_OVERRIDES = {
    87: {
        "name": "small_replacement_087_b1_h2x2_t24",
        "dtype": "bf16",
        "gtype": "fp32",
        "B": 1,
        "HK": 2,
        "HV": 2,
        "T": 24,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "varlen": False,
        "size_class": "small",
    }
}


def _normalize_profile(case, dtype=None, gtype=None, chunk_size=None):
    profile = {
        "name": case["name"],
        "dtype": dtype or case["dtype"],
        "gtype": gtype or case["gtype"],
        "B": int(case["B"]),
        "HK": int(case["query_head"]),
        "HV": int(case["value_head"]),
        "T": int(case["T"]),
        "K": int(case["Kdim"]),
        "V": int(case["Vdim"]),
        "chunk_size": int(chunk_size or case["chunk_size"]),
        "varlen": bool(case["varlen"]),
    }
    if profile["varlen"]:
        profile["mean_len"] = int(case["mean_len"])
    return profile


def _profile_key(profile):
    return tuple(
        profile.get(key)
        for key in (
            "dtype",
            "gtype",
            "B",
            "HK",
            "HV",
            "T",
            "K",
            "V",
            "chunk_size",
            "varlen",
            "mean_len",
        )
    )


def _shape_elements(profile):
    return profile["B"] * profile["T"] * (
        profile["HK"] * profile["K"]
        + profile["HV"]
        * (2 * profile["V"] + profile["K"] + profile["chunk_size"] + 2)
    )


def _is_filtered_profile(profile):
    return (
        profile["dtype"] == "fp16"
        and profile["V"] == 256
        and profile["chunk_size"] == 128
    )


def _fit_large_profile_to_cap(profile):
    profile = deepcopy(profile)
    if _shape_elements(profile) <= LARGE_ELEMENT_CAP:
        return profile

    if profile["B"] > 1:
        elements_per_batch = _shape_elements({**profile, "B": 1})
        profile["B"] = min(
            profile["B"], max(1, LARGE_ELEMENT_CAP // elements_per_batch)
        )

    if _shape_elements(profile) > LARGE_ELEMENT_CAP:
        elements_per_token = _shape_elements({**profile, "T": 1})
        profile["T"] = min(
            profile["T"], max(1, LARGE_ELEMENT_CAP // elements_per_token)
        )

    elements = _shape_elements(profile)
    if not SMALL_ELEMENT_LIMIT < elements <= LARGE_ELEMENT_CAP:
        raise RuntimeError(
            f"unable to fit large profile {profile['name']} within element range: {elements}"
        )
    return profile


def _build_small_profiles():
    buckets = {
        (False, "eq"): [],
        (False, "gva"): [],
        (True, "eq"): [],
        (True, "gva"): [],
    }
    for varlen in (False, True):
        batches = (1,) if varlen else SMALL_BATCHES
        for B in batches:
            for head_index, (HK, HV) in enumerate(SMALL_HEAD_PAIRS):
                relation = "eq" if HK == HV else "gva"
                for T in SMALL_T_VALUES:
                    for V in SMALL_V_VALUES:
                        for chunk_size in CHUNK_SIZES:
                            for dtype, gtype in DTYPE_GTYPE_PAIRS:
                                profile = {
                                    "name": "small",
                                    "dtype": dtype,
                                    "gtype": gtype,
                                    "B": B,
                                    "HK": HK,
                                    "HV": HV,
                                    "T": T,
                                    "K": 128,
                                    "V": V,
                                    "chunk_size": chunk_size,
                                    "varlen": varlen,
                                    "size_class": "small",
                                }
                                if varlen:
                                    mean_index = (head_index + T + V + chunk_size) % len(SMALL_MEAN_LENGTHS)
                                    profile["mean_len"] = SMALL_MEAN_LENGTHS[mean_index]
                                if (
                                    not _is_filtered_profile(profile)
                                    and _shape_elements(profile) <= SMALL_ELEMENT_LIMIT
                                ):
                                    buckets[(varlen, relation)].append(profile)

    selected = []
    per_bucket = SMALL_CASE_COUNT // len(buckets)
    for bucket_index, bucket_key in enumerate(buckets):
        bucket = buckets[bucket_key]
        random.Random(20260822 + bucket_index).shuffle(bucket)
        if len(bucket) < per_bucket:
            raise RuntimeError(f"not enough small profiles in bucket {bucket_key}: {len(bucket)}")
        for profile_index, profile in enumerate(bucket[:per_bucket]):
            profile = deepcopy(profile)
            profile["name"] = (
                f"small_{bucket_index}_{profile_index:03d}_"
                f"b{profile['B']}_h{profile['HK']}x{profile['HV']}_t{profile['T']}"
            )
            selected.append(profile)
    return selected


def _build_large_profiles(reference_cases):
    candidates = []
    for case in reference_cases:
        profile = _normalize_profile(case)
        reference_elements = _shape_elements(profile)
        if (
            not _is_filtered_profile(profile)
            and SMALL_ELEMENT_LIMIT < reference_elements <= REFERENCE_LARGE_ELEMENT_CAP
        ):
            profile = _fit_large_profile_to_cap(profile)
            profile["name"] = f"large_{profile['name']}"
            profile["size_class"] = "large"
            candidates.append(profile)
    candidates.sort(key=lambda profile: (_shape_elements(profile), profile["name"]))
    if len(candidates) != LARGE_CASE_COUNT:
        raise RuntimeError(f"expected {LARGE_CASE_COUNT} reference large profiles, got {len(candidates)}")
    return candidates


def _build_profiles():
    reference_cases = json.loads(REFERENCE_CASES_PATH.read_text(encoding="utf-8"))
    profiles = _build_small_profiles() + _build_large_profiles(reference_cases)
    for index, replacement in SAFE_PROFILE_OVERRIDES.items():
        if not 0 <= index < SMALL_CASE_COUNT:
            raise RuntimeError(f"safe profile override index is not a small case: {index}")
        if _is_filtered_profile(replacement):
            raise RuntimeError(f"safe profile override is filtered: {index}")
        profiles[index] = deepcopy(replacement)

    keys = [_profile_key(profile) for profile in profiles]
    if len(profiles) != CASE_COUNT or len(set(keys)) != CASE_COUNT:
        raise RuntimeError(
            f"expected {CASE_COUNT} unique profiles, got {len(profiles)} total and {len(set(keys))} unique"
        )
    return profiles


PROFILES = _build_profiles()

def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")

def _spec(index):
    if not 0 <= index < len(PROFILES):
        raise IndexError(f"case index {index} is outside [0, {len(PROFILES)})")
    profile = deepcopy(PROFILES[index])
    profile.update({"op": OP_NAME, "case_id": index, "seed": 20260817 + index, "route": "ascendc", "soc": "ascend910b"})
    return profile

if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_prepare_wy_repr_bwd_da")
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
