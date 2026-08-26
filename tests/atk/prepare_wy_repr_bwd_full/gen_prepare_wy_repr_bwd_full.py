"""prepare_wy_repr_bwd_full 的 ATK 泛化用例生成器。"""

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


OP_NAME = "prepare_wy_repr_bwd_full"
CASE_COUNT = 200
SMALL_CASE_COUNT = 190
LARGE_CASE_COUNT = 10
SHRINK_FACTOR = 10
DA_SMALL_ELEMENT_LIMIT = 20_000_000
DA_LARGE_ELEMENT_CAP = 80_000_000
FULL_LARGE_ELEMENT_CAP = 9_000_000
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
DA_SMALL_T_VALUES = (24, 128, 196, 256, 512)
SMALL_V_VALUES = (128, 256)
DA_SMALL_MEAN_LENGTHS = (2, 3, 4, 5, 9, 16)
SMALL_BUCKET_COUNTS = {
    (False, "eq"): 48,
    (False, "gva"): 48,
    (True, "eq"): 47,
    (True, "gva"): 47,
}
LARGE_REFERENCE_NAMES = (
    "var_hk_eq_hv_4",
    "fix_hk_eq_hv_5",
    "fix_hk_eq_hv_4",
    "phase_1_fix_1",
    "phase_1_var_3",
    "gva_fix_4",
    "gva_fix_1",
    "gva_var_1",
    "gva_var_2",
    "phase_1_var_5",
)
LARGE_PROFILE_OVERRIDES = {
    # Avoid the ratio=32 BF16 dk accumulation case while keeping this profile
    # large and preserving its V=256 / chunk=64 tiling coverage.
    "gva_fix_4": {"B": 16, "HV": 16, "T": 32},
}


def _normalize_profile(case):
    profile = {
        "name": case["name"],
        "dtype": case["dtype"],
        "gtype": case["gtype"],
        "B": int(case["B"]),
        "HK": int(case["query_head"]),
        "HV": int(case["value_head"]),
        "T": int(case["T"]),
        "K": int(case["Kdim"]),
        "V": int(case["Vdim"]),
        "chunk_size": int(case["chunk_size"]),
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


def _da_shape_elements(profile):
    return profile["B"] * profile["T"] * (
        profile["HK"] * profile["K"]
        + profile["HV"]
        * (2 * profile["V"] + profile["K"] + profile["chunk_size"] + 2)
    )


def _full_shape_elements(profile):
    return profile["B"] * profile["T"] * (
        profile["HK"] * profile["K"]
        + profile["HV"]
        * (2 * profile["V"] + profile["K"] + 2 * profile["chunk_size"] + 2)
    )


def _tiling_key(profile):
    return 2 if profile["V"] == 256 else 1


def _is_filtered_profile(profile):
    return (
        profile["dtype"] == "fp16"
        and profile["V"] == 256
        and profile["chunk_size"] == 128
    )


def _shrink_profile(profile):
    profile = deepcopy(profile)
    original_t = profile["T"]
    original_elements = _da_shape_elements(profile)
    target_elements = max(1, round(original_elements / SHRINK_FACTOR))
    elements_per_token = _full_shape_elements({**profile, "T": 1})
    profile["T"] = max(1, min(original_t, round(target_elements / elements_per_token)))
    if profile["varlen"]:
        mean_len = max(1, int(profile.get("mean_len", 1)))
        profile["mean_len"] = max(
            1,
            min(profile["T"], round(mean_len * profile["T"] / original_t)),
        )
    profile["tiling_key"] = _tiling_key(profile)
    profile["shrink_factor"] = SHRINK_FACTOR
    profile["source_T"] = original_t
    profile["source_da_elements"] = original_elements
    profile["full_input_elements"] = _full_shape_elements(profile)
    profile["element_ratio_to_da"] = round(
        profile["full_input_elements"] / original_elements, 6
    )
    return profile


def _fit_to_da_large_cap(profile):
    profile = deepcopy(profile)
    if _da_shape_elements(profile) <= DA_LARGE_ELEMENT_CAP:
        return profile

    if profile["B"] > 1:
        elements_per_batch = _da_shape_elements({**profile, "B": 1})
        profile["B"] = min(
            profile["B"], max(1, DA_LARGE_ELEMENT_CAP // elements_per_batch)
        )

    if _da_shape_elements(profile) > DA_LARGE_ELEMENT_CAP:
        elements_per_token = _da_shape_elements({**profile, "T": 1})
        profile["T"] = min(
            profile["T"], max(1, DA_LARGE_ELEMENT_CAP // elements_per_token)
        )
    return profile


def _build_small_profiles():
    buckets = {key: [] for key in SMALL_BUCKET_COUNTS}
    for varlen in (False, True):
        batches = (1,) if varlen else SMALL_BATCHES
        for B in batches:
            for head_index, (HK, HV) in enumerate(SMALL_HEAD_PAIRS):
                relation = "eq" if HK == HV else "gva"
                for base_t in DA_SMALL_T_VALUES:
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
                                    "T": base_t,
                                    "K": 128,
                                    "V": V,
                                    "chunk_size": chunk_size,
                                    "varlen": varlen,
                                    "size_class": "small",
                                }
                                if varlen:
                                    mean_index = (
                                        head_index + base_t + V + chunk_size
                                    ) % len(DA_SMALL_MEAN_LENGTHS)
                                    profile["mean_len"] = DA_SMALL_MEAN_LENGTHS[mean_index]
                                if (
                                    not _is_filtered_profile(profile)
                                    and _da_shape_elements(profile) <= DA_SMALL_ELEMENT_LIMIT
                                ):
                                    buckets[(varlen, relation)].append(
                                        _shrink_profile(profile)
                                    )

    selected = []
    used_keys = set()
    for bucket_index, (bucket_key, required_count) in enumerate(
        SMALL_BUCKET_COUNTS.items()
    ):
        bucket = buckets[bucket_key]
        random.Random(20260822 + bucket_index).shuffle(bucket)
        bucket_selected = []
        for profile in bucket:
            key = _profile_key(profile)
            if key in used_keys:
                continue
            profile = deepcopy(profile)
            profile_index = len(bucket_selected)
            profile["name"] = (
                f"small_{bucket_index}_{profile_index:03d}_"
                f"b{profile['B']}_h{profile['HK']}x{profile['HV']}_t{profile['T']}"
            )
            bucket_selected.append(profile)
            used_keys.add(key)
            if len(bucket_selected) == required_count:
                break
        if len(bucket_selected) != required_count:
            raise RuntimeError(
                f"not enough unique small profiles in bucket {bucket_key}: "
                f"{len(bucket_selected)} < {required_count}"
            )
        selected.extend(bucket_selected)
    return selected


def _build_large_profiles(reference_cases):
    by_name = {case["name"]: case for case in reference_cases}
    missing = [name for name in LARGE_REFERENCE_NAMES if name not in by_name]
    if missing:
        raise RuntimeError(f"missing large reference profiles: {missing}")

    selected = []
    for profile_index, name in enumerate(LARGE_REFERENCE_NAMES):
        profile = _normalize_profile(by_name[name])
        if _is_filtered_profile(profile):
            raise RuntimeError(f"large reference profile is filtered: {name}")
        profile = _shrink_profile(_fit_to_da_large_cap(profile))
        profile.update(LARGE_PROFILE_OVERRIDES.get(name, {}))
        profile["tiling_key"] = _tiling_key(profile)
        profile["full_input_elements"] = _full_shape_elements(profile)
        profile["element_ratio_to_da"] = round(
            profile["full_input_elements"] / profile["source_da_elements"], 6
        )
        profile["name"] = f"large_{profile_index:02d}_{name}"
        profile["size_class"] = "large"
        elements = _full_shape_elements(profile)
        if elements > FULL_LARGE_ELEMENT_CAP:
            raise RuntimeError(
                f"large profile {name} exceeds full cap: {elements}"
            )
        selected.append(profile)
    return selected


def _build_profiles():
    reference_cases = json.loads(REFERENCE_CASES_PATH.read_text(encoding="utf-8"))
    small_profiles = _build_small_profiles()
    large_profiles = _build_large_profiles(reference_cases)
    profiles = small_profiles + large_profiles
    keys = [_profile_key(profile) for profile in profiles]
    if len(profiles) != CASE_COUNT or len(set(keys)) != CASE_COUNT:
        raise RuntimeError(
            f"expected {CASE_COUNT} unique profiles, got {len(profiles)} total "
            f"and {len(set(keys))} unique"
        )
    if {profile["tiling_key"] for profile in profiles} != {1, 2}:
        raise RuntimeError("profiles do not cover tiling keys 1 and 2")
    if len(small_profiles) != SMALL_CASE_COUNT or len(large_profiles) != LARGE_CASE_COUNT:
        raise RuntimeError("small/large profile distribution is invalid")
    return profiles


PROFILES = _build_profiles()


def _dtype(dtype):
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _spec(index):
    if not 0 <= index < len(PROFILES):
        raise IndexError(f"case index {index} is outside [0, {len(PROFILES)})")
    profile = deepcopy(PROFILES[index])
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


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_prepare_wy_repr_bwd_full")
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
                    cfg.range_values = json.dumps(
                        spec, ensure_ascii=False, separators=(",", ":")
                    )
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config
