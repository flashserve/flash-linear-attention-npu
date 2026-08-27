"""recurrent_kda 的 200 条 ATK 泛化用例生成器。"""

from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from itertools import product

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


OP_NAME = "recurrent_kda"
CASE_COUNT = 200
SMALL_CASE_COUNT = 190
LARGE_CASE_COUNT = 10
EXPECTED_TILING_KEYS = {0}
SMALL_STATE_ELEMENT_CAP = 5_000_000
LARGE_STATE_ELEMENT_CAP = 12_000_000

GATE_DTYPES = ("fp32", "bf16", "fp16")
BETA_DTYPES = ("fp32", "bf16", "fp16")
STATE_DTYPES = ("fp32", "bf16")
LAYOUTS = ("BSND", "TND")
STATE_DIRECTIONS = (False, True)
VALUE_DIMS = (128, 256)
INT_DTYPES = ("int32", "int64")

SMALL_SHAPES = (
    (1, 1, 1, 1),
    (1, 2, 1, 2),
    (1, 4, 2, 2),
    (1, 8, 2, 4),
    (2, 1, 1, 4),
    (2, 2, 2, 4),
    (2, 3, 2, 8),
    (2, 4, 4, 8),
    (3, 1, 4, 4),
    (3, 2, 4, 8),
    (4, 1, 4, 16),
    (4, 2, 8, 16),
)

LARGE_SHAPES = (
    (4, 8, 4, 16),
    (8, 8, 8, 16),
    (8, 4, 4, 32),
    (4, 8, 8, 32),
    (8, 6, 8, 32),
)

OPTIONAL_MODES = (
    {"cu_mode": "none", "ssm_mode": "none", "accepted": False},
    {"cu_mode": "uniform", "ssm_mode": "none", "accepted": False},
    {"cu_mode": "varlen", "ssm_mode": "none", "accepted": False},
    {"cu_mode": "varlen", "ssm_mode": "packed", "accepted": False},
    {"cu_mode": "varlen", "ssm_mode": "speculative", "accepted": False},
    {"cu_mode": "varlen", "ssm_mode": "speculative", "accepted": True},
    {"cu_mode": "padding", "ssm_mode": "none", "accepted": False},
    {"cu_mode": "zero", "ssm_mode": "none", "accepted": False},
    {"cu_mode": "uniform", "ssm_mode": "packed", "accepted": False},
    {"cu_mode": "varlen", "ssm_mode": "packed", "accepted": True},
)

KERNEL_FEATURE_MODES = (
    {},
    {"use_qk_l2norm_in_kernel": True},
    {"use_gate_in_kernel": True},
    {"use_gate_in_kernel": True, "dt_bias_mode": "flat"},
    {
        "use_gate_in_kernel": True,
        "dt_bias_mode": "matrix",
        "safe_gate": True,
        "lower_bound": -2.5,
    },
    {"use_beta_sigmoid_in_kernel": True},
    {"use_beta_sigmoid_in_kernel": True, "allow_neg_eigval": True},
    {
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "dt_bias_mode": "matrix",
        "safe_gate": True,
        "lower_bound": -4.0,
        "use_beta_sigmoid_in_kernel": True,
        "allow_neg_eigval": True,
    },
)

OUTPUT_MODES = (
    (True, False),
    (True, True),
    (False, False),
    (False, True),
)

BASE_COMBINATIONS = tuple(
    product(
        LAYOUTS,
        GATE_DTYPES,
        BETA_DTYPES,
        STATE_DTYPES,
        STATE_DIRECTIONS,
        VALUE_DIMS,
    )
)


def _partition(total: int, seed: int) -> list[int]:
    """把 token 数切成 1..8 的确定性变长序列。"""
    lengths = []
    remaining = int(total)
    step = 0
    while remaining > 0:
        upper = min(8, remaining)
        length = 1 + ((seed + 3 * step) % upper)
        lengths.append(length)
        remaining -= length
        step += 1
    if len(lengths) == 1 and total > 1:
        first = max(1, lengths[0] // 2)
        lengths = [first, lengths[0] - first]
    return lengths


def _sequence_lengths(layout: str, B: int, T: int, cu_mode: str, seed: int) -> list[int] | None:
    capacity = B * T
    if cu_mode == "none":
        return None
    if cu_mode == "uniform":
        return [T] * B
    if cu_mode == "varlen":
        return _partition(capacity, seed)
    if cu_mode == "padding":
        valid = max(1, capacity - 1 - (seed % min(4, capacity)))
        return _partition(valid, seed)
    if cu_mode == "zero":
        lengths = _partition(capacity, seed)
        lengths.insert((seed % (len(lengths) + 1)), 0)
        return lengths
    raise ValueError(f"unsupported cu_mode: {cu_mode}")


def _default_profile(index: int, size_class: str) -> dict:
    layout, gate_dtype, beta_dtype, state_dtype, state_v_first, V = BASE_COMBINATIONS[
        index % len(BASE_COMBINATIONS)
    ]
    shape_table = SMALL_SHAPES if size_class == "small" else LARGE_SHAPES
    B, T, H, HV = shape_table[index % len(shape_table)]
    if size_class == "large":
        layout = LAYOUTS[(index - SMALL_CASE_COUNT) % len(LAYOUTS)]

    optional = deepcopy(OPTIONAL_MODES[index % len(OPTIONAL_MODES)])
    if size_class == "large":
        optional = deepcopy(
            (
                {"cu_mode": "uniform", "ssm_mode": "none", "accepted": False},
                {"cu_mode": "uniform", "ssm_mode": "packed", "accepted": False},
                {"cu_mode": "uniform", "ssm_mode": "speculative", "accepted": True},
                {"cu_mode": "none", "ssm_mode": "none", "accepted": False},
            )[index % 4]
        )

    # TND 不传 cu_seqlens 时只能表示一条序列，保持总长度 <= 8。
    if layout == "TND" and optional["cu_mode"] == "none":
        if size_class == "large":
            optional = {"cu_mode": "uniform", "ssm_mode": "none", "accepted": False}
        else:
            B = 1

    lengths = _sequence_lengths(layout, B, T, optional["cu_mode"], 20260817 + index)
    if optional["accepted"] and (lengths is None or any(length <= 0 for length in lengths)):
        optional["accepted"] = False

    feature = {
        "use_qk_l2norm_in_kernel": False,
        "use_gate_in_kernel": False,
        "dt_bias_mode": "none",
        "use_beta_sigmoid_in_kernel": False,
        "allow_neg_eigval": False,
        "safe_gate": False,
        "lower_bound": -5.0,
    }
    feature.update(KERNEL_FEATURE_MODES[index % len(KERNEL_FEATURE_MODES)])
    output_final_state, inplace_final_state = OUTPUT_MODES[index % len(OUTPUT_MODES)]

    profile = {
        "name": f"{size_class}_{index:03d}",
        "size_class": size_class,
        "tiling_key": 0,
        "dtype": "bf16",
        "gate_dtype": gate_dtype,
        "beta_dtype": beta_dtype,
        "state_dtype": state_dtype,
        "B": B,
        "T": T,
        "H": H,
        "HV": HV,
        "K": 128,
        "V": V,
        "layout": layout,
        "state_v_first": state_v_first,
        "cu_mode": optional["cu_mode"],
        "seq_lengths": lengths,
        "cu_dtype": INT_DTYPES[index % len(INT_DTYPES)],
        "ssm_mode": optional["ssm_mode"],
        "ssm_dtype": INT_DTYPES[(index // 2) % len(INT_DTYPES)],
        "accepted_tokens": optional["accepted"],
        "accepted_dtype": INT_DTYPES[(index // 3) % len(INT_DTYPES)],
        "state_capacity_extra": 1 + (index % 3) if optional["ssm_mode"] != "none" else 0,
        "output_final_state": output_final_state,
        "inplace_final_state": inplace_final_state,
        "state_noncontiguous": index % 7 == 0,
        "input_noncontiguous": index % 13 == 0,
        "scale": (128.0 ** -0.5, 1.0, 0.5)[index % 3],
        **feature,
    }
    profile["initial_state_none"] = (
        index % 23 == 0
        and not inplace_final_state
        and profile["ssm_mode"] == "none"
    )
    if profile["initial_state_none"]:
        profile["state_dtype"] = "fp32"
        profile["state_noncontiguous"] = False
    return profile


def _logical_lengths(profile: dict) -> list[int]:
    if profile["seq_lengths"] is not None:
        return list(profile["seq_lengths"])
    if profile["layout"] == "BSND":
        return [profile["T"]] * profile["B"]
    return [profile["B"] * profile["T"]]


def _validate_profile(profile: dict) -> None:
    if profile["tiling_key"] not in EXPECTED_TILING_KEYS:
        raise RuntimeError(f"invalid tiling key: {profile['tiling_key']}")
    if profile["dtype"] != "bf16":
        raise RuntimeError("q/k/v must use bf16")
    if profile["gate_dtype"] not in GATE_DTYPES or profile["beta_dtype"] not in BETA_DTYPES:
        raise RuntimeError("invalid gate/beta dtype")
    if profile["state_dtype"] not in STATE_DTYPES:
        raise RuntimeError("invalid state dtype")
    if profile["layout"] not in LAYOUTS or profile["K"] != 128 or profile["V"] not in VALUE_DIMS:
        raise RuntimeError("invalid layout or K/V")
    if not (0 < profile["H"] <= 256 and 0 < profile["HV"] <= 256):
        raise RuntimeError("H/HV must be in (0, 256]")
    if profile["HV"] % profile["H"] != 0:
        raise RuntimeError("HV must be divisible by H")

    capacity = profile["B"] * profile["T"]
    lengths = _logical_lengths(profile)
    if not lengths or any(length < 0 or length > 8 for length in lengths):
        raise RuntimeError(f"invalid sequence lengths: {lengths}")
    if sum(lengths) > capacity:
        raise RuntimeError("valid tokens exceed physical capacity")
    if profile["layout"] == "TND" and profile["cu_mode"] == "none" and capacity > 8:
        raise RuntimeError("dense TND sequence length exceeds 8")
    if profile["accepted_tokens"]:
        if profile["ssm_mode"] == "none" or any(length <= 0 for length in lengths):
            raise RuntimeError("accepted tokens require non-empty indexed sequences")
    if profile["ssm_mode"] not in ("none", "packed", "speculative"):
        raise RuntimeError("invalid ssm mode")
    if profile["ssm_mode"] == "none" and profile["state_capacity_extra"] != 0:
        raise RuntimeError("state capacity extra requires ssm indices")
    if profile["initial_state_none"] and (
        profile["inplace_final_state"] or profile["ssm_mode"] != "none"
    ):
        raise RuntimeError("implicit state is only valid for non-inplace non-indexed cases")
    if not profile["use_gate_in_kernel"]:
        if profile["safe_gate"] or profile["dt_bias_mode"] != "none":
            raise RuntimeError("safe gate/dt bias require in-kernel gate")
    if profile["safe_gate"] and not -5.0 <= profile["lower_bound"] < 0.0:
        raise RuntimeError("safe gate lower bound is invalid")
    if profile["allow_neg_eigval"] and not profile["use_beta_sigmoid_in_kernel"]:
        raise RuntimeError("allow_neg_eigval requires beta sigmoid in generated cases")

    seq_num = len(lengths)
    state_capacity = seq_num + profile["state_capacity_extra"]
    state_elements = state_capacity * profile["HV"] * profile["K"] * profile["V"]
    cap = SMALL_STATE_ELEMENT_CAP if profile["size_class"] == "small" else LARGE_STATE_ELEMENT_CAP
    if state_elements > cap:
        raise RuntimeError(
            f"{profile['name']} state elements {state_elements} exceed cap {cap}"
        )
    profile["token_capacity"] = capacity
    profile["valid_tokens"] = sum(lengths)
    profile["seq_num"] = seq_num
    profile["state_capacity"] = state_capacity
    profile["state_elements"] = state_elements


def _build_profiles() -> list[dict]:
    profiles = []
    for index in range(CASE_COUNT):
        size_class = "small" if index < SMALL_CASE_COUNT else "large"
        profile = _default_profile(index, size_class)
        _validate_profile(profile)
        profile["name"] = (
            f"{size_class}_{index:03d}_{profile['layout'].lower()}_"
            f"b{profile['B']}_t{profile['T']}_h{profile['H']}x{profile['HV']}_v{profile['V']}"
        )
        profiles.append(profile)

    if len(profiles) != CASE_COUNT or len({profile["name"] for profile in profiles}) != CASE_COUNT:
        raise RuntimeError("profiles must contain exactly 200 unique names")
    if {profile["tiling_key"] for profile in profiles} != EXPECTED_TILING_KEYS:
        raise RuntimeError("profiles do not cover the complete tilingKey set {0}")
    if sum(profile["size_class"] == "small" for profile in profiles) != SMALL_CASE_COUNT:
        raise RuntimeError("small/large distribution is invalid")

    required_values = {
        "layout": set(LAYOUTS),
        "gate_dtype": set(GATE_DTYPES),
        "beta_dtype": set(BETA_DTYPES),
        "state_dtype": set(STATE_DTYPES),
        "state_v_first": set(STATE_DIRECTIONS),
        "V": set(VALUE_DIMS),
        "cu_mode": {mode["cu_mode"] for mode in OPTIONAL_MODES},
        "ssm_mode": {mode["ssm_mode"] for mode in OPTIONAL_MODES},
        "output_final_state": {False, True},
        "inplace_final_state": {False, True},
        "use_qk_l2norm_in_kernel": {False, True},
        "use_gate_in_kernel": {False, True},
        "use_beta_sigmoid_in_kernel": {False, True},
        "safe_gate": {False, True},
    }
    for field, expected in required_values.items():
        actual = {profile[field] for profile in profiles}
        if actual != expected:
            raise RuntimeError(f"field {field} coverage mismatch: expected {expected}, got {actual}")
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

    @GENERATOR_REGISTRY.register("generator_recurrent_kda")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            spec = _spec(index)
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = f"{OP_NAME}_{index:04d}_{spec['name']}"
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    # q/k/v 只支持 BF16；第二路 marker 仅用于让统一 -dt 100 生成 200 条。
                    cfg.dtype = "bf16"
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(
                        spec, ensure_ascii=False, separators=(",", ":")
                    )
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config


if __name__ == "__main__":
    summary_fields = (
        "tiling_key",
        "size_class",
        "layout",
        "gate_dtype",
        "beta_dtype",
        "state_dtype",
        "state_v_first",
        "V",
        "cu_mode",
        "ssm_mode",
    )
    summary = {
        field: dict(sorted(Counter(str(profile[field]) for profile in PROFILES).items()))
        for field in summary_fields
    }
    print(json.dumps({"case_count": len(PROFILES), "coverage": summary}, ensure_ascii=False, indent=2))
