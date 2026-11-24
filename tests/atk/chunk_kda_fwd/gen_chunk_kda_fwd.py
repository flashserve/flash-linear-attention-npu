"""Generate direct-input ATK cases for chunk_kda_fwd.

The generated JSON contains the real operator tensors and attributes. Marker
tensors and serialized case_spec metadata are deliberately not used.

正向矩阵只生成 chunk_size=64（算子只支持 64），共 1076 条：52 条 boundary + 1024 条 core。

chunk_size=128 属于被拦截的取值，不再进入正向矩阵；反向用例由 --negative-out 从正向 JSON 派生
（默认 12 条，带 expected_error_msg），用 atk --task run 执行并由 ATK 比对报错文案。
"""

from __future__ import annotations

import argparse
import json
import re
import math
from collections import Counter
from pathlib import Path
from itertools import combinations, product

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig, InputCaseConfig
except ModuleNotFoundError as exc:
    if exc.name not in {"atk", "torch"}:
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None
    InputCaseConfig = None


OP_NAME = "chunk_kda_fwd"
SEED_BASE = 20260903
Q_DTYPES = ("bf16", "fp16")
G_DTYPES = ("bf16", "fp32")
BETA_DTYPES = ("bf16", "fp32")
LAYOUTS = ("BSND", "BNSD", "TND", "NTD")
CHUNK_SIZES = (64, 128)
LONG_T_VALUES = (1024, 1536, 2048, 4096, 8192, 16384)
LONG_SEEDS = {
    total_t: 20260814 + 4 * index for index, total_t in enumerate(LONG_T_VALUES)
}
BF16_TOKEN_BOUNDARY_LABELS = {"t255_c128", "t256_c128", "t257_c128"}
TENSOR_INPUTS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "initial_state",
    "A_log",
    "dt_bias",
)
OPTIONAL_TENSORS = {"initial_state", "A_log", "dt_bias"}
BOOL_ATTRS = (
    "output_final_state",
    "safe_gate",
    "use_gate_in_kernel",
    "disable_recompute",
    "return_intermediate_states",
    "state_v_first",
)
# All six public booleans are independently supported. Keep the exemption table
# explicit so future semantic restrictions must be reviewed instead of silently
# weakening the pairwise gate.
BOOL_PAIRWISE_EXEMPTIONS: dict[tuple[str, str], set[tuple[bool, bool]]] = {}
EXPECTED_INPUTS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "scale",
    "chunk_size",
    "layout",
    "initial_state",
    "output_final_state",
    "cu_seqlens",
    "chunk_indices",
    "safe_gate",
    "lower_bound",
    "use_gate_in_kernel",
    "A_log",
    "dt_bias",
    "disable_recompute",
    "return_intermediate_states",
    "state_v_first",
)
FINITE_VALUES = (-1.0, 0.0, 1.0)
UNIFORM_RANGES = {
    "q": (-0.08, 0.08),
    "k": (-0.08, 0.08),
    "v": (-0.08, 0.08),
    "g": (-1.0, 1.0),
    "beta": (0.0, 1.0),
    "initial_state": (-0.02, 0.02),
    "A_log": (-6.0, -2.0),
    "dt_bias": (-2.0, 2.0),
}
NORMAL_PARAMS = {
    "v": (0.0, 0.05),
    "g": (0.0, 1.25),
    "beta": (0.5, 0.1),
    "initial_state": (0.0, 0.01),
    "A_log": (0.0, 0.12),
    "dt_bias": (-3.0, 1.65),
}
PREGATED_RANGE = (-0.02, -0.002)
HEAD_PROFILES = (
    (1, 1),
    (1, 2),
    (2, 4),
    (4, 4),
    (4, 8),
    (8, 32),
    (32, 128),
    (96, 96),
)
DIM_PROFILES = (
    (16, 16),
    (16, 256),
    (64, 128),
    (128, 128),
    (128, 256),
    (256, 16),
    (256, 256),
)
CU_MODES = ("single", "balanced", "front_short", "back_short", "zero_length")

# Ten real tensor/dtype roles per q dtype, each with -1, 0 and 1.
FINITE_TARGETS = (
    ("q", None),
    ("k", None),
    ("v", None),
    ("g", "bf16"),
    ("g", "fp32"),
    ("beta", "bf16"),
    ("beta", "fp32"),
    ("A_log", "fp32"),
    ("dt_bias", "fp32"),
    ("initial_state", "fp32"),
)


def _balanced(total: int, count: int) -> list[int]:
    base, extra = divmod(total, count)
    return [base + (1 if index < extra else 0) for index in range(count)]


def _sequence_lengths(total: int, mode: str, count: int) -> list[int]:
    if mode == "single":
        return [total]
    if mode == "many_1024":
        return [0] * 1023 + [total]
    count = max(2, int(count))
    if mode == "balanced":
        return _balanced(total, count)
    if mode == "front_short":
        return [1] + _balanced(max(0, total - 1), count - 1)
    if mode == "back_short":
        return _balanced(max(0, total - 1), count - 1) + [1]
    if mode == "zero_length":
        return [0] + _balanced(total, count - 1)
    if mode == "mixed_tail8":
        if total < 1024 or total % 8:
            raise ValueError("mixed_tail8 requires a total divisible by 8 and >= 1024")
        base = total // 8
        return [
            base - 47,
            base + 31,
            base - 1,
            base + 17,
            base - 33,
            base + 49,
            base - 15,
            base - 1,
        ]
    raise ValueError(f"unsupported cu mode: {mode}")


def _prefix_sum(lengths: list[int]) -> list[int]:
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + int(length))
    return offsets


def _canonical_chunk_indices(
    cu_seqlens: list[int] | None, chunk_size: int
) -> list[int] | None:
    if cu_seqlens is None:
        return None
    indices = []
    for seq_id, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        chunk_count = (end - start + chunk_size - 1) // chunk_size
        for chunk_id in range(chunk_count):
            indices.extend((seq_id, chunk_id))
    return indices


def _finalize_profile(profile: dict) -> dict:
    result = dict(profile)
    layout = result["layout"]
    cu_mode = result.get("cu_mode")
    if layout in {"TND", "NTD"} and cu_mode is None:
        cu_mode = "single"
    if cu_mode is not None:
        lengths = _sequence_lengths(
            int(result["T"]), str(cu_mode), int(result.get("seq_count", 4))
        )
        result["cu_seqlens"] = _prefix_sum(lengths)
        result["B"] = 1
    else:
        result["cu_seqlens"] = None

    if not result["use_gate_in_kernel"]:
        result["dt_bias"] = False
    result["chunk_indices"] = (
        _canonical_chunk_indices(result["cu_seqlens"], int(result["chunk_size"]))
        if result.get("explicit_chunk_indices")
        else None
    )
    result["case_key"] = (
        f"{result['category']}_{result['label']}_{result['q_dtype']}_"
        f"{layout.lower()}_c{result['chunk_size']}"
    )
    _validate_profile(result)
    return result


def _base_profile(q_dtype: str, category: str, label: str, ordinal: int) -> dict:
    h_num, hv_num = HEAD_PROFILES[ordinal % len(HEAD_PROFILES)]
    k_dim, v_dim = DIM_PROFILES[ordinal % len(DIM_PROFILES)]
    use_gate = ordinal % 3 != 0
    return {
        "q_dtype": q_dtype,
        "g_dtype": G_DTYPES[(ordinal // 2) % len(G_DTYPES)],
        "beta_dtype": BETA_DTYPES[ordinal % len(BETA_DTYPES)],
        "category": category,
        "label": label,
        "distribution": "uniform" if ordinal % 2 == 0 else "normal",
        "layout": LAYOUTS[ordinal % len(LAYOUTS)],
        "B": 1 if ordinal % 3 == 0 else 2,
        "H": h_num,
        "HV": hv_num,
        "T": (7, 16, 31, 65, 127, 129)[ordinal % 6],
        "K": k_dim,
        "V": v_dim,
        "chunk_size": CHUNK_SIZES[ordinal % len(CHUNK_SIZES)],
        "scale": 1.0 / math.sqrt(k_dim),
        "initial_state": ordinal % 2 == 0,
        "output_final_state": ordinal % 3 == 0,
        "cu_mode": None,
        "seq_count": 2 + ordinal % 3,
        "explicit_chunk_indices": False,
        "safe_gate": ordinal % 2 == 0,
        "lower_bound": (-5.0, -1.0, -0.001)[ordinal % 3],
        "use_gate_in_kernel": use_gate,
        "dt_bias": use_gate and ordinal % 2 == 0,
        "disable_recompute": ordinal % 2 == 1,
        "return_intermediate_states": ordinal % 4 < 2,
        "state_v_first": ordinal % 2 == 0,
    }


def _build_core_profiles(q_dtype: str) -> list[dict]:
    profiles = []
    t_values = {
        64: (63, 64, 65, 129),
        128: (127, 128, 129, 257),
    }
    for ordinal, (layout, chunk_size, g_dtype, beta_dtype) in enumerate(
        product(LAYOUTS, CHUNK_SIZES, G_DTYPES, BETA_DTYPES)
    ):
        dtype_slot = G_DTYPES.index(g_dtype) * len(BETA_DTYPES) + BETA_DTYPES.index(
            beta_dtype
        )
        profile = _base_profile(
            q_dtype,
            "core",
            f"{layout.lower()}_c{chunk_size}_{g_dtype}_{beta_dtype}",
            ordinal,
        )
        h_num, hv_num = HEAD_PROFILES[
            (ordinal + dtype_slot) % len(HEAD_PROFILES)
        ]
        k_dim, v_dim = DIM_PROFILES[
            (ordinal // 2 + dtype_slot) % len(DIM_PROFILES)
        ]
        if layout in {"TND", "NTD"}:
            cu_mode = CU_MODES[dtype_slot % len(CU_MODES)]
        elif dtype_slot >= 2:
            cu_mode = ("balanced", "zero_length")[dtype_slot - 2]
        else:
            cu_mode = None
        profile.update(
            {
                "g_dtype": g_dtype,
                "beta_dtype": beta_dtype,
                "layout": layout,
                "B": 1 if cu_mode is not None else 1 + dtype_slot % 2,
                "H": h_num,
                "HV": hv_num,
                "T": t_values[chunk_size][dtype_slot],
                "K": k_dim,
                "V": v_dim,
                "chunk_size": chunk_size,
                "scale": 1.0 / math.sqrt(k_dim),
                "initial_state": dtype_slot == 1
                and hv_num * k_dim * v_dim <= 524_288,
                "output_final_state": dtype_slot in {2, 3},
                "cu_mode": cu_mode,
                "seq_count": 4,
                "explicit_chunk_indices": cu_mode is not None
                and dtype_slot % 2 == 0,
                "safe_gate": dtype_slot in {0, 1},
                "lower_bound": (-5.0, -1.0, -0.001, -5.0)[dtype_slot],
                "use_gate_in_kernel": dtype_slot % 2 == 0,
                "dt_bias": dtype_slot == 0,
                "disable_recompute": (ordinal + dtype_slot) % 2 == 1,
                "return_intermediate_states": (ordinal + dtype_slot) % 4 < 2,
                "state_v_first": (ordinal + dtype_slot) % 2 == 0,
            }
        )
        profiles.append(_finalize_profile(profile))
    assert len(profiles) == 32
    return profiles


def _build_finite_profiles(q_dtype: str) -> list[dict]:
    profiles = []
    for target_index, (input_name, target_dtype) in enumerate(FINITE_TARGETS):
        for value_index, finite_value in enumerate(FINITE_VALUES):
            ordinal = target_index * len(FINITE_VALUES) + value_index
            profile = _base_profile(
                q_dtype,
                "finite_value",
                f"{input_name}_{target_dtype or q_dtype}_{finite_value:g}",
                ordinal,
            )
            layout = LAYOUTS[(target_index + value_index) % len(LAYOUTS)]
            use_gate = (
                input_name in {"A_log", "dt_bias"}
                or (input_name == "g" and finite_value > 0)
                or ordinal % 2 == 0
            )
            cu_mode = (
                CU_MODES[(target_index + value_index) % len(CU_MODES)]
                if layout in {"TND", "NTD"}
                else ("balanced" if ordinal % 5 == 0 else None)
            )
            g_dtype = (
                target_dtype
                if input_name == "g"
                else G_DTYPES[(target_index + value_index) % len(G_DTYPES)]
            )
            beta_dtype = (
                target_dtype
                if input_name == "beta"
                else BETA_DTYPES[(target_index + value_index + 1) % len(BETA_DTYPES)]
            )
            profile.update(
                {
                    "finite_input": input_name,
                    "finite_dtype": target_dtype or q_dtype,
                    "finite_value": finite_value,
                    "g_dtype": g_dtype,
                    "beta_dtype": beta_dtype,
                    "layout": layout,
                    "B": 1 if cu_mode is not None else 1 + ordinal % 2,
                    "T": (7, 16, 31, 65, 127, 129)[ordinal % 6],
                    "chunk_size": CHUNK_SIZES[(target_index + value_index) % 2],
                    "initial_state": input_name == "initial_state",
                    "output_final_state": ordinal % 3 == 1,
                    "cu_mode": cu_mode,
                    "seq_count": 2 + ordinal % 3,
                    "explicit_chunk_indices": cu_mode is not None
                    and ordinal % 2 == 0,
                    "use_gate_in_kernel": use_gate,
                    "dt_bias": input_name == "dt_bias"
                    or (use_gate and ordinal % 3 == 0),
                    "safe_gate": ordinal % 2 == 0,
                    "disable_recompute": ordinal % 2 == 1,
                    "return_intermediate_states": ordinal % 4 < 2,
                    "state_v_first": ordinal % 2 == 0,
                }
            )
            profiles.append(_finalize_profile(profile))
    assert len(profiles) == 30
    return profiles


def _model_anchor(layout: str) -> dict:
    return {
        "label": f"model_{layout.lower()}",
        "layout": layout,
        "B": 1,
        "H": 96,
        "HV": 96,
        "T": 1024,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "g_dtype": "fp32",
        "beta_dtype": "bf16",
        "initial_state": False,
        "output_final_state": False,
        "cu_mode": "balanced" if layout in {"TND", "NTD"} else None,
        "seq_count": 4,
        "explicit_chunk_indices": layout in {"TND", "NTD"},
        "safe_gate": True,
        "lower_bound": -5.0,
        "use_gate_in_kernel": True,
        "dt_bias": True,
        "disable_recompute": False,
        "return_intermediate_states": False,
        "state_v_first": True,
    }


def _mss_anchor(layout: str, gate_in_kernel: bool, ordinal: int) -> dict:
    chunk_size = CHUNK_SIZES[ordinal % 2]
    if layout in {"TND", "NTD"}:
        cu_mode = ("balanced", "zero_length")[ordinal % 2]
    elif ordinal % 2 == 0:
        cu_mode = None
    else:
        cu_mode = "balanced"
    return {
        "label": f"mss_{layout.lower()}_gate{int(gate_in_kernel)}",
        "layout": layout,
        "B": 1 if cu_mode is not None else 2,
        "H": 2,
        "HV": 4,
        "T": 65 if chunk_size == 64 else 129,
        "K": 128,
        "V": 128,
        "chunk_size": chunk_size,
        "initial_state": True,
        "output_final_state": ordinal % 2 == 0,
        "cu_mode": cu_mode,
        "seq_count": 3,
        "explicit_chunk_indices": cu_mode is not None and ordinal % 2 == 1,
        "safe_gate": ordinal % 2 == 0,
        "lower_bound": (-5.0, -1.0)[ordinal % 2],
        "use_gate_in_kernel": gate_in_kernel,
        "dt_bias": gate_in_kernel and ordinal % 2 == 0,
        "disable_recompute": ordinal % 2 == 1,
        "return_intermediate_states": ordinal % 2 == 0,
        "state_v_first": ordinal % 2 == 0,
    }


def _boundary_scenarios() -> list[dict]:
    scenarios = []
    for chunk_size, token_values in (
        (64, (1, 63, 64, 65, 127, 128, 129, 255)),
        (128, (1, 127, 128, 129, 255, 256, 257)),
    ):
        for total_t in token_values:
            scenarios.append(
                {
                    "label": f"t{total_t}_c{chunk_size}",
                    "T": total_t,
                    "chunk_size": chunk_size,
                }
            )

    scenarios.extend(_model_anchor(layout) for layout in LAYOUTS)
    for ordinal, (layout, gate_in_kernel) in enumerate(
        product(LAYOUTS, (False, True))
    ):
        scenarios.append(_mss_anchor(layout, gate_in_kernel, ordinal))

    scenarios.extend(
        [
            {"label": "head_min", "H": 1, "HV": 1, "K": 16, "V": 16},
            {
                "label": "gqa_max_ratio",
                "H": 1,
                "HV": 128,
                "K": 16,
                "V": 16,
                "safe_gate": True,
                "disable_recompute": True,
                "state_v_first": True,
            },
            {
                "label": "head_dim_max",
                "H": 128,
                "HV": 128,
                "T": 1,
                "K": 256,
                "V": 256,
                "layout": "BNSD",
                "initial_state": False,
            },
            {
                "label": "kmin_vmax",
                "K": 16,
                "V": 256,
                "safe_gate": True,
                "state_v_first": False,
            },
            {"label": "kmax_vmin", "K": 256, "V": 16},
            {"label": "k128_v256", "K": 128, "V": 256},
            {
                "label": "rank4_zero_length",
                "layout": "BSND",
                "B": 1,
                "cu_mode": "zero_length",
                "initial_state": True,
                "explicit_chunk_indices": True,
            },
            {
                "label": "many_1024_sequences",
                "layout": "TND",
                "T": 1,
                "H": 1,
                "HV": 1,
                "K": 16,
                "V": 16,
                "cu_mode": "many_1024",
                "initial_state": False,
                "explicit_chunk_indices": True,
            },
            {
                "label": "state_kv_output",
                "initial_state": True,
                "output_final_state": True,
                "state_v_first": False,
            },
            {
                "label": "state_vk_output",
                "initial_state": True,
                "output_final_state": True,
                "state_v_first": True,
            },
            {
                "label": "gate_no_dt_intermediate",
                "use_gate_in_kernel": True,
                "dt_bias": False,
                "disable_recompute": False,
                "return_intermediate_states": True,
            },
        ]
    )
    assert len(scenarios) == 38
    return scenarios


def _build_boundary_profiles(q_dtype: str) -> list[dict]:
    profiles = []
    for ordinal, scenario in enumerate(_boundary_scenarios()):
        profile = _base_profile(
            q_dtype, "boundary", str(scenario["label"]), ordinal
        )
        profile.update(scenario)
        if "initial_state" not in scenario:
            profile["initial_state"] = False
        layout = profile["layout"]
        if "cu_mode" not in scenario and layout in {"TND", "NTD"}:
            profile["cu_mode"] = "single"
        if profile.get("cu_mode") is not None:
            profile["B"] = 1
        profile["g_dtype"] = scenario.get(
            "g_dtype", G_DTYPES[(ordinal // 2) % len(G_DTYPES)]
        )
        profile["beta_dtype"] = scenario.get(
            "beta_dtype", BETA_DTYPES[ordinal % len(BETA_DTYPES)]
        )
        profile["distribution"] = "uniform" if ordinal % 2 == 0 else "normal"
        profile["scale"] = 1.0 / math.sqrt(int(profile["K"]))
        profiles.append(_finalize_profile(profile))
    assert len(profiles) == 38
    return profiles


def _build_long_regression_profiles() -> list[dict]:
    profiles = []
    for ordinal, (total_t, disable_recompute) in enumerate(
        product(LONG_T_VALUES, (False, True))
    ):
        mode = "export" if disable_recompute else "recompute"
        profile = _base_profile(
            "bf16",
            "long_regression",
            f"mixed_tail_t{total_t}_{mode}",
            ordinal,
        )
        profile.update(
            {
                "distribution": (
                    "uniform"
                    if LONG_T_VALUES.index(total_t) % 2 == 0
                    else "normal"
                ),
                "layout": "BSND",
                "B": 1,
                "H": 96,
                "HV": 96,
                "T": total_t,
                "K": 128,
                "V": 128,
                "chunk_size": 64,
                "scale": 1.0 / math.sqrt(128),
                "g_dtype": "fp32",
                "beta_dtype": "bf16",
                "initial_state": False,
                "output_final_state": False,
                "cu_mode": "mixed_tail8",
                "seq_count": 8,
                "explicit_chunk_indices": False,
                "safe_gate": True,
                "lower_bound": -5.0,
                "use_gate_in_kernel": True,
                "dt_bias": True,
                "disable_recompute": disable_recompute,
                "return_intermediate_states": False,
                "state_v_first": True,
            }
        )
        profiles.append(_finalize_profile(profile))
    assert len(profiles) == 12
    return profiles


def _profile_tensor_dtype(profile: dict, name: str) -> str:
    if name in {"q", "k", "v"}:
        return str(profile["q_dtype"])
    if name == "g":
        return str(profile["g_dtype"])
    if name == "beta":
        return str(profile["beta_dtype"])
    return "fp32"


def _tensor_present(profile: dict, name: str) -> bool:
    if name == "initial_state":
        return bool(profile["initial_state"])
    if name == "A_log":
        return bool(profile["use_gate_in_kernel"])
    if name == "dt_bias":
        return bool(profile["dt_bias"])
    return True


def _case_shapes(profile: dict) -> dict[str, list[int]]:
    layout = str(profile["layout"])
    batch = int(profile["B"])
    total_t = int(profile["T"])
    h_num = int(profile["H"])
    hv_num = int(profile["HV"])
    k_dim = int(profile["K"])
    v_dim = int(profile["V"])

    if layout == "BSND":
        q_shape = [batch, total_t, h_num, k_dim]
        v_shape = [batch, total_t, hv_num, v_dim]
        g_shape = [batch, total_t, hv_num, k_dim]
        beta_shape = [batch, total_t, hv_num]
    elif layout == "BNSD":
        q_shape = [batch, h_num, total_t, k_dim]
        v_shape = [batch, hv_num, total_t, v_dim]
        g_shape = [batch, hv_num, total_t, k_dim]
        beta_shape = [batch, hv_num, total_t]
    elif layout == "TND":
        q_shape = [total_t, h_num, k_dim]
        v_shape = [total_t, hv_num, v_dim]
        g_shape = [total_t, hv_num, k_dim]
        beta_shape = [total_t, hv_num]
    elif layout == "NTD":
        q_shape = [h_num, total_t, k_dim]
        v_shape = [hv_num, total_t, v_dim]
        g_shape = [hv_num, total_t, k_dim]
        beta_shape = [hv_num, total_t]
    else:
        raise ValueError(f"unsupported layout: {layout}")

    cu_seqlens = profile["cu_seqlens"]
    seq_num = len(cu_seqlens) - 1 if cu_seqlens is not None else batch
    state_shape = (
        [seq_num, hv_num, v_dim, k_dim]
        if profile["state_v_first"]
        else [seq_num, hv_num, k_dim, v_dim]
    )
    return {
        "q": q_shape,
        "k": list(q_shape),
        "v": v_shape,
        "g": g_shape,
        "beta": beta_shape,
        "initial_state": state_shape,
        "A_log": [hv_num],
        "dt_bias": [hv_num * k_dim],
    }


def _is_finite_target(profile: dict, name: str) -> bool:
    return (
        profile["category"] == "finite_value"
        and profile["finite_input"] == name
        and profile["finite_dtype"] == _profile_tensor_dtype(profile, name)
    )


def _normal_range_for(profile: dict, name: str) -> dict:
    if name in {"q", "k"}:
        mean, std = 0.0, 1.0 / math.sqrt(int(profile["K"]))
    else:
        mean, std = NORMAL_PARAMS[name]
    return {"name": "nd", "mean": [mean], "std": [std]}


def _range_for(profile: dict, name: str):
    if _is_finite_target(profile, name):
        return float(profile["finite_value"])
    if name == "g" and not profile["use_gate_in_kernel"]:
        return list(PREGATED_RANGE)
    distribution = profile.get("tensor_distributions", {}).get(
        name, profile["distribution"]
    )
    if distribution == "normal":
        return _normal_range_for(profile, name)
    return list(UNIFORM_RANGES[name])


def _validate_profile(profile: dict) -> None:
    if profile["q_dtype"] not in Q_DTYPES:
        raise ValueError(f"unsupported q dtype: {profile['q_dtype']}")
    if profile["g_dtype"] not in G_DTYPES:
        raise ValueError(f"unsupported g dtype: {profile['g_dtype']}")
    if profile["beta_dtype"] not in BETA_DTYPES:
        raise ValueError(f"unsupported beta dtype: {profile['beta_dtype']}")
    if profile["layout"] not in LAYOUTS:
        raise ValueError(f"unsupported layout: {profile['layout']}")
    if profile["chunk_size"] not in CHUNK_SIZES:
        raise ValueError(f"unsupported chunk size: {profile['chunk_size']}")
    if not 0 < profile["H"] <= profile["HV"] <= 128:
        raise ValueError(f"invalid head relation: H={profile['H']}, HV={profile['HV']}")
    if profile["HV"] % profile["H"] != 0:
        raise ValueError(f"HV must be divisible by H: {profile}")
    for name in ("K", "V"):
        value = int(profile[name])
        if not 16 <= value <= 256 or value % 16:
            raise ValueError(f"{name} must be a 16-aligned value in [16, 256]")
    if int(profile["T"]) <= 0 or int(profile["B"]) <= 0:
        raise ValueError(f"B and T must be positive: {profile}")
    if profile["safe_gate"] and not -5.0 <= profile["lower_bound"] < 0.0:
        raise ValueError(f"invalid safe-gate lower_bound: {profile['lower_bound']}")
    if not profile["use_gate_in_kernel"] and profile["dt_bias"]:
        raise ValueError("dt_bias must be absent when gate is already precomputed")

    cu_seqlens = profile["cu_seqlens"]
    if profile["layout"] in {"TND", "NTD"} and cu_seqlens is None:
        raise ValueError(f"{profile['layout']} requires cu_seqlens")
    if cu_seqlens is not None:
        if (
            len(cu_seqlens) < 2
            or cu_seqlens[0] != 0
            or cu_seqlens[-1] != profile["T"]
            or any(left > right for left, right in zip(cu_seqlens, cu_seqlens[1:]))
        ):
            raise ValueError(f"invalid cu_seqlens: {cu_seqlens}")
        if len(cu_seqlens) - 1 > 1024:
            raise ValueError("cu_seqlens exceeds the 1024-sequence limit")
        if profile["layout"] in {"BSND", "BNSD"} and profile["B"] != 1:
            raise ValueError("rank-4 varlen input requires B=1")

    expected_indices = _canonical_chunk_indices(
        cu_seqlens, int(profile["chunk_size"])
    )
    if profile["explicit_chunk_indices"]:
        if profile["chunk_indices"] != expected_indices:
            raise ValueError("chunk_indices must use canonical sequence-major order")
    elif profile["chunk_indices"] is not None:
        raise ValueError("implicit chunk_indices must be null")

    if profile["category"] == "finite_value":
        target_name = profile["finite_input"]
        if not _tensor_present(profile, target_name):
            raise ValueError(f"finite target {target_name} must be present")
        if _profile_tensor_dtype(profile, target_name) != profile["finite_dtype"]:
            raise ValueError(f"finite target dtype mismatch: {profile}")
        if target_name == "g" and profile["finite_value"] > 0:
            if not profile["use_gate_in_kernel"]:
                raise ValueError("positive g constants are valid only for raw-g cases")


def _is_distribution_sample(profile: dict, name: str) -> bool:
    if not _tensor_present(profile, name):
        return False
    if name == "g" and not profile["use_gate_in_kernel"]:
        return False
    return not _is_finite_target(profile, name)


def _distribution_key(profile: dict, name: str) -> tuple[str, str, str]:
    return (
        str(profile["q_dtype"]),
        name,
        _profile_tensor_dtype(profile, name),
    )


def _assign_tensor_distributions(profiles: list[dict]) -> None:
    counts = Counter()
    for profile in profiles:
        assigned = {}
        for name in TENSOR_INPUTS:
            if not _is_distribution_sample(profile, name):
                continue
            key = _distribution_key(profile, name)
            if profile["category"] == "long_regression":
                assigned[name] = profile["distribution"]
            else:
                assigned[name] = "uniform" if counts[key] % 2 == 0 else "normal"
            counts[key] += 1
        profile["tensor_distributions"] = assigned


def _profile_seed(profile: dict, index: int) -> int:
    if profile["category"] == "long_regression":
        return LONG_SEEDS[int(profile["T"])]
    return SEED_BASE + index


def _distribution_counts(
    profiles: list[dict],
) -> dict[tuple[str, str, str], Counter]:
    counts = {}
    for profile in profiles:
        for name in TENSOR_INPUTS:
            if not _is_distribution_sample(profile, name):
                continue
            key = _distribution_key(profile, name)
            distribution = profile["tensor_distributions"].get(name)
            if distribution not in {"uniform", "normal"}:
                raise AssertionError(
                    f"missing regular distribution for {key}: {profile['case_key']}"
                )
            counts.setdefault(key, Counter())[distribution] += 1
    return counts


def _expand_view_profiles(profiles: list[dict]) -> list[dict]:
    """为部分用例派生“尾轴非连续视图”变体（slice / offset）。

    变体与 base 共用 seed 与值域，只改变 q/k/v 的传入形态：
      slice  —— qkv 拼接后按 head 轴切分，视图整体非连续；
      offset —— 再前置一段无关 head，使 view 的 storage offset 非 0。
    """
    expanded = []
    for index, profile in enumerate(profiles):
        base = dict(profile)
        base.setdefault("view_mode", "dense")
        expanded.append(base)
        if index % 4 != 0:
            continue
        for mode in ("slice", "offset"):
            variant = dict(base)
            variant["view_mode"] = mode
            variant["label"] = f"{base['label']}_{mode}"
            variant["case_key"] = f"{base['case_key']}_{mode}"
            expanded.append(variant)
    return expanded


def _build_profiles() -> list[dict]:
    bf16_boundaries = [
        profile
        for profile in _build_boundary_profiles("bf16")
        if not str(profile["label"]).startswith("t")
        or profile["label"] in BF16_TOKEN_BOUNDARY_LABELS
    ]
    assert len(bf16_boundaries) == 26
    by_dtype = {
        "bf16": (
            _build_core_profiles("bf16")
            + _build_finite_profiles("bf16")
            + bf16_boundaries
            + _build_long_regression_profiles()
        ),
        "fp16": (
            _build_core_profiles("fp16")
            + _build_finite_profiles("fp16")
            + _build_boundary_profiles("fp16")
        ),
    }
    assert all(len(profiles) == 100 for profiles in by_dtype.values())
    profiles = [
        by_dtype[q_dtype][profile_index]
        for profile_index in range(100)
        for q_dtype in Q_DTYPES
    ]
    _assign_tensor_distributions(profiles)
    _validate_coverage(profiles)
    return _expand_view_profiles(profiles)


def _validate_coverage(profiles: list[dict]) -> None:
    if len(profiles) != 200:
        raise AssertionError(f"expected 200 profiles, got {len(profiles)}")
    general_profiles = [
        profile for profile in profiles if profile["category"] != "long_regression"
    ]
    if Counter(profile["q_dtype"] for profile in profiles) != Counter(
        {"bf16": 100, "fp16": 100}
    ):
        raise AssertionError("each q dtype must own exactly 100 profiles")
    if len({profile["case_key"] for profile in profiles}) != len(profiles):
        raise AssertionError("case keys must be unique")

    long_profiles = [
        profile for profile in profiles if profile["category"] == "long_regression"
    ]
    expected_long = set(product(LONG_T_VALUES, (False, True)))
    actual_long = {
        (int(profile["T"]), bool(profile["disable_recompute"]))
        for profile in long_profiles
    }
    if len(long_profiles) != len(expected_long) or actual_long != expected_long:
        raise AssertionError("long regression must cover every T/recompute pair")
    for profile in long_profiles:
        expected = {
            "q_dtype": "bf16",
            "g_dtype": "fp32",
            "beta_dtype": "bf16",
            "layout": "BSND",
            "B": 1,
            "H": 96,
            "HV": 96,
            "K": 128,
            "V": 128,
            "chunk_size": 64,
            "initial_state": False,
            "output_final_state": False,
            "explicit_chunk_indices": False,
            "safe_gate": True,
            "lower_bound": -5.0,
            "use_gate_in_kernel": True,
            "dt_bias": True,
            "return_intermediate_states": False,
            "state_v_first": True,
        }
        if any(profile[key] != value for key, value in expected.items()):
            raise AssertionError(f"bad long regression profile: {profile['case_key']}")
        expected_cu = _prefix_sum(
            _sequence_lengths(int(profile["T"]), "mixed_tail8", 8)
        )
        if profile["cu_seqlens"] != expected_cu:
            raise AssertionError(f"bad long regression offsets: {profile['case_key']}")
        expected_distribution = (
            "uniform" if LONG_T_VALUES.index(int(profile["T"])) % 2 == 0 else "normal"
        )
        if profile["distribution"] != expected_distribution:
            raise AssertionError(f"bad long regression distribution: {profile['case_key']}")
        for name in TENSOR_INPUTS:
            if _is_distribution_sample(profile, name) and profile[
                "tensor_distributions"
            ].get(name) != expected_distribution:
                raise AssertionError(
                    f"bad long tensor distribution for {name}: {profile['case_key']}"
                )

    for key, counts in sorted(_distribution_counts(profiles).items()):
        difference = abs(counts["uniform"] - counts["normal"])
        if difference > 1:
            raise AssertionError(
                f"unbalanced regular distributions for {key}: {dict(counts)}"
            )

    all_bool_pairs = set(product((False, True), repeat=2))
    for left, right in combinations(BOOL_ATTRS, 2):
        observed = {
            (bool(profile[left]), bool(profile[right]))
            for profile in general_profiles
        }
        required = all_bool_pairs - BOOL_PAIRWISE_EXEMPTIONS.get((left, right), set())
        missing = sorted(required - observed)
        if missing:
            raise AssertionError(
                f"incomplete bool pairwise coverage for {left}/{right}: {missing}"
            )

    expected_core = set(product(LAYOUTS, CHUNK_SIZES, G_DTYPES, BETA_DTYPES))
    token_boundaries = {1, 63, 64, 65, 127, 128, 129, 255, 256, 257}
    required_heads = {(1, 1), (1, 128), (32, 128), (96, 96), (128, 128)}
    required_dims = {
        (16, 16),
        (16, 256),
        (128, 128),
        (128, 256),
        (256, 16),
        (256, 256),
    }
    for q_dtype in Q_DTYPES:
        selected = [
            profile
            for profile in profiles
            if profile["q_dtype"] == q_dtype
        ]
        categories = Counter(profile["category"] for profile in selected)
        expected_categories = (
            Counter(
                {
                    "core": 32,
                    "finite_value": 30,
                    "boundary": 26,
                    "long_regression": 12,
                }
            )
            if q_dtype == "bf16"
            else Counter({"core": 32, "finite_value": 30, "boundary": 38})
        )
        if categories != expected_categories:
            raise AssertionError(f"bad category quota for {q_dtype}: {categories}")
        core_profiles = [
            profile for profile in selected if profile["category"] == "core"
        ]
        core = {
            (
                profile["layout"],
                profile["chunk_size"],
                profile["g_dtype"],
                profile["beta_dtype"],
            )
            for profile in core_profiles
        }
        if len(core_profiles) != len(expected_core) or core != expected_core:
            raise AssertionError(f"incomplete core product for {q_dtype}")
        finite = {
            (
                profile["finite_input"],
                profile["finite_dtype"],
                float(profile["finite_value"]),
            )
            for profile in selected
            if profile["category"] == "finite_value"
        }
        expected_finite = {
            (name, dtype or q_dtype, float(value))
            for name, dtype in FINITE_TARGETS
            for value in FINITE_VALUES
        }
        if finite != expected_finite:
            raise AssertionError(f"incomplete finite-value product for {q_dtype}")
        regular = [
            profile for profile in selected if profile["category"] != "finite_value"
        ]
        if Counter(profile["distribution"] for profile in regular) != Counter(
            {"uniform": 35, "normal": 35}
        ):
            raise AssertionError(f"regular distributions are not 50/50 for {q_dtype}")
        if not token_boundaries.issubset({profile["T"] for profile in selected}):
            raise AssertionError(f"token boundaries are incomplete for {q_dtype}")
        if not required_heads.issubset(
            {(profile["H"], profile["HV"]) for profile in selected}
        ):
            raise AssertionError(f"head boundaries are incomplete for {q_dtype}")
        if not required_dims.issubset(
            {(profile["K"], profile["V"]) for profile in selected}
        ):
            raise AssertionError(f"K/V boundaries are incomplete for {q_dtype}")
        for key in (
            "initial_state",
            "output_final_state",
            "explicit_chunk_indices",
            "safe_gate",
            "use_gate_in_kernel",
            "dt_bias",
            "disable_recompute",
            "return_intermediate_states",
            "state_v_first",
        ):
            if {bool(profile[key]) for profile in selected} != {False, True}:
                raise AssertionError(f"{key} does not cover both values for {q_dtype}")
        if {profile["layout"] for profile in selected} != set(LAYOUTS):
            raise AssertionError(f"layout coverage is incomplete for {q_dtype}")
        if {profile["chunk_size"] for profile in selected} != set(CHUNK_SIZES):
            raise AssertionError(f"chunk coverage is incomplete for {q_dtype}")
        if {profile["g_dtype"] for profile in selected} != set(G_DTYPES):
            raise AssertionError(f"g dtype coverage is incomplete for {q_dtype}")
        if {profile["beta_dtype"] for profile in selected} != set(BETA_DTYPES):
            raise AssertionError(f"beta dtype coverage is incomplete for {q_dtype}")

    model_profiles = [
        profile
        for profile in profiles
        if profile["category"] == "boundary"
        and str(profile["label"]).startswith("model_")
    ]
    expected_models = set(product(Q_DTYPES, LAYOUTS))
    actual_models = {
        (profile["q_dtype"], profile["layout"]) for profile in model_profiles
    }
    if len(model_profiles) != len(expected_models) or actual_models != expected_models:
        raise AssertionError("model anchors must cover every q dtype/layout pair")
    for profile in model_profiles:
        expected = {
            "g_dtype": "fp32",
            "beta_dtype": "bf16",
            "B": 1,
            "T": 1024,
            "H": 96,
            "HV": 96,
            "K": 128,
            "V": 128,
            "chunk_size": 64,
            "scale": 1.0 / math.sqrt(128),
            "initial_state": False,
            "output_final_state": False,
            "safe_gate": True,
            "lower_bound": -5.0,
            "use_gate_in_kernel": True,
            "dt_bias": True,
            "disable_recompute": False,
            "return_intermediate_states": False,
            "state_v_first": True,
        }
        if any(profile[key] != value for key, value in expected.items()):
            raise AssertionError(f"bad model anchor: {profile['case_key']}")
        packed = profile["layout"] in {"TND", "NTD"}
        expected_cu = [0, 256, 512, 768, 1024] if packed else None
        expected_indices = (
            _canonical_chunk_indices(expected_cu, 64) if packed else None
        )
        if profile["cu_seqlens"] != expected_cu:
            raise AssertionError(f"bad model offsets: {profile['case_key']}")
        if profile["chunk_indices"] != expected_indices:
            raise AssertionError(f"bad model chunk indices: {profile['case_key']}")

    bf16_mss = [
        profile
        for profile in profiles
        if profile["q_dtype"] == "bf16"
        and profile["category"] == "boundary"
        and str(profile["label"]).startswith("mss_")
    ]
    expected_mss = set(product(LAYOUTS, CHUNK_SIZES))
    actual_mss = {
        (profile["layout"], profile["chunk_size"]) for profile in bf16_mss
    }
    if len(bf16_mss) != 8 or actual_mss != expected_mss:
        raise AssertionError("BF16 MSS anchors must cover every layout/chunk pair")
    for profile in bf16_mss:
        if not profile["initial_state"]:
            raise AssertionError("every MSS anchor requires initial_state")
        if profile["T"] != profile["chunk_size"] + 1:
            raise AssertionError("MSS anchors must cross a chunk boundary by one token")
        if profile["use_gate_in_kernel"] is not (
            profile["chunk_size"] == 128
        ):
            raise AssertionError("MSS chunk pairs must cover both gate modes")
    if {profile["state_v_first"] for profile in bf16_mss} != {False, True}:
        raise AssertionError("MSS anchors must cover both state orientations")
    if not any(profile["cu_seqlens"] is None for profile in bf16_mss):
        raise AssertionError("MSS anchors require a dense case")
    if not any(profile["cu_seqlens"] is not None for profile in bf16_mss):
        raise AssertionError("MSS anchors require a varlen case")
    if not any(profile["chunk_indices"] is not None for profile in bf16_mss):
        raise AssertionError("MSS anchors require explicit chunk indices")

    for profile in profiles:
        shapes = _case_shapes(profile)
        total_numel = 0
        for name, shape in shapes.items():
            if _tensor_present(profile, name):
                size = 1
                for dimension in shape:
                    size *= int(dimension)
                total_numel += size
        large_anchor = profile["category"] == "long_regression" or str(
            profile["label"]
        ).startswith("model_")
        if total_numel > 13_000_000 and not large_anchor:
            raise AssertionError(
                f"profile exceeds the 13M-element input budget: {profile['case_key']}"
            )


PROFILES = _build_profiles()


def build_profiles() -> list[dict]:
    """Return copies of the frozen profiles for tests and subset builders."""

    return [dict(profile) for profile in PROFILES]


def _input_name(item) -> str:
    config = item[0] if isinstance(item, list) else item
    return str(config.name)


def _attrs_configs(name: str, values: list[int] | None):
    if InputCaseConfig is None:
        raise RuntimeError("ATK is required to construct attrs inputs")
    if values is None:
        # attrs cannot be empty. The executor normalizes ["null"] to None.
        return [
            InputCaseConfig(
                name=name,
                type="attrs",
                required=False,
                dtype="string",
                shape=None,
                range_values="null",
                backward=False,
            )
        ]
    return [
        InputCaseConfig(
            name=name,
            type="attrs",
            required=False,
            dtype="int",
            shape=None,
            range_values=int(value),
            backward=False,
        )
        for value in values
    ]


def _configure_case(case_config, profile: dict, index: int):
    if [_input_name(item) for item in case_config.inputs] != list(EXPECTED_INPUTS):
        raise ValueError("YAML input order no longer matches the direct-input contract")
    shapes = _case_shapes(profile)
    list_values = {
        "cu_seqlens": profile["cu_seqlens"],
        "chunk_indices": profile["chunk_indices"],
    }
    for position, item in enumerate(case_config.inputs):
        name = _input_name(item)
        if name in list_values:
            case_config.inputs[position] = _attrs_configs(name, list_values[name])
            continue
        config = item[0] if isinstance(item, list) else item
        config.backward = False
        config.align_32B = None
        config.outlier_values = None
        if name in TENSOR_INPUTS:
            config.shape = shapes[name]
            config.dtype = _profile_tensor_dtype(profile, name)
            config.required = name not in OPTIONAL_TENSORS
            config.range_values = (
                _range_for(profile, name)
                if _tensor_present(profile, name)
                else "null"
            )
            continue
        config.shape = None
        config.required = True
        if name == "scale":
            config.dtype = "float"
            config.range_values = float(profile["scale"])
        elif name == "chunk_size":
            config.dtype = "int"
            config.range_values = int(profile["chunk_size"])
        elif name == "layout":
            config.dtype = "string"
            config.range_values = str(profile["layout"])
        elif name == "lower_bound":
            config.dtype = "float"
            config.range_values = float(profile["lower_bound"])
        elif name in BOOL_ATTRS:
            config.dtype = "attr_bool"
            config.range_values = bool(profile[name])
        else:
            raise ValueError(f"unhandled direct input: {name}")

    case_config.id = index
    case_config.default_seed = _profile_seed(profile, index)
    case_config.name = (
        f"{OP_NAME}_{index:04d}_{profile['case_key']}_"
        f"h{profile['H']}_hv{profile['HV']}_t{profile['T']}_"
        f"k{profile['K']}_v{profile['V']}"
    )
    case_config.aclnn_name = "ChunkKdaFwd"
    case_config.api_type = "executor_chunk_kda_fwd"
    case_config.backward = False
    case_config.expected_error_msg = ""
    case_config.outputs = None
    case_config.save_name = OP_NAME
    case_config.is_boundary = profile["category"] != "core"
    return case_config

OP_NAME = OP_NAME
DTYPE_COMBOS = tuple(
    (q_dtype, g_dtype, beta_dtype)
    for q_dtype in Q_DTYPES
    for g_dtype in G_DTYPES
    for beta_dtype in BETA_DTYPES
)
LAYOUTS = tuple(LAYOUTS)
KV_TIERS = (64, 128)
CHUNK_SIZES = (64,)  # 只生成 64；128 属拦截项，反向用例见 build_negative_cases
HEADS = (
    (1, 1),
    (1, 2),
    (2, 2),
    (2, 4),
    (4, 4),
    (4, 8),
    (8, 8),
    (16, 16),
    (1, 4),
    (2, 8),
    (4, 16),
    (8, 32),
)
SMALL_HEADS = HEADS[:10]
SEQ_COUNTS = (2, 4, 8)

# label, T in chunk units, head slot, cu_mode, initial_state, out_final_state,
# safe_gate, lower_bound slot, use_gate_in_kernel, dt_bias, disable_recompute,
# return_intermediate_states, state_v_first, explicit chunk_indices,
# distribution, seq_count
VARIANTS = (
    ("t_c1_dense_bal", 1, 0, None, False, False, False, 0, False, False, False, False, False, False, "uniform", 2),
    ("t_c1m1_dense_bal", 1, 1, None, True, True, True, 1, True, True, True, True, True, False, "normal", 2),
    ("t_c1p1_dense_varlen", 1, 2, "balanced", True, False, True, 2, True, False, False, True, False, True, "uniform", 4),
    ("t_c2_dense_bal", 2, 3, None, False, True, False, 0, True, False, True, False, True, False, "normal", 2),
    ("t_c2m1_varlen_front", 2, 4, "front_short", True, False, True, 1, True, True, False, True, False, True, "uniform", 4),
    ("t_c2p1_dense_bal", 2, 5, None, True, True, False, 2, False, False, True, True, True, False, "normal", 2),
    ("t_c3_varlen_back", 3, 6, "back_short", False, True, True, 0, True, False, False, False, True, True, "uniform", 4),
    ("t_c3_dense_bal", 3, 7, None, True, False, False, 1, True, True, True, False, False, False, "normal", 2),
    ("t_c4_varlen_zero", 4, 8, "zero_length", True, True, True, 2, True, False, False, True, True, True, "uniform", 4),
    ("t_c4_dense_bal", 4, 9, None, False, False, True, 0, False, False, False, True, False, False, "normal", 2),
    ("t_c1_varlen_single", 1, 10, "single", True, True, False, 1, True, True, True, False, True, True, "uniform", 2),
    ("t_c2_dense_bal", 2, 11, None, True, False, True, 2, True, False, False, True, False, False, "normal", 2),
    ("t_c5_varlen_bal", 5, 0, "balanced", False, True, False, 0, True, True, True, False, True, False, "uniform", 8),
    ("t_c5_dense_bal", 5, 1, None, True, True, True, 1, False, False, False, True, True, False, "normal", 2),
    ("t_c6_varlen_front", 6, 2, "front_short", True, False, False, 2, True, False, True, False, False, True, "uniform", 8),
    ("t_c6_dense_bal", 6, 3, None, False, True, True, 0, True, True, False, True, True, False, "normal", 2),
)

BOUNDARY_T = (1, 3, 7, 15, 31)


def _profile(
    q_dtype: str,
    g_dtype: str,
    beta_dtype: str,
    layout: str,
    kv_dim: int,
    chunk_size: int,
    total_t: int,
    heads: tuple[int, int],
    cu_mode,
    initial_state: bool,
    output_final_state: bool,
    safe_gate: bool,
    lower_bound: float,
    use_gate_in_kernel: bool,
    dt_bias: bool,
    disable_recompute: bool,
    return_intermediate_states: bool,
    state_v_first: bool,
    explicit_chunk_indices: bool,
    distribution: str,
    seq_count: int,
    category: str,
    label: str,
    batch: int = 1,
) -> dict:
    h_num, hv_num = heads
    profile = {
        "q_dtype": q_dtype,
        "g_dtype": g_dtype,
        "beta_dtype": beta_dtype,
        "category": category,
        "label": label,
        "distribution": distribution,
        "layout": layout,
        "B": int(batch),
        "H": h_num,
        "HV": hv_num,
        "T": int(total_t),
        "K": int(kv_dim),
        "V": int(kv_dim),
        "chunk_size": int(chunk_size),
        "scale": 1.0 / (int(kv_dim) ** 0.5),
        "initial_state": bool(initial_state),
        "output_final_state": bool(output_final_state),
        "cu_mode": cu_mode,
        "seq_count": int(seq_count),
        "explicit_chunk_indices": bool(explicit_chunk_indices),
        "safe_gate": bool(safe_gate),
        "lower_bound": float(lower_bound),
        "use_gate_in_kernel": bool(use_gate_in_kernel),
        "dt_bias": bool(dt_bias),
        "disable_recompute": bool(disable_recompute),
        "return_intermediate_states": bool(return_intermediate_states),
        "state_v_first": bool(state_v_first),
    }
    return _finalize_profile(profile)


def _build_profiles() -> list[dict]:
    profiles: list[dict] = []
    lower_bounds = (-5.0, -1.0, -0.001)
    for q_dtype, g_dtype, beta_dtype in DTYPE_COMBOS:
        for layout in LAYOUTS:
            for kv_dim in KV_TIERS:
                for chunk_size in CHUNK_SIZES:
                    for slot, variant in enumerate(VARIANTS):
                        (label, units, head_slot, cu_mode, initial_state,
                         output_final_state, safe_gate, lb_slot, use_gate,
                         dt_bias, disable_recompute, return_intermediate,
                         state_v_first, explicit_indices, distribution,
                         seq_count) = variant
                        mode = cu_mode
                        if layout in {"TND", "NTD"} and mode is None:
                            mode = "single"
                        if layout not in {"TND", "NTD"} and slot % 2 == 0:
                            mode = None
                        profiles.append(
                            _profile(
                                q_dtype, g_dtype, beta_dtype, layout, kv_dim,
                                chunk_size, units * chunk_size + (slot % 3),
                                HEADS[head_slot], mode, initial_state,
                                output_final_state, safe_gate,
                                lower_bounds[lb_slot], use_gate, dt_bias,
                                disable_recompute, return_intermediate,
                                state_v_first, explicit_indices, distribution,
                                SEQ_COUNTS[seq_count % len(SEQ_COUNTS)],
                                "core", f"{label}_{layout.lower()}_kv{kv_dim}",
                                batch=1 if mode is not None else 1 + slot % 3,
                            )
                        )
        assert len(profiles) % 128 == 0, len(profiles)

    # Boundary cases: token-count edges, head extremes, empty sequences and the
    # gate/state attribute corners.  They are additive, not orthogonal, so the
    # regular part above stays a clean cross product.
    boundary_specs = []
    for layout in LAYOUTS:
        for kv_dim in KV_TIERS:
            for total_t in (1, 63, 65):
                boundary_specs.append((layout, kv_dim, 64, total_t, 1))
    for layout in LAYOUTS:
        for kv_dim in KV_TIERS:
            boundary_specs.append((layout, kv_dim, 64, 64, 2))
    for layout in LAYOUTS:
        for kv_dim in KV_TIERS:
            boundary_specs.append((layout, kv_dim, 64, 128, 4))
    for layout in LAYOUTS:
        for total_t, mode, explicit in (
            (64, "zero_length", True),
            (128, "balanced", True),
            (192, "front_short", False),
        ):
            boundary_specs.append((layout, 128, 64, total_t, 3))
    assert len(boundary_specs) == 52, len(boundary_specs)

    base_index = len(profiles)
    for offset, (layout, kv_dim, chunk_size, total_t, kind) in enumerate(boundary_specs):
        q_dtype = "bf16" if offset % 2 == 0 else "fp16"
        g_dtype = G_DTYPES[offset % len(G_DTYPES)]
        beta_dtype = BETA_DTYPES[offset % len(BETA_DTYPES)]
        if kind == 1:
            heads = SMALL_HEADS[offset % len(SMALL_HEADS)]
            cu_mode = None if layout in {"BSND", "BNSD"} else "single"
            initial_state, output_final_state = offset % 2 == 0, offset % 3 == 0
            safe_gate, lb = offset % 2 == 0, (-5.0, -1.0, -0.001)[offset % 3]
            use_gate, dt_bias = offset % 2 == 1, offset % 4 == 1
            disable_recompute, return_intermediate = offset % 2 == 1, offset % 3 != 2
            state_v_first, explicit = offset % 2 == 0, False
            distribution = "normal" if offset % 2 else "uniform"
            seq_count = 2
        elif kind == 2:
            heads = (1, 128) if offset % 2 == 0 else (128, 128)
            if offset % 4 == 2:
                heads = (1, 1)
            elif offset % 4 == 3:
                heads = (64, 128)
            cu_mode = None if layout in {"BSND", "BNSD"} else "single"
            initial_state, output_final_state = True, True
            safe_gate, lb = True, -1.0
            use_gate, dt_bias = True, offset % 2 == 0
            disable_recompute, return_intermediate = False, True
            state_v_first, explicit = offset % 2 == 0, False
            distribution = "uniform"
            seq_count = 2
        else:
            heads = SMALL_HEADS[offset % len(SMALL_HEADS)]
            cu_mode = mode
            initial_state, output_final_state = True, offset % 2 == 0
            safe_gate, lb = offset % 2 == 1, (-5.0, -0.001)[offset % 2]
            use_gate, dt_bias = True, offset % 2 == 0
            disable_recompute, return_intermediate = offset % 2 == 0, offset % 2 == 1
            state_v_first, explicit = offset % 2 == 1, explicit
            distribution = "normal" if offset % 2 else "uniform"
            seq_count = 4
        if kind == 4:
            heads = (128, 128) if offset % 2 == 0 else (1, 1)
            cu_mode = None if layout in {"BSND", "BNSD"} else "single"
            initial_state, output_final_state = True, False
            safe_gate, lb = False, -5.0
            use_gate, dt_bias = False, False
            disable_recompute, return_intermediate = True, False
            state_v_first, explicit = False, False
            distribution = "uniform"
            seq_count = 2
        profiles.append(
            _profile(
                q_dtype, g_dtype, beta_dtype, layout, kv_dim, chunk_size,
                total_t, heads, cu_mode, initial_state, output_final_state,
                safe_gate, lb, use_gate, dt_bias, disable_recompute,
                return_intermediate, state_v_first, explicit, distribution,
                seq_count, "boundary", f"bnd_{kind}_{layout.lower()}_t{total_t}",
                batch=1 + offset % 2,
            )
        )
        assert len(profiles) == base_index + offset + 1

    return profiles


PROFILES = _build_profiles()
assert len(PROFILES) == 1076, f"expected 1076 profiles, built {len(PROFILES)}"


def build_profiles() -> list[dict]:
    return [dict(profile) for profile in PROFILES]


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_chunk_kda_fwd")
    class ChunkKdaFwdGenerator(CaseGenerator):
        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            if index >= len(PROFILES):
                raise IndexError(
                    f"ATK requested profile {index}, but only {len(PROFILES)} exist"
                )
            return _configure_case(case_config, PROFILES[index], index)

NEGATIVE_CHUNK_SIZE = 128
NEGATIVE_LAYOUTS = ("BSND", "BNSD", "TND", "NTD")
NEGATIVE_EXPECTED = "chunkSize only supports 64|aclnnStatus=161002"


def _attr(profile, name):
    for item in profile["inputs"]:
        if item.get("name") == name:
            return item.get("range_values")
    return None


def _set_attr(profile, name, value):
    for item in profile["inputs"]:
        if item.get("name") == name:
            item["range_values"] = value
            return True
    return False


def build_negative_cases(positive_path, out_path):
    # 从正向用例派生 chunk_size=128 的反向拦截用例。
    # 选用例规则：每种 layout 各取 bf16/fp16 且 K=V=128 的一条，再加四种 layout 的
    # K=V=64（bf16）一条，共 12 条；只把 chunk_size 改成 128 并写入 expected_error_msg，
    # 其余输入与正向用例保持一致。算子的 chunk_size 校验先于 layout/shape 校验，
    # 因此这些用例必然命中参数拦截。
    with open(positive_path, encoding="utf-8") as handle:
        positive = json.load(handle)

    def q_dtype(case):
        return next(item["dtype"] for item in case["inputs"] if item.get("name") == "q")

    def kv_dim(case):
        return next(item["shape"][-1] for item in case["inputs"] if item.get("name") == "q")

    def total_t(case):
        match = re.search(r"_t(\d+)_k(\d+)_v(\d+)$", case["name"])
        return int(match.group(1)) if match else 0

    wanted = [(layout, dtype, 128) for layout in NEGATIVE_LAYOUTS for dtype in ("bf16", "fp16")]
    wanted += [(layout, "bf16", 64) for layout in NEGATIVE_LAYOUTS]

    used = set()
    negatives = []
    for layout, dtype, kv in wanted:
        pool = [
            case for case in positive
            if _attr(case, "layout") == layout
            and q_dtype(case) == dtype
            and kv_dim(case) == kv
            and not case.get("is_boundary")
        ]
        dense = [case for case in pool if "_dense_" in case["name"] and total_t(case) >= 128]
        long_enough = [case for case in pool if total_t(case) >= 128]
        picked = None
        for candidate in (dense or long_enough or pool):
            if candidate["name"] not in used:
                picked = candidate
                break
        if picked is None:
            raise RuntimeError(
                "no negative template for layout=" + layout + " dtype=" + dtype + " kv=" + str(kv)
            )
        used.add(picked["name"])
        negatives.append(picked)

    for index, case in enumerate(negatives):
        case["id"] = index
        case["name"] = case["name"] + "_neg_c128"
        case["is_boundary"] = False
        if not _set_attr(case, "chunk_size", NEGATIVE_CHUNK_SIZE):
            raise RuntimeError("case " + case["name"] + " has no chunk_size input")
        case["expected_error_msg"] = NEGATIVE_EXPECTED

    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(negatives, handle, ensure_ascii=False, separators=(",", ":"))
    return negatives


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the frozen direct-input chunk_kda_fwd ATK matrix."
    )
    parser.add_argument("--summary", action="store_true")
    parser.add_argument(
        "--negative-out",
        metavar="PATH",
        help="从正向用例 JSON 派生 chunk_size=128 的反向拦截用例并写入该路径",
    )
    parser.add_argument(
        "--positive-json",
        metavar="PATH",
        default=str(Path(__file__).resolve().with_name("atk_chunk_kda_fwd.json")),
        help="用于派生反向用例的正向用例 JSON（默认与本脚本同目录）",
    )
    args = parser.parse_args()
    if args.negative_out:
        negatives = build_negative_cases(args.positive_json, args.negative_out)
        print(json.dumps({"negative": len(negatives), "out": args.negative_out}, sort_keys=True))
        return
    if args.summary:
        print(
            json.dumps(
                {
                    "total": len(PROFILES),
                    "q_dtype": dict(
                        sorted(Counter(item["q_dtype"] for item in PROFILES).items())
                    ),
                    "category": dict(
                        sorted(Counter(item["category"] for item in PROFILES).items())
                    ),
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
