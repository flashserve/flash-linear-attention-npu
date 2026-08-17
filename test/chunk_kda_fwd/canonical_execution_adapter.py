#!/usr/bin/env python3
"""Materialize the 124 non-accuracy cases from the canonical KDA matrix.

The manifest keeps the human-readable design rows.  This adapter deliberately
does not infer executable values from that prose: every shape, mutation,
threshold, tool and repeat policy is assigned by canonical design ID below.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from canonical_case_adapter import (
    DEFAULT_MANIFEST,
    EXPECTED_SOURCE_SHA256,
    L1_STANDARD,
    SEED_BASE,
    _atk_payload as _positive_atk_payload,
    _balanced,
    _base_spec,
    _cu,
    _mixed,
)


EXPECTED_KIND_IDS = {
    "run": tuple(
        [f"KDA-FWD-N{index:03d}" for index in range(1, 85)]
        + [f"KDA-FWD-G{index:03d}" for index in range(81, 89)]
    ),
    "msopprof": tuple(
        [f"KDA-FWD-M{index:03d}" for index in range(1, 13)]
        + [f"KDA-FWD-G{index:03d}" for index in range(89, 97)]
    ),
    "stress": tuple(
        [f"KDA-FWD-S{index:03d}" for index in range(1, 5)]
        + [f"KDA-FWD-G{index:03d}" for index in range(97, 99)]
    ),
    "sanitizer": tuple(
        [f"KDA-FWD-S{index:03d}" for index in range(5, 9)]
        + [f"KDA-FWD-G{index:03d}" for index in range(99, 101)]
    ),
}
EXPECTED_KIND_COUNTS = {kind: len(ids) for kind, ids in EXPECTED_KIND_IDS.items()}
EXPECTED_NON_ACCURACY_IDS = tuple(
    design_id
    for prefix in ("N", "M", "S", "G")
    for design_id in (
        [f"KDA-FWD-N{index:03d}" for index in range(1, 85)]
        if prefix == "N"
        else [f"KDA-FWD-M{index:03d}" for index in range(1, 13)]
        if prefix == "M"
        else [f"KDA-FWD-S{index:03d}" for index in range(1, 9)]
        if prefix == "S"
        else [f"KDA-FWD-G{index:03d}" for index in range(81, 101)]
    )
)
SUPPORTED_SOCS = ("ascend910b", "ascend910_93", "ascend950")
SUPPORTED_ROUTES = ("ascendc", "aclnn", "direct_launch")
SUPPORTED_KINDS = tuple(EXPECTED_KIND_IDS)
EXPECTED_CASES_SHA256 = "a4e669910843b8eb3f93a0e6a8c0143d79d94a19bad111925f55c2e0ee57f747"
INPUT_SPEC_KEYS = (
    "B", "H", "HV", "T", "K", "V", "chunk_size", "layout",
    "q_dtype", "g_dtype", "beta_dtype", "a_log_dtype", "dt_bias_dtype",
    "initial_state", "cu_seqlens", "explicit_chunk_indices", "safe_gate",
    "lower_bound", "use_gate_in_kernel", "dt_bias", "state_v_first",
    "data_profile", "data_variant", "traceable_head_mapping", "input_storage", "data_scale",
    "qk_scale", "v_scale", "gate_scale", "beta_scale", "beta_bias",
    "a_log_scale", "dt_bias_scale", "dt_bias_mean", "seed",
)


def _number(design_id: str) -> int:
    return int(design_id[-3:])


def _stable_id(design_id: str) -> int:
    return {
        "N": 3000,
        "M": 4000,
        "S": 5000,
        "G": 6000,
    }[design_id[8]] + _number(design_id)


def _logical_soc(platforms: list[str]) -> str:
    return platforms[0] if len(platforms) == 1 else "multi_soc"


def _execution_base(row: dict, ordinal: int) -> dict:
    spec = _base_spec(row, 1000 + ordinal)
    kind = row["kind"]
    spec.update(
        {
            "profile": f"canonical_{kind}_v1",
            "tags": f"{kind},canonical_300",
            "execution_kind": kind,
            "route": row["routes"][0],
            "soc": _logical_soc(row["platforms"]),
            "target_routes": list(row["routes"]),
            "target_platforms": list(row["platforms"]),
            "design_variants": list(row["variants"]),
            "materialized_variant": row["variants"][0],
            "seed": SEED_BASE + 1000 + ordinal,
        }
    )
    return spec


_N_RUN_MUTATIONS = dict(
        enumerate(
            (
                "null_q", "null_k", "null_v", "null_g", "null_beta",
                "null_attn", "null_aqk", "null_akk", "layout_null",
                "layout_lower", "layout_invalid", "rank_bsnd_inputs_rank3",
                "rank_tnd_inputs_rank4", "beta_rank_invalid", "qk_shape",
                "v_batch_mismatch", "v_token_mismatch", "v_head_mismatch",
                "g_batch_mismatch", "g_token_mismatch", "g_head_mismatch",
                "g_key_dim_mismatch", "beta_batch_mismatch",
                "beta_token_mismatch", "beta_head_mismatch",
                "tnd_token_mismatch", "ntd_head_mismatch",
                "bnsd_shape_mismatch", "hv_lt_h", "hv_not_divisible",
                "h_gt_128", "hv_gt_128", "qkv_fp32", "k_dtype",
                "v_dtype", "g_fp16", "g_int32", "beta_fp16",
                "beta_int32", "alog_dtype", "dtbias_dtype", "state_dtype",
                "attn_dtype", "gk_dtype", "aqk_dtype", "akk_dtype",
                "w_dtype", "u_dtype", "final_dtype", "h_dtype", "k_zero",
                "k_15", "k_17", "k_272", "v_zero", "v_15", "v_17",
                "v_272", "chunk_zero", "chunk_32", "chunk_96",
                "missing_alog", "alog_shape_plus_one",
                "dtbias_shape_minus_one", "lower_low", "lower_high",
                "cu_short", "cu_start", "cu_end", "cu_order",
                "seq_gt_1024", "varlen_b2", "indices_without_cu",
                "indices_missing_pair", "indices_extra_pair", "indices_order",
                "state_shape_kv", "state_shape_vk", "final_shape",
                "gk_sequence_major", "aqk_last_dim", "w_last_dim_v",
                "u_last_dim_k", "h_layout_or_state",
            ),
            start=1,
        )
    )
_G_RUN_MUTATIONS = {
    81: "h_zero",
    82: "hv_zero",
    83: "hv_lt_h_4_2",
    84: "hv_not_divisible_3_8",
    85: "h_gt_128",
    86: "hv_gt_128",
    87: "g_beta_key_heads",
    88: "initial_state_key_heads",
}

_ACLNN_MESSAGES = {
    **{index: "q, k, v, g and beta must not be nullptr" for index in range(1, 6)},
    6: "attnOut must not be nullptr",
    7: "aqkOut and akkOut must not be nullptr",
    8: "aqkOut and akkOut must not be nullptr",
    9: "layout must be uppercase and one of BSND, BNSD, TND or NTD",
    10: "layout must be uppercase and one of BSND, BNSD, TND or NTD",
    11: "layout must be uppercase and one of BSND, BNSD, TND or NTD",
    12: "q/k/v/g and beta ranks must match layout",
    13: "q/k/v/g and beta ranks must match layout",
    14: "q/k/v/g and beta ranks must match layout",
    15: "q and k must have identical shape",
    **{index: "BSND expects v/g/beta" for index in range(16, 26)},
    26: "TND expects v/g/beta",
    27: "NTD expects v/g/beta",
    28: "BNSD expects v/g/beta",
    29: "H and HV must be positive",
    30: "HV must be divisible by H",
    31: "H and HV must be less than or equal to 128",
    32: "H and HV must be less than or equal to 128",
    **{index: "q, k and v must use the same float16 or bfloat16 dtype" for index in range(33, 36)},
    36: "g must be float32 or bfloat16",
    37: "g must be float32 or bfloat16",
    38: "beta must be float32 or bfloat16",
    39: "beta must be float32 or bfloat16",
    40: "aLogOptional must be float32 or bfloat16",
    41: "dtBiasOptional must be float32 or bfloat16",
    42: "initialStateOptional must be float32",
    43: "attnOut must match q dtype",
    44: "gkOut must be float32",
    45: "Aqk/Akk must match q dtype",
    46: "Aqk/Akk must match q dtype",
    47: "w/qg/kg must match q dtype",
    48: "u/vNew must match q dtype",
    49: "finalStateOut must be float32",
    50: "hOut must match q dtype",
    **{index: "K/V must be multiples of 16" for index in range(51, 59)},
    59: "chunkSize must be 64 or 128",
    60: "chunkSize must be 64 or 128",
    61: "chunkSize must be 64 or 128",
    62: "aLogOptional is required when useGateInKernel is true",
    63: "aLogOptional must have shape [HV]",
    64: "dtBiasOptional must have shape [HV*K]",
    65: "lowerBound must be in [-5, 0)",
    66: "lowerBound must be in [-5, 0)",
    67: "cuSeqlensOptional must contain at least",
    68: "cuSeqlensOptional[0] must be 0",
    69: "cuSeqlensOptional last element must equal the sequence length",
    70: "cuSeqlensOptional must be nondecreasing",
    71: "varlen input supports at most 1024 sequences",
    72: "rank4 varlen input with cuSeqlensOptional requires B=1",
    73: "chunkIndicesOptional requires cuSeqlensOptional",
    74: "chunkIndicesOptional must contain exactly one",
    75: "chunkIndicesOptional must contain exactly one",
    76: "chunkIndicesOptional must use canonical sequence-major chunk order",
    77: "initialStateOptional must be [N,HV,K,V]",
    78: "initialStateOptional must be [N,HV,K,V]",
    79: "finalStateOut must be [N,HV,K,V]",
    80: "gkOut must be float32 in fixed head-major",
    81: "Aqk/Akk must match q dtype",
    82: "w/qg/kg must match q dtype",
    83: "u/vNew must match q dtype",
    84: "hOut must match q dtype",
}

_PUBLIC_MESSAGES = {
    10: "layout must be uppercase and one of BSND, BNSD, TND, NTD",
    11: "layout must be uppercase and one of BSND, BNSD, TND, NTD",
    12: "q/k/v/g rank does not match layout",
    13: "q/k/v/g rank does not match layout",
    81: "H/HV must satisfy 0 < H <= HV <= 128",
    82: "H/HV must satisfy 0 < H <= HV <= 128",
    83: "H/HV must satisfy 0 < H <= HV <= 128",
    84: "H/HV must satisfy 0 < H <= HV <= 128",
    85: "H/HV must satisfy 0 < H <= HV <= 128",
    86: "H/HV must satisfy 0 < H <= HV <= 128",
    87: "v/g/beta shapes do not match the selected layout",
    88: "initial_state shape/dtype does not match state_v_first",
}


def _apply_outcome(spec: dict, route: str) -> None:
    outcome = spec["expected_outcomes"][route]
    spec["route"] = route
    spec["expected_code_name"] = outcome["code_name"]
    spec["expected_return_code"] = outcome["return_code"]
    spec["expected_message"] = outcome["message"]


def _apply_run(row: dict, spec: dict) -> None:
    design_id = row["id"]
    prefix, index = design_id[8], _number(design_id)
    mutation_index = index if prefix == "N" else index
    spec.update(
        {
            "B": 1,
            "H": 2,
            "HV": 4,
            "T": 128,
            "K": 128,
            "V": 64,
            "layout": "BSND",
            "q_dtype": "bf16",
            "g_dtype": "fp32",
            "beta_dtype": "fp32",
            "a_log_dtype": "fp32",
            "dt_bias_dtype": "fp32",
            "initial_state": True,
            "output_final_state": True,
            "use_gate_in_kernel": True,
            "dt_bias": True,
            "safe_gate": True,
            "disable_recompute": True,
            "return_intermediate_states": True,
            "state_v_first": False,
            "mutation": (
                _N_RUN_MUTATIONS[mutation_index]
                if prefix == "N"
                else _G_RUN_MUTATIONS[mutation_index]
            ),
            "tags": "run,negative,canonical_300",
        }
    )
    if prefix == "N":
        if index == 13:
            spec["layout"] = "TND"
            spec["cu_seqlens"] = _cu([128])
        elif index == 26:
            spec["layout"] = "TND"
            spec["cu_seqlens"] = _cu([64, 64])
        elif index == 27:
            spec["layout"] = "NTD"
            spec["cu_seqlens"] = _cu([64, 64])
        elif index == 28:
            spec["layout"] = "BNSD"
        elif index in {67, 68, 69, 70, 71, 73, 74, 75, 76}:
            spec["cu_seqlens"] = _cu([64, 64])
        elif index == 72:
            spec["cu_seqlens"] = _cu([128])
        if index == 78:
            spec["state_v_first"] = True
    else:
        spec.update({"H": 2, "HV": 4})
        if index == 84:
            # The canonical G084 contract is exactly H/HV=3/8.
            spec.update({"H": 3, "HV": 8})
        if index == 88:
            spec["initial_state"] = True

    aclnn_index = index
    if prefix == "G":
        aclnn_message = {
            81: "H and HV must be positive",
            82: "H and HV must be positive",
            83: "HV must be greater than or equal to H",
            84: "HV must be divisible by H",
            85: "H and HV must be less than or equal to 128",
            86: "H and HV must be less than or equal to 128",
            87: "BSND expects v/g/beta",
            88: "initialStateOptional must be [N,HV,K,V]",
        }[index]
    else:
        aclnn_message = _ACLNN_MESSAGES[aclnn_index]
    aclnn_code = 161001 if prefix == "N" and index in {*range(1, 9), 62} else 161002
    outcomes = {
        "aclnn": {
            "code_name": "ACLNN_ERR_PARAM_NULLPTR" if aclnn_code == 161001 else "ACLNN_ERR_PARAM_INVALID",
            "return_code": aclnn_code,
            "message": aclnn_message,
        }
    }
    if "ascendc" in row["routes"]:
        outcomes["ascendc"] = {
            "code_name": "RuntimeError",
            "return_code": "RuntimeError",
            "message": _PUBLIC_MESSAGES[index],
        }
    spec["expected_outcomes"] = outcomes
    _apply_outcome(spec, row["routes"][0])


def _model_case(spec: dict, *, total: int, heads: int = 96, value_heads: int = 96) -> None:
    spec.update(
        {
            "B": 1,
            "H": heads,
            "HV": value_heads,
            "T": total,
            "K": 128,
            "V": 128,
            "chunk_size": 64,
            "layout": "BSND",
            "q_dtype": "bf16",
            "g_dtype": "fp32",
            "beta_dtype": "bf16",
            "a_log_dtype": "fp32",
            "dt_bias_dtype": "fp32",
            "initial_state": False,
            "output_final_state": False,
            "cu_seqlens": "",
            "explicit_chunk_indices": False,
            "safe_gate": True,
            "lower_bound": -5.0,
            "use_gate_in_kernel": True,
            "dt_bias": True,
            "disable_recompute": True,
            "return_intermediate_states": False,
            "state_v_first": True,
            "data_profile": "model_h96",
            "distribution": "dense",
        }
    )
    spec["scale"] = 1.0 / math.sqrt(spec["K"])


def _set_varlen(spec: dict, lengths: list[int], distribution: str) -> None:
    spec.update(
        {
            "layout": "TND",
            "cu_seqlens": _cu(lengths),
            "distribution": distribution,
        }
    )


def _apply_msopprof(row: dict, spec: dict) -> None:
    design_id, index = row["id"], _number(row["id"])
    spec.update(
        {
            "profiler": {
                "tool": "msopprof",
                "aic_metrics": "BasicInfo",
                "launch_count": 20,
                "warm_up": 5,
                "replay_mode": "application",
                "kill": "off",
                "kernel_name": "chunk_kda_fwd",
                "metric": "device_duration_us",
            },
            "performance_expectation": {
                "baseline_design_id": None,
                "max_relative_regression": None,
                "absolute_ms_lt": None,
                "short_chain_exemption": False,
            },
            "variant_settings": {
                variant: {
                    "required_build_variant": (
                        "l2_streaming_single_read_disabled"
                        if variant == "l2_streaming_single_read_disabled"
                        else "baseline"
                    )
                }
                for variant in row["variants"]
            },
        }
    )
    if design_id.startswith("KDA-FWD-M"):
        total = 16384 if index in range(3, 9) else 8192
        _model_case(spec, total=total)
        spec["disable_recompute"] = index not in {2, 4}
        if index == 5:
            _set_varlen(spec, [16384], "single")
        elif index == 6:
            _set_varlen(spec, _balanced(16384, 8), "balanced8")
        elif index == 7:
            _set_varlen(spec, _mixed(16384), "mixed")
        elif index == 8:
            _set_varlen(spec, [64] * 256, "short64")
            spec["performance_expectation"]["short_chain_exemption"] = True
        elif index == 10:
            _set_varlen(spec, _balanced(8192, 8), "balanced8")
        elif index == 12:
            _set_varlen(spec, _mixed(8192), "mixed")
        comparison = {
            5: "KDA-FWD-M003",
            6: "KDA-FWD-M003",
            7: "KDA-FWD-M003",
            10: "KDA-FWD-M009",
            12: "KDA-FWD-M011",
        }.get(index)
        if comparison:
            spec["performance_expectation"].update(
                {
                    "baseline_design_id": comparison,
                    "max_relative_regression": 0.05,
                    "absolute_ms_lt": 12.0 if index in {5, 6, 7} else None,
                }
            )
    else:
        total = 16384 if index <= 92 else 8192
        heads, value_heads = {
            89: (48, 96), 90: (48, 96),
            91: (32, 96), 92: (32, 96),
            93: (24, 96), 94: (24, 96),
            95: (12, 96), 96: (12, 96),
        }[index]
        _model_case(spec, total=total, heads=heads, value_heads=value_heads)
        if index in {90, 94}:
            _set_varlen(spec, _balanced(total, 8), "balanced8")
        elif index in {92, 96}:
            _set_varlen(spec, _mixed(total), "mixed")
        comparison = {90: 89, 92: 91, 94: 93, 96: 95}.get(index)
        if comparison:
            spec["performance_expectation"].update(
                {
                    "baseline_design_id": f"KDA-FWD-G{comparison:03d}",
                    "max_relative_regression": 0.05,
                    "absolute_ms_lt": 12.0 if index in {90, 92} else None,
                }
            )


def _mask_settings(name: str) -> dict:
    if name == "all_outputs":
        return {
            "output_final_state": True,
            "disable_recompute": True,
            "return_intermediate_states": True,
        }
    if name == "hidden_outputs":
        return {
            "output_final_state": False,
            "disable_recompute": False,
            "return_intermediate_states": False,
        }
    return {}


def _apply_stress(row: dict, spec: dict) -> None:
    design_id, index = row["id"], _number(row["id"])
    repeat_count = int(row["repeat_count"])
    if design_id.startswith("KDA-FWD-S"):
        _model_case(spec, total=8192)
        if index == 2:
            _set_varlen(spec, _balanced(8192, 8), "balanced8")
        elif index in {3, 4}:
            _set_varlen(spec, _mixed(8192), "mixed")
    else:
        _model_case(spec, total=8192, heads=32, value_heads=96)
        spec["data_variant"] = "gva_head_traceable"
        if index == 98:
            _set_varlen(spec, _mixed(8192), "mixed")
    spec.update(
        {
            "repeat_count": repeat_count,
            "comparison": "all_tensor_outputs_bitwise_against_run_0",
            "cross_variant_common_outputs_bitwise": len(row["variants"]) > 1,
            "variant_settings": {
                variant: _mask_settings(variant) for variant in row["variants"]
            },
        }
    )
    if row["variants"] == ["fixed_input"]:
        spec.update(_mask_settings("all_outputs"))
        spec["variant_settings"]["fixed_input"] = {}


def _apply_sanitizer(row: dict, spec: dict) -> None:
    design_id, index = row["id"], _number(row["id"])
    _model_case(spec, total=8192)
    if design_id.startswith("KDA-FWD-S"):
        if index == 5:
            variant_settings = {
                "dense_key2": {},
                "mixed_tail_key2": {"varlen": "mixed"},
            }
        elif index == 6:
            variant_settings = {
                "dense_key2": {},
                "mixed_tail_key2": {"varlen": "mixed"},
                "max_kv_key1": {"K": 256, "V": 256},
            }
        elif index == 7:
            variant_settings = {
                "tail_initial_none_hidden_outputs": {
                    "T": 65,
                    "H": 2,
                    "HV": 2,
                    **_mask_settings("hidden_outputs"),
                }
            }
        else:
            variant_settings = {
                "long_chain": {},
                "mixed_tail": {"varlen": "mixed"},
            }
    elif index == 99:
        spec.update({"H": 2, "HV": 6, "T": 1536})
        variant_settings = {
            "aligned": {"varlen": "balanced8"},
            "mixed_tail": {"varlen": "mixed"},
        }
    else:
        spec.update({"H": 1, "HV": 128, "T": 65, "K": 256, "V": 256})
        variant_settings = {"max_group_tail_initial_none": {}}
    spec.update(
        {
            "sanitizer_tools": list(row["sanitizer_tools"]),
            "variant_settings": variant_settings,
            "sanitizer": {
                "object_symbol_regex": "sanitizer",
                "kernel_regex": "chunk_kda_fwd",
                "require_active_banner": True,
                "reject_no_active_banner": True,
                "tool_options": {
                    "racecheck": [],
                    "memcheck": ["--leak-check=yes"],
                    "initcheck": [],
                    "synccheck": [],
                },
            },
        }
    )


def _apply_variant(spec: dict, variant: str) -> None:
    settings = copy.deepcopy(spec.get("variant_settings", {}).get(variant, {}))
    varlen = settings.pop("varlen", None)
    spec.update(settings)
    if varlen == "mixed":
        _set_varlen(spec, _mixed(int(spec["T"])), "mixed")
    elif varlen == "balanced8":
        _set_varlen(spec, _balanced(int(spec["T"]), 8), "balanced8")
    spec["materialized_variant"] = variant
    spec["scale"] = 1.0 / math.sqrt(int(spec["K"]))


def _validate_source(path: Path) -> list[dict]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("op") != "chunk_kda_fwd":
        raise ValueError("canonical source is not the chunk_kda_fwd manifest")
    matrix = manifest.get("design_matrix", {})
    source = matrix.get("source", {})
    if source.get("row_sha256") != EXPECTED_SOURCE_SHA256 or source.get("row_count") != 300:
        raise ValueError("canonical design source changed; update the explicit executable rules")
    cases_digest = hashlib.sha256(
        json.dumps(
            matrix.get("cases", []),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if cases_digest != EXPECTED_CASES_SHA256:
        raise ValueError("canonical 300 case rows changed; update the explicit executable rules")
    rows = [case for case in matrix.get("cases", []) if case.get("kind") != "accuracy"]
    if tuple(row.get("id") for row in rows) != EXPECTED_NON_ACCURACY_IDS:
        raise ValueError("canonical non-accuracy IDs are not the exact ordered 124-case sequence")
    counts = Counter(row.get("kind") for row in rows)
    if dict(counts) != EXPECTED_KIND_COUNTS:
        raise ValueError(f"canonical non-accuracy kind counts changed: {dict(counts)}")
    return rows


def materialize(path=DEFAULT_MANIFEST, *, kind: str | None = None) -> list[dict]:
    """Return logical non-accuracy ``{'id': int, 'spec': dict}`` records."""
    if kind is not None and kind not in SUPPORTED_KINDS:
        raise ValueError(f"unsupported execution kind: {kind}")
    rows = _validate_source(Path(path))
    records = []
    for ordinal, row in enumerate(rows, start=1):
        if kind is not None and row["kind"] != kind:
            continue
        spec = _execution_base(row, ordinal)
        {
            "run": _apply_run,
            "msopprof": _apply_msopprof,
            "stress": _apply_stress,
            "sanitizer": _apply_sanitizer,
        }[row["kind"]](row, spec)
        _apply_variant(spec, row["variants"][0])
        spec["shape_spec"] = (
            f"B={spec['B']},H={spec['H']},HV={spec['HV']},T={spec['T']},"
            f"K={spec['K']},V={spec['V']}"
        )
        records.append({"id": _stable_id(row["id"]), "spec": spec})
    expected = 124 if kind is None else EXPECTED_KIND_COUNTS[kind]
    if len(records) != expected or len({record["id"] for record in records}) != expected:
        raise ValueError(f"materializer did not produce {expected} unique {kind or 'non-accuracy'} records")
    return records


def materialize_all(path=DEFAULT_MANIFEST) -> list[dict]:
    """Return the exact ordered 300-case cache catalog materialization."""
    from canonical_case_adapter import materialize as materialize_accuracy

    by_design_id = {
        record["spec"]["design_id"]: record
        for record in [*materialize_accuracy(path), *materialize(path)]
    }
    rows = json.loads(Path(path).read_text(encoding="utf-8"))["design_matrix"]["cases"]
    records = [by_design_id[row["id"]] for row in rows]
    if len(records) != 300 or len({record["id"] for record in records}) != 300:
        raise ValueError("combined cache adapter did not produce 300 unique records")
    return records


def materialize_input_variants(spec: dict) -> dict:
    """Collapse execution variants that produce byte-identical cached inputs."""
    aliases: dict[str, str] = {}
    variants: dict[str, dict] = {}
    fingerprints: dict[str, str] = {}
    for variant in spec.get("design_variants", [spec.get("materialized_variant", "default")]):
        projected = copy.deepcopy(spec)
        _apply_variant(projected, variant)
        input_spec = {key: projected.get(key) for key in INPUT_SPEC_KEYS}
        fingerprint = json.dumps(input_spec, sort_keys=True, separators=(",", ":"))
        primary = fingerprints.get(fingerprint)
        if primary is None:
            primary = variant
            fingerprints[fingerprint] = primary
            variants[primary] = projected
        aliases[variant] = primary
    return {
        "schema": "chunk_kda_fwd.canonical_input_variants.v1",
        "aliases": aliases,
        "variant_specs": variants,
    }


def project_records(
    path=DEFAULT_MANIFEST,
    *,
    kind: str | None = None,
    soc: str | None = None,
    route: str | None = None,
    variant: str | None = None,
    sanitizer_tool: str | None = None,
    include_not_applicable: bool = False,
) -> list[dict]:
    if soc is not None and soc not in SUPPORTED_SOCS:
        raise ValueError(f"unsupported SOC: {soc}")
    if route is not None and route not in SUPPORTED_ROUTES:
        raise ValueError(f"unsupported route: {route}")
    projected = []
    for record in materialize(path, kind=kind):
        base = record["spec"]
        if soc is not None and soc not in base["target_platforms"]:
            if include_not_applicable:
                projected.append(
                    {
                        "id": record["id"],
                        "spec": {
                            "design_id": base["design_id"],
                            "execution_kind": base["execution_kind"],
                            "soc": soc,
                            "status": "not_applicable",
                            "reason": f"case targets {','.join(base['target_platforms'])}",
                        },
                    }
                )
            continue
        routes = [route] if route is not None else base["target_routes"]
        variants = [variant] if variant is not None else base["design_variants"]
        tools = (
            [sanitizer_tool]
            if sanitizer_tool is not None
            else base.get("sanitizer_tools", [None])
        )
        for target_route in routes:
            if target_route not in base["target_routes"]:
                continue
            for target_variant in variants:
                if target_variant not in base["design_variants"]:
                    continue
                for tool in tools:
                    if tool is not None and tool not in base.get("sanitizer_tools", []):
                        continue
                    spec = copy.deepcopy(base)
                    spec["soc"] = soc or _logical_soc(base["target_platforms"])
                    spec["route"] = target_route
                    spec["status"] = "planned"
                    _apply_variant(spec, target_variant)
                    if spec["execution_kind"] == "run":
                        _apply_outcome(spec, target_route)
                    if tool is not None:
                        spec["sanitizer_tool"] = tool
                    projected.append({"id": record["id"], "spec": spec})
    return projected


def _run_atk_payload(record: dict) -> dict:
    spec = record["spec"]
    payload = _positive_atk_payload(record)
    expected = f"{spec['expected_code_name']}({spec['expected_return_code']}): {spec['expected_message']}"
    payload["expected_error_msg"] = expected
    for item in payload["inputs"]:
        if item["name"] == "negative_case":
            item["range_values"] = True
    return payload


def build_run_atk_payloads(path=DEFAULT_MANIFEST, *, soc: str, route: str) -> list[dict]:
    return [
        _run_atk_payload(record)
        for record in project_records(path, kind="run", soc=soc, route=route)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--kind", choices=SUPPORTED_KINDS)
    parser.add_argument("--soc", choices=SUPPORTED_SOCS)
    parser.add_argument("--route", choices=SUPPORTED_ROUTES)
    parser.add_argument("--variant")
    parser.add_argument("--sanitizer-tool", choices=("racecheck", "memcheck", "initcheck", "synccheck"))
    parser.add_argument("--format", choices=("records", "run-atk"), default="records")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.format == "run-atk":
        if not args.soc or not args.route:
            parser.error("--format run-atk requires --soc and --route")
        payload = build_run_atk_payloads(args.source, soc=args.soc, route=args.route)
    else:
        payload = project_records(
            args.source,
            kind=args.kind,
            soc=args.soc,
            route=args.route,
            variant=args.variant,
            sanitizer_tool=args.sanitizer_tool,
            include_not_applicable=args.soc is not None,
        )
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
