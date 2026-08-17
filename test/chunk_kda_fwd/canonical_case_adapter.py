#!/usr/bin/env python3
"""Materialize the 176 canonical ChunkKdaFwd numeric accuracy cases.

The design matrix intentionally keeps human-readable source rows.  This module
does not parse those rows.  Every executable field is produced from an explicit
ID/range table, and the source-row digest prevents a changed design from being
silently paired with stale executable rules.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_MANIFEST = ROOT / "tests" / "op_cases" / "chunk_kda_fwd.json"
DEFAULT_ATK_JSON = HERE / "atk_chunk_kda_fwd_canonical_accuracy.json"

EXPECTED_SOURCE_SHA256 = "b4bd2735051423d80773719b12fa23e4d1fc43767c4347c60b69f0b3daed946d"
EXPECTED_ACCURACY_IDS = tuple(
    [f"KDA-FWD-P{index:03d}" for index in range(1, 97)]
    + [f"KDA-FWD-G{index:03d}" for index in range(1, 81)]
)
SEED_BASE = 202608140
TRACEABLE_PHYSICAL_ID_OFFSET = 10000

L1_STANDARD = {
    "acc": {
        "cv_fused_double_benchmark": {
            "max_re_ratio": 5,
            "avg_re_ratio": 1.5,
            "root_mean_squared_ratio": 1.5,
        }
    },
    "perf": "not_key",
}

EXECUTOR_REQUIRED_FIELDS = frozenset(
    {
        "design_id",
        "case_key",
        "tags",
        "route",
        "soc",
        "B",
        "H",
        "HV",
        "T",
        "K",
        "V",
        "chunk_size",
        "layout",
        "q_dtype",
        "g_dtype",
        "beta_dtype",
        "a_log_dtype",
        "dt_bias_dtype",
        "scale",
        "initial_state",
        "output_final_state",
        "cu_seqlens",
        "explicit_chunk_indices",
        "safe_gate",
        "lower_bound",
        "use_gate_in_kernel",
        "dt_bias",
        "disable_recompute",
        "return_intermediate_states",
        "state_v_first",
        "data_profile",
        "traceable_head_mapping",
        "data_scale",
        "qk_scale",
        "v_scale",
        "gate_scale",
        "beta_scale",
        "beta_bias",
        "a_log_scale",
        "dt_bias_scale",
        "dt_bias_mean",
        "seed",
    }
)


def _case_number(design_id: str) -> int:
    return int(design_id[-3:])


def _atk_id(design_id: str) -> int:
    prefix = design_id[8]
    base = {"P": 1000, "G": 2000}[prefix]
    return base + _case_number(design_id)


def _cu(lengths: list[int]) -> str:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return ",".join(str(value) for value in values)


def _balanced(total: int, count: int) -> list[int]:
    base, extra = divmod(total, count)
    return [base + (index < extra) for index in range(count)]


def _mixed(total: int) -> list[int]:
    fixed = [65, 127, 129, 191, 64, 128, 193]
    remainder = total - sum(fixed)
    if remainder <= 0:
        raise ValueError(f"mixed distribution requires T > {sum(fixed)}")
    return fixed + [remainder]


def _logical_soc(platforms: list[str]) -> str:
    return platforms[0] if len(platforms) == 1 else "multi_soc"


def _base_spec(row: dict, ordinal: int) -> dict:
    design_id = row["id"]
    tags = ["accuracy", "canonical_300", "regression"]
    if design_id.startswith("KDA-FWD-G"):
        tags.append("gva")
    return {
        "design_id": design_id,
        "design_variants": list(row["variants"]),
        "materialized_variant": "random",
        "target_platforms": list(row["platforms"]),
        "target_routes": list(row["routes"]),
        "case_key": design_id.lower().replace("-", "_"),
        "profile": "canonical_accuracy_v1",
        "tags": ",".join(tags),
        "route": "ascendc",
        "soc": _logical_soc(row["platforms"]),
        "B": 1,
        "H": 2,
        "HV": 2,
        "T": 128,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "layout": "BNSD",
        "q_dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "fp32",
        "a_log_dtype": "fp32",
        "dt_bias_dtype": "fp32",
        "scale": 1.0 / math.sqrt(128),
        "initial_state": False,
        "output_final_state": True,
        "cu_seqlens": "",
        "explicit_chunk_indices": False,
        "safe_gate": True,
        "lower_bound": -5.0,
        "use_gate_in_kernel": False,
        "dt_bias": False,
        "disable_recompute": True,
        "return_intermediate_states": True,
        "state_v_first": False,
        "data_profile": "uniform",
        "data_variant": "random",
        "traceable_head_mapping": False,
        "input_storage": [],
        "distribution": "dense",
        "data_scale": 0.08,
        "qk_scale": 0.05,
        "v_scale": 0.05,
        "gate_scale": 1.0,
        "beta_scale": 0.35,
        "beta_bias": 1.5,
        "a_log_scale": 0.12,
        "dt_bias_scale": 1.65,
        "dt_bias_mean": -3.0,
        "seed": SEED_BASE + ordinal,
    }


def _raw_gate(
    spec: dict,
    *,
    safe: bool,
    a_log_dtype: str = "fp32",
    dt_bias_dtype: str = "fp32",
    dt_bias: bool = True,
) -> None:
    spec.update(
        {
            "use_gate_in_kernel": True,
            "safe_gate": safe,
            "a_log_dtype": a_log_dtype,
            "dt_bias_dtype": dt_bias_dtype,
            "dt_bias": dt_bias,
        }
    )


def _apply_p_rule(index: int, spec: dict) -> None:
    if 1 <= index <= 16:
        bits = index - 1
        spec["output_final_state"] = bool(bits & 8)
        spec["use_gate_in_kernel"] = bool(bits & 4)
        spec["dt_bias"] = bool(bits & 4)
        spec["disable_recompute"] = bool(bits & 2)
        spec["return_intermediate_states"] = bool(bits & 1)
        return

    if 17 <= index <= 24:
        offset = index - 17
        spec["layout"] = ("BSND", "BNSD", "TND", "NTD")[offset % 4]
        spec["q_dtype"] = "bf16" if index <= 20 else "fp16"
        if index in {19, 20}:
            spec["cu_seqlens"] = _cu([128])
            spec["distribution"] = "single"
        elif index == 23:
            spec["cu_seqlens"] = _cu([64, 64])
            spec["distribution"] = "aligned2"
        if index == 24:
            spec["input_storage"] = ["q", "k", "v", "g", "beta"]
        return

    if 25 <= index <= 32:
        offset = index - 25
        spec["q_dtype"] = "bf16" if offset < 4 else "fp16"
        pair = offset % 4
        spec["g_dtype"] = "bf16" if pair >= 2 else "fp32"
        spec["beta_dtype"] = "bf16" if pair % 2 else "fp32"
        return

    if 33 <= index <= 40:
        offset = index - 33
        a_log_dtype = "bf16" if offset % 4 in {1, 3} else "fp32"
        dt_bias_dtype = "bf16" if offset % 4 in {2, 3} else "fp32"
        _raw_gate(
            spec,
            safe=index >= 37,
            a_log_dtype=a_log_dtype,
            dt_bias_dtype=dt_bias_dtype,
        )
        if index == 38:
            spec["lower_bound"] = -0.001
        if index == 40:
            spec["gate_scale"] = 64.0
        return

    if index in {41, 42}:
        spec["initial_state"] = True
        spec["state_v_first"] = index == 42
        return

    if index == 43:
        spec.update({"H": 1, "HV": 2})
        return
    if index == 44:
        spec.update({"H": 2, "HV": 8})
        return
    if index == 45:
        spec.update({"H": 1, "HV": 3})
        _raw_gate(spec, safe=True)
        return
    if index == 46:
        spec.update(
            {
                "H": 96,
                "HV": 96,
                "T": 1024,
                "disable_recompute": False,
                "return_intermediate_states": False,
            }
        )
        _raw_gate(spec, safe=True)
        return
    if index == 47:
        spec.update({"H": 128, "HV": 128, "T": 64})
        return
    if index == 48:
        spec.update({"B": 2, "T": 128})
        return
    if index in {49, 50, 51, 52}:
        spec["T"] = {49: 1, 50: 63, 51: 65, 52: 129}[index]
        return

    boundary_shapes = {
        53: (128, 127, 128, 128, 2, 2),
        54: (128, 128, 128, 128, 2, 2),
        55: (128, 129, 128, 128, 2, 2),
        56: (128, 256, 128, 128, 2, 2),
        57: (128, 257, 128, 256, 2, 2),
        58: (128, 256, 64, 128, 2, 2),
        59: (128, 256, 256, 128, 2, 2),
        60: (128, 128, 16, 16, 2, 2),
        61: (64, 128, 128, 256, 2, 2),
        62: (64, 128, 256, 128, 2, 2),
        63: (64, 128, 64, 128, 2, 2),
        64: (64, 128, 16, 128, 2, 2),
        65: (64, 128, 128, 16, 2, 2),
        66: (64, 128, 128, 64, 2, 2),
        67: (64, 128, 64, 64, 2, 2),
        68: (64, 193, 64, 128, 2, 2),
        69: (128, 384, 128, 128, 2, 2),
        70: (128, 384, 128, 128, 2, 2),
        71: (128, 128, 256, 256, 1, 4),
        72: (64, 64, 16, 256, 1, 1),
    }
    if 53 <= index <= 72:
        chunk, total, key, value, head, value_head = boundary_shapes[index]
        spec.update(
            {
                "chunk_size": chunk,
                "T": total,
                "K": key,
                "V": value,
                "H": head,
                "HV": value_head,
            }
        )
        if index == 68:
            spec["cu_seqlens"] = _cu([65, 128])
            spec["distribution"] = "mixed2"
        elif index == 69:
            spec["layout"] = "BSND"
        elif index == 70:
            spec["layout"] = "TND"
            spec["cu_seqlens"] = _cu([128, 256])
            spec["distribution"] = "packed2"
        return

    if 73 <= index <= 96:
        group = (index - 73) // 4
        distribution_index = (index - 73) % 4
        total = (1024, 1536, 2048, 4096, 8192, 16384)[group]
        distribution = ("single", "balanced8", "mixed", "short64")[distribution_index]
        lengths = {
            "single": [total],
            "balanced8": _balanced(total, 8),
            "mixed": _mixed(total),
            "short64": [64] * (total // 64),
        }[distribution]
        spec.update(
            {
                "H": 96,
                "HV": 96,
                "T": total,
                "K": 128,
                "V": 128,
                "chunk_size": 64,
                "layout": "TND",
                "q_dtype": "bf16",
                "g_dtype": "fp32",
                "beta_dtype": "bf16",
                "initial_state": False,
                "output_final_state": False,
                "cu_seqlens": _cu(lengths),
                "explicit_chunk_indices": index % 2 == 0,
                "disable_recompute": index % 2 == 0,
                "return_intermediate_states": False,
                "state_v_first": True,
                "data_profile": "model_h96",
                "distribution": distribution,
            }
        )
        _raw_gate(spec, safe=True)
        return

    raise ValueError(f"no positive rule for KDA-FWD-P{index:03d}")


GVA_HEADS = (
    (1, 2),
    (2, 4),
    (3, 6),
    (48, 96),
    (64, 128),
    (1, 3),
    (2, 6),
    (32, 96),
    (1, 4),
    (2, 8),
    (24, 96),
    (1, 8),
    (6, 96),
    (3, 96),
    (1, 128),
    (96, 96),
)


def _apply_g_rule(index: int, spec: dict) -> None:
    if 1 <= index <= 16:
        spec["H"], spec["HV"] = GVA_HEADS[index - 1]
        return

    if 17 <= index <= 24:
        mapping = {
            17: ("BSND", "bf16", 2, 8, ""),
            18: ("BNSD", "bf16", 2, 8, ""),
            19: ("TND", "bf16", 2, 8, _cu([128])),
            20: ("NTD", "bf16", 2, 8, _cu([128])),
            21: ("BSND", "fp16", 2, 6, ""),
            22: ("BNSD", "fp16", 2, 6, ""),
            23: ("TND", "fp16", 2, 6, _cu([64, 64])),
            24: ("NTD", "fp16", 2, 6, _cu([65, 63])),
        }
        layout, q_dtype, head, value_head, cu_seqlens = mapping[index]
        spec.update(
            {
                "layout": layout,
                "q_dtype": q_dtype,
                "H": head,
                "HV": value_head,
                "cu_seqlens": cu_seqlens,
                "distribution": "packed" if cu_seqlens else "dense",
            }
        )
        return

    if 25 <= index <= 28:
        offset = index - 25
        spec.update(
            {
                "H": 4,
                "HV": 8,
                "g_dtype": "bf16" if offset in {1, 3} else "fp32",
                "beta_dtype": "bf16" if offset in {2, 3} else "fp32",
            }
        )
        return
    if index == 29:
        spec.update({"H": 1, "HV": 8, "q_dtype": "fp16"})
        return
    if index == 30:
        spec.update(
            {
                "H": 3,
                "HV": 96,
                "q_dtype": "fp16",
                "g_dtype": "bf16",
                "beta_dtype": "bf16",
            }
        )
        return
    if index in {31, 32}:
        spec.update({"H": 2, "HV": 8, "layout": "BSND" if index == 31 else "NTD"})
        spec["input_storage"] = ["q", "k"] if index == 31 else ["v", "g", "beta"]
        return

    if index in {33, 34}:
        spec.update({"H": 2, "HV": 8, "safe_gate": index == 34})
        return
    if index == 35:
        spec.update({"H": 2, "HV": 8})
        _raw_gate(spec, safe=False)
        return
    if index in {36, 37, 38}:
        spec.update({"H": 2, "HV": 6})
        _raw_gate(
            spec,
            safe=True,
            a_log_dtype="bf16" if index in {36, 38} else "fp32",
            dt_bias_dtype="bf16" if index == 37 else "fp32",
            dt_bias=index != 38,
        )
        if index == 37:
            spec["lower_bound"] = -0.001
        return
    if index == 39:
        spec.update({"H": 24, "HV": 96, "data_variant": "head_distinct_a_log"})
        _raw_gate(spec, safe=True, a_log_dtype="bf16", dt_bias_dtype="bf16")
        return
    if index == 40:
        spec.update({"H": 32, "HV": 96, "data_variant": "head_distinct_dt_bias"})
        _raw_gate(spec, safe=True, a_log_dtype="bf16", dt_bias_dtype="bf16")
        return

    if index in {41, 42}:
        spec.update(
            {
                "H": 2,
                "HV": 8,
                "initial_state": True,
                "state_v_first": index == 42,
            }
        )
        return
    if index in {43, 44}:
        spec.update(
            {
                "H": 2,
                "HV": 6,
                "initial_state": True,
                "data_variant": f"initial_state_pulse_hv_{index - 41}",
            }
        )
        return
    if index == 45:
        spec.update({"H": 48, "HV": 96, "output_final_state": False})
        return
    if index == 46:
        spec.update(
            {
                "H": 48,
                "HV": 96,
                "disable_recompute": False,
                "return_intermediate_states": False,
            }
        )
        return
    if index == 47:
        spec.update({"H": 32, "HV": 96})
        return
    if index == 48:
        spec.update({"H": 24, "HV": 96, "return_intermediate_states": False})
        return

    chunk_shapes = {
        49: (64, 1, 128, 128, 2, 6),
        50: (64, 63, 128, 128, 2, 6),
        51: (64, 64, 128, 128, 2, 6),
        52: (64, 65, 128, 128, 2, 6),
        53: (64, 127, 128, 128, 2, 8),
        54: (64, 128, 128, 128, 2, 8),
        55: (64, 129, 128, 128, 2, 8),
        56: (64, 193, 128, 128, 32, 96),
        57: (128, 127, 128, 128, 2, 8),
        58: (128, 128, 128, 128, 2, 8),
        59: (128, 129, 128, 128, 2, 8),
        60: (64, 128, 16, 16, 2, 8),
        61: (64, 128, 64, 128, 2, 6),
        62: (64, 128, 128, 256, 2, 8),
        63: (64, 128, 256, 128, 2, 8),
        64: (128, 129, 256, 256, 1, 8),
    }
    if 49 <= index <= 64:
        chunk, total, key, value, head, value_head = chunk_shapes[index]
        spec.update(
            {
                "chunk_size": chunk,
                "T": total,
                "K": key,
                "V": value,
                "H": head,
                "HV": value_head,
            }
        )
        return

    varlen_cases = {
        65: ("TND", 128, [64, 64], 2, 8, False, "aligned2"),
        66: ("NTD", 128, [64, 64], 2, 8, True, "aligned2"),
        67: ("BSND", 128, [64, 64], 2, 8, False, "aligned2"),
        68: ("BNSD", 128, [64, 64], 2, 8, False, "aligned2"),
        69: ("TND", 128, [63, 65], 2, 6, False, "tail2"),
        70: ("TND", 128, [65, 63], 2, 6, False, "full_tail_then_tail"),
        71: ("TND", 128, [64, 0, 64], 2, 8, False, "middle_zero"),
        72: ("TND", 128, [0, 64, 64], 2, 8, False, "leading_zero"),
        73: ("TND", 128, [64, 64, 0], 2, 8, False, "trailing_zero"),
        74: ("TND", 129, [1, 64, 64], 2, 6, True, "single_then_aligned"),
        75: ("TND", 1024, _balanced(1024, 8), 48, 96, False, "balanced8"),
        76: ("TND", 1536, _mixed(1536), 32, 96, False, "mixed"),
        77: ("TND", 2048, [2048], 24, 96, False, "single"),
        78: ("TND", 4096, [64] * 64, 12, 96, False, "short64"),
        79: ("TND", 8192, _balanced(8192, 8), 6, 96, False, "balanced8"),
        80: ("TND", 16384, _mixed(16384), 3, 96, False, "mixed"),
    }
    if 65 <= index <= 80:
        layout, total, lengths, head, value_head, explicit, distribution = varlen_cases[index]
        spec.update(
            {
                "layout": layout,
                "T": total,
                "H": head,
                "HV": value_head,
                "cu_seqlens": _cu(lengths),
                "explicit_chunk_indices": explicit,
                "distribution": distribution,
            }
        )
        return

    raise ValueError(f"no GVA accuracy rule for KDA-FWD-G{index:03d}")


def _validate_spec(spec: dict) -> None:
    missing = EXECUTOR_REQUIRED_FIELDS.difference(spec)
    if missing:
        raise ValueError(f"{spec.get('design_id')} missing executor fields {sorted(missing)}")
    if "accuracy" not in spec["tags"].split(",") or "negative" in spec["tags"].split(","):
        raise ValueError(f"{spec['design_id']} is not a positive accuracy spec")
    if spec["layout"] not in {"BSND", "BNSD", "TND", "NTD"}:
        raise ValueError(f"{spec['design_id']} has invalid layout")
    if spec["q_dtype"] not in {"bf16", "fp16"}:
        raise ValueError(f"{spec['design_id']} has invalid q_dtype")
    for key in ("g_dtype", "beta_dtype", "a_log_dtype", "dt_bias_dtype"):
        if spec[key] not in {"bf16", "fp32"}:
            raise ValueError(f"{spec['design_id']} has invalid {key}")
    if int(spec["H"]) <= 0 or int(spec["HV"]) % int(spec["H"]):
        raise ValueError(f"{spec['design_id']} has invalid GVA heads")
    if int(spec["chunk_size"]) not in {64, 128}:
        raise ValueError(f"{spec['design_id']} has invalid chunk_size")
    # The upstream GPU Triton KDA kernel supports chunks 32/64 only.  Keep
    # chunk-128 design rows executable with an independent GPU Torch
    # same-precision control, and expose that exception in the generated spec.
    spec["gpu_control_reference"] = (
        "triton_same_precision"
        if int(spec["chunk_size"]) == 64
        else "torch_same_precision"
    )
    cu_text = str(spec["cu_seqlens"])
    if cu_text:
        cu_values = [int(value) for value in cu_text.split(",")]
        if cu_values[0] != 0 or cu_values[-1] != int(spec["T"]):
            raise ValueError(f"{spec['design_id']} has invalid cu_seqlens endpoints")
        if any(left > right for left, right in zip(cu_values, cu_values[1:])):
            raise ValueError(f"{spec['design_id']} has decreasing cu_seqlens")
        if int(spec["B"]) != 1:
            raise ValueError(f"{spec['design_id']} varlen requires B=1")
    elif spec["explicit_chunk_indices"]:
        raise ValueError(f"{spec['design_id']} has indices without cu_seqlens")
    spec["scale"] = 1.0 / math.sqrt(int(spec["K"]))
    spec["shape_spec"] = (
        f"B={spec['B']},H={spec['H']},HV={spec['HV']},T={spec['T']},"
        f"K={spec['K']},V={spec['V']}"
    )
    spec["optional_spec"] = (
        f"initial={spec['initial_state']},final={spec['output_final_state']},"
        f"varlen={bool(cu_text)},disable_recompute={spec['disable_recompute']}"
    )


def materialize(path=DEFAULT_MANIFEST) -> list[dict]:
    """Return exact ``{'id': int, 'spec': dict}`` records for 176 cases."""
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("op") != "chunk_kda_fwd":
        raise ValueError("canonical source is not the chunk_kda_fwd manifest")
    matrix = manifest.get("design_matrix", {})
    source = matrix.get("source", {})
    if source.get("row_sha256") != EXPECTED_SOURCE_SHA256 or source.get("row_count") != 300:
        raise ValueError("canonical design source changed; update the explicit executable rules")
    accuracy_rows = [case for case in matrix.get("cases", []) if case.get("kind") == "accuracy"]
    actual_ids = tuple(row.get("id") for row in accuracy_rows)
    if actual_ids != EXPECTED_ACCURACY_IDS:
        raise ValueError("canonical accuracy IDs are not the exact ordered 176-case sequence")

    records = []
    for ordinal, row in enumerate(accuracy_rows, start=1):
        spec = _base_spec(row, ordinal)
        design_id = row["id"]
        prefix, index = design_id[8], _case_number(design_id)
        if prefix == "P":
            _apply_p_rule(index, spec)
        elif prefix == "G":
            _apply_g_rule(index, spec)
        else:
            raise ValueError(f"unsupported accuracy prefix in {design_id}")
        _validate_spec(spec)
        records.append({"id": _atk_id(design_id), "spec": spec})

    if len(records) != 176 or len({record["id"] for record in records}) != 176:
        raise ValueError("materializer did not produce 176 unique records")
    if tuple(record["spec"]["design_id"] for record in records) != EXPECTED_ACCURACY_IDS:
        raise ValueError("materializer lost canonical design ID order")
    return records


def _apply_accuracy_variant(spec: dict, variant: str) -> dict:
    if variant not in spec["design_variants"]:
        raise ValueError(
            f"{spec['design_id']} does not declare accuracy variant {variant!r}"
        )
    projected = copy.deepcopy(spec)
    projected["materialized_variant"] = variant
    projected["traceable_head_mapping"] = variant == "traceable_metamorphic"
    if variant == "traceable_metamorphic":
        projected["case_key"] = f"{projected['case_key']}_traceable"
    return projected


def materialize_cache_variants(spec: dict) -> list[dict]:
    """Return every distinct numerical accuracy variant for one logical case."""
    return [
        {"variant": variant, "spec": _apply_accuracy_variant(spec, variant)}
        for variant in spec["design_variants"]
    ]


def _physical_case_id(logical_case_id: int, variant: str) -> int:
    if variant == "random":
        return logical_case_id
    if variant == "traceable_metamorphic":
        return logical_case_id + TRACEABLE_PHYSICAL_ID_OFFSET
    raise ValueError(f"unsupported accuracy variant {variant!r}")


def _input(name: str, dtype: str, value, *, input_type="attr", shape=None) -> dict:
    return {
        "name": name,
        "type": input_type,
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": value,
        "backward": False,
    }


def _atk_payload(record: dict) -> dict:
    case_id, spec = int(record["id"]), record["spec"]
    inputs = [
        _input("low_precision_marker", spec["q_dtype"], [0, 0], input_type="tensor", shape=[1]),
        _input("fp32_marker", "fp32", [0, 0], input_type="tensor", shape=[1]),
        _input(
            "case_spec",
            "non_param",
            json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ),
        _input("design_id", "string", spec["design_id"]),
        _input("soc", "string", spec["soc"]),
        _input("route", "string", spec["route"]),
        _input("batch", "int", spec["B"]),
        _input("head", "int", spec["H"]),
        _input("value_head", "int", spec["HV"]),
        _input("total_tokens", "int", spec["T"]),
        _input("key_dim", "int", spec["K"]),
        _input("value_dim", "int", spec["V"]),
        _input("chunk_size", "int", spec["chunk_size"]),
        _input("layout", "string", spec["layout"]),
        _input("q_dtype", "string", spec["q_dtype"]),
        _input("g_dtype", "string", spec["g_dtype"]),
        _input("beta_dtype", "string", spec["beta_dtype"]),
        _input("a_log_dtype", "string", spec["a_log_dtype"]),
        _input("dt_bias_dtype", "string", spec["dt_bias_dtype"]),
        _input("initial_state", "bool", spec["initial_state"]),
        _input("output_final_state", "bool", spec["output_final_state"]),
        _input("varlen", "bool", bool(spec["cu_seqlens"])),
        _input("safe_gate", "bool", spec["safe_gate"]),
        _input("lower_bound", "float", spec["lower_bound"]),
        _input("use_gate_in_kernel", "bool", spec["use_gate_in_kernel"]),
        _input("disable_recompute", "bool", spec["disable_recompute"]),
        _input("return_intermediate_states", "bool", spec["return_intermediate_states"]),
        _input("state_v_first", "bool", spec["state_v_first"]),
        _input("negative_case", "bool", False),
    ]
    return {
        "id": case_id,
        "default_seed": int(spec["seed"]),
        "name": "chunk_kda_fwd",
        "aclnn_name": "ChunkKdaFwd",
        "version": "v2.1",
        "api": "pytorch",
        "api_type": "executor_chunk_kda_fwd",
        "expected_error_msg": None,
        "backward": False,
        "standard": L1_STANDARD,
        "outputs": None,
        "inputs": inputs,
        "save_name": "chunk_kda_fwd",
    }


def project_records(
    path=DEFAULT_MANIFEST,
    *,
    soc: str | None = None,
    route: str | None = None,
    variant: str | None = None,
) -> list[dict]:
    if soc is not None and soc not in {"ascend910b", "ascend910_93", "ascend950"}:
        raise ValueError(f"unsupported SOC: {soc}")
    if route is not None and route not in {"ascendc", "aclnn", "direct_launch"}:
        raise ValueError(f"unsupported route: {route}")
    if variant is not None and variant not in {"random", "traceable_metamorphic"}:
        raise ValueError(f"unsupported accuracy variant: {variant}")
    records = materialize(path)
    projected_records = []
    for record in records:
        spec = record["spec"]
        if soc is not None and soc not in spec["target_platforms"]:
            continue
        if route is not None and route not in spec["target_routes"]:
            continue
        variants = [variant] if variant is not None else spec["design_variants"]
        for target_variant in variants:
            if target_variant not in spec["design_variants"]:
                continue
            projected_spec = _apply_accuracy_variant(spec, target_variant)
            if soc is not None:
                projected_spec["soc"] = soc
            if route is not None:
                projected_spec["route"] = route
            projected_records.append(
                {
                    "id": _physical_case_id(record["id"], target_variant),
                    "spec": projected_spec,
                }
            )
    return projected_records


def build_atk_payloads(
    path=DEFAULT_MANIFEST,
    *,
    soc: str | None = None,
    route: str | None = None,
    variant: str | None = None,
) -> list[dict]:
    return [
        _atk_payload(record)
        for record in project_records(path, soc=soc, route=route, variant=variant)
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_ATK_JSON)
    parser.add_argument("--soc", choices=("ascend910b", "ascend910_93", "ascend950"))
    parser.add_argument("--route", choices=("ascendc", "aclnn", "direct_launch"))
    parser.add_argument("--variant", choices=("random", "traceable_metamorphic"))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    payloads = build_atk_payloads(
        args.source,
        soc=args.soc,
        route=args.route,
        variant=args.variant,
    )
    args.output.write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.summary:
        design_ids = [
            json.loads(next(item["range_values"] for item in case["inputs"] if item["name"] == "case_spec"))[
                "design_id"
            ]
            for case in payloads
        ]
        print(
            f"physical_accuracy_cases={len(payloads)} soc={args.soc or 'all'} "
            f"route={args.route or 'ascendc-primary'} "
            f"variant={args.variant or 'all'} "
            f"first={design_ids[0] if design_ids else '-'} last={design_ids[-1] if design_ids else '-'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
