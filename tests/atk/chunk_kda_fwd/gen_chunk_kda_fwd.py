"""Generate canonical ATK manifests for the modern chunk_kda_fwd ABI.

The host implementation has two tiling keys:

* key 2: ``chunk_size == 64`` and ``K == V == 128``;
* key 1: every other valid shape.

The checked-in accuracy manifest is fixed at 200 positive cases.  MSS,
determinism, and performance use separate compact manifests so both keys
have ordinary and boundary coverage.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:  # Keep repository-only linting usable.
    if exc.name not in {"atk", "torch"}:
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None


OP_NAME = "chunk_kda_fwd"
SEED_BASE = 20260831
ACCURACY_PROFILE_COUNT = 100
ACCURACY_COUNT = ACCURACY_PROFILE_COUNT * 2
TILING_KEYS = (1, 2)
SOCS = ("ascend910b", "ascend910_93", "ascend950")
STANDARD = {"acc": "mixed_tolerance_bm", "perf": "not_key"}
MSS_STANDARD = {"acc": "mixed_tolerance_bm", "perf": "not_key", "mem": 1.1}
_T_VALUES = (64, 65, 96, 127, 128, 129, 192, 256, 512, 1024)
_DATA_SCALES = (0.03, 0.08, 0.2, 0.5)
_GATE_SCALES = (0.01, 0.02, 0.04, 0.08)
_BETA_RANGES = ((0.1, 0.9), (0.05, 0.95), (0.2, 0.8), (0.01, 0.99))


def _tiling_key(chunk_size: int, key_dim: int, value_dim: int) -> int:
    return 2 if (int(chunk_size), int(key_dim), int(value_dim)) == (64, 128, 128) else 1


def _cu_seqlens(total_tokens: int, profile_id: int) -> str:
    first = max(1, total_tokens // (3 + profile_id % 3))
    second = max(first + 1, total_tokens * 2 // 3)
    second = min(second, total_tokens - 1)
    first = min(first, second - 1)
    return f"0,{first},{second},{total_tokens}"


def _accuracy_spec(profile_id: int, q_dtype: str, case_id: int) -> dict[str, Any]:
    layouts = ("BSND", "BNSD", "TND", "NTD")
    layout = layouts[profile_id % len(layouts)]
    total_tokens = _T_VALUES[(profile_id // 4) % len(_T_VALUES)]
    h_values = (1, 2, 4, 8)
    h_num = h_values[profile_id % len(h_values)]
    hv_num = min(h_num * (1, 1, 2, 2, 4)[(profile_id // 4) % 5], 32)
    if layout == "TND":
        h_num = 1
        hv_num = (1, 2, 4, 8)[(profile_id // 4) % 4]
    batch = 1 if layout in {"TND", "NTD"} else 1 + (profile_id // 16) % 3
    varlen = profile_id % 5 in {1, 4}
    if varlen:
        batch = 1

    # Rotate the template key with two representative key-1 combinations.
    shape_variant = profile_id % 3
    if shape_variant == 0:
        chunk_size, key_dim, value_dim = 64, 128, 128
    elif shape_variant == 1:
        chunk_size, key_dim, value_dim = 128, 128, 128
    else:
        chunk_size, key_dim, value_dim = 64, 128, 256
    cu = _cu_seqlens(total_tokens, profile_id) if varlen else ""
    initial_state = profile_id % 7 in {0, 3}
    output_final_state = profile_id % 3 != 1
    disable_recompute = profile_id % 2 == 1
    beta_low, beta_high = _BETA_RANGES[profile_id % len(_BETA_RANGES)]
    key = _tiling_key(chunk_size, key_dim, value_dim)
    return {
        "case_id": case_id,
        "case_key": f"accuracy_{profile_id:03d}_{layout}_t{total_tokens}_{q_dtype}",
        "design_id": f"KDA-FWD-A{profile_id:03d}-{q_dtype}",
        "profile": "accuracy_100x2",
        "tags": "accuracy,regression" + (",varlen" if varlen else ",dense"),
        "soc": "all",
        "target_platforms": list(SOCS),
        "target_routes": ["ascendc"],
        "route": "ascendc",
        "execution_mode": "public_api",
        "coverage_only": False,
        "runtime_status": "public_api_reachable",
        "B": batch,
        "H": h_num,
        "HV": hv_num,
        "T": total_tokens,
        "K": key_dim,
        "V": value_dim,
        "chunk_size": chunk_size,
        "layout": layout,
        "q_dtype": q_dtype,
        "g_dtype": "fp32",
        "beta_dtype": "bf16" if profile_id % 2 == 0 else "fp32",
        "scale": 1.0 / math.sqrt(key_dim),
        "initial_state": initial_state,
        "output_final_state": output_final_state,
        "cu_seqlens": cu,
        "explicit_chunk_indices": bool(varlen and profile_id % 2 == 0),
        "safe_gate": True,
        "lower_bound": -5.0,
        "use_gate_in_kernel": True,
        "dt_bias": True,
        "disable_recompute": disable_recompute,
        "return_intermediate_states": True,
        "state_v_first": False,
        "negative_case": False,
        "data_profile": "uniform",
        "data_scale": _DATA_SCALES[profile_id % len(_DATA_SCALES)],
        "gate_scale": _GATE_SCALES[profile_id % len(_GATE_SCALES)],
        "qk_scale": 0.25,
        "v_scale": 0.25,
        "beta_scale": 0.35,
        "beta_bias": 1.5,
        "a_log_scale": 0.12,
        "dt_bias_scale": 0.5,
        "dt_bias_mean": -3.0,
        "beta_low": beta_low,
        "beta_high": beta_high,
        "state_scale": 0.02,
        "tiling_key": key,
        "expected_tiling_key": key,
        "tiling_key_condition": (
            "chunk_size=64 and K=128 and V=128" if key == 2
            else "otherwise (valid chunk/K/V combination)"
        ),
        "tiling_selection_evidence": "chunk_kda_fwd_tiling.cpp:SetTilingKey",
        "seed": SEED_BASE + case_id,
    }


def build_accuracy_specs(*, seed_base: int = SEED_BASE) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    case_id = 0
    for profile_id in range(ACCURACY_PROFILE_COUNT):
        for q_dtype in ("bf16", "fp16"):
            spec = _accuracy_spec(profile_id, q_dtype, case_id)
            spec["seed"] = int(seed_base) + case_id
            specs.append(spec)
            case_id += 1
    return specs


def build_specs(*, seed_base: int = SEED_BASE) -> list[dict[str, Any]]:
    """Return the canonical accuracy matrix for legacy tooling callers."""
    return build_accuracy_specs(seed_base=seed_base)


def _coverage_spec(key: int, boundary: bool, case_id: int, *, seed_base: int) -> dict[str, Any]:
    if key == 2:
        chunk_size, key_dim, value_dim = 64, 128, 128
    elif boundary:
        chunk_size, key_dim, value_dim = 128, 128, 128
    else:
        chunk_size, key_dim, value_dim = 64, 128, 256
    total_tokens = 65 if boundary else 64
    return {
        "case_id": case_id,
        "case_key": f"mss_key{key}_{'boundary' if boundary else 'ordinary'}",
        "design_id": f"KDA-FWD-MSS-K{key}-{int(boundary)}",
        "profile": "mss_tiling_key_coverage",
        "tags": f"mss,tiling_key_{key},{'boundary' if boundary else 'ordinary'}",
        "soc": "all",
        "target_platforms": list(SOCS),
        "target_routes": ["ascendc"],
        "route": "ascendc",
        "execution_mode": "public_api",
        "coverage_only": False,
        "runtime_status": "public_api_reachable",
        "B": 1,
        "H": 1,
        "HV": 1,
        "T": total_tokens,
        "K": key_dim,
        "V": value_dim,
        "chunk_size": chunk_size,
        "layout": "BSND",
        "q_dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "fp32",
        "scale": 1.0 / math.sqrt(key_dim),
        "initial_state": boundary,
        "output_final_state": True,
        "cu_seqlens": "",
        "explicit_chunk_indices": False,
        "safe_gate": True,
        "lower_bound": -5.0,
        "use_gate_in_kernel": True,
        "dt_bias": True,
        "disable_recompute": True,
        "return_intermediate_states": True,
        "state_v_first": False,
        "negative_case": False,
        "data_profile": "uniform",
        "data_scale": 0.08,
        "gate_scale": 0.02,
        "qk_scale": 0.25,
        "v_scale": 0.25,
        "beta_scale": 0.35,
        "beta_bias": 1.5,
        "a_log_scale": 0.12,
        "dt_bias_scale": 0.5,
        "dt_bias_mean": -3.0,
        "beta_low": 0.1,
        "beta_high": 0.9,
        "state_scale": 0.02,
        "tiling_key": key,
        "expected_tiling_key": key,
        "tiling_key_condition": (
            "chunk_size=64 and K=128 and V=128" if key == 2
            else "otherwise (valid chunk/K/V combination)"
        ),
        "tiling_selection_evidence": "chunk_kda_fwd_tiling.cpp:SetTilingKey",
        "seed": int(seed_base) + 1000 + case_id,
    }


def build_mss_specs(*, seed_base: int = SEED_BASE) -> list[dict[str, Any]]:
    entries = ((key, boundary) for key in TILING_KEYS for boundary in (False, True))
    return [_coverage_spec(key, boundary, case_id, seed_base=seed_base)
            for case_id, (key, boundary) in enumerate(entries)]


def build_perf_specs(*, seed_base: int = SEED_BASE) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for case_id, key in enumerate(TILING_KEYS):
        spec = _coverage_spec(key, False, case_id, seed_base=seed_base)
        spec.update({
            "case_key": f"perf_key{key}",
            "design_id": f"KDA-FWD-PERF-K{key}",
            "profile": "perf_tiling_key_coverage",
            "tags": f"performance,tiling_key_{key}",
            "T": 512,
            "seed": int(seed_base) + 2000 + case_id,
        })
        specs.append(spec)
    return specs


def _input(name: str, dtype: str, value: Any, *, input_type: str = "attr", shape=None) -> dict[str, Any]:
    return {"name": name, "type": input_type, "required": True, "dtype": dtype,
            "shape": shape, "range_values": value, "backward": False}


_ATTRS = (
    ("design_id", "string"), ("soc", "string"), ("route", "string"),
    ("batch", "int"), ("head", "int"), ("value_head", "int"),
    ("total_tokens", "int"), ("key_dim", "int"), ("value_dim", "int"),
    ("chunk_size", "int"), ("layout", "string"), ("scale", "float"),
    ("q_dtype", "string"), ("g_dtype", "string"), ("beta_dtype", "string"),
    ("initial_state", "bool"), ("output_final_state", "bool"), ("varlen", "bool"),
    ("cu_seqlens", "string"), ("explicit_chunk_indices", "bool"),
    ("safe_gate", "bool"), ("lower_bound", "float"),
    ("use_gate_in_kernel", "bool"), ("dt_bias", "bool"),
    ("disable_recompute", "bool"), ("return_intermediate_states", "bool"),
    ("state_v_first", "bool"), ("negative_case", "bool"),
    ("tiling_key", "int"), ("expected_tiling_key", "int"),
    ("tiling_key_condition", "string"), ("execution_mode", "string"),
    ("coverage_only", "bool"), ("runtime_status", "string"),
    ("tiling_selection_evidence", "string"), ("data_profile", "string"),
    ("data_scale", "float"), ("gate_scale", "float"), ("qk_scale", "float"),
    ("v_scale", "float"), ("beta_scale", "float"), ("beta_bias", "float"),
    ("a_log_scale", "float"), ("dt_bias_scale", "float"),
    ("dt_bias_mean", "float"), ("beta_low", "float"), ("beta_high", "float"),
    ("state_scale", "float"), ("case_id", "int"), ("profile", "string"),
    ("tags", "string"),
)

_ATTR_ALIASES = {
    "batch": "B",
    "head": "H",
    "value_head": "HV",
    "total_tokens": "T",
    "key_dim": "K",
    "value_dim": "V",
}


def _case_payload(spec: dict[str, Any], *, manifest: str) -> dict[str, Any]:
    case_id = int(spec["case_id"])
    metadata = {
        "op": OP_NAME,
        "manifest": manifest,
        "shape_spec": (
            f"B={spec['B']},H={spec['H']},HV={spec['HV']},T={spec['T']},"
            f"K={spec['K']},V={spec['V']},chunk={spec['chunk_size']}"
        ),
        "optional_spec": (
            f"initial={spec['initial_state']},final={spec['output_final_state']},"
            f"varlen={bool(spec['cu_seqlens'])},indices={spec['explicit_chunk_indices']}"
        ),
        **spec,
    }
    inputs = [
        _input("low_precision_marker", spec["q_dtype"], [0, 0], input_type="tensor", shape=[1]),
        _input("fp32_marker", "fp32", [0, 0], input_type="tensor", shape=[1]),
        _input("case_spec", "non_param", json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"))),
    ]
    for name, dtype in _ATTRS:
        source_name = _ATTR_ALIASES.get(name, name)
        value = (
            bool(spec["cu_seqlens"])
            if name == "varlen"
            else spec.get(source_name, False if dtype == "bool" else 0)
        )
        inputs.append(_input(name, dtype, value))
    return {
        "id": case_id,
        "default_seed": int(spec["seed"]),
        "name": f"{OP_NAME}_{manifest}_{case_id:04d}",
        "aclnn_name": "ChunkKdaFwd",
        "version": "v2.1",
        "api": "pytorch",
        "api_type": "executor_chunk_kda_fwd",
        "expected_error_msg": None,
        "backward": False,
        "standard": MSS_STANDARD if manifest == "mss" else STANDARD,
        "outputs": None,
        "inputs": inputs,
        "save_name": OP_NAME,
    }


def _validate_specs(specs: list[dict[str, Any]], manifest: str) -> None:
    if [int(item["case_id"]) for item in specs] != list(range(len(specs))):
        raise ValueError(f"{manifest} IDs must be contiguous from zero")
    for spec in specs:
        expected = _tiling_key(spec["chunk_size"], spec["K"], spec["V"])
        if int(spec["tiling_key"]) != expected or int(spec["expected_tiling_key"]) != expected:
            raise ValueError(f"{spec['case_key']} has a stale tiling key")
    if manifest == "accuracy" and (len(specs) != ACCURACY_COUNT or {str(s["soc"]) for s in specs} != {"all"}):
        raise ValueError("accuracy must contain exactly 200 cross-SoC cases")
    if manifest == "mss" and {(int(s["tiling_key"]), bool(s["initial_state"])) for s in specs} != {
        (key, boundary) for key in TILING_KEYS for boundary in (False, True)
    }:
        raise ValueError("MSS must contain ordinary and boundary records for both keys")
    if manifest == "perf" and {int(s["tiling_key"]) for s in specs} != set(TILING_KEYS):
        raise ValueError("performance must contain both tiling keys")


def build_cases() -> list[CaseConfig]:
    if CaseConfig is None:
        raise RuntimeError("ATK and PyTorch are required to instantiate CaseConfig objects")
    return [CaseConfig(**_case_payload(spec, manifest="accuracy")) for spec in build_accuracy_specs()]


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_chunk_kda_fwd")
    class ChunkKdaFwdGenerator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)
            self.cases = build_cases()
            self.length = len(self.cases)
            self.index = 0

        def generate(self):
            if self.index >= self.length:
                raise StopIteration
            case = self.cases[self.index]
            self.index += 1
            return case


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize chunk_kda_fwd ATK manifests")
    parser.add_argument("--output", "--atk-output-json", type=Path, default=None)
    parser.add_argument("--mss-output", type=Path, default=None)
    parser.add_argument("--perf-output", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--positive", type=int, default=ACCURACY_COUNT)
    parser.add_argument("--negative", type=int, default=0)
    parser.add_argument("--seed", type=int, default=SEED_BASE)
    parser.add_argument("--atk-template-json", type=Path, default=None)
    parser.add_argument("--print-summary", "--summary", action="store_true")
    args = parser.parse_args()
    if args.positive != ACCURACY_COUNT or args.negative:
        raise ValueError("chunk_kda_fwd has a frozen 200-positive-case manifest and no negative rows")
    if args.atk_template_json is not None and not args.atk_template_json.is_file():
        raise FileNotFoundError(args.atk_template_json)
    root = args.out_dir or Path(__file__).resolve().parent
    paths = (
        args.output or root / "atk_chunk_kda_fwd.json",
        args.mss_output or root / "atk_chunk_kda_fwd_mss.json",
        args.perf_output or root / "atk_chunk_kda_fwd_perf.json",
    )
    specs_by_manifest = (
        (build_accuracy_specs(seed_base=args.seed), "accuracy"),
        (build_mss_specs(seed_base=args.seed), "mss"),
        (build_perf_specs(seed_base=args.seed), "perf"),
    )
    for specs, manifest in specs_by_manifest:
        _validate_specs(specs, manifest)
    for path, (specs, manifest) in zip(paths, specs_by_manifest):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([_case_payload(s, manifest=manifest) for s in specs], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.print_summary:
        print(
            f"seed={args.seed} accuracy={len(specs_by_manifest[0][0])} "
            f"mss={len(specs_by_manifest[1][0])} perf={len(specs_by_manifest[2][0])} "
            f"accuracy_keys={sorted({s['tiling_key'] for s in specs_by_manifest[0][0]})} "
            f"mss_keys={sorted({s['tiling_key'] for s in specs_by_manifest[1][0]})} "
            f"perf_keys={sorted({s['tiling_key'] for s in specs_by_manifest[2][0]})}"
        )


if __name__ == "__main__":
    main()
