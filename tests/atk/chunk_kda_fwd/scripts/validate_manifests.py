#!/usr/bin/env python3
"""Validate the checked-in chunk_kda_fwd ATK manifests without an NPU."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


HERE = Path(__file__).resolve()
OP_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[4]
GEN_PATH = OP_DIR / "gen_chunk_kda_fwd.py"
PREPARE_PATH = (
    REPO_ROOT
    / "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/arch35/chunk_kda_fwd_prepare.h"
)
SOCS = {"ascend910b", "ascend910_93", "ascend950"}
GATE_VARIANTS = {
    (False, False, False),
    (True, False, False),
    (True, False, True),
    (True, True, False),
    (True, True, True),
}
OUTPUT_POLICIES = {(False, False), (False, True), (True, False), (True, True)}
SHAPE_VARIANTS = {
    (64, 128, 128),
    (128, 128, 128),
    (64, 128, 256),
    (64, 16, 128),
    (64, 256, 128),
}
ENUMERATED_YAML_INPUTS = {
    "soc", "batch", "head", "value_head", "total_tokens", "key_dim", "value_dim",
    "chunk_size", "layout", "scale", "q_dtype", "g_dtype", "beta_dtype",
    "initial_state", "output_final_state", "varlen", "explicit_chunk_indices",
    "safe_gate", "lower_bound", "use_gate_in_kernel", "dt_bias",
    "disable_recompute", "return_intermediate_states", "state_v_first",
    "negative_case", "tiling_key", "expected_tiling_key", "execution_mode",
    "coverage_only", "runtime_status", "data_profile", "data_scale", "gate_scale",
    "qk_scale", "v_scale", "beta_scale", "beta_bias", "a_log_scale",
    "dt_bias_scale", "dt_bias_mean", "beta_low", "beta_high", "state_scale", "profile",
}


def _load_generator():
    spec = importlib.util.spec_from_file_location("chunk_kda_fwd_generator", GEN_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {GEN_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path.name} must contain a JSON list")
    return value


def _spec(payload: dict) -> dict:
    for item in payload.get("inputs", []):
        if item.get("name") == "case_spec":
            value = json.loads(item["range_values"])
            if not isinstance(value, dict):
                raise ValueError("case_spec must decode to an object")
            return value
    raise ValueError(f"payload {payload.get('id')} has no case_spec")


def _input_values(payload: dict) -> dict[str, object]:
    return {
        str(item["name"]): item.get("range_values")
        for item in payload.get("inputs", [])
        if isinstance(item, dict) and "name" in item
    }


def _check_manifest(path: Path, expected_count: int, expected_keys: set[int]) -> list[dict]:
    cases = _read(path)
    if len(cases) != expected_count:
        raise ValueError(f"{path.name}: expected {expected_count} cases, got {len(cases)}")
    if [int(case["id"]) for case in cases] != list(range(expected_count)):
        raise ValueError(f"{path.name}: ids must be contiguous from zero")
    specs = []
    for case in cases:
        spec = _spec(case)
        specs.append(spec)
        values = _input_values(case)
        if int(case["id"]) != int(spec["case_id"]):
            raise ValueError(f"{path.name}: id/case_spec mismatch")
        aliases = {
            "batch": "B",
            "head": "H",
            "value_head": "HV",
            "total_tokens": "T",
            "key_dim": "K",
            "value_dim": "V",
        }
        for input_name, spec_name in aliases.items():
            if values.get(input_name) != spec[spec_name]:
                raise ValueError(
                    f"{path.name}: {input_name} does not match case_spec.{spec_name}"
                )
        key = int(spec["tiling_key"])
        expected = 2 if (int(spec["chunk_size"]), int(spec["K"]), int(spec["V"])) == (64, 128, 128) else 1
        if key != expected or int(spec["expected_tiling_key"]) != expected:
            raise ValueError(f"{path.name}: stale tiling key in {spec['case_key']}")
        if key not in expected_keys:
            raise ValueError(f"{path.name}: unexpected key {key}")
        if spec.get("soc") != "all":
            raise ValueError(f"{path.name}: canonical manifests must use soc=all")
        if set(spec.get("target_platforms", [])) != SOCS:
            raise ValueError(f"{path.name}: target platform matrix is incomplete")
        if bool(spec.get("coverage_only")):
            raise ValueError(f"{path.name}: runtime manifests cannot be coverage-only")
    return specs


def _check_source_evidence() -> None:
    tiling = (REPO_ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd/op_host/chunk_kda_fwd_tiling.cpp").read_text(encoding="utf-8")
    kernel = (REPO_ROOT / "fla/ops/ascendc/kda/chunk_kda_fwd/op_kernel/chunk_kda_fwd.cpp").read_text(encoding="utf-8")
    if "SetTilingKey(useChunk64K128V128Template ? 2 : 1)" not in tiling:
        raise ValueError("host tiling key predicate is missing")
    for key in (1, 2):
        if f"TILING_KEY_IS({key})" not in kernel:
            raise ValueError(f"kernel source has no TILING_KEY_IS({key}) dispatch")


def _source_section(source: str, start_marker: str, end_marker: str) -> str:
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    return " ".join(source[start:end].split())


def _check_kernel_sync_contract() -> None:
    source = PREPARE_PATH.read_text(encoding="utf-8")
    compile_gate = (
        "SAFE_GATE && COMPILE_BT == 64 && COMPILE_K == 128 && COMPILE_V == 128"
    )

    join = _source_section(
        source,
        "__aicore__ inline void JoinAivMte3()",
        "__aicore__ inline void RunAicAfterBothAivReady",
    )
    for token in (
        "if (!isAivOnly_)",
        "Catlass::Arch::CrossCoreBarrier<0x1, PIPE_MTE3>();",
        "PipeBarrier<PIPE_MTE3>();",
    ):
        if token not in join:
            raise ValueError(f"A5 AIV join is missing {token}")
    if "if (!headPairMode_)" in join:
        raise ValueError("head-pair AIV lanes must join before publishing a shared ready token")

    aiv_direct = _source_section(
        source, "const bool useDirectScoreUb =", "bool firstSolveRowsPrepared"
    )
    aic_direct = _source_section(
        source,
        "__aicore__ inline void ProcessChunkPreAicHeadPairFp32",
        "if (!directScoreDispatched)",
    )
    initializer = _source_section(
        source,
        "__aicore__ inline void ProcessPreAivHeadPair()",
        "for (uint64_t task = coreIdx",
    )
    for name, section in (
        ("AIV direct-score eligibility", aiv_direct),
        ("AIC direct-score eligibility", aic_direct),
        ("direct-score credit initialization", initializer),
    ):
        if compile_gate not in section or "KDA_ARCH35_ENABLE_DIRECT_SCORE_UB" not in section:
            raise ValueError(f"{name} drifted from the key2-only compile gate")
    for token in (
        "curT == 64",
        "scoreBlockSize == KDA_DIRECT_SCORE_ROWS",
        "scoreBlockCount == KDA_SCORE_QUEUE_DEPTH",
    ):
        if token not in aiv_direct or token not in aic_direct:
            raise ValueError(f"A5 direct-score producer/consumer mismatch: {token}")
    if "rowCount == KDA_DIRECT_SCORE_ROWS" not in aic_direct:
        raise ValueError("AIC direct-score dispatch must require one direct-score row block")
    if "InitializeDirectScoreUbArch35();" not in initializer:
        raise ValueError("key2 direct-score queue has no initial free credits")

    aiv_pipeline = _source_section(
        source,
        "__aicore__ inline void ProcessChunkPreAivFp32",
        "__aicore__ inline void ProcessChunkPreAicFp32",
    )
    join_index = aiv_pipeline.index("JoinAivMte3();")
    ready_index = aiv_pipeline.index(
        "CrossCoreSetFlagWithReverse<0x2, PIPE_MTE3>(scoreReadyFlag_)"
    )
    if join_index > ready_index:
        raise ValueError("A5 score-ready token is published before both AIV lanes join")


def _check_generator(module, manifests: dict[str, list[dict]]) -> None:
    expected = {
        "accuracy": module.build_accuracy_specs(),
        "mss": module.build_mss_specs(),
        "perf": module.build_perf_specs(),
    }
    for name, specs in expected.items():
        path = OP_DIR / f"atk_chunk_kda_fwd{'_' + name if name != 'accuracy' else ''}.json"
        actual = manifests[name]
        generated = [module._case_payload(item, manifest=name) for item in specs]
        if actual != generated:
            raise ValueError(f"{path.name}: materialized payloads drifted from generator")


def _contains_exact(specs: list[dict], expected: dict[str, object]) -> bool:
    return any(all(spec.get(name) == value for name, value in expected.items()) for spec in specs)


def _check_accuracy_coverage(specs: list[dict]) -> None:
    profile_dtypes: dict[str, set[str]] = {}
    for spec in specs:
        profile = str(spec["design_id"]).rsplit("-", 1)[0]
        profile_dtypes.setdefault(profile, set()).add(str(spec["q_dtype"]))
    if len(profile_dtypes) != 100 or any(dtypes != {"bf16", "fp16"} for dtypes in profile_dtypes.values()):
        raise ValueError("accuracy must contain 100 structural profiles paired with BF16/FP16")

    gate_variants = {
        (bool(spec["use_gate_in_kernel"]), bool(spec["safe_gate"]), bool(spec["dt_bias"]))
        for spec in specs
    }
    if gate_variants != GATE_VARIANTS:
        raise ValueError(f"accuracy gate coverage mismatch: {sorted(gate_variants)}")
    if {str(spec["g_dtype"]) for spec in specs} != {"fp32", "bf16"}:
        raise ValueError("accuracy must cover FP32 and BF16 gate tensors")
    gate_dtype_variants = {
        (
            bool(spec["use_gate_in_kernel"]), bool(spec["safe_gate"]), bool(spec["dt_bias"]),
            str(spec["g_dtype"]),
        )
        for spec in specs
    }
    expected_gate_dtype_variants = {
        variant + (g_dtype,)
        for variant in GATE_VARIANTS
        for g_dtype in ("fp32", "bf16")
    }
    if gate_dtype_variants != expected_gate_dtype_variants:
        raise ValueError("each gate variant must cover FP32 and BF16 gate tensors")
    output_policies = {
        (bool(spec["disable_recompute"]), bool(spec["return_intermediate_states"]))
        for spec in specs
    }
    if output_policies != OUTPUT_POLICIES:
        raise ValueError(f"accuracy output-policy coverage mismatch: {sorted(output_policies)}")
    if {bool(spec["state_v_first"]) for spec in specs} != {False, True}:
        raise ValueError("accuracy must cover both state layouts")
    if {int(spec["K"]) for spec in specs} != {16, 128, 256}:
        raise ValueError("accuracy must cover K=16/128/256")
    actual_shapes = {
        (int(spec["chunk_size"]), int(spec["K"]), int(spec["V"])) for spec in specs
    }
    if not SHAPE_VARIANTS.issubset(actual_shapes):
        raise ValueError(f"accuracy shape coverage mismatch: {sorted(actual_shapes)}")
    if not any(
        bool(spec["state_v_first"])
        and int(spec["K"]) != int(spec["V"])
        and any(
            bool(spec[name])
            for name in (
                "initial_state", "output_final_state",
                "disable_recompute", "return_intermediate_states",
            )
        )
        for spec in specs
    ):
        raise ValueError("state_v_first=true must exercise an observable K/V-asymmetric state path")
    if {int(spec["tiling_key"]) for spec in specs} != {1, 2}:
        raise ValueError("accuracy must exercise both tiling keys")

    a5_fusion = {
        "q_dtype": "bf16", "g_dtype": "fp32", "B": 1, "H": 1, "HV": 2,
        "T": 256, "K": 128, "V": 128, "chunk_size": 64, "layout": "BSND",
        "initial_state": False, "output_final_state": False, "cu_seqlens": "",
        "explicit_chunk_indices": False, "safe_gate": True,
        "use_gate_in_kernel": True, "dt_bias": True, "disable_recompute": False,
        "return_intermediate_states": False, "state_v_first": False, "tiling_key": 2,
    }
    if not _contains_exact(specs, a5_fusion):
        raise ValueError("accuracy is missing the fixed A5 key2 fusion candidate")

    hang_regression = {
        "q_dtype": "bf16", "g_dtype": "fp32", "beta_dtype": "fp32",
        "B": 1, "H": 2, "HV": 2, "T": 65, "K": 128, "V": 256,
        "chunk_size": 64, "layout": "BNSD", "initial_state": False,
        "output_final_state": True, "cu_seqlens": "", "explicit_chunk_indices": False,
        "safe_gate": True, "use_gate_in_kernel": True, "dt_bias": True,
        "disable_recompute": True, "return_intermediate_states": True,
        "state_v_first": False, "tiling_key": 1,
    }
    if not _contains_exact(specs, hang_regression):
        raise ValueError("accuracy is missing the fixed key1 hang regression")


def _check_yaml_input_contract(manifests: dict[str, list[dict]]) -> None:
    design = yaml.safe_load((OP_DIR / "chunk_kda_fwd.yaml").read_text(encoding="utf-8"))
    yaml_inputs = design.get("inputs")
    if not isinstance(yaml_inputs, list):
        raise ValueError("chunk_kda_fwd.yaml must declare an inputs list")
    yaml_contract = []
    for item in yaml_inputs:
        if not isinstance(item, dict):
            raise ValueError("chunk_kda_fwd.yaml inputs must be mappings")
        dtype_values = item.get("dtypes", {}).get("values", [])
        if not isinstance(dtype_values, list) or not dtype_values:
            raise ValueError(f"YAML input {item.get('name')}: dtypes.values must be non-empty")
        valid_values = item.get("ranges", {}).get("valid", {}).get("values", [])
        if not isinstance(valid_values, list):
            raise ValueError(f"YAML input {item.get('name')}: ranges.valid.values must be a list")
        yaml_contract.append(
            (
                str(item.get("name")),
                str(item.get("type")),
                bool(item.get("required")),
                {str(dtype) for dtype in dtype_values},
                valid_values,
            )
        )
    yaml_names = [item[0] for item in yaml_contract]
    if len(yaml_names) != len(set(yaml_names)):
        raise ValueError("chunk_kda_fwd.yaml contains duplicate input names")
    for manifest_name, cases in manifests.items():
        for case in cases:
            actual_inputs = case.get("inputs", [])
            if not isinstance(actual_inputs, list) or any(
                not isinstance(item, dict) for item in actual_inputs
            ):
                raise ValueError(f"{manifest_name} case {case.get('id')}: invalid inputs list")
            actual_names = [str(item.get("name")) for item in actual_inputs]
            if len(actual_names) != len(set(actual_names)):
                raise ValueError(
                    f"{manifest_name} case {case.get('id')}: duplicate input names"
                )
            if len(actual_inputs) != len(yaml_contract):
                raise ValueError(
                    f"{manifest_name} case {case.get('id')}: input count mismatch; "
                    f"YAML={len(yaml_contract)}, JSON={len(actual_inputs)}"
                )
            for index, (actual, expected) in enumerate(zip(actual_inputs, yaml_contract)):
                expected_name, expected_type, expected_required, allowed_dtypes, valid_values = expected
                actual_name = str(actual.get("name"))
                actual_type = str(actual.get("type"))
                actual_required = bool(actual.get("required"))
                actual_dtype = str(actual.get("dtype"))
                if (
                    actual_name != expected_name
                    or actual_type != expected_type
                    or actual_required != expected_required
                    or actual_dtype not in allowed_dtypes
                ):
                    raise ValueError(
                        f"{manifest_name} case {case.get('id')} input {index}: "
                        f"expected name/type/required/dtype={expected_name}/{expected_type}/"
                        f"{expected_required}/{sorted(allowed_dtypes)}, got "
                        f"{actual_name}/{actual_type}/{actual_required}/{actual_dtype}"
                    )
                if expected_name in ENUMERATED_YAML_INPUTS and actual.get("range_values") not in valid_values:
                    raise ValueError(
                        f"{manifest_name} case {case.get('id')} input {index}: "
                        f"{expected_name} value {actual.get('range_values')!r} is missing from YAML valid values"
                    )


def main() -> int:
    _check_source_evidence()
    _check_kernel_sync_contract()
    module = _load_generator()
    manifests = {
        "accuracy": _read(OP_DIR / "atk_chunk_kda_fwd.json"),
        "mss": _read(OP_DIR / "atk_chunk_kda_fwd_mss.json"),
        "perf": _read(OP_DIR / "atk_chunk_kda_fwd_perf.json"),
    }
    accuracy_specs = _check_manifest(OP_DIR / "atk_chunk_kda_fwd.json", 200, {1, 2})
    mss_specs = _check_manifest(OP_DIR / "atk_chunk_kda_fwd_mss.json", 4, {1, 2})
    perf_specs = _check_manifest(OP_DIR / "atk_chunk_kda_fwd_perf.json", 2, {1, 2})
    _check_generator(module, manifests)
    _check_yaml_input_contract(manifests)
    _check_accuracy_coverage(accuracy_specs)
    if {(int(item["tiling_key"]), bool(item["initial_state"])) for item in mss_specs} != {
        (key, boundary) for key in (1, 2) for boundary in (False, True)
    }:
        raise ValueError("MSS must contain ordinary and boundary rows for each key")
    if {int(item["tiling_key"]) for item in perf_specs} != {1, 2}:
        raise ValueError("performance must exercise both tiling keys")
    print("chunk_kda_fwd manifests valid: accuracy=200, mss=4, perf=2, keys=[1, 2]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
