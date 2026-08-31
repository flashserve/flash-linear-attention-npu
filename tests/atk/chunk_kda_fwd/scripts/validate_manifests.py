#!/usr/bin/env python3
"""Validate the checked-in chunk_kda_fwd ATK manifests without an NPU."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve()
OP_DIR = HERE.parents[1]
REPO_ROOT = HERE.parents[4]
GEN_PATH = OP_DIR / "gen_chunk_kda_fwd.py"
SOCS = {"ascend910b", "ascend910_93", "ascend950"}


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


def main() -> int:
    _check_source_evidence()
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
    if {int(item["tiling_key"]) for item in accuracy_specs} != {1, 2}:
        raise ValueError("accuracy must exercise both tiling keys")
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
