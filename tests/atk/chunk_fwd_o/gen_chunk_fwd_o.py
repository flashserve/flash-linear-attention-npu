"""ATK generator and reviewed 200-case matrix for ``chunk_fwd_o``.

This is deliberately the single source of truth for the checked-in ATK JSON
and for the executor's case-id fallback on ATK 26.7.8.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
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


OP_NAME = "chunk_fwd_o"
CASE_COUNT = 200
_DTYPES = ("bf16", "fp16")
_HEAD_PAIRS = ((1, 1), (1, 2), (2, 2), (2, 4), (4, 4))
_FIXED_TOKENS = (1, 7, 31, 63, 64, 65, 95, 127, 128, 129, 191, 192, 255, 256, 257, 320)
_VARLEN_SEQUENCES = (
    (1,), (63, 1), (64, 1), (65, 63), (128, 1, 63),
    (64, 65, 127), (127, 128, 129), (1, 255), (129, 64, 63),
    (192, 1), (31, 33, 65), (128, 128), (7, 57, 64),
    (255, 65), (1, 1, 1, 1), (65, 128, 1),
)


def _base_spec(case_id: int, dtype: str, *, mode: str, **kwargs: Any) -> dict[str, Any]:
    spec = {
        "op": OP_NAME, "case_id": case_id, "seed": 20260817 + case_id,
        "route": "ascendc", "soc": "ascend910b", "dtype": dtype,
        "g_dtype": "fp32" if case_id % 3 == 0 else dtype, "mode": mode, "K": 128,
    }
    spec.update(kwargs)
    spec["scale"] = (0.5 if case_id % 10 == 0 else 1.0) / math.sqrt(spec["K"])
    return spec


def _build_profiles() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for index in range(100):
        dtype = _DTYPES[index % len(_DTYPES)]
        hk, hv = _HEAD_PAIRS[index % len(_HEAD_PAIRS)]
        profiles.append(_base_spec(
            index, dtype, mode="fixed", name=f"fixed_{dtype}_{index:03d}",
            B=2 if index % 5 == 0 else 1, HK=hk, HV=hv,
            T=_FIXED_TOKENS[index % len(_FIXED_TOKENS)],
            V=256 if index % 5 in (1, 4) else 128,
            chunk_size=128 if index % 2 else 64,
        ))
    for offset in range(100):
        case_id = 100 + offset
        dtype = _DTYPES[offset % len(_DTYPES)]
        hk, hv = _HEAD_PAIRS[offset % len(_HEAD_PAIRS)]
        seqlens = _VARLEN_SEQUENCES[offset % len(_VARLEN_SEQUENCES)]
        profiles.append(_base_spec(
            case_id, dtype, mode="varlen", name=f"varlen_{dtype}_{offset:03d}",
            B=1, HK=hk, HV=hv, T=sum(seqlens),
            V=256 if offset % 5 in (2, 4) else 128,
            chunk_size=128 if offset % 2 else 64, seqlens=list(seqlens),
        ))
    _validate_profiles(profiles)
    return profiles


def _validate_profiles(profiles: list[dict[str, Any]]) -> None:
    if len(profiles) != CASE_COUNT:
        raise ValueError(f"expected {CASE_COUNT} cases, got {len(profiles)}")
    if [int(spec["case_id"]) for spec in profiles] != list(range(CASE_COUNT)):
        raise ValueError("case_id must be contiguous from 0 through 199")
    for spec in profiles:
        if spec["dtype"] not in _DTYPES or spec["g_dtype"] not in (*_DTYPES, "fp32"):
            raise ValueError(f"unsupported dtype combination: {spec}")
        if spec["K"] != 128 or spec["V"] not in (128, 256) or spec["chunk_size"] not in (64, 128):
            raise ValueError(f"unsupported ChunkFwdO shape: {spec}")
        if spec["HV"] % spec["HK"]:
            raise ValueError(f"invalid GVA head mapping: {spec}")
        if spec["mode"] == "fixed":
            if "seqlens" in spec or spec["B"] not in (1, 2):
                raise ValueError(f"invalid fixed-length case: {spec}")
        elif spec["mode"] == "varlen":
            seqlens = spec.get("seqlens")
            if spec["B"] != 1 or not seqlens or sum(seqlens) != spec["T"] or min(seqlens) <= 0:
                raise ValueError(f"invalid varlen case: {spec}")
        else:
            raise ValueError(f"unsupported mode: {spec}")
    if Counter(spec["mode"] for spec in profiles) != Counter({"fixed": 100, "varlen": 100}):
        raise ValueError("expected 100 fixed and 100 varlen cases")
    if Counter(spec["dtype"] for spec in profiles) != Counter({"bf16": 100, "fp16": 100}):
        raise ValueError("expected 100 bf16 and 100 fp16 cases")


PROFILES = _build_profiles()


def _spec(index: int) -> dict[str, Any]:
    if not 0 <= index < CASE_COUNT:
        raise IndexError(f"{OP_NAME} only defines {CASE_COUNT} reviewed ATK cases, got index {index}")
    return PROFILES[index]


def _dtype(dtype: str) -> str:
    return {"bf16": "bf16", "fp16": "fp16", "fp32": "fp32"}.get(dtype, "bf16")


def _tensor(name: str, dtype: str) -> dict[str, Any]:
    return {"name": name, "type": "tensor", "required": True, "dtype": dtype,
            "shape": [1], "range_values": [0, 0], "backward": False}


def _attr(name: str, dtype: str, value: Any) -> dict[str, Any]:
    return {"name": name, "type": "attr", "required": True, "dtype": dtype,
            "shape": None, "range_values": value, "backward": False}


def _record(spec: dict[str, Any]) -> dict[str, Any]:
    inputs = [_tensor("low_precision_marker", spec["dtype"]), _tensor("fp32_marker", "fp32"),
              _attr("case_spec", "non_param", json.dumps(spec, ensure_ascii=False, separators=(",", ":")))]
    for name, dtype in (("dtype", "string"), ("B", "int"), ("HK", "int"), ("HV", "int"),
                        ("T", "int"), ("K", "int"), ("V", "int"), ("chunk_size", "int"),
                        ("case_id", "int"), ("seed", "int"), ("route", "string"), ("soc", "string")):
        inputs.append(_attr(name, dtype, spec[name]))
    return {
        "id": spec["case_id"], "default_seed": spec["seed"], "name": spec["name"],
        "aclnn_name": "", "version": "v2.1", "api": "pytorch", "api_type": f"executor_{OP_NAME}",
        "expected_error_msg": None, "backward": False,
        "standard": {"acc": {"cv_fused_double_benchmark": {"max_re_ratio": 5, "avg_re_ratio": 1.5,
                     "root_mean_squared_ratio": 1.5}}, "perf": "not_key"}, "outputs": None, "inputs": inputs,
    }


def build_records() -> list[dict[str, Any]]:
    return [_record(spec) for spec in PROFILES]


def validate_records(records: list[dict[str, Any]]) -> None:
    if len(records) != CASE_COUNT:
        raise ValueError(f"expected {CASE_COUNT} ATK records, got {len(records)}")
    specs = []
    for expected_id, record in enumerate(records):
        if record.get("id") != expected_id or record.get("api_type") != f"executor_{OP_NAME}":
            raise ValueError(f"invalid record header at index {expected_id}: {record.get('id')!r}")
        case_input = next((item for item in record["inputs"] if item["name"] == "case_spec"), None)
        if case_input is None:
            raise ValueError(f"case {expected_id} has no case_spec")
        specs.append(json.loads(case_input["range_values"]))
    _validate_profiles(specs)
    if specs != PROFILES:
        raise ValueError("ATK JSON case specs drifted from generator profiles")


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
            case_config.name = f"{OP_NAME}_{index:04d}_{spec['name']}"
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "low_precision_marker":
                    cfg.dtype = _dtype(spec["dtype"])
                elif cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config


def main() -> int:
    parser = argparse.ArgumentParser(description="write or validate ChunkFwdO's reviewed ATK JSON")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name(f"atk_{OP_NAME}.json"))
    parser.add_argument("--check", action="store_true", help="validate an existing JSON")
    args = parser.parse_args()
    if args.check:
        with args.output.open(encoding="utf-8") as handle:
            records = json.load(handle)
        validate_records(records)
        print(f"{args.output}: {len(records)}/{CASE_COUNT} ChunkFwdO ATK cases validated")
        return 0
    records = build_records()
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    validate_records(records)
    print(f"wrote {len(records)} ChunkFwdO ATK cases to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
