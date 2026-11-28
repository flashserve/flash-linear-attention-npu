"""Generate the ATK case matrix for A3 ``ChunkKdaBwdPrepare``.

The matrix is deliberately restricted to the public wrapper/host-tiling
contract: BF16 inputs, C=64, K=V=128, chunk_size=64 and A3.  Preserve the
original 200 shapes and seeds, adding explicit BNSD/BSND
dense and NTD/TND packed layouts.  JSON files can be generated with --kind
full/mss/perf; --check verifies them against this matrix. Accuracy uses the
ATK native L1 double benchmark: CPU FP64 Golden and CPU FP32 Bench.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

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


OP_NAME = "chunk_kda_bwd_prepare"
SUITE_NAME = "chunk_kda_bwd_prepare_a3"
CASE_COUNT = 200
SEED_BASE = 20260831
# Keep the generated records aligned with the YAML's native L1 standard.
ACCURACY_STANDARD = {
    "cv_fused_double_benchmark": {
        "max_re_ratio": 5,
        "avg_re_ratio": 1.5,
        "root_mean_squared_ratio": 1.5,
    }
}
_DTYPE = "bf16"
_DENSE_TOKENS = (1, 7, 31, 63, 64, 65, 95, 127, 128, 129, 191, 192, 255, 256, 257, 320, 511, 512, 1024)
_HEADS = (1, 2, 3, 4, 5, 7, 8, 16)
_VARLEN_SEQUENCES = (
    (1,), (0, 1), (1, 0), (63, 1), (64, 1), (65, 63),
    (128, 1, 63), (64, 65, 127), (127, 128, 129), (1, 255),
    (129, 64, 63), (192, 1), (31, 33, 65), (128, 128),
    (7, 57, 64), (255, 65), (1, 1, 1, 1), (65, 128, 1),
    (0, 64, 1), (64, 0, 65), (1,) * 32,
)
_SCALES = (0.0, 0.125, -0.125, 0.5 / math.sqrt(128.0), 1.0 / math.sqrt(128.0), 1.0)


def _base_spec(case_id: int, *, mode: str, **kwargs: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {
        "op": OP_NAME,
        "case_id": case_id,
        "seed": SEED_BASE + case_id,
        "soc": "ascend910_93",
        "route": "aclnn",
        "mode": mode,
        "layout": ("BNSD", "BSND")[case_id % 2] if mode == "dense" else ("NTD", "TND")[case_id % 2],
        "dtype": _DTYPE,
        "C": 64,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "scale": float(_SCALES[case_id % len(_SCALES)]),
        "state_v_first": bool((case_id // 3) % 2),
    }
    spec.update(kwargs)
    return spec


def _build_profiles() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for index in range(98):
        profiles.append(_base_spec(
            index,
            mode="dense",
            name=f"dense_bf16_{index:03d}",
            B=2 if index % 7 == 0 else 1,
            H=_HEADS[index % len(_HEADS)],
            T=_DENSE_TOKENS[index % len(_DENSE_TOKENS)],
            tags=["dense", "optional_none", "tail"],
        ))
    for offset in range(100):
        case_id = 98 + offset
        seqlens = _VARLEN_SEQUENCES[offset % len(_VARLEN_SEQUENCES)]
        profiles.append(_base_spec(
            case_id,
            mode="varlen",
            name=f"varlen_bf16_{offset:03d}",
            B=1,
            H=_HEADS[offset % len(_HEADS)],
            T=sum(seqlens),
            seqlens=list(seqlens),
            tags=["varlen", "optional_metadata", "uneven_sequence"],
        ))

    # The two production shapes used for KernelA performance work.  Keep them
    # in the accuracy matrix as requested, rather than testing only a reduced
    # surrogate shape.
    profiles.append(_base_spec(
        198, mode="dense", name="model_b1_h96_t8192_state_v_last",
        B=1, H=96, T=8192, state_v_first=False,
        scale=1.0 / math.sqrt(128.0), tags=["model", "8k", "h96", "dense"],
    ))
    profiles.append(_base_spec(
        199, mode="dense", name="model_b1_h96_t16384_state_v_first",
        B=1, H=96, T=16384, state_v_first=True,
        scale=1.0 / math.sqrt(128.0), tags=["model", "16k", "h96", "dense"],
    ))
    _validate_profiles(profiles)
    return profiles


def _validate_profiles(profiles: list[dict[str, Any]]) -> None:
    if len(profiles) != CASE_COUNT:
        raise ValueError(f"expected {CASE_COUNT} cases, got {len(profiles)}")
    if [int(spec["case_id"]) for spec in profiles] != list(range(CASE_COUNT)):
        raise ValueError("case_id must be stable and contiguous from 0 through 199")
    for spec in profiles:
        if spec["dtype"] != "bf16" or spec["C"] != 64 or spec["K"] != 128 or spec["V"] != 128:
            raise ValueError(f"unsupported dtype or dimensions: {spec}")
        if spec["chunk_size"] != 64 or int(spec["H"]) <= 0 or int(spec["T"]) <= 0:
            raise ValueError(f"unsupported chunk/head/token setting: {spec}")
        if not math.isfinite(float(spec["scale"])):
            raise ValueError(f"scale must be finite: {spec}")
        if spec["mode"] == "dense":
            if spec["layout"] not in ("BNSD", "BSND"):
                raise ValueError(f"dense layout must be BNSD or BSND: {spec}")
            if int(spec["B"]) not in (1, 2) or "seqlens" in spec:
                raise ValueError(f"invalid dense case: {spec}")
        elif spec["mode"] == "varlen":
            if spec["layout"] not in ("NTD", "TND"):
                raise ValueError(f"varlen layout must be NTD or TND: {spec}")
            seqlens = spec.get("seqlens")
            if int(spec["B"]) != 1 or not seqlens or min(seqlens) < 0 or sum(seqlens) != int(spec["T"]):
                raise ValueError(f"invalid varlen case: {spec}")
            if sum((length + 63) // 64 for length in seqlens) <= 0:
                raise ValueError(f"varlen case has no chunks: {spec}")
        else:
            raise ValueError(f"unsupported mode: {spec}")
    if Counter(spec["mode"] for spec in profiles) != Counter({"dense": 100, "varlen": 100}):
        raise ValueError("expected 100 dense and 100 varlen cases")
    if {False, True} != {bool(spec["state_v_first"]) for spec in profiles}:
        raise ValueError("both state_v_first branches must be covered")
    if Counter(spec["layout"] for spec in profiles) != Counter({"BNSD": 50, "BSND": 50, "NTD": 50, "TND": 50}):
        raise ValueError("expected 50 cases for each input layout")
    if {(spec["layout"], spec["state_v_first"]) for spec in profiles} != {
        (layout, state) for layout in ("BNSD", "BSND", "NTD", "TND") for state in (False, True)
    }:
        raise ValueError("both state_v_first branches must be covered for every layout")
    for tokens in (63, 64, 65):
        if not any(spec["mode"] == "dense" and spec["T"] == tokens for spec in profiles):
            raise ValueError(f"missing dense boundary T={tokens}")
    if [(profiles[-2]["B"], profiles[-2]["H"], profiles[-2]["T"]),
        (profiles[-1]["B"], profiles[-1]["H"], profiles[-1]["T"])] != [(1, 96, 8192), (1, 96, 16384)]:
        raise ValueError("the two production model shapes are missing")


PROFILES = _build_profiles()
# 25 small dense + 23 varlen examples, plus both production shapes. Include
# non-trivial head/tail/state and
# both production shapes.  The preserved original ids are intentional: the
# executor uses case_id as the ATK-26.7.8-stable transport for the spec.
MSS_CASE_IDS = tuple(list(range(0, 25)) + list(range(100, 123)) + [198, 199])
PERF_CASE_IDS = (198, 199)


def _tensor(name: str, dtype: str) -> dict[str, Any]:
    return {"name": name, "type": "tensor", "required": True, "dtype": dtype,
            "shape": [1], "range_values": [0, 0], "backward": False}


def _attr(name: str, dtype: str, value: Any) -> dict[str, Any]:
    return {"name": name, "type": "attr", "required": True, "dtype": dtype,
            "shape": None, "range_values": value, "backward": False}


def _record(spec: dict[str, Any]) -> dict[str, Any]:
    inputs = [
        _tensor("low_precision_marker", "bf16"),
        _tensor("fp32_marker", "fp32"),
        _attr("case_spec", "non_param", json.dumps(spec, ensure_ascii=False, separators=(",", ":"))),
    ]
    for name, dtype in (("dtype", "string"), ("mode", "string"), ("layout", "string"), ("B", "int"), ("H", "int"),
                        ("T", "int"), ("C", "int"), ("K", "int"), ("V", "int"),
                        ("chunk_size", "int"), ("state_v_first", "bool"), ("scale", "float"),
                        ("case_id", "int"), ("seed", "int"), ("route", "string"), ("soc", "string")):
        inputs.append(_attr(name, dtype, spec[name]))
    return {
        "id": spec["case_id"], "default_seed": spec["seed"], "name": spec["name"],
        "aclnn_name": "ChunkKdaBwdPrepare", "version": "v2.1", "api": "pytorch",
        "api_type": f"executor_{SUITE_NAME}", "expected_error_msg": None, "backward": False,
        "standard": {"acc": copy.deepcopy(ACCURACY_STANDARD), "perf": "not_key"},
        "outputs": None, "inputs": inputs,
    }


def build_records(case_ids: Iterable[int] = range(CASE_COUNT)) -> list[dict[str, Any]]:
    return [_record(PROFILES[index]) for index in case_ids]


def expand_seeds(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("seed count must be positive")
    expanded = []
    for iteration in range(count):
        for original in records:
            row = copy.deepcopy(original)
            row["id"] = len(expanded)
            row["name"] += f"_seed{iteration}"
            row["default_seed"] += 10000 * iteration
            for item in row["inputs"]:
                item["backward"] = False
                if item["name"] == "seed":
                    item["range_values"] += 10000 * iteration
                elif item["name"] == "case_spec":
                    spec = json.loads(item["range_values"])
                    spec["seed"] += 10000 * iteration
                    item["range_values"] = json.dumps(spec, separators=(",", ":"))
            expanded.append(row)
    return expanded


def validate_records(records: list[dict[str, Any]], expected_ids: Iterable[int]) -> None:
    ids = list(expected_ids)
    if [record.get("id") for record in records] != ids:
        raise ValueError("ATK record ids drifted from the reviewed matrix")
    for record, expected_id in zip(records, ids):
        if record.get("api_type") != f"executor_{SUITE_NAME}":
            raise ValueError(f"api_type mismatch for case {expected_id}")
        if record.get("standard", {}).get("acc") != ACCURACY_STANDARD:
            raise ValueError(
                f"case {expected_id} does not use the native L1 double benchmark; "
                "regenerate the ATK template with the updated YAML"
            )
        item = next((entry for entry in record["inputs"] if entry["name"] == "case_spec"), None)
        if item is None or json.loads(item["range_values"]) != PROFILES[expected_id]:
            raise ValueError(f"case_spec drift for case {expected_id}")


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_chunk_kda_bwd_prepare_a3")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)

        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = max(int(self.index) - 1, 0)
            spec = PROFILES[index]
            case_config.id = index
            case_config.default_seed = spec["seed"]
            case_config.name = spec["name"]
            for item in case_config.inputs:
                cfg = item[0] if isinstance(item, list) else item
                if cfg.name == "case_spec":
                    cfg.range_values = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
                elif cfg.name in spec:
                    cfg.range_values = spec[cfg.name]
            return case_config


def main() -> int:
    parser = argparse.ArgumentParser(description="write or validate KernelA prepare ATK JSON")
    parser.add_argument("--kind", choices=("full", "mss", "perf"), default="full")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--atk-template-json", type=Path)
    args = parser.parse_args()
    ids = {"full": tuple(range(CASE_COUNT)), "mss": MSS_CASE_IDS, "perf": PERF_CASE_IDS}[args.kind]
    output = args.output or Path(__file__).with_name(
        f"atk_{SUITE_NAME}{'' if args.kind == 'full' else '_' + args.kind}.json")
    records = (json.loads(args.atk_template_json.read_text(encoding="utf-8"))
               if args.atk_template_json else build_records())
    validate_records(records, range(CASE_COUNT))
    records = [records[index] for index in ids]
    if args.kind == "full":
        records = expand_seeds(records, args.seed_count)
    if args.check:
        actual = json.loads(output.read_text(encoding="utf-8"))
        if actual != records:
            raise ValueError("ATK records drifted from the reviewed matrix")
        print(f"{output}: {len(records)} reviewed {args.kind} cases validated")
        return 0
    output.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(records)} {args.kind} cases to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
