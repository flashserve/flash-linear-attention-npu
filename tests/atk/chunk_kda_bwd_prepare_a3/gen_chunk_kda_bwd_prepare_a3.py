"""Generate 200 small GVA cases (one seed), 20 DC/MSS cases and two design performance cases.

Validated on A2 910B3; the a3 suffix is inherited asset naming.
CPU FP64 Golden and FP32 Bench use the unchanged native double benchmark.
M2 performance uses the explicitly authorized 16x1024 development probe.
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


def _build_profiles():
    profiles = []
    for i in range(200):
        ratio = 2 + i % 15
        hk = (1, 3)[(i // 15) % 2]
        layout = ("BNSD", "BSND", "NTD", "TND")[(i // 30) % 4]
        dense = bool((i // 8) % 2)
        tokens = (1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 193, 257)[(i // 4) % 12]
        spec = _base_spec(i, mode="dense" if dense else "varlen",
            name=f"gva200_{i:03d}_r{ratio}", B=2 if dense and layout in ("BNSD", "BSND") and i % 11 == 0 else 1,
            H=hk * ratio, T=tokens, HK=hk, gva_ratio=ratio, layout=layout,
            state_v_first=bool((i // 4) % 2), seed=20261124+i,
            explicit_chunk_indices=bool(i % 2), tags=["gva", "small"])
        if not dense:
            spec["seqlens"] = [0, tokens // 2, tokens - tokens // 2]
        profiles.append(spec)
    return profiles


PROFILES = _build_profiles()
# 25 small dense + 23 varlen examples, plus both production shapes. Include
# non-trivial head/tail/state and
# both production shapes.  The preserved original ids are intentional: the
# executor uses case_id as the ATK-26.7.8-stable transport for the spec.
MSS_CASE_IDS = (136, 140, 128, 132, 40, 44, 32, 36, 88, 188, 80, 84, 90, 92, 112, 116, 199, 198, 197, 196)
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
    parser.add_argument("--seed-count", type=int, choices=(1,), default=1)
    parser.add_argument("--atk-template-json", type=Path)
    args = parser.parse_args()
    ids = {"full": tuple(range(CASE_COUNT)), "mss": MSS_CASE_IDS, "perf": PERF_CASE_IDS}[args.kind]
    output = args.output or Path(__file__).with_name(
        f"atk_{SUITE_NAME}{'' if args.kind == 'full' else '_' + args.kind}.json")
    records = (json.loads(args.atk_template_json.read_text(encoding="utf-8"))
               if args.atk_template_json else build_records())
    validate_records(records, range(CASE_COUNT))
    records = [records[index] for index in ids]
    if args.kind == "perf":
        records = [_record(spec) for spec in json.loads(Path(__file__).with_name("performance_profiles.json").read_text())]
    if args.kind == "full" and args.seed_count > 1:
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
