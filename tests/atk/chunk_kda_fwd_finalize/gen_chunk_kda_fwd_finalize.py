"""生成仅含公开入参的 ChunkKdaFwdFinalize ATK 用例。"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig, InputCaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None
    InputCaseConfig = None


OP_NAME = "chunk_kda_fwd_finalize"
SEED = 20260914
STANDARD = {"acc": "mixed_tolerance_bm", "perf": "not_key", "mem": 1.1}
INPUT_NAMES = (
    "qg_scaled", "aqk", "v_new", "h", "cu_seqlens", "chunk_indices",
    "output_layout", "state_v_first",
)
LAYOUTS = ("BNSD", "BSND", "NTD", "TND")


def _profiles():
    tails = (1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 129)
    heads = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48)
    profiles = [
        {"name": "tail_%03d" % tokens, "B": 1, "HV": hv, "T": tokens}
        for tokens, hv in zip(tails, heads)
    ]
    profiles.extend(
        {"name": "dense_h%d_t%d" % (hv, tokens), "B": batch, "HV": hv, "T": tokens}
        for batch, hv, tokens in (
            (384, 5, 17), (2, 13, 48),
            (1, 17, 66), (1, 33, 127), (1, 96, 192),
        )
    )
    varlen = (
        (1, (1, 63), False),
        (2, (15, 16, 17), True),
        (4, (31, 32, 33), False),
        (8, (63, 64, 65), True),
        (16, (1, 64, 1), False),
        (24, (17, 47, 64), True),
        (32, (2, 62, 65), False),
        (96, (1, 15, 16, 17, 31, 32, 33, 63, 64, 65), True),
    )
    profiles.extend(
        {
            "name": "varlen_h%d_%s" % (hv, "_".join(map(str, seqs))),
            "B": 1, "HV": hv, "T": sum(seqs), "seqs": seqs,
            "explicit_indices": explicit,
        }
        for hv, seqs, explicit in varlen
    )
    profiles.append({
        "name": "varlen_key2_h5_a1_a3_a4", "B": 1, "HV": 5, "T": 479,
        "seqs": (1,) * 382 + (33, 64), "explicit_indices": True,
    })
    assert len(profiles) == 25
    return profiles


def _input(name, dtype, value, shape=None, kind="attr", required=True):
    return {
        "name": name,
        "type": kind,
        "required": required,
        "dtype": dtype,
        "shape": shape,
        "range_values": value,
        "backward": False,
        "align_32B": None,
        "outlier_values": None,
    }


def _array(name, values):
    if values is None:
        return [_input(name, "string", "null", kind="attrs", required=False)]
    return [
        _input(name, "int", int(value), kind="attrs", required=False)
        for value in values
    ]


def _metadata(profile):
    seqs = profile.get("seqs")
    if seqs is None:
        return None, None
    cu = [0]
    pairs = []
    for sequence, length in enumerate(seqs):
        cu.append(cu[-1] + length)
        for chunk in range((length + 63) // 64):
            pairs.extend((sequence, chunk))
    return cu, pairs if profile.get("explicit_indices", False) else None


def _case(index, profile, layout, state_v_first):
    packed = layout in ("NTD", "TND")
    batch, heads, tokens = (1 if packed else profile["B"], profile["HV"], profile["T"])
    cu, pairs = _metadata(profile)
    chunks = sum((end - begin + 63) // 64 for begin, end in zip(cu, cu[1:])) if cu else (tokens + 63) // 64
    vector_shape = ([batch, heads, tokens, 128] if not packed else [heads, tokens, 128])
    score_shape = vector_shape[:-1] + [64]
    value_shape = [batch, heads, tokens, 128]
    state_shape = [batch, heads, chunks, 128, 128]
    name = "%s_%04d_%s_%s_%s" % (
        OP_NAME, index, profile["name"], layout.lower(),
        "vk" if state_v_first else "kv",
    )
    return {
        "id": index,
        "default_seed": SEED + index,
        "name": name,
        "aclnn_name": "ChunkKdaFwdFinalize",
        "version": "v2.1",
        "api": "pytorch",
        "api_type": "executor_chunk_kda_fwd_finalize",
        "expected_error_msg": "",
        "backward": False,
        "standard": STANDARD,
        "outputs": None,
        "inputs": [
            _input("qg_scaled", "bf16", [-1.0, 1.0], vector_shape, "tensor"),
            _input("aqk", "bf16", [-1.0, 1.0], score_shape, "tensor"),
            _input("v_new", "bf16", [-1.0, 1.0], value_shape, "tensor"),
            _input("h", "bf16", [-1.0, 1.0], state_shape, "tensor"),
            _array("cu_seqlens", cu),
            _array("chunk_indices", pairs),
            _input("output_layout", "string", layout),
            _input("state_v_first", "attr_bool", state_v_first),
        ],
        "save_name": name,
        "is_boundary": tokens in (1, 15, 16, 17, 31, 32, 33, 63, 64, 65),
    }


def accuracy_cases():
    cases = []
    for profile in _profiles():
        for layout in LAYOUTS:
            for state_v_first in (False, True):
                cases.append(_case(len(cases), profile, layout, state_v_first))
    assert len(cases) == 200
    return cases


def perf_cases():
    # 单算 profiling 用真实大 shape，不进入 CPU 精度集。
    settings = (
        (1, 16, 11264, None), (1, 32, 11264, None),
        (1, 96, 8192, None), (1, 96, 16384, (1024,) * 16),
        (2, 16, 11264, None), (4, 96, 128, None),
        (1, 32, 160, None), (6, 6, 1084, None),
        (1, 8, 32768, None), (1, 32, 65536, (1024,) * 64),
    )
    cases = []
    for index, (batch, heads, tokens, seqs) in enumerate(settings):
        profile = {"name": "perf_b%d_h%d_t%d" % (batch, heads, tokens),
                   "B": batch, "HV": heads, "T": tokens}
        if seqs:
            profile["seqs"] = seqs
            profile["explicit_indices"] = index == 9
        case = _case(index, profile, "BNSD", False)
        cases.append(case)
    return cases


def determinism_cases():
    cases = []
    for layout in LAYOUTS:
        for state_v_first in (False, True):
            profile = {"name": "determinism_tail_h13", "B": 1,
                       "HV": 13, "T": 65, "seqs": (1, 64),
                       "explicit_indices": True}
            cases.append(_case(len(cases), profile, layout, state_v_first))
    profile = {"name": "determinism_key2_dense_tail", "B": 384,
               "HV": 2, "T": 17}
    for state_v_first in (False, True):
        cases.append(_case(len(cases), profile, "BNSD", state_v_first))
    profile = {"name": "determinism_key2_packed_tail", "B": 1,
               "HV": 2, "T": 479, "seqs": (1,) * 382 + (33, 64),
               "explicit_indices": True}
    for state_v_first in (False, True):
        cases.append(_case(len(cases), profile, "TND", state_v_first))
    return cases


SUITES = {"accuracy": accuracy_cases, "perf": perf_cases, "mss": determinism_cases}


if GENERATOR_REGISTRY is not None:

    @GENERATOR_REGISTRY.register("generator_chunk_kda_fwd_finalize")
    class ChunkKdaFwdFinalizeGenerator(CaseGenerator):
        def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
            index = int(self.index) - 1
            cases = accuracy_cases()
            if not 0 <= index < len(cases):
                raise IndexError("ATK requested a case outside the 200-case accuracy suite")
            expected = [item[0].name if isinstance(item, list) else item.name
                        for item in case_config.inputs]
            if expected != list(INPUT_NAMES):
                raise ValueError("YAML public input order does not match frozen cases")
            saved = cases[index]
            case_config.inputs = [
                [InputCaseConfig(**part) for part in item] if isinstance(item, list)
                else InputCaseConfig(**item)
                for item in saved["inputs"]
            ]
            for field in ("id", "default_seed", "name", "aclnn_name", "api_type",
                          "backward", "expected_error_msg", "outputs", "save_name", "is_boundary"):
                setattr(case_config, field, deepcopy(saved[field]))
            return case_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if not args.output_dir and not args.summary:
        parser.error("specify --summary or --output-dir")
    for suite, build in SUITES.items():
        cases = build()
        names = [case["name"] for case in cases]
        if len(set(names)) != len(names):
            raise ValueError("duplicate %s case name" % suite)
        if args.summary:
            print("%s=%d" % (suite, len(cases)))
        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            suffix = "" if suite == "accuracy" else "_" + suite
            path = args.output_dir / ("atk_%s%s.json" % (OP_NAME, suffix))
            with path.open("w", encoding="utf-8") as stream:
                json.dump(cases, stream, ensure_ascii=False, indent=2)
                stream.write("\n")


if __name__ == "__main__":
    main()
