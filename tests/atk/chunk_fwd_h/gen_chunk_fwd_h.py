"""Materialize the frozen ATK matrices for chunk_fwd_h."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

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


OP_NAME = "chunk_fwd_h"
SEED_BASE = 20260831
STANDARD = {"acc": "mixed_tolerance_bm", "perf": "not_key", "mem": 1.1}


def _positive(case_key: str, **updates) -> dict:
    spec = {
        "case_key": case_key,
        "tags": "accuracy,regression",
        "route": "ascendc",
        "soc": "all",
        "dtype": "bf16",
        "B": 1,
        "HK": 1,
        "HV": 1,
        "T": 65,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "mode": "g",
        "gate_dtype": "bf16",
        "state_dtype": "none",
        "output_final_state": False,
        "save_new_value": True,
        "use_exp2": False,
        "state_v_first": False,
        "seqlens": "",
        "explicit_chunk_indices": False,
        "non_contiguous_u": False,
        "input_scale": 0.05,
        "gate_step_scale": 0.02,
        "state_scale": 0.05,
    }
    spec.update(updates)
    return spec


def _negative(mutation: str, expected_message: str) -> dict:
    return _positive(
        f"negative_{mutation}",
        tags="negative,boundary",
        mutation=mutation,
        expected_return_code=0,
        expected_exception="RuntimeError",
        expected_message=expected_message,
    )


POSITIVE_SPECS = [
    _positive(
        "dense_active1_terminal_vnew_only",
        tags="accuracy,boundary,no_cube",
        T=1,
        state_v_first=True,
    ),
    _positive(
        "dense_active2_g_ratio3_final",
        tags="accuracy,boundary,head_partition",
        HK=11,
        HV=33,
        T=1,
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
    ),
    _positive(
        "dense_g_ratio7_final",
        tags="accuracy,boundary,head_partition,key_reuse",
        HK=8,
        HV=56,
        T=1,
        state_dtype="fp32",
        output_final_state=True,
        use_exp2=True,
        state_v_first=True,
    ),
    _positive(
        "dense_active3_cross_sequence",
        tags="accuracy,boundary,head_partition,cross_sequence",
        B=5,
        HK=1,
        HV=13,
        T=1,
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
        state_v_first=True,
    ),
    _positive(
        "dense_active4_g_cross_round_key_reuse",
        tags="accuracy,boundary,head_partition,key_reuse",
        HK=16,
        HV=160,
        T=1,
        gate_dtype="fp32",
        state_dtype="fp32",
        output_final_state=True,
        use_exp2=True,
        state_v_first=True,
    ),
    _positive(
        "dense_active4_gk_distinct_keys",
        tags="accuracy,boundary,head_partition,gk",
        HK=160,
        HV=160,
        T=1,
        mode="gk",
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
    ),
    _positive(
        "g_full_chunks_bf16_state",
        HK=2,
        HV=4,
        T=128,
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
    ),
    _positive(
        "g_fp32_resident_lookahead_tail63",
        tags="accuracy,boundary,a5_resident,a5_lookahead,tail63",
        T=319,
        gate_dtype="fp32",
        output_final_state=True,
    ),
    _positive(
        "g_fp32_initial_terminal_tail1_no_final",
        tags="accuracy,boundary,a5_resident,a5_lookahead,tail1",
        T=257,
        state_dtype="fp32",
        use_exp2=True,
        state_v_first=True,
    ),
    _positive(
        "gk_full_chunks_bf16_state",
        mode="gk",
        HK=2,
        HV=2,
        T=128,
        state_dtype="bf16",
        output_final_state=True,
    ),
    _positive(
        "gk_tail1_fp32_state",
        tags="accuracy,boundary,gk,tail1",
        mode="gk",
        T=129,
        gate_dtype="fp32",
        state_dtype="fp32",
        output_final_state=True,
        use_exp2=True,
        state_v_first=True,
    ),
    _positive(
        "g_non_contiguous_u_view",
        tags="accuracy,boundary,non_contiguous_input",
        T=65,
        output_final_state=True,
        non_contiguous_u=True,
    ),
    _positive(
        "g_long_credit_reuse",
        tags="accuracy,regression,long_credit_reuse",
        T=1025,
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
    ),
    _positive(
        "g_varlen_explicit_indices",
        tags="accuracy,boundary,varlen,explicit_indices",
        HK=1,
        HV=3,
        T=259,
        gate_dtype="fp32",
        state_dtype="fp32",
        output_final_state=True,
        state_v_first=True,
        seqlens="1,64,65,129",
        explicit_chunk_indices=True,
    ),
    _positive(
        "gk_varlen_auto_indices",
        tags="accuracy,boundary,varlen,auto_indices,gk",
        mode="gk",
        HK=2,
        HV=2,
        T=192,
        state_dtype="bf16",
        output_final_state=True,
        seqlens="63,64,65",
    ),
    _positive(
        "g_dense_total64_head_partition_cross_sequence",
        tags="accuracy,boundary,dense,cross_sequence",
        B=2,
        HK=16,
        HV=32,
        T=1,
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
    ),
    _positive(
        "g_varlen_total64_head_partition_cross_sequence",
        tags="accuracy,boundary,varlen,cross_sequence",
        HK=16,
        HV=32,
        T=2,
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
        seqlens="1,1",
        explicit_chunk_indices=True,
    ),
]


for mode in ("g", "gk"):
    for state_dtype in ("bf16", "fp32"):
        for use_exp2 in (False, True):
            POSITIVE_SPECS.append(
                _positive(
                    f"matrix_{mode}_{state_dtype}_{'exp2' if use_exp2 else 'exp'}",
                    tags="accuracy,gate_state_exponent_matrix",
                    mode=mode,
                    T=129,
                    gate_dtype="fp32",
                    state_dtype=state_dtype,
                    output_final_state=True,
                    use_exp2=use_exp2,
                    state_v_first=True,
                )
            )


NEGATIVE_SPECS = [
    _negative("both_gate_inputs", "exactly one of g and gk must be provided"),
    _negative("missing_gate_inputs", "exactly one of g and gk must be provided"),
    _negative("unsupported_chunk_size", "chunk_size must be 64"),
    _negative("save_new_value_false", "save_new_value must be True"),
    _negative("invalid_input_rank", "k, w and u must be rank-4 BNSD tensors"),
    _negative("non_positive_dimension", "B, HK, HV and T must all be positive"),
    _negative("unsupported_input_dtype", "k, w and u must all use bfloat16"),
    _negative("mismatched_input_dtype", "k, w and u must all use bfloat16"),
    _negative("unsupported_kv_dimension", "K and V must both be 128"),
    _negative("mismatched_w_u_shape", "w/u must be [B, HV, T, K/V]"),
    _negative("invalid_g_head_ratio", "g-only mode requires HV >= HK and HV % HK == 0"),
    _negative("invalid_g_shape", "g must be [B, HV, T]"),
    _negative("invalid_gk_head_count", "gk-only mode requires prepared kg to have HV heads"),
    _negative("invalid_gk_shape", "gk must be [B, HV, T, K]"),
    _negative("unsupported_gate_dtype", "g/gk must use bfloat16 or float32"),
    _negative("unsupported_state_dtype", "initial_state must use bfloat16 or float32"),
    _negative("invalid_state_shape", "initial_state shape does not match state_v_first"),
    _negative("varlen_batch_not_one", "variable-length BNSD input requires B=1"),
    _negative(
        "invalid_cu_size",
        "cu_seqlens must be strictly increasing, start at 0 and end at T",
    ),
    _negative(
        "invalid_cu_start",
        "cu_seqlens must be strictly increasing, start at 0 and end at T",
    ),
    _negative(
        "invalid_cu_end",
        "cu_seqlens must be strictly increasing, start at 0 and end at T",
    ),
    _negative(
        "non_increasing_cu",
        "cu_seqlens must be strictly increasing, start at 0 and end at T",
    ),
    _negative(
        "chunk_indices_without_cu",
        "chunk_indices must use canonical sequence-major order",
    ),
    _negative(
        "invalid_chunk_indices_length",
        "chunk_indices must use canonical sequence-major order",
    ),
    _negative(
        "invalid_chunk_indices_order",
        "chunk_indices must use canonical sequence-major order",
    ),
]


VARLEN_65_CU = (
    "0,2365,2409,2536,3008,3681,6545,8416,11615,12599,12844,12982,13209,"
    "15561,16291,16669,17755,21365,21416,25267,26084,26364,26833,27900,"
    "28033,29011,29291,29576,30471,30733,30906,31202,32264,32393,34233,"
    "34274,34589,40060,41075,41272,42123,42257,43054,44349,46396,46875,"
    "48850,49338,49457,50062,50509,54275,55763,56093,56473,56941,56964,"
    "57218,57274,58855,58926,60953,63436,63524,65536"
)


PERF_SPECS = [
    _positive("a5_g_h4_t512", tags="performance", HK=4, HV=4, T=512, seed=201),
    _positive(
        "a5_g_gva_1_to_6_t2048",
        tags="performance",
        HK=1,
        HV=6,
        T=2048,
        state_dtype="fp32",
        output_final_state=True,
        seed=202,
    ),
    _positive(
        "a5_gk_h4_t2048",
        tags="performance",
        mode="gk",
        HK=4,
        HV=4,
        T=2048,
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
        use_exp2=True,
        state_v_first=True,
        seed=203,
    ),
    _positive(
        "a5_b2_hk16_hv32_t11264",
        tags="performance,model_target",
        B=2,
        HK=16,
        HV=32,
        T=11264,
        gate_dtype="fp32",
        output_final_state=True,
        use_exp2=True,
        state_v_first=True,
        seed=204,
    ),
    _positive(
        "a5_b1_hk16_hv32_t11264",
        tags="performance,model_target",
        HK=16,
        HV=32,
        T=11264,
        gate_dtype="fp32",
        output_final_state=True,
        use_exp2=True,
        state_v_first=True,
        seed=205,
    ),
    _positive(
        "a5_b1_hk16_hv32_t11264_bf16_initial",
        tags="performance,model_target,initial_state",
        HK=16,
        HV=32,
        T=11264,
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
        use_exp2=True,
        state_v_first=True,
        seed=206,
    ),
    _positive(
        "a5_b1_hk16_hv32_t11264_fp32_initial",
        tags="performance,model_target,initial_state",
        HK=16,
        HV=32,
        T=11264,
        gate_dtype="fp32",
        state_dtype="fp32",
        output_final_state=True,
        use_exp2=True,
        state_v_first=True,
        seed=207,
    ),
]


for state_dtype in ("bf16", "fp32"):
    for mode in ("g", "gk"):
        for use_exp2 in (False, True):
            PERF_SPECS.append(
                _positive(
                    f"a5_b1_hk32_hv32_t11264_{mode}_{'exp2' if use_exp2 else 'exp'}_{state_dtype}_initial",
                    tags="performance,model_target,gate_state_exponent_matrix",
                    mode=mode,
                    HK=32,
                    HV=32,
                    T=11264,
                    gate_dtype="fp32",
                    state_dtype=state_dtype,
                    output_final_state=True,
                    use_exp2=use_exp2,
                    state_v_first=True,
                    seed=208,
                )
            )


PERF_SPECS.extend(
    [
        _positive(
            "a5_varlen_h32_t65536_s64",
            tags="performance,model_target,varlen",
            HK=32,
            HV=32,
            T=65536,
            gate_dtype="fp32",
            output_final_state=True,
            use_exp2=True,
            state_v_first=True,
            seqlens=",".join(
                str(end - begin)
                for begin, end in zip(
                    [int(value) for value in VARLEN_65_CU.split(",")][:-1],
                    [int(value) for value in VARLEN_65_CU.split(",")][1:],
                )
            ),
            explicit_chunk_indices=True,
            seed=202,
        ),
        _positive(
            "a5_b4_hk96_hv96_t128",
            tags="performance,model_target",
            B=4,
            HK=96,
            HV=96,
            T=128,
            gate_dtype="fp32",
            output_final_state=True,
            use_exp2=True,
            state_v_first=True,
            seed=209,
        ),
        _positive(
            "a5_b1_hk32_hv32_t160",
            tags="performance,model_target,tail",
            HK=32,
            HV=32,
            T=160,
            gate_dtype="fp32",
            output_final_state=True,
            use_exp2=True,
            state_v_first=True,
            seed=210,
        ),
        _positive(
            "a5_b6_hk6_hv6_t1084",
            tags="performance,model_target,tail",
            B=6,
            HK=6,
            HV=6,
            T=1084,
            gate_dtype="fp32",
            output_final_state=True,
            use_exp2=True,
            state_v_first=True,
            seed=211,
        ),
        _positive(
            "a5_b1_hk12_hv12_t1084",
            tags="performance,model_target,tail",
            HK=12,
            HV=12,
            T=1084,
            gate_dtype="fp32",
            output_final_state=True,
            use_exp2=True,
            state_v_first=True,
            seed=212,
        ),
    ]
)


MSS_CASE_KEYS = (
    "dense_active1_terminal_vnew_only",
    "dense_active2_g_ratio3_final",
    "dense_g_ratio7_final",
    "dense_active3_cross_sequence",
    "dense_active4_g_cross_round_key_reuse",
    "dense_active4_gk_distinct_keys",
    "g_fp32_resident_lookahead_tail63",
    "g_fp32_initial_terminal_tail1_no_final",
    "gk_tail1_fp32_state",
    "g_varlen_explicit_indices",
    "g_dense_total64_head_partition_cross_sequence",
    "g_varlen_total64_head_partition_cross_sequence",
)


def build_accuracy_specs() -> list[dict]:
    specs = deepcopy(POSITIVE_SPECS)
    for case_id, spec in enumerate(specs):
        spec["case_id"] = case_id
        spec.setdefault("seed", SEED_BASE + case_id)
    return specs


def build_negative_specs() -> list[dict]:
    specs = deepcopy(NEGATIVE_SPECS)
    for case_id, spec in enumerate(specs):
        spec["case_id"] = case_id
        spec.setdefault("seed", SEED_BASE + len(POSITIVE_SPECS) + case_id)
    return specs


def build_perf_specs() -> list[dict]:
    specs = deepcopy(PERF_SPECS)
    for case_id, spec in enumerate(specs):
        spec["case_id"] = case_id
        spec.setdefault("seed", SEED_BASE + 1000 + case_id)
    return specs


def build_mss_specs() -> list[dict]:
    by_key = {spec["case_key"]: spec for spec in POSITIVE_SPECS}
    specs = [deepcopy(by_key[key]) for key in MSS_CASE_KEYS]
    for case_id, spec in enumerate(specs):
        spec["case_id"] = case_id
        spec["tags"] = f"{spec['tags']},determinism,sanitizer"
        spec.setdefault("seed", SEED_BASE + 2000 + case_id)
    return specs


def _input(
    name: str,
    dtype: str,
    value,
    *,
    input_type: str = "attr",
    shape=None,
) -> dict:
    return {
        "name": name,
        "type": input_type,
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": value,
        "backward": False,
    }


def _case_payload(case_id: int, spec: dict) -> dict:
    metadata = deepcopy(spec)
    metadata["case_id"] = case_id
    inputs = [
        _input(
            "low_precision_marker",
            "bf16",
            [0, 0],
            input_type="tensor",
            shape=[1],
        ),
        _input(
            "fp32_marker",
            "fp32",
            [0, 0],
            input_type="tensor",
            shape=[1],
        ),
        _input(
            "case_spec",
            "non_param",
            json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        ),
        _input("case_key", "string", metadata["case_key"]),
        _input("soc", "string", metadata["soc"]),
        _input("route", "string", metadata["route"]),
        _input("dtype", "string", metadata["dtype"]),
        _input("B", "int", metadata["B"]),
        _input("HK", "int", metadata["HK"]),
        _input("HV", "int", metadata["HV"]),
        _input("T", "int", metadata["T"]),
        _input("K", "int", metadata["K"]),
        _input("V", "int", metadata["V"]),
        _input("chunk_size", "int", metadata["chunk_size"]),
        _input("mode", "string", metadata["mode"]),
        _input("gate_dtype", "string", metadata["gate_dtype"]),
        _input("state_dtype", "string", metadata["state_dtype"]),
        _input("output_final_state", "bool", metadata["output_final_state"]),
        _input("save_new_value", "bool", metadata["save_new_value"]),
        _input("use_exp2", "bool", metadata["use_exp2"]),
        _input("state_v_first", "bool", metadata["state_v_first"]),
        _input("seqlens", "string", metadata["seqlens"]),
        _input(
            "explicit_chunk_indices",
            "bool",
            metadata["explicit_chunk_indices"],
        ),
        _input("non_contiguous_u", "bool", metadata["non_contiguous_u"]),
        _input("negative_case", "bool", "negative" in metadata["tags"].split(",")),
        _input("tags", "string", metadata["tags"]),
        _input("seed", "int", metadata["seed"]),
    ]
    return {
        "id": case_id,
        "default_seed": metadata["seed"],
        "name": f"{OP_NAME}_{case_id:04d}_{metadata['case_key']}",
        "aclnn_name": None,
        "version": "v2.1",
        "api": "pytorch",
        "api_type": "executor_chunk_fwd_h",
        "expected_error_msg": None,
        "backward": False,
        "standard": STANDARD,
        "outputs": None,
        "inputs": inputs,
        "save_name": OP_NAME,
    }


def _payloads(specs: list[dict]) -> list[dict]:
    return [_case_payload(case_id, spec) for case_id, spec in enumerate(specs)]


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_chunk_fwd_h")
    class ChunkFwdHGenerator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)
            if CaseConfig is None:
                raise RuntimeError("ATK is required to build CaseConfig objects")
            self.cases = [
                CaseConfig(**payload) for payload in _payloads(build_accuracy_specs())
            ]
            self.length = len(self.cases)
            self.index = 0

        def generate(self) -> CaseConfig:
            case = self.cases[self.index]
            self.index += 1
            return case


def _write(path: Path, specs: list[dict]) -> None:
    path.write_text(
        json.dumps(_payloads(specs), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accuracy = build_accuracy_specs()
    negative = build_negative_specs()
    perf = build_perf_specs()
    mss = build_mss_specs()
    _write(args.output_dir / f"atk_{OP_NAME}.json", accuracy)
    _write(args.output_dir / f"atk_{OP_NAME}_negative.json", negative)
    _write(args.output_dir / f"atk_{OP_NAME}_perf.json", perf)
    _write(args.output_dir / f"atk_{OP_NAME}_mss.json", mss)
    if args.summary:
        print(
            f"accuracy={len(accuracy)} positive={len(POSITIVE_SPECS)} "
            f"negative={len(negative)} perf={len(perf)} mss={len(mss)}"
        )


if __name__ == "__main__":
    main()
