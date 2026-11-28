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
TEMPLATE_GATE_DTYPES = ("bf16", "fp32")
TEMPLATE_GATE_MODES = ("g", "gk")
TEMPLATE_EXP2_VALUES = (False, True)
TEMPLATE_STATE_DTYPES = ("bf16", "fp32")
TEMPLATE_STATE_LAYOUTS = (False, True)


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


def _template_signature(spec: dict) -> tuple:
    """Return the symbolic top-level kernel template arguments for a case."""
    gate_dtype = str(spec["gate_dtype"])
    mode = str(spec["mode"])
    state_dtype = str(spec.get("state_dtype", "none"))
    if gate_dtype not in TEMPLATE_GATE_DTYPES:
        raise ValueError(f"unsupported template gate dtype: {gate_dtype}")
    if mode not in TEMPLATE_GATE_MODES:
        raise ValueError(f"unsupported template gate mode: {mode}")
    if state_dtype not in {"none", *TEMPLATE_STATE_DTYPES}:
        raise ValueError(f"unsupported template state dtype: {state_dtype}")
    return (
        gate_dtype,
        int(spec["V"]),
        mode == "gk",
        bool(spec["use_exp2"]),
        state_dtype != "bf16",
        bool(spec["state_v_first"]),
    )


EXPECTED_TEMPLATE_SIGNATURES = frozenset(
    (
        gate_dtype,
        128,
        mode == "gk",
        use_exp2,
        state_dtype == "fp32",
        state_v_first,
    )
    for gate_dtype in TEMPLATE_GATE_DTYPES
    for mode in TEMPLATE_GATE_MODES
    for use_exp2 in TEMPLATE_EXP2_VALUES
    for state_dtype in TEMPLATE_STATE_DTYPES
    for state_v_first in TEMPLATE_STATE_LAYOUTS
)


def _template_matrix_specs(prefix: str, tags: str, **updates) -> list[dict]:
    specs = []
    for gate_dtype in TEMPLATE_GATE_DTYPES:
        for mode in TEMPLATE_GATE_MODES:
            for use_exp2 in TEMPLATE_EXP2_VALUES:
                for state_dtype in TEMPLATE_STATE_DTYPES:
                    for state_v_first in TEMPLATE_STATE_LAYOUTS:
                        values = {"output_final_state": True, **updates}
                        values.update(
                            mode=mode,
                            gate_dtype=gate_dtype,
                            state_dtype=state_dtype,
                            use_exp2=use_exp2,
                            state_v_first=state_v_first,
                        )
                        specs.append(
                            _positive(
                                (
                                    f"{prefix}_{gate_dtype}_{mode}_"
                                    f"{'exp2' if use_exp2 else 'exp'}_{state_dtype}_"
                                    f"{'vk' if state_v_first else 'kv'}"
                                ),
                                tags=tags,
                                **values,
                            )
                        )
    return specs


def _assert_complete_template_matrix(name: str, specs: list[dict]) -> None:
    signatures = [_template_signature(spec) for spec in specs]
    actual = frozenset(signatures)
    if len(specs) != len(EXPECTED_TEMPLATE_SIGNATURES):
        raise AssertionError(
            f"{name}: expected {len(EXPECTED_TEMPLATE_SIGNATURES)} cases, got {len(specs)}"
        )
    if len(actual) != len(signatures):
        raise AssertionError(f"{name}: duplicate top-level template signatures")
    if actual != EXPECTED_TEMPLATE_SIGNATURES:
        missing = sorted(EXPECTED_TEMPLATE_SIGNATURES - actual)
        unexpected = sorted(actual - EXPECTED_TEMPLATE_SIGNATURES)
        raise AssertionError(
            f"{name}: template signature mismatch; missing={missing}, unexpected={unexpected}"
        )


def _assert_unique_specs(name: str, specs: list[dict]) -> None:
    case_keys = [str(spec["case_key"]) for spec in specs]
    if len(case_keys) != len(set(case_keys)):
        raise AssertionError(f"{name}: duplicate case_key")

    ignored = {"case_id", "case_key", "seed", "tags"}
    fingerprints = [
        json.dumps(
            {key: value for key, value in spec.items() if key not in ignored},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for spec in specs
    ]
    if len(fingerprints) != len(set(fingerprints)):
        raise AssertionError(f"{name}: structurally duplicate cases")


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
        "g_fp32_initial_terminal_tail2_no_final",
        tags="accuracy,boundary,a5_resident,a5_lookahead,tail2",
        T=258,
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
        "gk_tail2_fp32_state",
        tags="accuracy,boundary,gk,tail2",
        mode="gk",
        T=130,
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


ACCURACY_GENERALIZATION_SPECS = [
    _positive(
        "dense_no_initial_final_fp32_g",
        tags="accuracy,generalization,no_initial",
        T=65,
        output_final_state=True,
    ),
    _positive(
        "dense_no_initial_final_fp32_gk_vk",
        tags="accuracy,generalization,no_initial,gk,tail1",
        mode="gk",
        T=129,
        gate_dtype="fp32",
        output_final_state=True,
        use_exp2=True,
        state_v_first=True,
    ),
    _positive(
        "dense_no_initial_no_final_full_chunk",
        tags="accuracy,generalization,no_initial,no_final,full_chunk",
        T=64,
    ),
    _positive(
        "dense_bf16_initial_no_final_tail1",
        tags="accuracy,generalization,no_final,tail1",
        T=65,
        state_dtype="bf16",
    ),
    _positive(
        "dense_fp32_initial_no_final_gk_full_chunks",
        tags="accuracy,generalization,no_final,gk,full_chunk",
        mode="gk",
        T=128,
        gate_dtype="fp32",
        state_dtype="fp32",
        use_exp2=True,
        state_v_first=True,
    ),
    _positive(
        "dense_tail2_fp32_state",
        tags="accuracy,generalization,tail2",
        T=130,
        gate_dtype="fp32",
        state_dtype="fp32",
        output_final_state=True,
    ),
    _positive(
        "dense_tail32_gk_bf16_state",
        tags="accuracy,generalization,gk,tail32",
        mode="gk",
        T=160,
        state_dtype="bf16",
        output_final_state=True,
        state_v_first=True,
    ),
    _positive(
        "dense_tail63_gk_fp32_state",
        tags="accuracy,generalization,gk,tail63",
        mode="gk",
        T=191,
        gate_dtype="fp32",
        state_dtype="fp32",
        output_final_state=True,
        use_exp2=True,
    ),
    _positive(
        "dense_four_full_chunks",
        tags="accuracy,generalization,full_chunk",
        T=256,
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
    ),
    _positive(
        "dense_credit_reuse_33_chunks_gk",
        tags="accuracy,generalization,gk,long_credit_reuse,tail1",
        mode="gk",
        T=2113,
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
        use_exp2=True,
    ),
    _positive(
        "dense_gva_ratio2",
        tags="accuracy,generalization,head_partition,gva",
        HK=2,
        HV=4,
        T=65,
        state_dtype="bf16",
        output_final_state=True,
    ),
    _positive(
        "dense_gva_ratio3",
        tags="accuracy,generalization,head_partition,gva",
        HK=3,
        HV=9,
        T=65,
        gate_dtype="fp32",
        state_dtype="fp32",
        output_final_state=True,
    ),
    _positive(
        "dense_gva_ratio4",
        tags="accuracy,generalization,head_partition,gva",
        HK=4,
        HV=16,
        T=65,
        state_dtype="bf16",
        output_final_state=True,
        use_exp2=True,
    ),
    _positive(
        "dense_gva_ratio8",
        tags="accuracy,generalization,head_partition,gva",
        HK=2,
        HV=16,
        T=65,
        gate_dtype="fp32",
        state_dtype="fp32",
        output_final_state=True,
        state_v_first=True,
    ),
    _positive(
        "dense_total20_heads",
        tags="accuracy,generalization,head_partition",
        HK=5,
        HV=20,
        T=1,
        output_final_state=True,
    ),
    _positive(
        "dense_total24_heads",
        tags="accuracy,generalization,head_partition",
        HK=6,
        HV=24,
        T=1,
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
    ),
    _positive(
        "dense_total28_heads",
        tags="accuracy,generalization,head_partition",
        HK=7,
        HV=28,
        T=1,
        state_dtype="fp32",
        output_final_state=True,
        use_exp2=True,
    ),
    _positive(
        "dense_total32_heads",
        tags="accuracy,generalization,head_partition",
        HK=8,
        HV=32,
        T=1,
        gate_dtype="fp32",
        state_dtype="fp32",
        output_final_state=True,
        state_v_first=True,
    ),
    _positive(
        "dense_total48_heads_cross_sequence",
        tags="accuracy,generalization,head_partition,cross_sequence",
        B=2,
        HK=6,
        HV=24,
        T=1,
        state_dtype="bf16",
        output_final_state=True,
    ),
    _positive(
        "dense_total96_heads_cross_sequence",
        tags="accuracy,generalization,head_partition,cross_sequence",
        B=3,
        HK=8,
        HV=32,
        T=1,
        gate_dtype="fp32",
        state_dtype="fp32",
        output_final_state=True,
        use_exp2=True,
    ),
    _positive(
        "varlen_mixed_boundaries_explicit_indices",
        tags="accuracy,generalization,varlen,explicit_indices",
        HK=2,
        HV=4,
        T=454,
        gate_dtype="fp32",
        state_dtype="fp32",
        output_final_state=True,
        state_v_first=True,
        seqlens="1,2,3,63,64,65,127,129",
        explicit_chunk_indices=True,
    ),
    _positive(
        "varlen_32_single_token_sequences",
        tags="accuracy,generalization,varlen,cross_sequence,auto_indices",
        HK=1,
        HV=2,
        T=32,
        state_dtype="bf16",
        output_final_state=True,
        seqlens=",".join(["1"] * 32),
    ),
    _positive(
        "gk_non_contiguous_u_view",
        tags="accuracy,generalization,gk,non_contiguous_input",
        mode="gk",
        T=65,
        gate_dtype="fp32",
        state_dtype="bf16",
        output_final_state=True,
        non_contiguous_u=True,
    ),
]
POSITIVE_SPECS.extend(ACCURACY_GENERALIZATION_SPECS)


ACCURACY_TEMPLATE_MATRIX_GROUPS = [
    (
        "dense_tail1",
        _template_matrix_specs(
            "matrix_dense_tail1",
            "accuracy,template_key_matrix,dense,boundary,tail1",
            T=129,
        ),
    ),
    (
        "dense_full_chunks",
        _template_matrix_specs(
            "matrix_dense_full_chunks",
            "accuracy,template_key_matrix,dense,full_chunk",
            T=128,
        ),
    ),
    (
        "dense_tail63",
        _template_matrix_specs(
            "matrix_dense_tail63",
            "accuracy,template_key_matrix,dense,boundary,tail63",
            T=127,
        ),
    ),
    (
        "dense_terminal_no_final",
        _template_matrix_specs(
            "matrix_dense_terminal_no_final",
            "accuracy,template_key_matrix,dense,boundary,no_final,tail1",
            T=257,
            output_final_state=False,
        ),
    ),
    (
        "varlen_mixed_boundaries",
        _template_matrix_specs(
            "matrix_varlen_mixed_boundaries",
            "accuracy,template_key_matrix,varlen,boundary,explicit_indices",
            T=259,
            seqlens="1,64,65,129",
            explicit_chunk_indices=True,
        ),
    ),
]

for _, matrix_specs in ACCURACY_TEMPLATE_MATRIX_GROUPS:
    POSITIVE_SPECS.extend(matrix_specs)


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


PERF_TEMPLATE_SPECS = _template_matrix_specs(
    "a5_b1_hk32_hv32_t11264",
    "performance,model_target,template_key_matrix",
    HK=32,
    HV=32,
    T=11264,
    seed=208,
)
PERF_SPECS.extend(PERF_TEMPLATE_SPECS)


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


MSS_TEMPLATE_SPECS = _template_matrix_specs(
    "mss_matrix",
    "determinism,sanitizer,template_key_matrix,boundary,tail1",
    T=129,
)

# Keep one case per template signature while distributing the synchronization,
# partitioning and boundary shapes that are most useful to sanitizer runs.
MSS_TEMPLATE_OVERRIDES = {
    ("bf16", 128, False, False, False, False): {
        "case_key": "mss_dense_total64_cross_sequence",
        "B": 2,
        "HK": 16,
        "HV": 32,
        "T": 1,
        "extra_tags": "dense,cross_sequence,head_partition",
    },
    ("bf16", 128, False, False, False, True): {
        "case_key": "mss_varlen_total64_cross_sequence",
        "HK": 16,
        "HV": 32,
        "T": 2,
        "seqlens": "1,1",
        "explicit_chunk_indices": True,
        "extra_tags": "varlen,cross_sequence,head_partition,explicit_indices",
    },
    ("bf16", 128, False, False, True, True): {
        "case_key": "mss_terminal_vnew_only_no_cube",
        "T": 1,
        "state_dtype": "none",
        "output_final_state": False,
        "extra_tags": "no_cube",
    },
    ("bf16", 128, False, True, False, False): {
        "case_key": "mss_long_credit_reuse",
        "T": 1025,
        "extra_tags": "long_credit_reuse",
    },
    ("bf16", 128, False, True, True, True): {
        "case_key": "mss_fp32_initial_terminal_no_final",
        "T": 257,
        "output_final_state": False,
        "extra_tags": "a5_resident,a5_lookahead,tail1",
    },
    ("fp32", 128, False, False, False, False): {
        "case_key": "mss_active2_g_ratio3",
        "HK": 11,
        "HV": 33,
        "T": 1,
        "extra_tags": "head_partition",
    },
    ("fp32", 128, False, False, False, True): {
        "case_key": "mss_active3_cross_sequence",
        "B": 5,
        "HK": 1,
        "HV": 13,
        "T": 1,
        "extra_tags": "head_partition,cross_sequence",
    },
    ("fp32", 128, False, False, True, False): {
        "case_key": "mss_fp32_resident_lookahead_tail63",
        "T": 319,
        "remove_tags": "tail1",
        "extra_tags": "a5_resident,a5_lookahead,tail63",
    },
    ("fp32", 128, False, False, True, True): {
        "case_key": "mss_varlen_explicit_indices",
        "HK": 1,
        "HV": 3,
        "T": 259,
        "seqlens": "1,64,65,129",
        "explicit_chunk_indices": True,
        "extra_tags": "varlen,explicit_indices",
    },
    ("fp32", 128, False, True, True, True): {
        "case_key": "mss_active4_g_cross_round_key_reuse",
        "HK": 16,
        "HV": 160,
        "T": 1,
        "extra_tags": "head_partition,key_reuse",
    },
    ("fp32", 128, True, False, False, False): {
        "case_key": "mss_active4_gk_distinct_keys",
        "HK": 160,
        "HV": 160,
        "T": 1,
        "extra_tags": "head_partition,gk",
    },
    ("fp32", 128, True, True, True, True): {
        "case_key": "mss_gk_tail1_fp32_state",
        "extra_tags": "gk,tail1",
    },
}

for spec in MSS_TEMPLATE_SPECS:
    expected_signature = _template_signature(spec)
    override = MSS_TEMPLATE_OVERRIDES.get(expected_signature)
    if override is None:
        continue
    metadata_keys = {"extra_tags", "remove_tags"}
    spec.update({key: value for key, value in override.items() if key not in metadata_keys})
    removed_tags = set(override.get("remove_tags", "").split(",")) - {""}
    base_tags = [tag for tag in spec["tags"].split(",") if tag not in removed_tags]
    extra_tags = [tag for tag in override.get("extra_tags", "").split(",") if tag]
    spec["tags"] = ",".join(base_tags + extra_tags)
    if _template_signature(spec) != expected_signature:
        raise AssertionError(
            f"mss override changed template signature for {spec['case_key']}"
        )


def build_accuracy_specs() -> list[dict]:
    for matrix_name, matrix_specs in ACCURACY_TEMPLATE_MATRIX_GROUPS:
        _assert_complete_template_matrix(f"accuracy/{matrix_name}", matrix_specs)
    specs = deepcopy(POSITIVE_SPECS)
    if len(specs) != 200:
        raise AssertionError(f"accuracy: expected 200 cases, got {len(specs)}")
    _assert_unique_specs("accuracy", specs)
    for case_id, spec in enumerate(specs):
        spec["case_id"] = case_id
        spec.setdefault("seed", SEED_BASE + case_id)
    return specs


def build_perf_specs() -> list[dict]:
    _assert_complete_template_matrix("performance", PERF_TEMPLATE_SPECS)
    specs = deepcopy(PERF_SPECS)
    if len(specs) != 44:
        raise AssertionError(f"performance: expected 44 cases, got {len(specs)}")
    for case_id, spec in enumerate(specs):
        spec["case_id"] = case_id
        spec.setdefault("seed", SEED_BASE + 1000 + case_id)
    return specs


def build_mss_specs() -> list[dict]:
    _assert_complete_template_matrix("determinism", MSS_TEMPLATE_SPECS)
    _assert_complete_template_matrix("mss", MSS_TEMPLATE_SPECS)
    specs = deepcopy(MSS_TEMPLATE_SPECS)
    _assert_unique_specs("determinism/mss", specs)
    for case_id, spec in enumerate(specs):
        spec["case_id"] = case_id
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
    perf = build_perf_specs()
    mss = build_mss_specs()
    _write(args.output_dir / f"atk_{OP_NAME}.json", accuracy)
    _write(args.output_dir / f"atk_{OP_NAME}_perf.json", perf)
    _write(args.output_dir / f"atk_{OP_NAME}_mss.json", mss)
    if args.summary:
        print(
            f"accuracy={len(accuracy)} positive={len(POSITIVE_SPECS)} "
            f"perf={len(perf)} determinism={len(mss)} mss={len(mss)}"
        )


if __name__ == "__main__":
    main()
