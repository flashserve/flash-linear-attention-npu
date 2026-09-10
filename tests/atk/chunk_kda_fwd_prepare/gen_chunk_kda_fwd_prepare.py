"""生成 chunk_kda_fwd_prepare 的冻结 ATK 用例矩阵。"""

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


OP_NAME = "chunk_kda_fwd_prepare"
SEED_BASE = 20260910
STANDARD = {"acc": "mixed_tolerance_bm", "perf": "not_key", "mem": 1.1}
GATE_DTYPES = ("bf16", "fp32")
BETA_DTYPES = ("bf16", "fp32")
NORM_VALUES = (False, True)
BETA_MODES = ("raw", "sigmoid", "two_sigmoid")
GATE_MODES = ("precomputed", "softplus", "safe")
EXP_VALUES = (False, True)


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
        "layout": "BNSD",
        "chunk_size": 64,
        "gate_dtype": "bf16",
        "beta_dtype": "bf16",
        "scale": 1.0,
        "epsilon": 1e-6,
        "use_qk_l2norm_in_kernel": False,
        "use_gate_in_kernel": False,
        "use_beta_sigmoid_in_kernel": False,
        "allow_neg_eigval": False,
        "safe_gate": False,
        "lower_bound": -5.0,
        "use_exp2": False,
        "dt_bias": False,
        "cu_seqlens": "",
        "explicit_chunk_indices": False,
        "data_scale": 0.08,
        "gate_scale": 1.0,
        "beta_scale": 1.0,
    }
    spec.update(updates)
    return spec


def _mode_values(beta_mode: str, gate_mode: str) -> dict:
    if beta_mode == "raw":
        use_beta_sigmoid = False
        allow_neg = False
    elif beta_mode == "sigmoid":
        use_beta_sigmoid = True
        allow_neg = False
    elif beta_mode == "two_sigmoid":
        use_beta_sigmoid = True
        allow_neg = True
    else:
        raise ValueError(f"unknown beta mode: {beta_mode}")

    if gate_mode == "precomputed":
        use_gate = False
        safe_gate = False
    elif gate_mode == "softplus":
        use_gate = True
        safe_gate = False
    elif gate_mode == "safe":
        use_gate = True
        safe_gate = True
    else:
        raise ValueError(f"unknown gate mode: {gate_mode}")
    return {
        "use_beta_sigmoid_in_kernel": use_beta_sigmoid,
        "allow_neg_eigval": allow_neg,
        "use_gate_in_kernel": use_gate,
        "safe_gate": safe_gate,
    }


def _template_signature(spec: dict) -> tuple:
    if not _as_template_spec(spec):
        raise ValueError(f"invalid template spec: {spec['case_key']}")
    beta_mode = (
        "raw"
        if not spec["use_beta_sigmoid_in_kernel"]
        else ("two_sigmoid" if spec["allow_neg_eigval"] else "sigmoid")
    )
    gate_mode = (
        "precomputed"
        if not spec["use_gate_in_kernel"]
        else ("safe" if spec["safe_gate"] else "softplus")
    )
    return (
        spec["gate_dtype"],
        spec["beta_dtype"],
        bool(spec["use_qk_l2norm_in_kernel"]),
        beta_mode,
        gate_mode,
        bool(spec["use_exp2"]),
    )


def _as_template_spec(spec: dict) -> bool:
    return (
        spec["gate_dtype"] in GATE_DTYPES
        and spec["beta_dtype"] in BETA_DTYPES
        and (not spec["allow_neg_eigval"] or spec["use_beta_sigmoid_in_kernel"])
        and (
            spec["use_gate_in_kernel"]
            or (not spec["safe_gate"] and not spec["dt_bias"])
        )
    )


EXPECTED_TEMPLATE_SIGNATURES = frozenset(
    (
        gate_dtype,
        beta_dtype,
        norm,
        beta_mode,
        gate_mode,
        use_exp2,
    )
    for gate_dtype in GATE_DTYPES
    for beta_dtype in BETA_DTYPES
    for norm in NORM_VALUES
    for beta_mode in BETA_MODES
    for gate_mode in GATE_MODES
    for use_exp2 in EXP_VALUES
)


def _template_specs(prefix: str, tags: str, **updates) -> list[dict]:
    specs = []
    for gate_dtype in GATE_DTYPES:
        for beta_dtype in BETA_DTYPES:
            for norm in NORM_VALUES:
                for beta_mode in BETA_MODES:
                    for gate_mode in GATE_MODES:
                        for use_exp2 in EXP_VALUES:
                            values = {
                                "gate_dtype": gate_dtype,
                                "beta_dtype": beta_dtype,
                                "use_qk_l2norm_in_kernel": norm,
                                "use_exp2": use_exp2,
                                **_mode_values(beta_mode, gate_mode),
                                **updates,
                            }
                            values["dt_bias"] = (
                                gate_mode != "precomputed"
                                and len(specs) % 2 == 0
                            )
                            key = (
                                f"{prefix}_{gate_dtype}_{beta_dtype}_"
                                f"{'l2' if norm else 'identity'}_{beta_mode}_"
                                f"{gate_mode}_{'exp2' if use_exp2 else 'exp'}"
                            )
                            specs.append(_positive(key, tags=tags, **values))
    return specs


def _assert_template_matrix(name: str, specs: list[dict]) -> None:
    signatures = [_template_signature(spec) for spec in specs]
    if len(specs) != 144:
        raise AssertionError(f"{name}: expected 144 cases, got {len(specs)}")
    if len(signatures) != len(set(signatures)):
        raise AssertionError(f"{name}: duplicate template signatures")
    if frozenset(signatures) != EXPECTED_TEMPLATE_SIGNATURES:
        raise AssertionError(f"{name}: incomplete template matrix")


def _assert_unique(name: str, specs: list[dict]) -> None:
    keys = [spec["case_key"] for spec in specs]
    if len(keys) != len(set(keys)):
        raise AssertionError(f"{name}: duplicate case_key")


FUNCTIONAL_SPECS = [
    _positive("dense_bnsd_min", tags="accuracy,boundary,min", T=1),
    _positive("dense_bsnd_tail15", layout="BSND", T=15),
    _positive("dense_ntd_tail16", layout="NTD", T=16),
    _positive("dense_tnd_tail17", layout="TND", T=17),
    _positive("dense_bnsd_tail31", T=31),
    _positive("dense_bnsd_tail32", T=32),
    _positive("dense_bnsd_tail33", T=33),
    _positive("dense_bnsd_tail47", T=47),
    _positive("dense_bnsd_tail48", T=48),
    _positive("dense_bnsd_tail49", T=49),
    _positive("dense_bnsd_full_chunk", T=64),
    _positive("dense_two_chunks_tail1", T=65),
    _positive(
        "dense_bsnd_batch2",
        layout="BSND",
        B=2,
        HK=3,
        HV=15,
        T=65,
        gate_dtype="fp32",
        beta_dtype="fp32",
    ),
    _positive(
        "dense_gva_ratio5_cross_wave",
        HK=3,
        HV=15,
        T=33,
        use_qk_l2norm_in_kernel=True,
    ),
    _positive(
        "dense_gva_ratio8",
        HK=2,
        HV=16,
        T=17,
        use_exp2=True,
    ),
    _positive(
        "dense_head_partition_33",
        HK=11,
        HV=33,
        T=1,
        gate_dtype="fp32",
        beta_dtype="fp32",
    ),
    _positive(
        "dense_hv160_no_artificial_limit",
        tags="accuracy,boundary,head_partition,gva",
        HK=32,
        HV=160,
        T=1,
        gate_dtype="fp32",
    ),
    _positive(
        "varlen_tnd_auto_indices",
        layout="TND",
        T=146,
        cu_seqlens="0,1,17,81,146",
        HK=1,
        HV=5,
    ),
    _positive(
        "varlen_ntd_explicit_indices",
        layout="NTD",
        T=146,
        cu_seqlens="0,1,17,81,146",
        explicit_chunk_indices=True,
        HK=2,
        HV=8,
        use_exp2=True,
    ),
    _positive(
        "varlen_bnsd_rank4",
        layout="BNSD",
        T=129,
        cu_seqlens="0,64,129",
        explicit_chunk_indices=True,
    ),
    _positive(
        "varlen_more_than_1024_sequences",
        tags="accuracy,boundary,varlen",
        layout="TND",
        T=1,
        cu_seqlens=",".join(["0"] * 1025 + ["1"]),
        explicit_chunk_indices=True,
    ),
    _positive(
        "gate_softplus_without_bias",
        use_gate_in_kernel=True,
        gate_dtype="fp32",
        T=65,
    ),
    _positive(
        "gate_softplus_with_bias",
        use_gate_in_kernel=True,
        dt_bias=True,
        gate_dtype="bf16",
        T=65,
    ),
    _positive(
        "gate_safe_exp2",
        use_gate_in_kernel=True,
        safe_gate=True,
        dt_bias=True,
        lower_bound=-3.0,
        use_exp2=True,
        T=65,
    ),
    _positive(
        "beta_sigmoid",
        use_beta_sigmoid_in_kernel=True,
        beta_dtype="fp32",
    ),
    _positive(
        "beta_two_sigmoid",
        use_beta_sigmoid_in_kernel=True,
        allow_neg_eigval=True,
        beta_dtype="bf16",
    ),
    _positive(
        "l2norm_small_epsilon",
        use_qk_l2norm_in_kernel=True,
        epsilon=1e-12,
    ),
]

ACCURACY_TEMPLATE_SPECS = _template_specs(
    "matrix",
    "accuracy,template_key_matrix,full_chunk",
    T=64,
)
MSS_TEMPLATE_SPECS = _template_specs(
    "mss_matrix",
    "determinism,sanitizer,template_key_matrix,slot_reuse,tail1",
    T=65,
)

PERF_SPECS = [
    _positive(
        "model_b2_hk16_hv32_t11264",
        tags="performance,model_target",
        B=2,
        HK=16,
        HV=32,
        T=11264,
        gate_dtype="fp32",
        beta_dtype="bf16",
        scale=0.08838834764831845,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        dt_bias=True,
        use_exp2=True,
    ),
    _positive(
        "model_b1_hk16_hv32_t11264",
        tags="performance,model_target",
        HK=16,
        HV=32,
        T=11264,
        gate_dtype="fp32",
        beta_dtype="bf16",
        scale=0.08838834764831845,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
        dt_bias=True,
        use_exp2=True,
    ),
    _positive(
        "model_b1_h32_t11264",
        tags="performance,model_target",
        HK=32,
        HV=32,
        T=11264,
        gate_dtype="fp32",
        beta_dtype="fp32",
        use_exp2=True,
    ),
    _positive(
        "dense_b4_h96_t128",
        tags="performance,head_partition",
        B=4,
        HK=96,
        HV=96,
        T=128,
        gate_dtype="fp32",
        use_exp2=True,
    ),
    _positive(
        "dense_gva_ratio8_t2048",
        tags="performance,gva,cross_wave",
        HK=4,
        HV=32,
        T=2048,
        gate_dtype="fp32",
        beta_dtype="fp32",
    ),
    _positive(
        "dense_bsnd_tail_t1084",
        tags="performance,layout,tail",
        layout="BSND",
        HK=12,
        HV=12,
        T=1084,
        gate_dtype="fp32",
        use_gate_in_kernel=True,
        safe_gate=True,
        dt_bias=True,
        use_exp2=True,
    ),
    _positive(
        "varlen_tnd_t65536",
        tags="performance,varlen",
        layout="TND",
        HK=1,
        HV=4,
        T=65536,
        cu_seqlens="0,8192,16384,24576,32768,40960,49152,57344,65536",
        explicit_chunk_indices=True,
        gate_dtype="fp32",
        use_exp2=True,
    ),
]


def _number_specs(specs: list[dict], seed_offset: int) -> list[dict]:
    numbered = deepcopy(specs)
    _assert_unique("numbered", numbered)
    for case_id, spec in enumerate(numbered):
        spec["case_id"] = case_id
        spec.setdefault("seed", SEED_BASE + seed_offset + case_id)
    return numbered


def build_accuracy_specs() -> list[dict]:
    _assert_template_matrix("accuracy", ACCURACY_TEMPLATE_SPECS)
    return _number_specs(FUNCTIONAL_SPECS + ACCURACY_TEMPLATE_SPECS, 0)


def build_perf_specs() -> list[dict]:
    return _number_specs(PERF_SPECS, 1000)


def build_mss_specs() -> list[dict]:
    _assert_template_matrix("mss", MSS_TEMPLATE_SPECS)
    return _number_specs(MSS_TEMPLATE_SPECS, 2000)


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


ATTR_DTYPES = {
    "case_key": "string",
    "soc": "string",
    "route": "string",
    "dtype": "string",
    "B": "int",
    "HK": "int",
    "HV": "int",
    "T": "int",
    "K": "int",
    "V": "int",
    "layout": "string",
    "chunk_size": "int",
    "gate_dtype": "string",
    "beta_dtype": "string",
    "scale": "float",
    "epsilon": "float",
    "use_qk_l2norm_in_kernel": "bool",
    "use_gate_in_kernel": "bool",
    "use_beta_sigmoid_in_kernel": "bool",
    "allow_neg_eigval": "bool",
    "safe_gate": "bool",
    "lower_bound": "float",
    "use_exp2": "bool",
    "dt_bias": "bool",
    "cu_seqlens": "string",
    "explicit_chunk_indices": "bool",
    "tags": "string",
    "seed": "int",
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
    ]
    inputs.extend(
        _input(name, dtype, metadata[name])
        for name, dtype in ATTR_DTYPES.items()
    )
    return {
        "id": case_id,
        "default_seed": metadata["seed"],
        "name": f"{OP_NAME}_{case_id:04d}_{metadata['case_key']}",
        "aclnn_name": None,
        "version": "v2.1",
        "api": "pytorch",
        "api_type": "executor_chunk_kda_fwd_prepare",
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
    @GENERATOR_REGISTRY.register("generator_chunk_kda_fwd_prepare")
    class ChunkKdaFwdPrepareGenerator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)
            if CaseConfig is None:
                raise RuntimeError("ATK is required to build CaseConfig objects")
            self.cases = [
                CaseConfig(**payload)
                for payload in _payloads(build_accuracy_specs())
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
            f"accuracy={len(accuracy)} perf={len(perf)} "
            f"determinism={len(mss)} mss={len(mss)}"
        )


if __name__ == "__main__":
    main()
