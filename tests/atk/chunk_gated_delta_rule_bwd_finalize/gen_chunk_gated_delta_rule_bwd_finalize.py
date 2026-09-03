"""Generate frozen ATK matrices for ChunkGatedDeltaRuleBwdFinalize."""

from __future__ import annotations

import argparse
import json
import random
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


OP_NAME = "chunk_gated_delta_rule_bwd_finalize"
SEED_BASE = 20260901
STANDARD = {"acc": "mixed_tolerance_bm", "perf": "not_key", "mem": 1.1}
QK_RANGE = [-0.2, 0.2]
V_RANGE = [-0.5, 0.5]
VECTOR_GRAD_RANGE = [-0.03, 0.03]
SCALAR_RANGE = [-0.09, 0.09]
VECTOR_GRAD_INPUTS = frozenset({"v_new", "do", "du", "h", "dh", "a"})
SCALAR_DTYPES = ("bf16", "fp32")
BOOL_VALUES = (False, True)
EXPECTED_TEMPLATE_SIGNATURES = frozenset(
    (scalar_dtype, use_qk_l2norm, use_beta_sigmoid)
    for scalar_dtype in SCALAR_DTYPES
    for use_qk_l2norm in BOOL_VALUES
    for use_beta_sigmoid in BOOL_VALUES
)


def _base(case_key: str, **updates) -> dict:
    spec = {
        "case_key": case_key,
        "tags": "accuracy,regression",
        "route": "ascendc",
        "soc": "ascend950",
        "dtype": "bf16",
        "scalar_dtype": "bf16",
        "B": 1,
        "HK": 2,
        "HV": 4,
        "T": 65,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "varlen": False,
        "use_qk_l2norm": False,
        "use_beta_sigmoid": False,
        "state_v_first": False,
        "use_gate_in_kernel": False,
        "use_exp2": True,
        "scale": 0.08838,
    }
    spec.update(updates)
    return spec


def _template_matrix(prefix: str, tags: str, **updates) -> list[dict]:
    specs = []
    for scalar_dtype in SCALAR_DTYPES:
        for use_qk_l2norm in BOOL_VALUES:
            for use_beta_sigmoid in BOOL_VALUES:
                state_v_first = use_beta_sigmoid
                specs.append(
                    _base(
                        f"{prefix}_{scalar_dtype}_qk{int(use_qk_l2norm)}_"
                        f"beta{int(use_beta_sigmoid)}_state{int(state_v_first)}",
                        tags=tags,
                        scalar_dtype=scalar_dtype,
                        use_qk_l2norm=use_qk_l2norm,
                        use_beta_sigmoid=use_beta_sigmoid,
                        state_v_first=state_v_first,
                        **updates,
                    )
                )
    return specs


def _signature(spec: dict) -> tuple:
    return (
        str(spec["scalar_dtype"]),
        bool(spec["use_qk_l2norm"]),
        bool(spec["use_beta_sigmoid"]),
    )


def _assert_template_matrix(name: str, specs: list[dict]) -> None:
    signatures = [_signature(spec) for spec in specs]
    if len(signatures) != len(EXPECTED_TEMPLATE_SIGNATURES):
        raise AssertionError(
            f"{name}: expected {len(EXPECTED_TEMPLATE_SIGNATURES)} cases, "
            f"got {len(signatures)}")
    if frozenset(signatures) != EXPECTED_TEMPLATE_SIGNATURES:
        raise AssertionError(f"{name}: incomplete or duplicate template matrix")
    if {bool(spec["state_v_first"]) for spec in specs} != set(BOOL_VALUES):
        raise AssertionError(f"{name}: state_v_first must cover true and false")


ACCURACY_MATRICES = {
    "dense_full_chunk": _template_matrix(
        "dense_full_chunk", "accuracy,template,dense", B=1, HK=2, HV=2, T=64),
    "dense_tail_ratio2": _template_matrix(
        "dense_tail_ratio2", "accuracy,template,tail,gva", B=1, HK=2, HV=4, T=65),
    "dense_tail_ratio3": _template_matrix(
        "dense_tail_ratio3",
        "accuracy,template,tail,gva",
        B=1,
        HK=2,
        HV=6,
        T=65,
    ),
    "varlen_tail_ratio2": _template_matrix(
        "varlen_tail_ratio2",
        "accuracy,template,varlen,tail,gva",
        B=1,
        HK=2,
        HV=4,
        T=65,
        varlen=True,
    ),
}

MODEL_CASES = (
    ("model1_t2816", 2, 16, 32, 2816, False),
    ("model2_varlen_t8192", 1, 32, 32, 8192, True),
    ("model3_t64", 4, 96, 96, 64, False),
    ("model4_t65", 1, 32, 32, 65, False),
    ("model5_t271", 6, 6, 6, 271, False),
    ("model6_t271", 1, 12, 12, 271, False),
    ("model7_gva_1_3_t8192", 1, 8, 24, 8192, False),
    ("model8_gva_1_2_t2816", 1, 16, 32, 2816, False),
)

GENERALIZATION_CASES = (
    ("general_b1_h4_128", 1, 4, 4, 128, False),
    ("general_b1_h8_192", 1, 8, 8, 192, False),
    ("general_b2_h4_256", 2, 4, 4, 256, False),
    ("general_b1_gva_1_2_512", 1, 8, 16, 512, False),
    ("general_b1_gva_1_3_512", 1, 8, 24, 512, False),
    ("general_b2_gva_1_2_1024", 2, 8, 16, 1024, False),
    ("general_b1_h16_1024", 1, 16, 16, 1024, False),
    ("general_b1_h32_1024", 1, 32, 32, 1024, False),
    ("general_b1_gva_1_2_2048", 1, 16, 32, 2048, False),
    ("general_b1_gva_1_3_2048", 1, 16, 48, 2048, False),
    ("general_b1_varlen_h8_512", 1, 8, 8, 512, True),
    ("general_b1_varlen_gva_1_2_1024", 1, 8, 16, 1024, True),
    ("general_b1_varlen_h16_1024", 1, 16, 16, 1024, True),
)

PERF_MATRICES = {
    name: _template_matrix(
        name,
        "performance,model_target,template",
        B=batch,
        HK=key_heads,
        HV=value_heads,
        T=tokens,
        varlen=varlen,
    )
    for name, batch, key_heads, value_heads, tokens, varlen in MODEL_CASES
}

# Keep small boundary cases and add every model shape to the accuracy matrix.
# This prevents the accuracy suite from exercising only toy allocations while
# the larger model shapes are covered by performance tests alone.
for _model_name, _batch, _key_heads, _value_heads, _tokens, _varlen in MODEL_CASES:
    ACCURACY_MATRICES[f"{_model_name}_generalization"] = _template_matrix(
        f"{_model_name}_generalization",
        "accuracy,model_target,template,generalization",
        B=_batch,
        HK=_key_heads,
        HV=_value_heads,
        T=_tokens,
        varlen=_varlen,
    )

for _name, _batch, _key_heads, _value_heads, _tokens, _varlen in GENERALIZATION_CASES:
    ACCURACY_MATRICES[_name] = _template_matrix(
        _name,
        "accuracy,generalization,large_shape,template",
        B=_batch,
        HK=_key_heads,
        HV=_value_heads,
        T=_tokens,
        varlen=_varlen,
    )

MSS_TEMPLATE_SPECS = _template_matrix(
    "mss_template",
    "determinism,mss,template,tail,gva",
    B=1,
    HK=2,
    HV=4,
    T=65,
)


def _finalize(name: str, specs: list[dict], seed_offset: int) -> list[dict]:
    result = deepcopy(specs)
    keys = [str(spec["case_key"]) for spec in result]
    if len(keys) != len(set(keys)):
        raise AssertionError(f"{name}: duplicate case_key")
    for case_id, spec in enumerate(result):
        spec["case_id"] = case_id
        spec["seed"] = SEED_BASE + seed_offset + case_id
    return result


def build_accuracy_specs() -> list[dict]:
    specs = []
    for name, matrix in ACCURACY_MATRICES.items():
        _assert_template_matrix(f"accuracy/{name}", matrix)
        specs.extend(matrix)
    return _finalize("accuracy", specs, 0)


def build_perf_specs() -> list[dict]:
    specs = []
    for name, matrix in PERF_MATRICES.items():
        _assert_template_matrix(f"performance/{name}", matrix)
        specs.extend(matrix)
    return _finalize("performance", specs, 1000)


def build_mss_specs() -> list[dict]:
    _assert_template_matrix("determinism/mss", MSS_TEMPLATE_SPECS)
    return _finalize("determinism/mss", MSS_TEMPLATE_SPECS, 2000)


def _input(name: str, dtype: str, value, *, input_type: str = "attr", shape=None) -> dict:
    return {
        "name": name,
        "type": input_type,
        "required": True,
        "dtype": dtype,
        "shape": shape,
        "range_values": value,
        "backward": False,
    }


def _state_chunk_num(spec: dict) -> int:
    total_tokens = int(spec["T"])
    chunk_size = int(spec["chunk_size"])
    if not bool(spec.get("varlen", False)):
        return (total_tokens + chunk_size - 1) // chunk_size
    rng = random.Random(20260822)
    cuts = sorted(rng.sample(range(1, total_tokens), 63))
    boundaries = [0, *cuts, total_tokens]
    return sum(
        (boundaries[index + 1] - boundaries[index] + chunk_size - 1) // chunk_size
        for index in range(64)
    )


def _case_payload(case_id: int, spec: dict) -> dict:
    metadata = deepcopy(spec)
    metadata["case_id"] = case_id
    batch = int(metadata["B"])
    key_heads = int(metadata["HK"])
    value_heads = int(metadata["HV"])
    total_tokens = int(metadata["T"])
    dim = int(metadata["K"])
    chunk_size = int(metadata["chunk_size"])
    chunks = _state_chunk_num(metadata)
    scalar_dtype = str(metadata["scalar_dtype"])
    tensor_specs = (
        ("q", "bf16", [batch, key_heads, total_tokens, dim]),
        ("k", "bf16", [batch, key_heads, total_tokens, dim]),
        ("v", "bf16", [batch, value_heads, total_tokens, dim]),
        ("v_new", "bf16", [batch, value_heads, total_tokens, dim]),
        ("do", "bf16", [batch, value_heads, total_tokens, dim]),
        ("g", scalar_dtype, [batch, value_heads, total_tokens]),
        ("beta", scalar_dtype, [batch, value_heads, total_tokens]),
        ("beta_raw", scalar_dtype, [batch, value_heads, total_tokens]),
        ("du", "bf16", [batch, value_heads, total_tokens, dim]),
        ("h", "bf16", [batch, value_heads, chunks, dim, dim]),
        ("dh", "bf16", [batch, value_heads, chunks, dim, dim]),
        ("a", "bf16", [batch, value_heads, total_tokens, chunk_size]),
    )
    def input_range(name: str) -> list[float]:
        if name in {"q", "k"}:
            return QK_RANGE
        if name == "v":
            return V_RANGE
        if name in VECTOR_GRAD_INPUTS:
            return VECTOR_GRAD_RANGE
        return SCALAR_RANGE

    inputs = [
        _input(
            name,
            dtype,
            input_range(name),
            input_type="tensor",
            shape=shape,
        )
        for name, dtype, shape in tensor_specs
    ] + [
        _input(
            "case_spec",
            "non_param",
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ),
    ]
    for name, dtype in (
        ("case_key", "string"),
        ("tags", "string"),
        ("route", "string"),
        ("soc", "string"),
        ("dtype", "string"),
        ("scalar_dtype", "string"),
        ("B", "int"),
        ("HK", "int"),
        ("HV", "int"),
        ("T", "int"),
        ("K", "int"),
        ("V", "int"),
        ("chunk_size", "int"),
        ("varlen", "bool"),
        ("use_qk_l2norm", "bool"),
        ("use_beta_sigmoid", "bool"),
        ("state_v_first", "bool"),
        ("scale", "float"),
        ("seed", "int"),
    ):
        inputs.append(_input(name, dtype, metadata[name]))
    return {
        "id": case_id,
        "default_seed": metadata["seed"],
        "name": f"{OP_NAME}_{case_id:04d}_{metadata['case_key']}",
        "aclnn_name": None,
        "version": "v2.1",
        "api": "pytorch",
        "api_type": "executor_chunk_gated_delta_rule_bwd_finalize",
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
    @GENERATOR_REGISTRY.register("generator_chunk_gated_delta_rule_bwd_finalize")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)
            if CaseConfig is None:
                raise RuntimeError("ATK is required to build CaseConfig objects")
            self.cases = [CaseConfig(**payload) for payload in _payloads(build_accuracy_specs())]
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
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
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
        print(f"accuracy={len(accuracy)} perf={len(perf)} determinism={len(mss)} mss={len(mss)}")


if __name__ == "__main__":
    main()
