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
    state_shape = [batch, chunks, value_heads, dim, dim]
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
        ("h", "bf16", state_shape),
        ("dh", "bf16", state_shape),
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


ARCH22_API = "executor_chunk_gated_delta_rule_bwd_finalize_arch22"
ARCH22_SEEDS = (20261030, 20261031, 20261032)


def expected_arch22_key(spec: dict) -> int:
    """静态预期；实际选择必须由 Host UT 或运行记录另行核对。"""
    owner = spec["chunk_size"] == 64 and spec["V"] == 128 and spec["HK"] == spec["HV"]
    return (10 | ((10 if spec["scalar_dtype"] == "bf16" else 30) << 8)
            | (int(spec["use_qk_l2norm"]) << 16)
            | (int(spec["use_beta_sigmoid"]) << 17)
            | (int(spec["use_exp2"]) << 18) | (int(owner) << 19))


def build_arch22_specs() -> list[dict]:
    """68 个逻辑代表；不交叉展开所有 shape 和开关。"""
    ordinary, boundary = [], []
    for owner in BOOL_VALUES:
        for gt in SCALAR_DTYPES:
            for norm in BOOL_VALUES:
                for sigmoid in BOOL_VALUES:
                    for exp2 in BOOL_VALUES:
                        suffix = f"o{int(owner)}_{gt}_n{int(norm)}_s{int(sigmoid)}_e{int(exp2)}"
                        common = dict(B=1, HK=2 if owner else 1, HV=2, T=64, K=128, V=128,
                                      chunk_size=64, scalar_dtype=gt, use_qk_l2norm=norm,
                                      use_beta_sigmoid=sigmoid, use_exp2=exp2,
                                      state_v_first=bool(norm != sigmoid), cu_seqlens=None)
                        ordinary.append(dict(common, case_key="a23_regular_" + suffix))
                        edge = dict(common, case_key="a23_boundary_" + suffix)
                        if owner:
                            # 奇数 head、31/33/1 尾块、合法空序列；owner key 内走 fallback。
                            edge.update(HK=1, HV=1, T=65, cu_seqlens=[0, 0, 31, 64, 65, 65])
                        else:
                            edge.update(T=129, V=256, chunk_size=128,
                                        cu_seqlens=[0, 0, 31, 129, 129])
                        boundary.append(edge)
    extras = [
        dict(ordinary[0], case_key="a23_bt64_v256_ratio3_tail", HV=3, V=256, T=65),
        dict(ordinary[15], case_key="a23_bt128_v128_batch2_tail", B=2, T=129, chunk_size=128),
        dict(ordinary[16], case_key="a23_owner_even_heads_dense_tail", HK=2, HV=2, T=65),
        dict(ordinary[31], case_key="a23_owner_multichunk_reordered", T=128,
             cu_seqlens=[0, 0, 128], chunk_indices=[1, 1, 1, 0]),
    ]
    return ordinary + boundary + extras


def _arch22_metadata(spec: dict):
    cu = spec.get("cu_seqlens")
    if cu is None:
        return None, None, (spec["T"] + spec["chunk_size"] - 1) // spec["chunk_size"]
    pairs = [value for seq, (begin, end) in enumerate(zip(cu, cu[1:]))
             for local in range((end - begin + spec["chunk_size"] - 1) // spec["chunk_size"])
             for value in (seq, local)]
    return cu, spec.get("chunk_indices", pairs), len(pairs) // 2


def _arch22_payload(case_id: int, spec: dict, seed: int) -> dict:
    b, hk, hv, t, k, v, bt = (spec[n] for n in ("B", "HK", "HV", "T", "K", "V", "chunk_size"))
    gt = spec["scalar_dtype"]
    cu, pairs, nt = _arch22_metadata(spec)
    state = [b, nt, hv, v, k] if spec["state_v_first"] else [b, nt, hv, k, v]
    inputs = []
    for name in ("q", "k", "v", "v_new", "do", "du", "g", "beta", "h", "dh", "a"):
        shape = ([b, hk, t, k] if name in ("q", "k") else
                 [b, hv, t, v] if name in ("v", "v_new", "do", "du") else
                 [b, hv, t] if name in ("g", "beta") else
                 state if name in ("h", "dh") else [b, hv, t, bt])
        span = QK_RANGE if name in ("q", "k") else V_RANGE if name == "v" else SCALAR_RANGE if name in ("g", "beta") else VECTOR_GRAD_RANGE
        inputs.append(_input(name, gt if name in ("g", "beta") else "bf16", span,
                             input_type="tensor", shape=shape))
    if spec["use_qk_l2norm"]:
        for name in ("q_rstd", "k_rstd"):
            inputs.append(_input(name, "fp32", [0.8, 1.2], input_type="tensor", shape=[b, hk, t]))
    if spec["use_beta_sigmoid"]:
        inputs.append(_input("beta_raw", gt, SCALAR_RANGE, input_type="tensor", shape=[b, hv, t]))
    # 这两个字段就是公开参数；non_param 仅承载 JSON int64 列表，不承载 case 控制元数据。
    if cu is not None:
        inputs.extend((_input("cu_seqlens", "non_param", json.dumps(cu)),
                       _input("chunk_indices", "non_param", json.dumps(pairs))))
    attrs = dict(scale=128 ** -0.5, chunk_size=bt,
                 use_qk_l2_norm_in_kernel=spec["use_qk_l2norm"],
                 use_beta_sigmoid_in_kernel=spec["use_beta_sigmoid"], use_gate_in_kernel=False,
                 state_v_first=spec["state_v_first"], use_exp2=spec["use_exp2"])
    for name, value in attrs.items():
        dtype = "bool" if isinstance(value, bool) else "int" if isinstance(value, int) else "float"
        inputs.append(_input(name, dtype, value))
    standard = deepcopy(STANDARD)
    standard["acc"] = {"mixed_tolerance_bm": {"output_dtype_overrides": {
        "0": "bf16", "1": "bf16", "2": "bf16", "3": gt, "4": gt,
    }}}
    return dict(id=case_id, default_seed=seed, name=f"{OP_NAME}_{case_id:04d}_{spec['case_key']}_seed{seed}",
                aclnn_name=None, version="v2.1", api="pytorch", api_type=ARCH22_API,
                expected_error_msg=None, backward=False, standard=standard,
                outputs=None, inputs=inputs, save_name=OP_NAME)


def formal_payloads(scope: str) -> list[dict]:
    builders = {"accuracy": build_accuracy_specs, "perf": build_perf_specs, "mss": build_mss_specs}
    original = _payloads(builders[scope]())
    if scope == "perf":
        return original
    specs = build_arch22_specs()
    if scope == "mss":
        # 每个 key 一条，另保留 4 种边界和 4 条 shape/runtime 分支。
        specs = specs[:32] + [specs[i] for i in (32, 39, 48, 63, 64, 65, 66, 67)]
    for spec in specs:
        for seed in ARCH22_SEEDS if scope == "accuracy" else ARCH22_SEEDS[:1]:
            original.append(_arch22_payload(len(original), spec, seed))
    return original


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_chunk_gated_delta_rule_bwd_finalize")
    class Generator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)
            if CaseConfig is None:
                raise RuntimeError("ATK is required to build CaseConfig objects")
            self.cases = [CaseConfig(**payload) for payload in formal_payloads("accuracy")]
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
    accuracy, perf, mss = (formal_payloads(scope) for scope in ("accuracy", "perf", "mss"))
    for suffix, payload in (("", accuracy), ("_perf", perf), ("_mss", mss)):
        (args.output_dir / f"atk_{OP_NAME}{suffix}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.summary:
        print(f"accuracy={len(accuracy)} perf={len(perf)} determinism={len(mss)} mss={len(mss)}")


if __name__ == "__main__":
    main()
