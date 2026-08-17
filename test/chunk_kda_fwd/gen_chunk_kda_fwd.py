"""Legacy deterministic ATK matrix for chunk_kda_fwd.

The generated IDs are frozen by profile:
  * 0-199: A2 positive accuracy cases
  * 200-249: A2 negative interception cases
  * 250-449: A5 positive accuracy cases
  * 450-499: A5 negative interception cases
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name not in {"atk", "torch"}:
        raise
    CaseGenerator = None
    GENERATOR_REGISTRY = None
    CaseConfig = None


SEED_BASE = 20260810
L1_STANDARD = {
    "acc": {
        "cv_fused_double_benchmark": {
            "max_re_ratio": 5,
            "avg_re_ratio": 1.5,
            "root_mean_squared_ratio": 1.5,
        }
    },
    "perf": "not_key",
}


def _balanced(total: int, count: int) -> list[int]:
    base, extra = divmod(total, count)
    return [base + (index < extra) for index in range(count)]


def _skewed(total: int, count: int) -> list[int]:
    weights = [count - index for index in range(count)]
    weight_sum = sum(weights)
    lengths = [max(1, total * weight // weight_sum) for weight in weights]
    lengths[0] += total - sum(lengths)
    return lengths


def _geometric(total: int) -> list[int]:
    lengths = []
    remaining = total
    while remaining > 1:
        value = max(1, 1 << (remaining.bit_length() - 2))
        lengths.append(value)
        remaining -= value
    if remaining:
        lengths.append(remaining)
    return lengths


def _alternating(total: int, count: int, short: int) -> list[int]:
    lengths = [short if index % 2 == 0 else 1 for index in range(count)]
    remaining = total - sum(lengths)
    long_indices = [index for index in range(count) if index % 2]
    base, extra = divmod(remaining, len(long_indices))
    for offset, index in enumerate(long_indices):
        lengths[index] += base + (offset < extra)
    return lengths


def _cu_seqlens(lengths: list[int]) -> str:
    values = [0]
    for length in lengths:
        values.append(values[-1] + int(length))
    return ",".join(str(value) for value in values)


def _positive_case(local_id: int, soc: str) -> dict:
    pair_id = local_id // 2
    disable_recompute = bool(local_id % 2)
    total_t = 8192 if pair_id % 2 == 0 else 16384
    chunk_size = 64
    profile_id = pair_id % 10
    sequence_count = (2, 4, 8, 16, 32)[(pair_id // 10) % 5]
    lengths = None
    scenario = "dense"
    layout = "BSND" if pair_id % 4 < 2 else "BNSD"
    if profile_id in {2, 3}:
        lengths = _balanced(total_t, sequence_count)
        scenario = f"packed_balanced{sequence_count}"
        layout = "TND" if pair_id % 2 == 0 else "NTD"
    elif profile_id in {4, 5}:
        lengths = _skewed(total_t, sequence_count)
        scenario = f"packed_skewed{sequence_count}"
        layout = "TND" if pair_id % 2 == 0 else "NTD"
    elif profile_id in {6, 7}:
        lengths = _geometric(total_t)
        scenario = "packed_geometric"
        layout = "TND" if pair_id % 2 == 0 else "NTD"
    elif profile_id in {8, 9}:
        lengths = _alternating(total_t, max(sequence_count, 4), chunk_size // 2)
        scenario = f"packed_alternating{max(sequence_count, 4)}"
        layout = "TND" if pair_id % 2 == 0 else "NTD"

    route = "ascendc" if pair_id < 84 else ("aclnn" if pair_id < 92 else "direct_launch")
    if route == "direct_launch":
        lengths = None
        scenario = "dense_direct"
        layout = "BNSD"

    tags = ["accuracy", "model_target", "regression"]
    if lengths is not None:
        tags.append("boundary")
    if not disable_recompute and pair_id in {0, 8}:
        tags.append("performance")
    if not disable_recompute and pair_id in {2, 9}:
        tags.append("determinism")
    if not disable_recompute and pair_id in {4, 8}:
        tags.append("sanitizer")
    if route != "ascendc":
        tags.append("route")

    return {
        "case_key": (
            f"{soc}_model_{pair_id:03d}_h96_t{total_t}_c{chunk_size}_{scenario}_"
            f"{'export' if disable_recompute else 'recompute'}"
        ),
        "tags": ",".join(tags),
        "route": route,
        "B": 1,
        "H": 96,
        "HV": 96,
        "T": total_t,
        "K": 128,
        "V": 128,
        "chunk_size": chunk_size,
        "layout": layout,
        "q_dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "bf16",
        "scale": 1.0 / math.sqrt(128),
        "initial_state": False,
        "output_final_state": False,
        "cu_seqlens": "" if lengths is None else _cu_seqlens(lengths),
        "explicit_chunk_indices": lengths is not None and pair_id % 2 == 0,
        "safe_gate": True,
        "lower_bound": -5.0,
        "use_gate_in_kernel": True,
        "dt_bias": True,
        "disable_recompute": disable_recompute,
        "return_intermediate_states": False,
        "state_v_first": True,
        "data_profile": "model_h96",
        "qk_scale": 0.05,
        "v_scale": 0.05,
        "gate_scale": 1.25,
        "beta_scale": 0.35,
        "beta_bias": 1.5,
        "a_log_scale": 0.12,
        "dt_bias_scale": 1.65,
        "dt_bias_mean": -3.0,
        "seed": SEED_BASE + pair_id,
    }


NEGATIVE_CASES = (
    ("null_q", "ACLNN_ERR_PARAM_NULLPTR", 161001, "q, k, v, g and beta must not be nullptr"),
    ("null_k", "ACLNN_ERR_PARAM_NULLPTR", 161001, "q, k, v, g and beta must not be nullptr"),
    ("null_v", "ACLNN_ERR_PARAM_NULLPTR", 161001, "q, k, v, g and beta must not be nullptr"),
    ("null_g", "ACLNN_ERR_PARAM_NULLPTR", 161001, "q, k, v, g and beta must not be nullptr"),
    ("null_beta", "ACLNN_ERR_PARAM_NULLPTR", 161001, "q, k, v, g and beta must not be nullptr"),
    ("null_attn", "ACLNN_ERR_PARAM_NULLPTR", 161001, "attnOut must not be nullptr"),
    ("null_aqk", "ACLNN_ERR_PARAM_NULLPTR", 161001, "aqkOut and akkOut must not be nullptr"),
    ("missing_alog", "ACLNN_ERR_PARAM_NULLPTR", 161001, "aLogOptional is required when useGateInKernel is true"),
    ("chunk_invalid", "ACLNN_ERR_PARAM_INVALID", 161002, "chunkSize must be 64 or 128"),
    ("layout_lower", "ACLNN_ERR_PARAM_INVALID", 161002, "layout must be uppercase and one of BSND, BNSD, TND or NTD"),
    ("layout_invalid", "ACLNN_ERR_PARAM_INVALID", 161002, "layout must be uppercase and one of BSND, BNSD, TND or NTD"),
    ("rank_invalid", "ACLNN_ERR_PARAM_INVALID", 161002, "q/k/v/g and beta ranks must match layout"),
    ("qk_shape", "ACLNN_ERR_PARAM_INVALID", 161002, "q and k must have identical shape"),
    ("v_shape", "ACLNN_ERR_PARAM_INVALID", 161002, "expects v/g/beta"),
    ("g_shape", "ACLNN_ERR_PARAM_INVALID", 161002, "expects v/g/beta"),
    ("beta_shape", "ACLNN_ERR_PARAM_INVALID", 161002, "expects v/g/beta"),
    ("h_zero", "ACLNN_ERR_PARAM_INVALID", 161002, "H and HV must be positive"),
    ("hv_lt_h", "ACLNN_ERR_PARAM_INVALID", 161002, "HV must be greater than or equal to H"),
    ("hv_not_divisible", "ACLNN_ERR_PARAM_INVALID", 161002, "HV must be divisible by H"),
    ("h_gt_128", "ACLNN_ERR_PARAM_INVALID", 161002, "H and HV must be less than or equal to 128"),
    ("k_lt_16", "ACLNN_ERR_PARAM_INVALID", 161002, "K/V must be multiples of 16"),
    ("k_gt_256", "ACLNN_ERR_PARAM_INVALID", 161002, "K/V must be multiples of 16"),
    ("k_unaligned", "ACLNN_ERR_PARAM_INVALID", 161002, "K/V must be multiples of 16"),
    ("v_lt_16", "ACLNN_ERR_PARAM_INVALID", 161002, "K/V must be multiples of 16"),
    ("v_gt_256", "ACLNN_ERR_PARAM_INVALID", 161002, "K/V must be multiples of 16"),
    ("v_unaligned", "ACLNN_ERR_PARAM_INVALID", 161002, "K/V must be multiples of 16"),
    ("q_fp32", "ACLNN_ERR_PARAM_INVALID", 161002, "q, k and v must use the same float16 or bfloat16 dtype"),
    ("k_dtype", "ACLNN_ERR_PARAM_INVALID", 161002, "q, k and v must use the same float16 or bfloat16 dtype"),
    ("v_dtype", "ACLNN_ERR_PARAM_INVALID", 161002, "q, k and v must use the same float16 or bfloat16 dtype"),
    ("g_fp16", "ACLNN_ERR_PARAM_INVALID", 161002, "g must be float32 or bfloat16"),
    ("beta_fp16", "ACLNN_ERR_PARAM_INVALID", 161002, "beta must be float32 or bfloat16"),
    ("alog_dtype", "ACLNN_ERR_PARAM_INVALID", 161002, "aLogOptional must be float32"),
    ("dtbias_dtype", "ACLNN_ERR_PARAM_INVALID", 161002, "dtBiasOptional must be float32"),
    ("state_dtype", "ACLNN_ERR_PARAM_INVALID", 161002, "initialStateOptional must be float32"),
    ("cu_short", "ACLNN_ERR_PARAM_INVALID", 161002, "cuSeqlensOptional must contain at least"),
    ("cu_start", "ACLNN_ERR_PARAM_INVALID", 161002, "cuSeqlensOptional[0] must be 0"),
    ("cu_end", "ACLNN_ERR_PARAM_INVALID", 161002, "cuSeqlensOptional last element must equal the sequence length"),
    ("cu_order", "ACLNN_ERR_PARAM_INVALID", 161002, "cuSeqlensOptional must be nondecreasing"),
    ("varlen_b2", "ACLNN_ERR_PARAM_INVALID", 161002, "rank4 varlen input with cuSeqlensOptional requires B=1"),
    ("seq_gt_1024", "ACLNN_ERR_PARAM_INVALID", 161002, "varlen input supports at most 1024 sequences"),
    ("indices_without_cu", "ACLNN_ERR_PARAM_INVALID", 161002, "chunkIndicesOptional requires cuSeqlensOptional"),
    ("indices_count", "ACLNN_ERR_PARAM_INVALID", 161002, "chunkIndicesOptional must contain exactly one"),
    ("indices_order", "ACLNN_ERR_PARAM_INVALID", 161002, "chunkIndicesOptional must use canonical sequence-major chunk order"),
    ("state_shape_kv", "ACLNN_ERR_PARAM_INVALID", 161002, "initialStateOptional must be [N,HV,K,V]"),
    ("state_shape_vk", "ACLNN_ERR_PARAM_INVALID", 161002, "initialStateOptional must be [N,HV,K,V]"),
    ("alog_shape", "ACLNN_ERR_PARAM_INVALID", 161002, "aLogOptional must have shape [HV]"),
    ("dtbias_shape", "ACLNN_ERR_PARAM_INVALID", 161002, "dtBiasOptional must have shape [HV*K]"),
    ("lower_low", "ACLNN_ERR_PARAM_INVALID", 161002, "lowerBound must be in [-5, 0)"),
    ("lower_high", "ACLNN_ERR_PARAM_INVALID", 161002, "lowerBound must be in [-5, 0)"),
    ("null_akk", "ACLNN_ERR_PARAM_NULLPTR", 161001, "aqkOut and akkOut must not be nullptr"),
)


def _negative_case(local_id: int, soc: str) -> dict:
    mutation, code_name, code, message = NEGATIVE_CASES[local_id]
    layout = "BSND"
    return {
        "case_key": f"{soc}_negative_{local_id:02d}_{mutation}",
        "tags": "negative,boundary",
        "route": "aclnn",
        "B": 1,
        "H": 2,
        "HV": 4,
        "T": 128,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
        "layout": layout,
        "q_dtype": "bf16" if local_id % 2 == 0 else "fp16",
        "g_dtype": "fp32",
        "beta_dtype": "fp32",
        "scale": 1.0 / math.sqrt(128),
        "initial_state": True,
        "output_final_state": True,
        "cu_seqlens": "",
        "explicit_chunk_indices": False,
        "safe_gate": True,
        "lower_bound": -5.0,
        "use_gate_in_kernel": True,
        "dt_bias": True,
        "disable_recompute": True,
        "return_intermediate_states": True,
        "state_v_first": mutation == "state_shape_vk",
        "data_scale": 0.08,
        "gate_scale": 1.0,
        "seed": SEED_BASE + 5000 + local_id,
        "mutation": mutation,
        "expected_code_name": code_name,
        "expected_return_code": code,
        "expected_message": message,
    }


def build_specs() -> list[dict]:
    specs = []
    for soc in ("ascend910b", "ascend950"):
        specs.extend(_positive_case(local_id, soc) for local_id in range(200))
        specs.extend(_negative_case(local_id, soc) for local_id in range(50))
    return specs


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


def _case_payload(case_id: int, spec: dict) -> dict:
    marker_dtype = spec["q_dtype"]
    metadata = {
        "profile": "a2_accuracy" if case_id < 200 else (
            "a2_negative" if case_id < 250 else ("a5_accuracy" if case_id < 450 else "a5_negative")
        ),
        "soc": "ascend910b" if case_id < 250 else "ascend950",
        "shape_spec": f"B={spec['B']},H={spec['H']},HV={spec['HV']},T={spec['T']},K={spec['K']},V={spec['V']}",
        "optional_spec": (
            f"initial={spec['initial_state']},final={spec['output_final_state']},"
            f"varlen={bool(spec['cu_seqlens'])},disable_recompute={spec['disable_recompute']}"
        ),
        **spec,
    }
    inputs = [
        _input("low_precision_marker", marker_dtype, [0, 0], input_type="tensor", shape=[1]),
        _input("fp32_marker", "fp32", [0, 0], input_type="tensor", shape=[1]),
        _input(
            "case_spec",
            "non_param",
            json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ),
        _input("soc", "string", metadata["soc"]),
        _input("route", "string", spec["route"]),
        _input("batch", "int", spec["B"]),
        _input("head", "int", spec["H"]),
        _input("value_head", "int", spec["HV"]),
        _input("total_tokens", "int", spec["T"]),
        _input("key_dim", "int", spec["K"]),
        _input("value_dim", "int", spec["V"]),
        _input("chunk_size", "int", spec["chunk_size"]),
        _input("layout", "string", spec["layout"]),
        _input("q_dtype", "string", spec["q_dtype"]),
        _input("g_dtype", "string", spec["g_dtype"]),
        _input("beta_dtype", "string", spec["beta_dtype"]),
        _input("initial_state", "bool", spec["initial_state"]),
        _input("output_final_state", "bool", spec["output_final_state"]),
        _input("varlen", "bool", bool(spec["cu_seqlens"])),
        _input("safe_gate", "bool", spec["safe_gate"]),
        _input("lower_bound", "float", spec["lower_bound"]),
        _input("use_gate_in_kernel", "bool", spec["use_gate_in_kernel"]),
        _input("disable_recompute", "bool", spec["disable_recompute"]),
        _input("return_intermediate_states", "bool", spec["return_intermediate_states"]),
        _input("state_v_first", "bool", spec["state_v_first"]),
        _input("negative_case", "bool", "negative" in str(spec["tags"]).split(",")),
    ]

    expected_error_msg = None
    if "expected_return_code" in spec:
        expected_error_msg = (
            f"{spec['expected_code_name']}({spec['expected_return_code']}): "
            f"{spec['expected_message']}"
        )
    return {
        "id": case_id,
        "default_seed": spec["seed"],
        "name": "chunk_kda_fwd",
        "aclnn_name": "ChunkKdaFwd",
        "version": "v2.1",
        "api": "pytorch",
        "api_type": "executor_chunk_kda_fwd",
        "expected_error_msg": expected_error_msg,
        "backward": False,
        "standard": L1_STANDARD,
        "outputs": None,
        "inputs": inputs,
        "save_name": "chunk_kda_fwd",
    }


def build_cases() -> list[CaseConfig]:
    if CaseConfig is None:
        raise RuntimeError("ATK and PyTorch are required to instantiate CaseConfig objects.")
    return [CaseConfig(**_case_payload(case_id, spec)) for case_id, spec in enumerate(build_specs())]


if GENERATOR_REGISTRY is not None:
    @GENERATOR_REGISTRY.register("generator_chunk_kda_fwd")
    class ChunkKdaFwdGenerator(CaseGenerator):
        def __init__(self, config):
            super().__init__(config)
            self.cases = build_cases()
            self.length = len(self.cases)
            self.index = 0

        def generate(self) -> CaseConfig:
            case = self.cases[self.index]
            self.index += 1
            return case


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize the frozen chunk_kda_fwd ATK matrix.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("atk_chunk_kda_fwd_legacy_500.json"),
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    payloads = [_case_payload(case_id, spec) for case_id, spec in enumerate(build_specs())]
    args.output.write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.summary:
        print("total=500 a2_accuracy=200 a2_negative=50 a5_accuracy=200 a5_negative=50")
        print("a2_accuracy_ids=0-199 a2_negative_ids=200-249")
        print("a5_accuracy_ids=250-449 a5_negative_ids=450-499")


if __name__ == "__main__":
    main()
