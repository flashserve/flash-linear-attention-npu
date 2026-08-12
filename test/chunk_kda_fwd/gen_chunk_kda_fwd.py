"""Deterministic eight-case A5 dense ATK matrix for chunk_kda_fwd.

Case IDs 250-257 cover sequence lengths 1K, 8K, 16K and 64K with
recomputation enabled and disabled for each length.
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


SEED_BASE = 20260812
FIRST_CASE_ID = 250
DENSE_SEQUENCE_LENGTHS = (1024, 8192, 16384, 65536)
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


def _dense_case(local_id: int) -> dict:
    length_id = local_id // 2
    disable_recompute = bool(local_id % 2)
    total_t = DENSE_SEQUENCE_LENGTHS[length_id]
    chunk_size = 64

    return {
        "case_key": (
            f"ascend950_h96_t{total_t}_c{chunk_size}_dense_"
            f"recompute_{str(not disable_recompute).lower()}"
        ),
        "tags": "accuracy,model_target,regression,dense",
        "route": "ascendc",
        "B": 1,
        "H": 96,
        "HV": 96,
        "T": total_t,
        "K": 128,
        "V": 128,
        "chunk_size": chunk_size,
        "layout": "BSND",
        "q_dtype": "bf16",
        "g_dtype": "fp32",
        "beta_dtype": "bf16",
        "scale": 1.0 / math.sqrt(128),
        "initial_state": False,
        "output_final_state": False,
        "cu_seqlens": "",
        "explicit_chunk_indices": False,
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
        "seed": SEED_BASE + length_id,
    }


def build_specs() -> list[dict]:
    return [_dense_case(local_id) for local_id in range(8)]


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
        "profile": "a5_accuracy",
        "soc": "ascend950",
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
    return [
        CaseConfig(**_case_payload(FIRST_CASE_ID + local_id, spec))
        for local_id, spec in enumerate(build_specs())
    ]


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
        default=Path(__file__).with_name("atk_chunk_kda_fwd.json"),
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    payloads = [
        _case_payload(FIRST_CASE_ID + local_id, spec)
        for local_id, spec in enumerate(build_specs())
    ]
    args.output.write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.summary:
        print("total=8 a5_dense_accuracy=8")
        print("a5_dense_accuracy_ids=250-257")


if __name__ == "__main__":
    main()
