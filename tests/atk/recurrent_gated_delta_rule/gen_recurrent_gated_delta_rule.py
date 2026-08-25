"""Generate the reviewed recurrent_gated_delta_rule ATK case matrix."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

try:
    from atk.case_generator.generator.base_generator import CaseGenerator
    from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
    from atk.configs.case_config import CaseConfig
except ModuleNotFoundError as exc:
    if exc.name != "atk":
        raise

    class CaseGenerator:
        """Fallback base used only by the standalone JSON materializer."""

    class CaseConfig:
        """Fallback marker used only by the standalone JSON materializer."""

    class _FallbackRegistry:
        def register(self, _name):
            return lambda generator: generator

    GENERATOR_REGISTRY = _FallbackRegistry()
    ATK_AVAILABLE = False
else:
    ATK_AVAILABLE = True


OP_NAME = "recurrent_gated_delta_rule"
SEED_BASE = 20260823
STANDARD = {
    "acc": {
        "cv_fused_double_benchmark": {
            "max_re_ratio": 10,
            "avg_re_ratio": 3,
            "root_mean_squared_ratio": 3,
        }
    },
    "perf": "not_key",
}

PROFILES = [
    {
        "name": "minimal_g_bf16_state",
        "seq_lengths": [1],
        "HK": 1,
        "HV": 1,
        "K": 64,
        "V": 64,
        "gate_mode": "g",
        "state_dtype": "bf16",
    },
    {
        "name": "baseline_g_fp32_state",
        "seq_lengths": [2],
        "HK": 1,
        "HV": 1,
        "K": 128,
        "V": 128,
        "gate_mode": "g",
        "state_dtype": "fp32",
    },
    {
        "name": "gk_tail_dimensions",
        "seq_lengths": [2],
        "HK": 1,
        "HV": 1,
        "K": 80,
        "V": 96,
        "gate_mode": "gk",
        "state_dtype": "bf16",
    },
    {
        "name": "dual_gate_gqa_varlen",
        "seq_lengths": [2, 1],
        "HK": 1,
        "HV": 2,
        "K": 128,
        "V": 64,
        "gate_mode": "both",
        "state_dtype": "bf16",
    },
    {
        "name": "accepted_tokens_gqa",
        "seq_lengths": [3, 2, 1],
        "accepted_tokens": [2, 1, 1],
        "HK": 2,
        "HV": 4,
        "K": 128,
        "V": 128,
        "gate_mode": "g",
        "state_dtype": "fp32",
    },
    {
        "name": "invalid_prefix_and_varlen",
        "prefix_tokens": 2,
        "seq_lengths": [1, 2],
        "HK": 1,
        "HV": 2,
        "K": 64,
        "V": 80,
        "gate_mode": "both",
        "state_dtype": "fp32",
    },
    {
        "name": "maximum_mtp_and_v256",
        "seq_lengths": [8],
        "HK": 2,
        "HV": 4,
        "K": 128,
        "V": 256,
        "gate_mode": "g",
        "state_dtype": "bf16",
    },
    {
        "name": "zero_length_batch",
        "seq_lengths": [2, 0, 1],
        "HK": 1,
        "HV": 2,
        "K": 64,
        "V": 80,
        "gate_mode": "g",
        "state_dtype": "bf16",
    },
    {
        "name": "multi_head_gk_noncontiguous_state",
        "seq_lengths": [2, 2],
        "HK": 4,
        "HV": 8,
        "K": 128,
        "V": 128,
        "gate_mode": "gk",
        "state_dtype": "fp32",
        "state_layout": "noncontiguous",
    },
    {
        "name": "maximum_key_value_dimensions",
        "seq_lengths": [1],
        "HK": 1,
        "HV": 1,
        "K": 512,
        "V": 512,
        "gate_mode": "both",
        "state_dtype": "fp32",
    },
]


def _spec(index: int) -> dict:
    profile = deepcopy(PROFILES[index % len(PROFILES)])
    prefix_tokens = int(profile.get("prefix_tokens", 0))
    seq_lengths = [int(value) for value in profile["seq_lengths"]]
    total_tokens = prefix_tokens + sum(seq_lengths)
    profile.update(
        {
            "op": OP_NAME,
            "case_id": index,
            "seed": SEED_BASE + index,
            "route": "ascendc",
            "soc": "all",
            "dtype": "bf16",
            "state_layout": str(profile.get("state_layout", "contiguous")),
            "B": len(seq_lengths),
            "T": total_tokens,
            "block_num": int(profile.get("block_num", total_tokens)),
            "scale": float(profile.get("scale", 1.0 / math.sqrt(profile["K"]))),
        }
    )
    return profile


def build_specs() -> list[dict]:
    return [_spec(index) for index in range(len(PROFILES))]


def _input(name: str, dtype: str, value, *, input_type: str = "attr", shape=None):
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
    inputs = [
        _input("low_precision_marker", "bf16", [0, 0], input_type="tensor", shape=[1]),
        _input("fp32_marker", "fp32", [0, 0], input_type="tensor", shape=[1]),
        _input(
            "case_spec",
            "string",
            json.dumps(spec, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        ),
        _input("dtype", "string", spec["dtype"]),
        _input("state_dtype", "string", spec["state_dtype"]),
        _input("state_layout", "string", spec["state_layout"]),
        _input("B", "int", spec["B"]),
        _input("T", "int", spec["T"]),
        _input("HK", "int", spec["HK"]),
        _input("HV", "int", spec["HV"]),
        _input("K", "int", spec["K"]),
        _input("V", "int", spec["V"]),
        _input("block_num", "int", spec["block_num"]),
        _input("gate_mode", "string", spec["gate_mode"]),
        _input("use_accepted_tokens", "bool", "accepted_tokens" in spec),
        _input("prefix_tokens", "int", int(spec.get("prefix_tokens", 0))),
        _input("case_id", "int", case_id),
        _input("seed", "int", spec["seed"]),
        _input("soc", "string", spec["soc"]),
        _input("route", "string", spec["route"]),
    ]
    return {
        "id": case_id,
        "default_seed": spec["seed"],
        "name": OP_NAME,
        "aclnn_name": "RecurrentGatedDeltaRule",
        "version": "v2.1",
        "api": "pytorch",
        "api_type": f"executor_{OP_NAME}",
        "expected_error_msg": None,
        "backward": False,
        "standard": STANDARD,
        "outputs": None,
        "inputs": inputs,
        "save_name": OP_NAME,
    }


def build_cases() -> list:
    if not ATK_AVAILABLE:
        raise RuntimeError("ATK is required to instantiate CaseConfig objects.")
    return [
        CaseConfig(**_case_payload(case_id, spec))
        for case_id, spec in enumerate(build_specs())
    ]


@GENERATOR_REGISTRY.register("generator_recurrent_gated_delta_rule")
class RecurrentGatedDeltaRuleGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        del case_config
        index = max(int(self.index) - 1, 0)
        spec = _spec(index)
        payload = _case_payload(index, spec)
        return CaseConfig(**payload)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize the recurrent_gated_delta_rule ATK matrix."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name(f"atk_{OP_NAME}.json"),
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    payloads = [
        _case_payload(case_id, spec)
        for case_id, spec in enumerate(build_specs())
    ]
    args.output.write_text(
        json.dumps(payloads, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.summary:
        print(f"total={len(payloads)} ids=0-{len(payloads) - 1}")
        for payload, spec in zip(payloads, build_specs()):
            print(
                f"case_id={payload['id']} name={spec['name']} "
                f"shape=T{spec['T']}-HK{spec['HK']}-HV{spec['HV']}-K{spec['K']}-V{spec['V']} "
                f"state={spec['state_dtype']}/{spec['state_layout']} gate={spec['gate_mode']}"
            )


if __name__ == "__main__":
    main()
