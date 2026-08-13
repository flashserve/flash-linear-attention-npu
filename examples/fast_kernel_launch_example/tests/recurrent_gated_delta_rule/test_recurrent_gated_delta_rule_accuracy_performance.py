#!/usr/bin/env python3
"""Run the shared RGDR accuracy suites through direct ``<<<>>>`` launch.

The cases, deterministic input files, validation checks, saved tensor format,
and CUDA-side golden/Triton comparison workflow come directly from the
operator PTA ``test_performance.py``.  This adapter replaces only its NPU
operator call with ``torch.ops.ascend_ops.recurrent_gated_delta_rule`` (or the
functional variant selected by ``--api-mode``).

Run this file on the NPU host.  To perform the cross-device CT comparison,
copy its output directory to the CUDA host and run the original PTA script
against that directory.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import ascend_ops  # noqa: F401  # Import _C and register torch.ops.ascend_ops.
import torch


_REPO_ROOT = Path(__file__).resolve().parents[4]
_PTA_DIR = (
    _REPO_ROOT
    / "fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/tests/pta"
)
_PTA_ACCURACY_FILE = _PTA_DIR / "test_performance.py"
_DEFAULT_OUTPUT_DIR = Path("recurrent_gdn_fast_launch_accuracy_outputs")


def load_pta_accuracy_workload() -> ModuleType:
    """Load the shared accuracy workload under a non-pytest module name."""

    module_name = "_rgdr_accuracy_performance_workload"
    spec = importlib.util.spec_from_file_location(module_name, _PTA_ACCURACY_FILE)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Unable to load PTA accuracy workload: {_PTA_ACCURACY_FILE}"
        )

    # test_performance.py imports golden.py from its own directory.
    pta_dir = str(_PTA_DIR)
    if pta_dir not in sys.path:
        sys.path.insert(0, pta_dir)

    module = importlib.util.module_from_spec(spec)
    # Dataclass processing expects the module to be visible in sys.modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


PTA_ACCURACY = load_pta_accuracy_workload()


def resolve_fast_launch_operator(api_mode: str) -> tuple[Any, str]:
    """Return the requested direct-launch torch operator."""

    op_name = "recurrent_gated_delta_rule"
    if api_mode == "functional":
        op_name += "_functional"

    namespace = torch.ops.ascend_ops
    if not hasattr(namespace, op_name):
        raise RuntimeError(
            f"torch.ops.ascend_ops.{op_name} is not registered; build and install "
            "the recurrent_gated_delta_rule fast-launch wheel first"
        )
    return getattr(namespace, op_name), f"torch.ops.ascend_ops.{op_name}"


@torch.inference_mode()
def run_fast_launch_npu(
    inputs: dict[str, Any],
    device: str,
    *,
    api_mode: str,
) -> dict[str, Any]:
    """Run the PTA NPU checks with the direct ``<<<>>>`` implementation."""

    PTA_ACCURACY.validate_accepted_tokens(inputs)
    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError("NPU execution requires torch_npu") from error

    operator, operator_entrypoint = resolve_fast_launch_operator(api_mode)
    npu_device = torch.device(device)
    torch_npu.npu.set_device(npu_device)

    names = (
        "query",
        "key",
        "value",
        "beta",
        "actual_seq_lengths",
        "ssm_state_indices",
    )
    device_inputs = {name: inputs[name].to(npu_device) for name in names}
    for name in ("g", "gk", "num_accepted_tokens"):
        value = inputs[name]
        device_inputs[name] = value.to(npu_device) if value is not None else None

    pristine_state = inputs["state"].to(npu_device)
    working_state = pristine_state.clone()

    def prepare_iteration() -> None:
        working_state.copy_(pristine_state)

    def invoke() -> Any:
        return operator(
            device_inputs["query"],
            device_inputs["key"],
            device_inputs["value"],
            working_state,
            beta=device_inputs["beta"],
            scale=inputs["scale"],
            actual_seq_lengths=device_inputs["actual_seq_lengths"],
            ssm_state_indices=device_inputs["ssm_state_indices"],
            num_accepted_tokens=device_inputs["num_accepted_tokens"],
            g=device_inputs["g"],
            gk=device_inputs["gk"],
        )

    prepare_iteration()
    output_1, state_1 = PTA_ACCURACY._extract_npu_result(
        invoke(), working_state
    )
    PTA_ACCURACY.synchronize("npu", torch_npu)
    output_1, state_1 = output_1.cpu(), state_1.cpu()

    prepare_iteration()
    output_2, state_2 = PTA_ACCURACY._extract_npu_result(
        invoke(), working_state
    )
    PTA_ACCURACY.synchronize("npu", torch_npu)
    output_2, state_2 = output_2.cpu(), state_2.cpu()

    if tuple(output_1.shape) != tuple(inputs["value"].shape):
        raise AssertionError(
            f"NPU output shape {tuple(output_1.shape)} != "
            f"{tuple(inputs['value'].shape)}"
        )

    deterministic_output = torch.equal(output_1, output_2)
    deterministic_state = torch.equal(state_1, state_2)
    modified_blocks = PTA_ACCURACY.modified_state_blocks(
        inputs["state"], state_1
    )
    expected_blocks = PTA_ACCURACY.expected_state_blocks(inputs)
    unexpected_blocks = set(modified_blocks) - set(
        expected_blocks["per_token_kernel_writeback"]
    )
    return {
        # Keep the original backend name so the PTA CUDA-side loader can read
        # these files without changing its cross-device comparison workflow.
        "backend": "npu",
        "implementation": "fast_launch",
        "operator": operator_entrypoint,
        "api_mode": api_mode,
        "device": device,
        "output": output_1.contiguous(),
        "final_state": state_1.contiguous(),
        "deterministic_output": deterministic_output,
        "deterministic_state": deterministic_state,
        "modified_state_blocks": modified_blocks,
        "expected_state_blocks": expected_blocks,
        "unexpected_state_blocks": sorted(unexpected_blocks),
        "checks": {
            "deterministic_output": deterministic_output,
            "deterministic_state": deterministic_state,
            "state_writeback_is_expected_subset": not unexpected_blocks,
        },
    }


def parse_adapter_args() -> tuple[argparse.Namespace, list[str]]:
    """Consume adapter-only arguments and leave PTA arguments untouched."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--api-mode",
        choices=("mutable", "functional"),
        default="mutable",
        help="Direct-launch API to validate (default: mutable)",
    )
    return parser.parse_known_args()


def main() -> None:
    adapter_args, pta_args = parse_adapter_args()
    user_set_output_dir = any(
        arg == "--output-dir" or arg.startswith("--output-dir=")
        for arg in pta_args
    )

    original_parse_args = PTA_ACCURACY.parse_args

    def parse_pta_args() -> argparse.Namespace:
        args = original_parse_args()
        if not user_set_output_dir:
            args.output_dir = _DEFAULT_OUTPUT_DIR
        if PTA_ACCURACY.infer_device_type(args.device) != "npu":
            raise ValueError(
                "This adapter runs the fast-launch NPU backend only; use the "
                "original PTA test_performance.py on the CUDA host"
            )
        return args

    # The shared main() owns suite selection, deterministic input caching,
    # negative cases, state-writeback checks, and result serialization.
    PTA_ACCURACY.parse_args = parse_pta_args
    PTA_ACCURACY.run_npu = partial(
        run_fast_launch_npu,
        api_mode=adapter_args.api_mode,
    )
    sys.argv = [sys.argv[0], *pta_args]

    operator_name = (
        "torch.ops.ascend_ops.recurrent_gated_delta_rule_functional"
        if adapter_args.api_mode == "functional"
        else "torch.ops.ascend_ops.recurrent_gated_delta_rule"
    )
    print(
        f"fast_launch_operator={operator_name} "
        f"api_mode={adapter_args.api_mode}",
        flush=True,
    )
    PTA_ACCURACY.main()


if __name__ == "__main__":
    main()
