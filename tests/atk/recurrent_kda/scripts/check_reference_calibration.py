"""Calibrate the unchanged ATK tolerance against FP64 and deliberate errors."""
import argparse
import ast
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor_recurrent_kda import build_inputs, call_options, recurrent_kda_reference, run_cpu
from gen_recurrent_kda import PROFILES, base_spec, contract_specs, typical_specs
from atk.tasks.post_process.mixed_tolerance_benchmark_compare import (
    MixedToleranceBenchmarkAccuracyCompare,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--reference", type=Path, help="Fixed PR917 executor source for CPU alignment")
    parser.add_argument("--typical", action="store_true", help="Calibrate representative and contract shapes")
    args = parser.parse_args()
    torch.set_num_threads(1)
    compare = MixedToleranceBenchmarkAccuracyCompare(None)
    reference = None
    aligned = 0
    if args.reference:
        # Extract only the pure CPU functions; do not register a second executor
        # or inherit the reference PR's backend/masking test adapter.
        module = ast.parse(args.reference.read_text())
        names = {"_flatten_bsnd", "_restore_layout", "_seq_ranges", "_state_slot",
                 "recurrent_kda_reference"}
        module.body = [node for node in module.body if
                       (isinstance(node, ast.FunctionDef) and node.name in names) or
                       (isinstance(node, ast.ImportFrom) and node.module == "__future__")]
        namespace = {"torch": torch, "F": torch.nn.functional}
        exec(compile(module, str(args.reference), "exec"), namespace)
        reference = namespace["recurrent_kda_reference"]
    profiles = typical_specs() + contract_specs()[:12] if args.typical else PROFILES
    for spec in profiles:
        low_inputs = build_inputs(spec, torch.device("cpu"))
        high_inputs = build_inputs(spec, torch.device("cpu"), high_precision=True)
        for name, value in low_inputs.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value.to(high_inputs[name].dtype), high_inputs[name]), (spec["name"], name)
        actual, golden = run_cpu(spec), run_cpu(spec, high_precision=True)
        for index, (value, target) in enumerate(zip(actual, golden)):
            result = compare.compute_accuracy_result(value, target.float(), f"output_{index}.pt")
            assert result.result, (spec["name"], index, result.error_info)
        if (reference and spec in PROFILES and not spec.get("padding")
                and not (spec.get("soc", "ascend950") == "ascend950"
                         and spec.get("state_dtype") == "bf16")):
            original = reference(**build_inputs(spec, torch.device("cpu"), high_precision=True),
                                 **call_options(spec))
            assert all(torch.equal(a, b) for a, b in zip(golden, original)), spec["name"]
            aligned += 1
    # An independent one-coordinate recurrence checks epsilon placement, scale,
    # and state update without reusing the reference's vector operations.
    spec = base_spec(B=1, T=1, H=1, HV=1, pattern="small", state_zero=True,
                     use_qk_l2norm_in_kernel=True)
    inputs = build_inputs(spec, torch.device("cpu"), high_precision=True)
    inputs["g"].zero_(); inputs["beta"].fill_(0.5); inputs["v"].fill_(2)
    out, state = recurrent_kda_reference(**inputs, **call_options(spec))
    x = 2.0 ** -12
    normalized = x / (x * x + 1e-6) ** 0.5
    assert torch.allclose(state[..., 0], torch.full_like(state[..., 0], normalized))
    assert torch.count_nonzero(state[..., 1:]) == 0
    expected = normalized ** 2 * 128 ** -0.5
    assert torch.allclose(out, torch.full_like(out, expected))
    # A5 BF16 State rounds after every update; out uses the update before
    # rounding. Check both layouts against an independent scalar recurrence.
    rounded_checks = 0
    for v_first in (True, False):
        spec = base_spec(B=1, T=5, H=1, HV=1, cu=False, state_dtype="bf16",
                         state_v_first=v_first, scale=1.0)
        inputs = build_inputs(spec, torch.device("cpu"), high_precision=True)
        inputs["q"].zero_(); inputs["q"][..., 0] = 1.0
        inputs["k"].zero_(); inputs["k"][..., 0] = 0.25
        inputs["g"].zero_(); inputs["beta"].fill_(0.5)
        inputs["initial_state"].zero_()
        inputs["initial_state"][0, 0, 0, 0] = 0.25
        values = torch.tensor([0.2, 0.3, 0.1, -0.2, 0.4]).bfloat16().double()
        inputs["v"].zero_(); inputs["v"][0, :, 0, 0] = values
        rounded_out, rounded_state = recurrent_kda_reference(
            **inputs, **call_options(spec), state_storage_dtype=torch.bfloat16)
        unrounded_out, _ = recurrent_kda_reference(**inputs, **call_options(spec))
        scalar_state, expected_out = 0.25, []
        for value in values.tolist():
            updated = scalar_state + (value - scalar_state * 0.25) * 0.5 * 0.25
            expected_out.append(updated)
            scalar_state = float(torch.tensor(updated, dtype=torch.float64).bfloat16())
        assert torch.equal(rounded_out[0, :, 0, 0], torch.tensor(expected_out, dtype=torch.float64))
        assert rounded_state[0, 0, 0, 0].item() == scalar_state
        assert torch.count_nonzero(rounded_state) == 1
        assert not torch.equal(rounded_out, unrounded_out)
        assert rounded_out[0, 0, 0, 0].item() != float(
            rounded_out[0, 0, 0, 0].bfloat16())
        rounded_checks += 1
    # Wrong output/state must fail the same comparator used for acceptance.
    signal, _ = run_cpu(PROFILES[-1], high_precision=True)
    zero = compare.compute_accuracy_result(torch.zeros_like(signal).bfloat16(), signal.float(), "output_0.pt")
    scaled = compare.compute_accuracy_result((state * 2).float(), state.float(), "output_1.pt")
    assert not zero.result and not scaled.result
    if args.artifacts:
        import numpy as np
        args.artifacts.mkdir(parents=True, exist_ok=True)
        for index, (value, target) in enumerate(zip(run_cpu(PROFILES[-1]),
                                                  run_cpu(PROFILES[-1], high_precision=True))):
            for role, tensor in [("correct", value), ("golden", target),
                                 ("wrong", torch.zeros_like(value) if index == 0 else value * 2)]:
                np.save(args.artifacts / f"{role}_{index}.npy", tensor.float().numpy())
    print(f"CALIBRATION_OK profiles={len(profiles)} reference_aligned={aligned} "
          f"analytic_epsilon=pass bf16_storage_checks={rounded_checks} "
          "wrong_zero=reject wrong_state=reject")


if __name__ == "__main__":
    main()
