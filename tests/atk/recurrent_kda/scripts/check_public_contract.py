"""Installed public API checks beyond ATK's always-return-state contract."""
import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor_recurrent_kda import build_inputs, call_options, run_cpu
from gen_recurrent_kda import base_spec
from fla_npu.ops.ascendc import recurrent_kda
from atk.tasks.post_process.mixed_tolerance_benchmark_compare import (
    MixedToleranceBenchmarkAccuracyCompare,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    compare = MixedToleranceBenchmarkAccuracyCompare(None)
    count = 0
    for layout in ["BSND", "TND"]:
        for v_first in [False, True]:
            for inplace in [False, True]:
                for return_state in [False, True]:
                    spec = base_spec(T=17, layout=layout, state_v_first=v_first,
                                     inplace=inplace, pattern="signal", state_zero=True,
                                     use_qk_l2norm_in_kernel=True)
                    inputs = build_inputs(spec, device)
                    state = inputs["initial_state"]
                    before, version = state.clone(), state._version
                    options = call_options(spec)
                    options["output_final_state"] = return_state
                    output, final = recurrent_kda(**inputs, **options)
                    torch.npu.synchronize()
                    golden = run_cpu(spec, high_precision=True)
                    assert compare.compute_accuracy_result(output.cpu(), golden[0].float(), "output_0.pt").result
                    assert state._version - version == int(inplace)
                    if inplace:
                        assert not torch.equal(state, before)
                    else:
                        assert torch.equal(state, before)
                    if return_state:
                        assert (final.data_ptr() == state.data_ptr()) == inplace
                        assert compare.compute_accuracy_result(final.cpu(), golden[1].float(), "output_1.pt").result
                    else:
                        assert final is None
                    count += 1
            for metadata in [False, True]:
                for return_state in [False, True]:
                    spec = base_spec(B=2, T=17, cu=metadata, V=128, layout=layout, state_v_first=v_first,
                                     initial_none=True, pattern="signal", use_qk_l2norm_in_kernel=True)
                    options = call_options(spec)
                    options["output_final_state"] = return_state
                    output, final = recurrent_kda(**build_inputs(spec, device), **options)
                    golden = run_cpu(spec, high_precision=True)
                    assert compare.compute_accuracy_result(output.cpu(), golden[0].float(), "output_0.pt").result
                    assert (final is not None) == return_state
                    if final is not None:
                        assert final.dtype == torch.float32
                        assert compare.compute_accuracy_result(final.cpu(), golden[1].float(), "output_1.pt").result
                    count += 1
    for tokens in [1, 9, 17]:
        for x in [0.0, 2.0 ** -12]:
            spec = base_spec(T=tokens, H=1, HV=1, pattern="small", state_zero=True,
                             use_qk_l2norm_in_kernel=True)
            inputs = build_inputs(spec, device)
            inputs["q"].zero_(); inputs["k"].zero_()
            inputs["q"][..., 0] = x; inputs["k"][..., 0] = x
            inputs["g"].zero_(); inputs["beta"].fill_(0.5); inputs["v"].fill_(2)
            inputs["initial_state"][..., 0] = 0.25
            output, final = recurrent_kda(**inputs, **call_options(spec))
            z, current, expected = x / (x*x + 1e-6)**0.5, 0.25, []
            for _ in range(tokens):
                current = current + 0.5 * (2 - current*z) * z
                expected.append(current * z * 128**-0.5)
            prefix = (1, tokens, 1, 1) if output.ndim == 4 else (tokens, 1, 1)
            target = torch.tensor(expected).reshape(prefix).expand_as(output.cpu())
            assert torch.allclose(output.cpu().float(), target, rtol=0.008, atol=1e-6)
            assert torch.allclose(final.cpu()[..., 0], torch.full_like(final.cpu()[..., 0], current),
                                  rtol=1e-5, atol=1e-6)
            assert torch.count_nonzero(final[..., 1:]) == 0
            count += 1
    for safe in [False, True]:
        spec = base_spec(T=9, a_log=False, ignored_gate_parameters=True,
                         safe_gate=safe, lower_bound=123.0)
        inputs = build_inputs(spec, device)
        baseline = {k: v for k, v in inputs.items() if k != "dt_bias"}
        opts = call_options(spec)
        expected = recurrent_kda(**baseline, **dict(opts, safe_gate=False, lower_bound=-5.0))
        actual = recurrent_kda(**inputs, **opts)
        torch.npu.synchronize()
        assert all(torch.equal(a, b) for a, b in zip(actual, expected))
        count += 1
    # Same descriptors, changing device values: the L2 cache must not reuse an
    # old packed boundary snapshot or an X selected for a previous length.
    for dtype in ["int32", "int64"]:
        for length in [0, 1, 2, 4, 8]:
            spec = base_spec(B=2, T=length, H=1, HV=1, lengths=[0, length],
                             padding=32-length, metadata_dtype=dtype, pattern="signal")
            inputs = build_inputs(spec, device)
            cu = inputs["cu_seqlens"]
            backing = torch.full((cu.numel()*3+2,), -100, dtype=cu.dtype, device=device)
            view = backing[1:1+cu.numel()*3:3]
            view.copy_(cu)
            golden = run_cpu(spec, high_precision=True)
            for metadata in [cu, view]:
                output, final = recurrent_kda(**dict(inputs, cu_seqlens=metadata), **call_options(spec))
                if length:
                    assert compare.compute_accuracy_result(
                        output[:length].cpu(), golden[0].float(), "output_0.pt").result
                assert compare.compute_accuracy_result(final.cpu(), golden[1].float(), "output_1.pt").result
            count += 1
    for state_dtype in ["fp32", "bf16"]:
        for gate_dtype in ["fp32", "bf16", "fp16"]:
            for beta_dtype in ["fp32", "bf16", "fp16"]:
                spec = base_spec(T=5, H=1, HV=1, state_dtype=state_dtype,
                                 state_v_first=state_dtype == "fp32", g_dtype=gate_dtype,
                                 beta_dtype=beta_dtype, use_gate_in_kernel=True,
                                 use_beta_sigmoid_in_kernel=True)
                actual = recurrent_kda(**build_inputs(spec, device), **call_options(spec))
                golden = run_cpu(spec, high_precision=True)
                assert all(compare.compute_accuracy_result(a.cpu(), b.float(), f"output_{i}.pt").result
                           for i, (a, b) in enumerate(zip(actual, golden)))
                count += 1
    expected_errors = {
        "head_ratio": ("161002", "HV must be divisible by H"),
        "q_dtype": ("161002", "query"),
        "state_stride": ("161002", "EZ0037", "initialStateRef", "inner state matrix dense"),
        "cu_start": ("161002", "cu_seqlens", "index=0", "actual=1", "capacity=17"),
        "cu_order": ("161002", "cu_seqlens", "index=1", "actual=-1", "capacity=17"),
        "cu_capacity": ("161002", "cu_seqlens", "index=1", "actual=18", "capacity=17"),
    }
    for invalid, fragments in expected_errors.items():
        spec = base_spec(T=17, inplace=True, HV=3 if invalid == "head_ratio" else 4)
        inputs = build_inputs(spec, device)
        if invalid == "q_dtype":
            inputs["q"] = inputs["q"].float()
        elif invalid == "state_stride":
            state = inputs["initial_state"]
            inputs["initial_state"] = torch.zeros((*state.shape[:-1], state.shape[-1] * 2),
                                                  device=device)[..., ::2]
        elif invalid.startswith("cu_"):
            offsets = {"cu_start": [1, 17], "cu_order": [0, -1], "cu_capacity": [0, 18]}[invalid]
            inputs["cu_seqlens"] = torch.tensor(offsets, dtype=torch.int64, device=device)
        try:
            recurrent_kda(**inputs, **call_options(spec))
            torch.npu.synchronize()
        except (RuntimeError, ValueError) as exc:
            message = str(exc)
            assert all(fragment in message for fragment in fragments), (invalid, fragments, message)
            count += 1
        else:
            raise AssertionError(f"accepted invalid {invalid}")
        # A rejected input must not poison the device or be an asynchronous
        # kernel failure disguised as successful validation.
        healthy = base_spec(T=1, H=1, HV=1, pattern="signal", state_zero=True)
        actual = recurrent_kda(**build_inputs(healthy, device), **call_options(healthy))
        torch.npu.synchronize()
        golden = run_cpu(healthy, high_precision=True)
        assert all(compare.compute_accuracy_result(a.cpu(), b.float(), f"output_{i}.pt").result
                   for i, (a, b) in enumerate(zip(actual, golden)))
    # Reuse the repository's mutation/grad regression for just the affected op.
    path = Path(__file__).resolve().parents[3] / "stable_abi/regression_mutation_contract.py"
    module_spec = importlib.util.spec_from_file_location("mutation_contract", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    module.scenario_recurrent_kda()
    print(f"PUBLIC_CONTRACT_OK cases={count} mutation_grad=pass")


if __name__ == "__main__":
    main()
