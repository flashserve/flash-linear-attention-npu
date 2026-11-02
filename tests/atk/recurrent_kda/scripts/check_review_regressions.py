"""Exercise published Python backends and raw ACLNN descriptors on the NPU."""
import argparse
import ctypes
import sys
import warnings
from pathlib import Path

import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from executor_recurrent_kda import build_inputs, call_options, run_cpu
from gen_recurrent_kda import base_spec
from fla_npu.ops.ascendc import _aclnn_ctypes, _stable
from atk.tasks.post_process.mixed_tolerance_benchmark_compare import MixedToleranceBenchmarkAccuracyCompare


def direct_aclnn(q, k, v, g, beta, initial_state, **options):
    """Bypass both wrappers' shape normalization and State allocation policies."""
    out = torch.empty_like(v, memory_format=torch.contiguous_format)
    shape, strides = initial_state.shape, initial_state.stride()
    span = sum((size - 1) * stride for size, stride in zip(shape, strides)) + 1
    storage = torch.full((span + 32,), 17.0, dtype=initial_state.dtype, device=initial_state.device)
    final = options.get("_final_state_override")
    if final is None:
        final = torch.as_strided(storage, shape, strides, storage_offset=16)
    names = ("cu_seqlens", "ssm_state_indices", "A_log", "dt_bias", "num_accepted_tokens")
    inputs = (q, k, v, g, beta, initial_state, *(options.get(name) for name in names))
    layout = ctypes.create_string_buffer(options.get("layout", "BSND").encode())

    def arguments(ctx):
        args = [ctx.tensor(tensor, name) for tensor, name in zip(
            inputs, ("q", "k", "v", "g", "beta", "initial_state", *names))]
        args += [ctypes.cast(layout, ctypes.c_char_p), ctypes.c_double(options["scale"])]
        args += [ctypes.c_bool(options.get(name, False)) for name in (
            "output_final_state", "inplace_final_state", "use_qk_l2norm_in_kernel",
            "use_gate_in_kernel", "use_beta_sigmoid_in_kernel", "allow_neg_eigval", "safe_gate")]
        return [*args, ctypes.c_double(options.get("lower_bound", -5.0)),
                ctypes.c_bool(options["state_v_first"]), ctx.tensor(out, "attn_out"),
                ctx.tensor(final, "final_state")]

    # Raw GetWorkspaceSize has no producer-stream argument. Fulfil its
    # metadata-ready precondition; wrapper-specific tests omit this wait.
    torch.npu.current_stream(q.device).synchronize()
    _aclnn_ctypes._call_aclnn("aclnnRecurrentKda", arguments, (out, initial_state, final))
    torch.npu.synchronize()
    assert torch.all(storage[:16] == 17) and torch.all(storage[-16:] == 17)
    return out, final


def strided_metadata(tensor):
    strides = (3,) if tensor.ndim == 1 else (tensor.shape[1] * 3 + 5, 3)
    span = sum((size - 1) * stride for size, stride in zip(tensor.shape, strides)) + 1
    backing = torch.full((span + 8,), -99, dtype=tensor.dtype, device=tensor.device)
    view = torch.as_strided(backing, tensor.shape, strides, storage_offset=3)
    view.copy_(tensor)
    return view


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.npu.set_device(args.device)
    device = torch.device(f"npu:{args.device}")
    compare = MixedToleranceBenchmarkAccuracyCompare(None)
    count = 0
    backends = {"stable": _stable.npu_recurrent_kda,
                "ctypes": _aclnn_ctypes.npu_recurrent_kda, "aclnn": direct_aclnn}
    for backend, call in backends.items():
        for v_first in (True, False):
            for dtype in ("fp32", "bf16"):
                for scenario in ("empty_request", "sparse_pool", "rank3_no_cu", "rank4_cu",
                                 "inplace_independent_dense", "inplace_independent_gapped"):
                    if scenario.startswith("inplace_independent") and backend != "aclnn":
                        continue
                    spec = base_spec(B=2, T=2, H=1, HV=1, state_dtype=dtype,
                                     state_v_first=v_first, inplace=scenario.startswith("inplace_independent"), pattern="signal",
                                     use_qk_l2norm_in_kernel=True, strided_state=True)
                    if scenario == "empty_request":
                        spec.update(lengths=[0, 2])
                    elif scenario == "sparse_pool":
                        spec.update(indices="packed", accepted=1)
                    elif scenario == "rank3_no_cu":
                        spec.update(B=1, cu=False)
                    inputs = build_inputs(spec, device)
                    state = inputs["initial_state"]
                    # Preserve values while exercising a nonzero logical offset
                    # and gapped outer stride on each entry.
                    backing = torch.zeros((state.shape[0] + 2, 2, 128, 128),
                                          dtype=state.dtype, device=device)
                    inputs["initial_state"] = (state.clone() if scenario == "inplace_independent_dense" else backing[1:-1, :1])
                    inputs["initial_state"].copy_(state)
                    before = inputs["initial_state"].clone()
                    if scenario.startswith("rank"):
                        for name in ("q", "k", "v", "g", "beta"):
                            tensor = inputs[name]
                            inputs[name] = (tensor.squeeze(0) if scenario == "rank3_no_cu" else
                                            tensor.reshape(spec["B"], spec["T"], *tensor.shape[1:]))
                    expected_shape = inputs["v"].shape
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always", DeprecationWarning)
                        output, final = call(**inputs, **call_options(spec))
                    torch.npu.synchronize()
                    assert output.shape == expected_shape, (backend, scenario, output.shape)
                    if backend != "aclnn":
                        deprecated = [w for w in caught if issubclass(w.category, DeprecationWarning)]
                        assert len(deprecated) == int(scenario.startswith("rank")), (backend, scenario, caught)
                    golden = run_cpu(spec, high_precision=True)
                    assert compare.compute_accuracy_result(
                        output.cpu().reshape(golden[0].shape), golden[0].float(), "output_0.pt").result
                    assert compare.compute_accuracy_result(final.cpu(), golden[1].float(), "output_1.pt").result
                    assert torch.equal(inputs["initial_state"], final if spec.get("inplace") else before)
                    assert final.data_ptr() != inputs["initial_state"].data_ptr()
                    if scenario == "empty_request":
                        assert torch.equal(final[0], before[0])
                    elif scenario == "sparse_pool":
                        used = set(inputs["ssm_state_indices"].cpu().tolist())
                        untouched = [i for i in range(before.shape[0]) if i not in used]
                        assert torch.equal(final[untouched], before[untouched])
                    count += 1
                    print("REVIEW_CASE_OK", backend, v_first, dtype, scenario, flush=True)
    rejected = 0
    for v_first in (True, False):
        for dtype in ("fp32", "bf16"):
            spec = base_spec(B=1, T=2, H=1, HV=1, state_dtype=dtype,
                             state_v_first=v_first, inplace=False, pattern="signal")
            inputs = build_inputs(spec, device)
            try:
                alias_options = call_options(spec)
                alias_options.update(_final_state_override=inputs["initial_state"],
                                     output_final_state=True, inplace_final_state=False)
                direct_aclnn(**inputs, **alias_options)
                raise AssertionError(("aclnn", "accepted non-inplace State alias", v_first, dtype))
            except (RuntimeError, ValueError) as exc:
                message = str(exc)
                assert "finalState" in message and "161002" in message, message
                rejected += 1
                print("REVIEW_ALIAS_REJECTED", v_first, dtype, flush=True)
    for backend, call in backends.items():
        spec = base_spec(B=3, T=2, H=1, HV=1, lengths=[0, 2, 1], padding=2,
                         indices="packed", accepted=1, layout="TND", state_v_first=True,
                         state_dtype="fp32", metadata_dtype="int32", inplace=False,
                         pattern="signal")
        inputs = build_inputs(spec, device)
        for name in ("cu_seqlens", "ssm_state_indices", "num_accepted_tokens"):
            inputs[name] = strided_metadata(inputs[name])
        inputs["num_accepted_tokens"][0] = -99
        torch.npu.synchronize()
        output, final = call(**inputs, **call_options(spec))
        torch.npu.synchronize()
        golden = run_cpu(spec, high_precision=True)
        assert compare.compute_accuracy_result(output.cpu()[:3], golden[0].float(), "output_0.pt").result
        assert compare.compute_accuracy_result(final.cpu(), golden[1].float(), "output_1.pt").result
        count += 1
        print("REVIEW_METADATA_VIEW_OK", backend, flush=True)
    for backend, call in backends.items():
        spec = base_spec(B=2, T=2, H=2, HV=2, cu=False, inplace=False, pattern="signal")
        inputs = build_inputs(spec, device)
        for name in ("q", "k", "v"):
            tensor = inputs[name]
            storage = torch.arange(896, dtype=torch.float32, device=device).to(tensor.dtype)
            inputs[name] = torch.as_strided(storage, tensor.shape, (384, 256, 128, 1))
        try:
            call(**inputs, **call_options(spec))
            torch.npu.synchronize()
        except (RuntimeError, ValueError) as exc:
            message = str(exc)
            assert "query" in message and "161002" in message, (backend, message)
            rejected += 1
            print("REVIEW_OVERLAP_REJECTED", backend, flush=True)
        else:
            raise AssertionError((backend, "accepted overlapping BSND QKV descriptor"))
    for backend, call in backends.items():
        for invalid in ("head_ratio", "q_dtype", "state_stride", "cu_start", "cu_order", "cu_capacity",
                        "slot_negative", "slot_capacity", "accepted_zero", "accepted_length",
                        "short_speculative", "lower_nan", "lower_neginf", "lower_posinf"):
            # Non-inplace stable calls intentionally clone/materialize State.
            # Exercise illegal kernel strides through the inplace path instead.
            spec = base_spec(T=2, HV=3 if invalid == "head_ratio" else 4, inplace=invalid == "state_stride")
            if invalid.startswith(("slot_", "accepted_")) or invalid == "short_speculative":
                spec.update(indices="speculative" if invalid == "short_speculative" else "packed")
            if invalid.startswith("lower_"):
                spec.update(a_log=True, safe_gate=True, lower_bound={
                    "lower_nan": float("nan"), "lower_neginf": -float("inf"), "lower_posinf": float("inf")}[invalid])
            inputs = build_inputs(spec, device)
            reason = "HV must be divisible by H"
            if invalid == "q_dtype":
                inputs["q"] = inputs["q"].float()
                reason = "bfloat16" if backend == "ctypes" else "query"
            elif invalid == "state_stride":
                state = inputs["initial_state"]
                inputs["initial_state"] = torch.zeros((*state.shape[:-1], 256),
                                                      dtype=state.dtype, device=device)[..., ::2]
                reason = "inner state matrix dense"
            elif invalid.startswith("cu_"):
                offsets = {"cu_start": [1, 2], "cu_order": [0, -1], "cu_capacity": [0, 3]}[invalid]
                inputs["cu_seqlens"] = torch.tensor(offsets, dtype=torch.int64, device=device)
                reason = "cu_seqlens"
            elif invalid.startswith("slot_"):
                inputs["ssm_state_indices"].fill_(-1 if invalid == "slot_negative" else inputs["initial_state"].shape[0])
                reason = "ssm_state_indices"
            elif invalid.startswith("accepted_"):
                inputs["num_accepted_tokens"].fill_(0 if invalid == "accepted_zero" else spec["T"] + 1)
                reason = "num_accepted_tokens"
            elif invalid == "short_speculative":
                inputs["ssm_state_indices"] = inputs["ssm_state_indices"][:, :1]
                reason = "ssm_state_indices"
            elif invalid.startswith("lower_"):
                reason = "lower_bound"
            try:
                call(**inputs, **call_options(spec))
                torch.npu.synchronize()
            except (RuntimeError, ValueError) as exc:
                message = str(exc)
                assert reason in message, (backend, invalid, message)
                python_guard = backend == "ctypes" and (invalid in ("head_ratio", "q_dtype") or invalid.startswith("lower_"))
                if not python_guard:
                    assert "161002" in message, (backend, invalid, message)
                    expected_code = "EZ0020" if invalid == "q_dtype" else "EZ0037"
                    assert expected_code in message, (backend, invalid, message)
                    assert not any(code in message for code in ("EZ9999", "EZ9903", "EZ1001")), message
                if invalid.startswith("cu_"):
                    assert all(text in message for text in ("EZ0037", "index=", "actual=", "capacity=2")), message
                rejected += 1
            else:
                raise AssertionError((backend, "accepted invalid", invalid))
            healthy = base_spec(T=1, H=1, HV=1, pattern="signal", state_zero=True)
            actual = call(**build_inputs(healthy, device), **call_options(healthy))
            torch.npu.synchronize()
            golden = run_cpu(healthy, high_precision=True)
            assert all(compare.compute_accuracy_result(a.cpu(), b.float(), f"output_{i}.pt").result
                       for i, (a, b) in enumerate(zip(actual, golden)))
    print(f"REVIEW_REGRESSIONS_OK cases={count} negative={rejected} backends=stable,ctypes,aclnn")


if __name__ == "__main__":
    main()
