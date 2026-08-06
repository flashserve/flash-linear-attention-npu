#!/usr/bin/env python3
"""Validate the model-facing Ascend C KDA adapter."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
from pathlib import Path


TRITON_ASCEND_VERSION = "3.2.1"
TRITON_ASCEND_KERNELS_COMMIT = "4cd4b506d4153ac18ac1ca8f4c770eac9fd3fcc8"
MAX_RELATIVE_L2 = 1e-2
MIN_COSINE = 0.9999

CASES = {
    "smoke": {
        "B": 1,
        "T": 128,
        "H": 2,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
    },
    "model_bwd_h96": {
        "B": 1,
        "T": 18432,
        "H": 96,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
    },
    "diagnostic_h96_short": {
        "B": 1,
        "T": 128,
        "H": 96,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
    },
    "diagnostic_h32_long": {
        "B": 1,
        "T": 18432,
        "H": 32,
        "K": 128,
        "V": 128,
        "chunk_size": 64,
    },
}


def _dtype_name(tensor):
    return str(tensor.dtype).removeprefix("torch.")


def _descriptor(tensor):
    return {
        "shape": list(tensor.shape),
        "dtype": _dtype_name(tensor),
    }


def _make_inputs(torch, case, device, seed):
    torch.manual_seed(seed)
    B, T, H, K, V = (case[name] for name in ("B", "T", "H", "K", "V"))
    q = (torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.02)
    k = (torch.randn(B, T, H, K, device=device, dtype=torch.bfloat16) * 0.02)
    v = (torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.02)
    beta = torch.sigmoid(
        torch.randn(B, T, H, device=device, dtype=torch.float32)
    ).to(torch.bfloat16)
    g = -torch.rand(B, T, H, K, device=device, dtype=torch.float32) * 0.01
    do = (torch.randn(B, T, H, V, device=device, dtype=torch.bfloat16) * 0.02)
    return q, k, v, g, beta, do


def _clone_for_grad(tensors):
    values = []
    for tensor in tensors:
        clone = tensor.detach().clone()
        if tensor is not tensors[-1]:
            clone.requires_grad_(True)
        values.append(clone)
    return values


def _install_backward_contract_capture():
    module = importlib.import_module(
        "triton_ascend_kernels.attention.fla.kda.chunk_bwd"
    )
    original = module.chunk_kda_bwd_intra
    captures = []

    def wrapper(*args, **kwargs):
        names = (
            "q",
            "k",
            "g",
            "beta",
            "dAqk",
            "dAkk",
            "dq",
            "dk",
            "db",
            "dg",
        )
        values = dict(zip(names, args))
        values.update(kwargs)
        result = original(*args, **kwargs)
        captures.append(
            {
                "inputs": {
                    name: _descriptor(values[name])
                    for name in names
                },
                "outputs": {
                    name: _descriptor(tensor)
                    for name, tensor in zip(("dq", "dk", "db", "dg"), result)
                },
            }
        )
        return result

    module.chunk_kda_bwd_intra = wrapper
    return module, original, captures


def _install_backward_stage_trace(torch):
    module = importlib.import_module(
        "triton_ascend_kernels.attention.fla.kda.chunk_bwd"
    )
    names = (
        "recompute_w_u_fwd",
        "chunk_gated_delta_rule_fwd_h",
        "chunk_kda_bwd_dAv",
        "chunk_gated_delta_rule_bwd_dhu",
        "chunk_kda_bwd_wy_dqkg_fused",
        "chunk_kda_bwd_intra",
        "kda_gate_bwd",
    )
    originals = {}
    for name in names:
        original = getattr(module, name)
        originals[name] = original

        def wrapper(*args, _name=name, _original=original, **kwargs):
            print(f"BACKWARD_STAGE_BEGIN {_name}", flush=True)
            result = _original(*args, **kwargs)
            torch.npu.synchronize()
            print(f"BACKWARD_STAGE_PASS {_name}", flush=True)
            return result

        setattr(module, name, wrapper)
    return module, originals


def _run_once(
    torch,
    chunk_kda,
    case,
    base,
    capture_backward,
    forward_only,
    disable_recompute,
    trace_backward_stages,
):
    q, k, v, g, beta, do = _clone_for_grad(base)
    capture = None
    hook = None
    if capture_backward:
        module, original, captures = _install_backward_contract_capture()
        hook = (module, original)
    trace_hook = (
        _install_backward_stage_trace(torch)
        if trace_backward_stages and not forward_only
        else None
    )

    try:
        o, final_state = chunk_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=case["K"] ** -0.5,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            use_gate_in_kernel=False,
            cu_seqlens=None,
            cu_seqlens_cpu=None,
            safe_gate=True,
            lower_bound=-5.0,
            disable_recompute=disable_recompute,
            return_intermediate_states=False,
            transpose_state_layout=False,
        )
        if final_state is not None:
            raise AssertionError("final_state must be None when not requested")
        if not forward_only:
            torch.autograd.backward(o, do)
        torch.npu.synchronize()
        if forward_only:
            if not torch.isfinite(o).all().item():
                raise AssertionError("o contains NaN or Inf")
            return {"o": o.detach()}, None
        if capture_backward:
            if len(captures) != 1:
                raise AssertionError(
                    f"expected one chunk_kda_bwd_intra call, got {len(captures)}"
                )
            capture = captures[0]
        outputs = {
            "o": o.detach(),
            "dq": q.grad.detach(),
            "dk": k.grad.detach(),
            "dv": v.grad.detach(),
            "dg": g.grad.detach(),
            "dbeta": beta.grad.detach(),
        }
        for name, tensor in outputs.items():
            if not torch.isfinite(tensor).all().item():
                raise AssertionError(f"{name} contains NaN or Inf")
        return outputs, capture
    finally:
        if trace_hook is not None:
            module, originals = trace_hook
            for name, original in originals.items():
                setattr(module, name, original)
        if hook is not None:
            module, original = hook
            module.chunk_kda_bwd_intra = original


def _assert_model_backward_contract(capture, case):
    B, T, H, K = (case[name] for name in ("B", "T", "H", "K"))
    BT = case["chunk_size"]
    expected = {
        "q": ([B, T, H, K], "bfloat16"),
        "k": ([B, T, H, K], "bfloat16"),
        "g": ([B, T, H, K], "float32"),
        "beta": ([B, T, H], "bfloat16"),
        "dAqk": ([B, T, H, BT], "float32"),
        "dAkk": ([B, T, H, BT], "float32"),
        "dq": ([B, T, H, K], "float32"),
        "dk": ([B, T, H, K], "float32"),
        "db": ([B, T, H], "float32"),
        "dg": ([B, T, H, K], "float32"),
    }
    for name, (shape, dtype) in expected.items():
        actual = capture["inputs"][name]
        if actual != {"shape": shape, "dtype": dtype}:
            raise AssertionError(
                f"{name} contract mismatch: expected shape={shape}, dtype={dtype}; "
                f"got {actual}"
            )


def _comparison_metrics(torch, actual, expected):
    actual = actual.float()
    expected = expected.float()
    diff = actual - expected
    norm = torch.linalg.vector_norm(expected)
    relative_l2 = (
        torch.linalg.vector_norm(diff) / norm
        if norm.item() != 0
        else torch.linalg.vector_norm(diff)
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual.reshape(1, -1),
        expected.reshape(1, -1),
        dim=1,
    )
    return {
        "max_abs": float(diff.abs().max().item()),
        "mean_abs": float(diff.abs().mean().item()),
        "relative_l2": float(relative_l2.item()),
        "cosine": float(cosine.item()),
    }


def _assert_binary_equal(torch, actual, expected, label):
    for name in expected:
        if not torch.equal(actual[name], expected[name]):
            raise AssertionError(f"{label}: {name} is not binary identical")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES), default="smoke")
    parser.add_argument(
        "--backend",
        choices=("ascendc", "triton", "compare"),
        default="ascendc",
    )
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--forward-only",
        action="store_true",
        help="validate the adapted forward without compiling upstream backward kernels",
    )
    parser.add_argument("--disable-recompute", action="store_true")
    parser.add_argument("--trace-backward-stages", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args()

    import torch
    from fla_npu.adapters import (
        install_triton_ascend_kda_adapter,
        remove_triton_ascend_kda_adapter,
    )

    # The pinned upstream package eagerly imports an unrelated attention
    # module through a legacy Triton extra-module name. Installing once first
    # applies the adapter's import compatibility bridge; backend selection
    # below still decides which KDA forward is exercised.
    install_triton_ascend_kda_adapter()
    from triton_ascend_kernels.attention.fla.kda import chunk_kda

    if args.runs < 1:
        raise ValueError("--runs must be positive")
    actual_triton_version = importlib.metadata.version("triton-ascend")
    if actual_triton_version != TRITON_ASCEND_VERSION:
        raise RuntimeError(
            "Triton-Ascend KDA validation requires "
            f"triton-ascend=={TRITON_ASCEND_VERSION}, got "
            f"{actual_triton_version}."
        )
    device = torch.device("npu")
    case = CASES[args.case]
    base = _make_inputs(torch, case, device, args.seed)
    result = {
        "case": args.case,
        "backend": args.backend,
        "shape": case,
        "runs": args.runs,
        "seed": args.seed,
        "forward_only": args.forward_only,
        "disable_recompute": args.disable_recompute,
        "triton_ascend_version": actual_triton_version,
        "triton_ascend_kernels_commit": TRITON_ASCEND_KERNELS_COMMIT,
    }

    if args.backend == "compare":
        if args.case != "smoke":
            raise ValueError("compare mode is intentionally limited to the smoke case")
        remove_triton_ascend_kda_adapter()
        triton_outputs, _ = _run_once(
            torch,
            chunk_kda,
            case,
            base,
            capture_backward=False,
            forward_only=args.forward_only,
            disable_recompute=args.disable_recompute,
            trace_backward_stages=args.trace_backward_stages,
        )
        install_triton_ascend_kda_adapter()
        ascendc_outputs, capture = _run_once(
            torch,
            chunk_kda,
            case,
            base,
            capture_backward=not args.forward_only,
            forward_only=args.forward_only,
            disable_recompute=args.disable_recompute,
            trace_backward_stages=args.trace_backward_stages,
        )
        metrics = {
            name: _comparison_metrics(
                torch, ascendc_outputs[name], triton_outputs[name]
            )
            for name in triton_outputs
        }
        failures = {
            name: values
            for name, values in metrics.items()
            if values["relative_l2"] > MAX_RELATIVE_L2
            or values["cosine"] < MIN_COSINE
        }
        if failures:
            raise AssertionError(
                "AscendC/Triton comparison exceeded "
                f"relative_l2<={MAX_RELATIVE_L2} or cosine>={MIN_COSINE}: "
                f"{json.dumps(failures, ensure_ascii=False)}"
            )
        result["metrics"] = metrics
        result["accuracy_thresholds"] = {
            "max_relative_l2": MAX_RELATIVE_L2,
            "min_cosine": MIN_COSINE,
        }
        result["backward_contract"] = capture
    else:
        if args.backend == "ascendc":
            install_triton_ascend_kda_adapter()
        else:
            remove_triton_ascend_kda_adapter()
        baseline = None
        captures = []
        for run in range(args.runs):
            outputs, capture = _run_once(
                torch,
                chunk_kda,
                case,
                base,
                capture_backward=(
                    args.backend == "ascendc" and not args.forward_only
                ),
                forward_only=args.forward_only,
                disable_recompute=args.disable_recompute,
                trace_backward_stages=args.trace_backward_stages,
            )
            if baseline is None:
                baseline = outputs
            else:
                _assert_binary_equal(
                    torch, outputs, baseline, f"run {run + 1}/{args.runs}"
                )
            if capture is not None:
                _assert_model_backward_contract(capture, case)
                captures.append(capture)
        result["binary_deterministic"] = True
        if captures:
            result["backward_contract"] = captures[0]

    result["status"] = "PASS"
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
