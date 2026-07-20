# Copyright (c) 2026 Tianjin University, Ltd.

from __future__ import annotations

import os
import pathlib
import sys

import torch
import torch_npu


ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.reference.recurrent_kda_reference import recurrent_kda_reference  # noqa: E402


def _device():
    device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    return device


def _make_inputs(*, layout="BSND", batch=2, seq_len=2, h=2, hv=4, kdim=64, vdim=64, seed=0,
                 with_initial_state=True):
    torch.manual_seed(seed)
    if layout == "BSND":
        q_shape = (batch, seq_len, h, kdim)
        v_shape = (batch, seq_len, hv, vdim)
        g_shape = (batch, seq_len, hv, kdim)
        beta_shape = (batch, seq_len, hv)
        cu_seqlens = None
        seq_num = batch
    elif layout == "TND":
        total_tokens = batch * seq_len
        q_shape = (total_tokens, h, kdim)
        v_shape = (total_tokens, hv, vdim)
        g_shape = (total_tokens, hv, kdim)
        beta_shape = (total_tokens, hv)
        cu_seqlens = [i * seq_len for i in range(batch + 1)]
        seq_num = batch
    else:
        raise ValueError(layout)

    initial_state = (
        torch.randn((seq_num, hv, vdim, kdim), dtype=torch.float32) * 0.02
        if with_initial_state else None
    )
    return {
        "q": torch.randn(q_shape, dtype=torch.bfloat16),
        "k": torch.randn(q_shape, dtype=torch.bfloat16),
        "v": torch.randn(v_shape, dtype=torch.bfloat16),
        "g": torch.randn(g_shape, dtype=torch.float32) * 0.5,
        "beta": torch.randn(beta_shape, dtype=torch.float32),
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "A_log": torch.randn((hv,), dtype=torch.float32) * 0.1,
        "dt_bias": torch.randn((hv, kdim), dtype=torch.float32) * 0.1,
        "layout": layout,
    }


def _assert_close(name, expected, actual, rtol=0.02, atol=0.01):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)
    diff = (actual.float() - expected.float()).abs()
    print(f"{name}: PASS max_abs={diff.max().item():.6f}")


def _run_case(desc, input_kwargs, op_kwargs):
    print(f"\n=== {desc} ===")
    inputs = _make_inputs(**input_kwargs)
    expected = recurrent_kda_reference(**inputs, output_final_state=True, **op_kwargs)

    device = _device()
    from fla_npu.ops.ascendc import recurrent_kda

    call_kwargs = {**op_kwargs, "output_final_state": True, "layout": inputs["layout"]}
    if inputs["cu_seqlens"] is not None:
        call_kwargs["cu_seqlens"] = inputs["cu_seqlens"]
    initial_state = inputs["initial_state"]
    out, final_state = recurrent_kda(
        inputs["q"].to(device),
        inputs["k"].to(device),
        inputs["v"].to(device),
        inputs["g"].to(device),
        inputs["beta"].to(device),
        initial_state.to(device) if initial_state is not None else None,
        A_log=inputs["A_log"].to(device) if op_kwargs.get("use_gate_in_kernel", False) else None,
        dt_bias=inputs["dt_bias"].to(device) if op_kwargs.get("use_gate_in_kernel", False) else None,
        **call_kwargs,
    )
    torch_npu.npu.synchronize()

    _assert_close("out", expected[0], out.cpu())
    _assert_close("final_state", expected[1], final_state.cpu())


def main():
    _run_case(
        "BSND raw gate, safe_gate=False, beta sigmoid",
        {"layout": "BSND", "batch": 2, "seq_len": 2, "seed": 1},
        {
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "use_beta_sigmoid_in_kernel": True,
            "safe_gate": False,
        },
    )
    _run_case(
        "BSND raw gate, safe_gate=True, allow_neg_eigval=True",
        {"layout": "BSND", "batch": 2, "seq_len": 2, "seed": 2},
        {
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "use_beta_sigmoid_in_kernel": True,
            "allow_neg_eigval": True,
            "safe_gate": True,
            "lower_bound": -4.0,
        },
    )
    _run_case(
        "TND precomputed log gate, zero initial state",
        {"layout": "TND", "batch": 2, "seq_len": 2, "seed": 3, "with_initial_state": False},
        {
            "use_gate_in_kernel": False,
            "use_beta_sigmoid_in_kernel": False,
            "safe_gate": False,
        },
    )


if __name__ == "__main__":
    main()
