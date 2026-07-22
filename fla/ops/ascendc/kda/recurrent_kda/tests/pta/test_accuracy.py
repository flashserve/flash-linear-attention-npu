"""Accuracy test for npu_recurrent_kda."""

from __future__ import annotations

import os
import sys

import torch
import torch_npu

from golden import recurrent_kda_golden
from utils import compare_tensors_by_ratio


def _device():
    device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    return torch.device(f"npu:{device_id}")


def make_inputs(*, layout="BSND", batch=2, seq_len=2, h=2, hv=4, kdim=128, vdim=128, seed=0,
                with_initial_state=True):
    torch.manual_seed(seed)
    if layout == "BSND":
        q_shape = (batch, seq_len, h, kdim)
        v_shape = (batch, seq_len, hv, vdim)
        g_shape = (batch, seq_len, hv, kdim)
        beta_shape = (batch, seq_len, hv)
        actual_seq_lengths = [0] + [seq_len] * batch
        seq_num = batch
    elif layout == "TND":
        total_tokens = batch * seq_len
        q_shape = (total_tokens, h, kdim)
        v_shape = (total_tokens, hv, vdim)
        g_shape = (total_tokens, hv, kdim)
        beta_shape = (total_tokens, hv)
        actual_seq_lengths = [0] + [seq_len] * batch
        seq_num = batch
    else:
        raise ValueError(layout)

    q = torch.randn(q_shape, dtype=torch.bfloat16)
    k = torch.randn(q_shape, dtype=torch.bfloat16)
    v = torch.randn(v_shape, dtype=torch.bfloat16)
    g = torch.randn(g_shape, dtype=torch.float32) * 0.5
    beta = torch.randn(beta_shape, dtype=torch.float32)
    initial_state = (
        torch.randn((seq_num, hv, vdim, kdim), dtype=torch.float32) * 0.02
        if with_initial_state else None
    )
    A_log = torch.randn((hv,), dtype=torch.float32) * 0.1
    dt_bias = torch.randn((hv, kdim), dtype=torch.float32) * 0.1
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "actual_seq_lengths": actual_seq_lengths,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "layout": layout,
    }


def run_case(desc, kwargs, op_kwargs, rtol=0.02, atol=0.01):
    print(f"\n=== {desc} ===")
    inp = make_inputs(**kwargs)
    golden = recurrent_kda_golden(**inp, output_final_state=True, **op_kwargs)

    dev = _device()
    torch_npu.npu.set_device(dev)
    from fla_npu.ops.ascendc import recurrent_kda

    call_kwargs = {**op_kwargs, "output_final_state": True, "layout": inp["layout"]}
    call_kwargs["actual_seq_lengths"] = torch.tensor(inp["actual_seq_lengths"], dtype=torch.int64, device=dev)
    initial_state_arg = inp["initial_state"].to(dev) if inp["initial_state"] is not None else None
    out, final_state = recurrent_kda(
        inp["q"].to(dev),
        inp["k"].to(dev),
        inp["v"].to(dev),
        inp["g"].to(dev),
        inp["beta"].to(dev),
        initial_state_arg,
        A_log=inp["A_log"].to(dev) if op_kwargs.get("use_gate_in_kernel", False) else None,
        dt_bias=inp["dt_bias"].to(dev) if op_kwargs.get("use_gate_in_kernel", False) else None,
        **call_kwargs,
    )
    torch_npu.npu.synchronize()

    out_ok = compare_tensors_by_ratio(golden[0], out.cpu(), "out", rtol=rtol, atol=atol)
    state_ok = compare_tensors_by_ratio(golden[1], final_state.cpu(), "final_state", rtol=rtol, atol=atol)
    return out_ok and state_ok


def main():
    results = [
        run_case(
            "BSND raw gate, safe_gate=False, beta sigmoid",
            {"layout": "BSND", "batch": 2, "seq_len": 2, "seed": 1},
            {
                "use_qk_l2norm_in_kernel": True,
                "use_gate_in_kernel": True,
                "use_beta_sigmoid_in_kernel": True,
                "allow_neg_eigval": False,
                "safe_gate": False,
            },
        ),
        run_case(
            "BSND raw gate, safe_gate=True",
            {"layout": "BSND", "batch": 2, "seq_len": 2, "seed": 2},
            {
                "use_qk_l2norm_in_kernel": True,
                "use_gate_in_kernel": True,
                "use_beta_sigmoid_in_kernel": True,
                "allow_neg_eigval": True,
                "safe_gate": True,
                "lower_bound": -4.0,
            },
        ),
        run_case(
            "TND precomputed log gate",
            {"layout": "TND", "batch": 2, "seq_len": 2, "seed": 3, "with_initial_state": False},
            {
                "use_qk_l2norm_in_kernel": False,
                "use_gate_in_kernel": False,
                "use_beta_sigmoid_in_kernel": False,
                "safe_gate": False,
            },
        ),
    ]
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
