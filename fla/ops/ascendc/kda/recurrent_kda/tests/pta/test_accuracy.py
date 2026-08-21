"""Accuracy test for npu_recurrent_kda."""

from __future__ import annotations

import os
import sys

import torch
import torch_npu

from fla_npu.ops.ascendc import recurrent_kda
from recurrent_kda_reference import recurrent_kda_reference as recurrent_kda_golden
from utils import compare_tensors_by_ratio


def _device():
    device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    return torch.device(f"npu:{device_id}")


def make_inputs(*, layout="BSND", batch=2, seq_len=2, h=2, hv=4, kdim=128, vdim=128, seed=0,
                with_initial_state=True, gate_dtype=torch.float32, beta_dtype=torch.float32,
                state_v_first=True, state_dtype=torch.float32, state_capacity=None,
                state_slots=None):
    torch.manual_seed(seed)
    if layout == "BSND":
        q_shape = (batch, seq_len, h, kdim)
        v_shape = (batch, seq_len, hv, vdim)
        g_shape = (batch, seq_len, hv, kdim)
        beta_shape = (batch, seq_len, hv)
        cu_seqlens = [seq_len * i for i in range(batch + 1)]
        seq_num = batch
    elif layout == "TND":
        total_tokens = batch * seq_len
        q_shape = (total_tokens, h, kdim)
        v_shape = (total_tokens, hv, vdim)
        g_shape = (total_tokens, hv, kdim)
        beta_shape = (total_tokens, hv)
        cu_seqlens = [seq_len * i for i in range(batch + 1)]
        seq_num = batch
    else:
        raise ValueError(layout)

    q = torch.randn(q_shape, dtype=torch.bfloat16)
    k = torch.randn(q_shape, dtype=torch.bfloat16)
    v = torch.randn(v_shape, dtype=torch.bfloat16)
    g = torch.randn(g_shape, dtype=gate_dtype) * 0.5
    beta = torch.randn(beta_shape, dtype=beta_dtype)
    state_tail = (vdim, kdim) if state_v_first else (kdim, vdim)
    state_capacity = seq_num if state_capacity is None else state_capacity
    initial_state = (
        torch.randn((state_capacity, hv, *state_tail), dtype=state_dtype) * 0.02
        if with_initial_state else None
    )
    ssm_state_indices = None
    if state_slots is not None:
        if len(state_slots) != seq_num:
            raise ValueError("state_slots must contain one slot per sequence")
        packed_slots = []
        for slot, (start, end) in zip(state_slots, zip(cu_seqlens, cu_seqlens[1:])):
            packed_slots.extend([slot] * (end - start))
        ssm_state_indices = torch.tensor(packed_slots, dtype=torch.int64)
    A_log = torch.randn((hv,), dtype=torch.float32) * 0.1
    dt_bias = torch.randn((hv, kdim), dtype=torch.float32) * 0.1
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "A_log": A_log,
        "dt_bias": dt_bias,
        "layout": layout,
    }


def make_non_contiguous_state(initial_state, dev):
    state = initial_state.to(dev)
    pool = torch.full(
        (state.shape[0], 2, *state.shape[1:]),
        7.0,
        dtype=state.dtype,
        device=dev,
    )
    view = pool[:, 0]
    view.copy_(state)
    guard = pool[:, 1].clone()
    if view.is_contiguous():
        raise AssertionError("constructed state view must be non-contiguous")
    return view, pool, guard


def run_case(desc, kwargs, op_kwargs, rtol=0.02, atol=0.01, metadata_dtype=torch.int64,
             use_cu_seqlens=True, non_contiguous_state=False):
    print(f"\n=== {desc} ===")
    inp = make_inputs(**kwargs)
    golden = recurrent_kda_golden(**inp, output_final_state=True, **op_kwargs)

    dev = _device()
    torch_npu.npu.set_device(dev)

    call_kwargs = {**op_kwargs, "output_final_state": True, "layout": inp["layout"]}
    call_kwargs["cu_seqlens"] = (
        torch.tensor(inp["cu_seqlens"], dtype=metadata_dtype, device=dev)
        if use_cu_seqlens else None
    )
    call_kwargs["ssm_state_indices"] = (
        inp["ssm_state_indices"].to(device=dev, dtype=metadata_dtype)
        if inp["ssm_state_indices"] is not None else None
    )
    state_pool = None
    state_guard = None
    if inp["initial_state"] is None:
        initial_state_arg = None
    elif non_contiguous_state:
        initial_state_arg, state_pool, state_guard = make_non_contiguous_state(inp["initial_state"], dev)
    else:
        initial_state_arg = inp["initial_state"].to(dev)
    initial_stride = initial_state_arg.stride() if initial_state_arg is not None else None
    initial_storage = initial_state_arg.untyped_storage().data_ptr() if initial_state_arg is not None else None
    initial_before = initial_state_arg.clone() if initial_state_arg is not None else None
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
    layout_ok = True
    if non_contiguous_state:
        layout_ok = (
            not initial_state_arg.is_contiguous()
            and initial_state_arg.stride() == initial_stride
            and initial_state_arg.untyped_storage().data_ptr() == initial_storage
            and torch.equal(state_pool[:, 1].cpu(), state_guard.cpu())
        )
        if op_kwargs.get("inplace_final_state", True):
            layout_ok = layout_ok and (
                final_state.untyped_storage().data_ptr() == initial_storage
                and final_state.stride() == initial_stride
            )
        else:
            layout_ok = layout_ok and torch.equal(initial_state_arg.cpu(), initial_before.cpu())
        if inp["ssm_state_indices"] is not None:
            used_slots = set(inp["ssm_state_indices"].tolist())
            untouched_slots = [
                slot for slot in range(initial_state_arg.shape[0]) if slot not in used_slots
            ]
            state_after = initial_state_arg if op_kwargs.get("inplace_final_state", True) else final_state
            layout_ok = layout_ok and torch.equal(
                state_after[untouched_slots].cpu(),
                initial_before[untouched_slots].cpu(),
            )
        if not layout_ok:
            print("state view/storage/guard validation failed")
    return out_ok and state_ok and layout_ok


def run_model_case():
    print("\n=== BSND model case with 470-slot state pool and accepted tokens ===")
    torch.manual_seed(41)
    batch, total_tokens = 1, 8
    key_heads = value_heads = 12
    key_dim = value_dim = 128
    state_capacity = 470

    q = torch.randn(
        (batch, total_tokens, key_heads, key_dim), dtype=torch.bfloat16
    )
    k = torch.randn(
        (batch, total_tokens, key_heads, key_dim), dtype=torch.bfloat16
    )
    v = torch.randn(
        (batch, total_tokens, value_heads, value_dim), dtype=torch.bfloat16
    )
    g = torch.randn(
        (batch, total_tokens, value_heads, key_dim), dtype=torch.float32
    ) * 0.5
    beta = torch.randn(
        (batch, total_tokens, value_heads), dtype=torch.float32
    )
    initial_state = torch.randn(
        (state_capacity, value_heads, value_dim, key_dim), dtype=torch.float32
    ) * 0.02
    cu_seqlens = list(range(total_tokens + 1))
    ssm_state_indices = torch.arange(total_tokens, dtype=torch.int64)
    num_accepted_tokens = torch.ones(total_tokens, dtype=torch.int64)
    a_log = torch.randn((value_heads,), dtype=torch.float32) * 0.1
    dt_bias = torch.randn((value_heads * key_dim,), dtype=torch.float32) * 0.1
    op_kwargs = {
        "layout": "BSND",
        "output_final_state": False,
        "use_qk_l2norm_in_kernel": False,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": False,
        "allow_neg_eigval": False,
        "safe_gate": False,
        "lower_bound": -5.0,
        "state_v_first": True,
    }

    golden = recurrent_kda_golden(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        A_log=a_log,
        dt_bias=dt_bias,
        **op_kwargs,
    )

    dev = _device()
    torch_npu.npu.set_device(dev)
    initial_state_npu = initial_state.to(dev)
    out, final_state = recurrent_kda(
        q.to(dev),
        k.to(dev),
        v.to(dev),
        g.to(dev),
        beta.to(dev),
        initial_state_npu,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int64, device=dev),
        ssm_state_indices=ssm_state_indices.to(dev),
        num_accepted_tokens=num_accepted_tokens.to(dev),
        A_log=a_log.to(dev),
        dt_bias=dt_bias.to(dev),
        **op_kwargs,
    )
    torch_npu.npu.synchronize()

    out_ok = compare_tensors_by_ratio(golden[0], out.cpu(), "out", rtol=0.02, atol=0.01)
    state_ok = compare_tensors_by_ratio(
        golden[1], initial_state_npu.cpu(), "inplace_state", rtol=0.02, atol=0.01
    )
    if final_state is not None:
        print("  [final_state] expected None when output_final_state=False")
        return False
    return out_ok and state_ok


def run_invalid_state_stride_case(desc, stride_kind):
    print(f"\n=== {desc} ===")
    inp = make_inputs(layout="BSND", batch=2, seq_len=2, seed=31)
    dev = _device()
    torch_npu.npu.set_device(dev)
    dense_state = inp["initial_state"].to(dev)
    if stride_kind == "inner":
        backing = torch.empty(
            (*dense_state.shape[:-1], dense_state.shape[-1] * 2),
            dtype=dense_state.dtype,
            device=dev,
        )
        invalid_state = backing[..., ::2]
        invalid_state.copy_(dense_state)
        if invalid_state.stride()[-1] == 1:
            raise AssertionError("inner-stride case was not constructed correctly")
    elif stride_kind == "overlap":
        invalid_state = torch.as_strided(
            dense_state,
            size=dense_state.shape,
            stride=(1, *dense_state.stride()[1:]),
        )
        if invalid_state.stride()[0] <= 0:
            raise AssertionError("overlapping-stride case was not constructed correctly")
    else:
        raise ValueError(stride_kind)

    try:
        recurrent_kda(
            inp["q"].to(dev),
            inp["k"].to(dev),
            inp["v"].to(dev),
            inp["g"].to(dev),
            inp["beta"].to(dev),
            invalid_state,
            cu_seqlens=torch.tensor(inp["cu_seqlens"], dtype=torch.int64, device=dev),
            output_final_state=True,
            inplace_final_state=True,
            state_v_first=True,
            layout=inp["layout"],
        )
        torch_npu.npu.synchronize()
    except RuntimeError as exc:
        print(f"expected host validation error: {exc}")
        return True
    print("invalid state stride was not rejected")
    return False


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
                "state_v_first": True,
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
                "state_v_first": True,
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
                "inplace_final_state": False,
                "state_v_first": True,
            },
        ),
        run_case(
            "BSND FP16 gate, BF16 beta, INT32 metadata",
            {
                "layout": "BSND", "batch": 2, "seq_len": 2, "vdim": 256, "seed": 4,
                "gate_dtype": torch.float16, "beta_dtype": torch.bfloat16,
            },
            {
                "use_gate_in_kernel": False,
                "use_beta_sigmoid_in_kernel": True,
                "state_v_first": True,
            },
            metadata_dtype=torch.int32,
        ),
        run_case(
            "BSND BF16 gate, FP16 beta, dense K-first state",
            {
                "layout": "BSND", "batch": 2, "seq_len": 2, "vdim": 256, "seed": 5,
                "gate_dtype": torch.bfloat16, "beta_dtype": torch.float16,
                "state_v_first": False,
            },
            {
                "use_gate_in_kernel": False,
                "use_beta_sigmoid_in_kernel": False,
                "state_v_first": False,
            },
            use_cu_seqlens=False,
        ),
        run_case(
            "BSND FP32 non-contiguous V-first state pool, inplace with state indices",
            {
                "layout": "BSND", "batch": 2, "seq_len": 2, "vdim": 128, "seed": 6,
                "state_v_first": True, "state_dtype": torch.float32,
                "state_capacity": 3, "state_slots": [2, 0],
            },
            {
                "use_gate_in_kernel": False,
                "use_beta_sigmoid_in_kernel": False,
                "inplace_final_state": True,
                "state_v_first": True,
            },
            non_contiguous_state=True,
        ),
        run_case(
            "TND BF16 non-contiguous K-first state, out of place",
            {
                "layout": "TND", "batch": 2, "seq_len": 2, "vdim": 256, "seed": 7,
                "state_v_first": False, "state_dtype": torch.bfloat16,
            },
            {
                "use_gate_in_kernel": False,
                "use_beta_sigmoid_in_kernel": False,
                "inplace_final_state": False,
                "state_v_first": False,
            },
            non_contiguous_state=True,
        ),
        run_model_case(),
        run_invalid_state_stride_case(
            "reject non-dense inner state matrix",
            "inner",
        ),
        run_invalid_state_stride_case(
            "reject overlapping outer state stride",
            "overlap",
        ),
    ]
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
