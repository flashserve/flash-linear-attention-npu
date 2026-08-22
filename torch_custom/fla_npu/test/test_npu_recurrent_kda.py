# Copyright (c) 2026 Tianjin University, Ltd.

from __future__ import annotations

import os
import pathlib
import sys

import torch
import torch_npu

from fla_npu.ops.ascendc import recurrent_kda  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
REFERENCE_DIR = ROOT / "fla/ops/ascendc/kda/recurrent_kda/tests/pta"
for path in (ROOT, REFERENCE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from recurrent_kda_reference import recurrent_kda_reference  # noqa: E402


def _device():
    device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    device = torch.device(f"npu:{device_id}")
    torch.npu.set_device(device)
    return device


def _make_inputs(*, layout="BSND", batch=2, seq_len=2, h=2, hv=4, kdim=128, vdim=128, seed=0,
                 with_initial_state=True, state_v_first=True, state_dtype=torch.float32):
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

    state_tail = (vdim, kdim) if state_v_first else (kdim, vdim)
    initial_state = (
        torch.randn((seq_num, hv, *state_tail), dtype=state_dtype) * 0.02
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


def _make_non_contiguous_state(initial_state, device):
    """Create a non-contiguous view whose inner [V,K] matrix stays dense.

    Mirrors ``make_non_contiguous_state`` in
    ``fla/ops/ascendc/kda/recurrent_kda/tests/pta/test_accuracy.py``: embed the
    state into a 5-D pool with a guard slot, then slice ``[:, 0]`` so that only
    the outer strides become non-contiguous while ``stride[-1] == 1`` and
    ``stride[-2] == kdim`` remain satisfied (required by tiling validation).
    """
    state = initial_state.to(device)
    pool = torch.full(
        (state.shape[0], 2, *state.shape[1:]),
        7.0,
        dtype=state.dtype,
        device=device,
    )
    view = pool[:, 0]
    view.copy_(state)
    guard = pool[:, 1].clone()
    assert not view.is_contiguous(), "constructed state view must be non-contiguous"
    return view, pool, guard


def _assert_close(name, expected, actual, rtol=0.02, atol=0.01):
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)
    diff = (actual.float() - expected.float()).abs()
    print(f"{name}: PASS max_abs={diff.max().item():.6f}")


def _run_case(desc, input_kwargs, op_kwargs, *, non_contiguous_state=False):
    print(f"\n=== {desc} ===")
    inputs = _make_inputs(**input_kwargs)
    expected = recurrent_kda_reference(**inputs, output_final_state=True, **op_kwargs)

    device = _device()

    call_kwargs = {**op_kwargs, "output_final_state": True, "layout": inputs["layout"]}
    call_kwargs["cu_seqlens"] = torch.tensor(
        inputs["cu_seqlens"], dtype=torch.int64, device=device
    )
    initial_state = inputs["initial_state"]
    state_pool = None
    state_guard = None
    initial_before = None
    if initial_state is None:
        state_npu = None
    elif non_contiguous_state:
        state_npu, state_pool, state_guard = _make_non_contiguous_state(initial_state, device)
        initial_before = state_npu.clone()
    else:
        state_npu = initial_state.to(device)

    initial_stride = state_npu.stride() if state_npu is not None else None
    initial_storage = (
        state_npu.untyped_storage().data_ptr() if state_npu is not None else None
    )

    out, final_state = recurrent_kda(
        inputs["q"].to(device),
        inputs["k"].to(device),
        inputs["v"].to(device),
        inputs["g"].to(device),
        inputs["beta"].to(device),
        state_npu,
        A_log=inputs["A_log"].to(device) if op_kwargs.get("use_gate_in_kernel", False) else None,
        dt_bias=inputs["dt_bias"].to(device) if op_kwargs.get("use_gate_in_kernel", False) else None,
        **call_kwargs,
    )
    torch_npu.npu.synchronize()

    _assert_close("out", expected[0], out.cpu())
    _assert_close("final_state", expected[1], final_state.cpu())

    if state_npu is None:
        return

    if non_contiguous_state:
        # The non-contiguous view, its backing storage and the guard slot must
        # all be preserved after the kernel runs.
        assert not state_npu.is_contiguous(), "state view became contiguous"
        assert state_npu.stride() == initial_stride, "state stride changed"
        assert (
            state_npu.untyped_storage().data_ptr() == initial_storage
        ), "state storage was reallocated"
        assert torch.equal(
            state_pool[:, 1].cpu(), state_guard.cpu()
        ), "guard slot was overwritten"
        if op_kwargs.get("inplace_final_state", True):
            # Inplace mode: final_state must alias the same storage and keep the
            # non-contiguous stride.
            assert (
                final_state.untyped_storage().data_ptr() == initial_storage
            ), "inplace final_state did not alias state storage"
            assert final_state.stride() == initial_stride, "inplace final_state stride changed"
        else:
            # Out-of-place mode: initial_state must be left untouched.
            assert torch.equal(
                state_npu.cpu(), initial_before.cpu()
            ), "out-of-place mode mutated initial_state"
        print("non-contiguous state layout: PASS")
    else:
        assert final_state.data_ptr() == state_npu.data_ptr()


def main():
    _run_case(
        "BSND raw gate, safe_gate=False, beta sigmoid",
        {"layout": "BSND", "batch": 2, "seq_len": 2, "seed": 1},
        {
            "use_qk_l2norm_in_kernel": True,
            "use_gate_in_kernel": True,
            "use_beta_sigmoid_in_kernel": True,
            "safe_gate": False,
            "state_v_first": True,
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
            "state_v_first": True,
        },
    )
    _run_case(
        "TND precomputed log gate, zero initial state",
        {"layout": "TND", "batch": 2, "seq_len": 2, "seed": 3, "with_initial_state": False},
        {
            "use_gate_in_kernel": False,
            "use_beta_sigmoid_in_kernel": False,
            "safe_gate": False,
            "inplace_final_state": False,
            "state_v_first": True,
        },
    )
    _run_case(
        "BSND non-contiguous V-first state, inplace",
        {
            "layout": "BSND", "batch": 2, "seq_len": 2, "vdim": 128, "seed": 6,
            "state_v_first": True, "state_dtype": torch.float32,
        },
        {
            "use_gate_in_kernel": False,
            "use_beta_sigmoid_in_kernel": False,
            "inplace_final_state": True,
            "state_v_first": True,
        },
        non_contiguous_state=True,
    )
    _run_case(
        "TND non-contiguous K-first state, out of place",
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
    )


if __name__ == "__main__":
    main()
