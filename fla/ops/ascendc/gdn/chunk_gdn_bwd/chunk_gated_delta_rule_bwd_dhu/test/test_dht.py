"""Exact terminal-gradient regression for #924, independent of the matmul golden."""

import itertools

import pytest
import torch

pytest.importorskip("torch_npu")
from fla_npu.ops import ascendc

pytestmark = [pytest.mark.npu, pytest.mark.skipif(not torch.npu.is_available(), reason="Ascend NPU required")]


def _check_terminal_gradient(dtype, use_gk, varlen, v_first, v_dim, chunk_size, seq_count=3):
    # q=w=dO=dv=gate=0 leaves the state unchanged. One-hot k traces dht into dv2.
    hk, hv, k_dim = 2, 6, 128
    lengths = [chunk_size + 1 + (i % 3) for i in range(seq_count)]
    if not varlen:
        lengths = [lengths[0]] * seq_count
    batch = 1 if varlen else seq_count
    tokens = sum(lengths) if varlen else lengths[0]
    q = torch.zeros(batch, hk, tokens, k_dim, dtype=dtype)
    k = torch.zeros_like(q)
    w = torch.zeros(batch, hv, tokens, k_dim, dtype=dtype)
    do = torch.zeros(batch, hv, tokens, v_dim, dtype=dtype)
    dv = torch.zeros_like(do)
    token_rows = torch.arange(tokens)
    for head in range(hk):
        k[:, head, token_rows, (token_rows * 3 + head * 7) % k_dim] = (head + 1) / 8
    n, h, r, c = torch.meshgrid(torch.arange(seq_count), torch.arange(hv),
                               torch.arange(k_dim), torch.arange(v_dim), indexing="ij")
    state = (((n * 3 + h * 5 + r + c * 3) % 17 - 8).float() / 16).to(dtype)
    boundary = state.transpose(-1, -2).contiguous() if v_first else state
    gate_shape = (batch, hv, tokens, k_dim) if use_gk else (batch, hv, tokens)
    gate = torch.zeros(gate_shape, dtype=torch.float32).npu()
    cu = [0]
    ci = []
    for seq, length in enumerate(lengths):
        cu.append(cu[-1] + length)
        for chunk in range((length + chunk_size - 1) // chunk_size):
            ci.extend((seq, chunk))
    device_inputs = [x.npu() for x in (q, k, w, do, dv)]
    zero_outputs = None
    for mode in ("none", "zero", "nonzero"):
        dht = None if mode == "none" else (torch.zeros_like(boundary) if mode == "zero" else boundary).npu()
        dh, dh0, dv2 = ascendc.chunk_gated_delta_rule_bwd_dhu(
            *device_inputs, 1.0, chunk_size, g=None if use_gk else gate, gK=gate if use_gk else None,
            h0=torch.zeros_like(boundary).npu(), dht=dht,
            cu_seqlens=cu if varlen else None, chunk_indices=ci if varlen else None,
            transpose_state_layout=v_first, use_exp2=use_gk,
        )
        actual = [x.cpu() for x in (dh, dh0, dv2)]
        if mode != "nonzero":
            for value in actual:
                assert torch.count_nonzero(value) == 0
            if zero_outputs is not None:
                for value, previous in zip(actual, zero_outputs):
                    torch.testing.assert_close(value, previous, rtol=0, atol=0)
            zero_outputs = actual
            continue

        expected_dh = torch.empty_like(actual[0])
        expected_dv2 = torch.empty_like(dv)
        chunk_base = 0
        for seq, length in enumerate(lengths):
            b = 0 if varlen else seq
            bos = cu[seq] if varlen else 0
            chunks = (length + chunk_size - 1) // chunk_size
            if varlen:
                expected_dh[0, chunk_base:chunk_base + chunks] = state[seq]
            else:
                expected_dh[seq] = state[seq]
            for head in range(hv):
                key_head = head // (hv // hk)
                rows = (torch.arange(bos, bos + length) * 3 + key_head * 7) % k_dim
                expected_dv2[b, head, bos:bos + length] = state[seq, head, rows] * ((key_head + 1) / 8)
            chunk_base += chunks
        for value, expected in zip(actual, (expected_dh, boundary, expected_dv2)):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,use_gk,varlen,v_first,v_dim,chunk_size", itertools.product(
    (torch.bfloat16, torch.float16), (False, True), (False, True), (False, True), (128, 256), (64, 128)))
def test_dht_terminal_gradient(dtype, use_gk, varlen, v_first, v_dim, chunk_size):
    _check_terminal_gradient(dtype, use_gk, varlen, v_first, v_dim, chunk_size)


@pytest.mark.parametrize("v_first", (False, True))
def test_dht_across_task_reuse(v_first):
    # More sequence/head-window tasks than cores, including a partial head window.
    _check_terminal_gradient(torch.bfloat16, True, True, v_first, 256, 64, seq_count=65)
