# Copyright (c) 2026 Tianjin University, Ltd.

"""A5 regression coverage for staged ChunkKdaFwd launches."""

from __future__ import annotations

import gc
import math

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from fla_npu.ops.ascendc import chunk_kda_fwd  # noqa: E402


OUTPUT_NAMES = (
    "attn_out",
    "final_state",
    "gk",
    "Aqk",
    "Akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
    "initial_state",
)


def _is_ascend950() -> bool:
    try:
        return "950" in torch.npu.get_device_name(0)
    except Exception:
        return False


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x_float = x.float()
    return (x_float * torch.rsqrt(x_float.square().sum(dim=-1, keepdim=True) + 1e-6)).to(dtype)


def _chunk_indices(cu_seqlens: tuple[int, ...], chunk_size: int) -> tuple[int, ...]:
    return tuple(
        value
        for seq_id, (start, end) in enumerate(zip(cu_seqlens, cu_seqlens[1:]))
        for chunk_id in range(math.ceil((end - start) / chunk_size))
        for value in (seq_id, chunk_id)
    )


def _chunked_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    initial_state_vk: torch.Tensor,
    cu_seqlens: tuple[int, ...],
) -> tuple[torch.Tensor, ...]:
    q, k, v, gate, beta, initial_state_vk = (
        tensor.detach().cpu()
        for tensor in (q, k, v, gate, beta, initial_state_vk)
    )
    batch, tokens, heads, head_dim = q.shape
    assert batch == 1
    chunk_size = 64
    total_chunks = sum(
        math.ceil((end - start) / chunk_size)
        for start, end in zip(cu_seqlens, cu_seqlens[1:])
    )
    dtype = q.dtype
    scale = head_dim**-0.5

    out = torch.empty_like(v)
    final_state_vk = torch.empty_like(initial_state_vk, dtype=torch.float32)
    gk = torch.empty((1, heads, tokens, head_dim), dtype=torch.float32)
    aqk = torch.zeros((1, heads, tokens, chunk_size), dtype=dtype)
    akk = torch.zeros_like(aqk)
    w = torch.empty((1, heads, tokens, head_dim), dtype=dtype)
    u = torch.empty_like(w)
    qg = torch.empty_like(w)
    kg = torch.empty_like(w)
    v_new = torch.empty_like(w)
    h = torch.empty((1, total_chunks, heads, head_dim, head_dim), dtype=dtype)

    global_chunk = 0
    for seq_id, (seq_start, seq_end) in enumerate(zip(cu_seqlens, cu_seqlens[1:])):
        state_kv = initial_state_vk[seq_id].float().transpose(-1, -2).contiguous()
        chunk_offset = 0
        for start in range(seq_start, seq_end, chunk_size):
            end = min(start + chunk_size, seq_end)
            length = end - start
            causal = torch.ones((length, length), dtype=torch.bool).tril()
            strict_causal = torch.ones_like(causal).tril(diagonal=-1)
            eye = torch.eye(length, dtype=torch.float32).expand(heads, -1, -1)
            q_block = q[0, start:end].float().transpose(0, 1)
            k_block = k[0, start:end].float().transpose(0, 1)
            v_block = v[0, start:end].float().transpose(0, 1)
            beta_block = beta[0, start:end].float().transpose(0, 1)
            gk_block = (
                torch.cumsum(gate[0, start:end].float(), dim=0) / math.log(2.0)
            ).transpose(0, 1)
            relative_gate = gk_block[:, :, None, :] - gk_block[:, None, :, :]
            gate_factor = torch.exp2(
                relative_gate.masked_fill(~causal[None, :, :, None], 0.0)
            )
            qk = torch.einsum("hik,hjk,hijk->hij", q_block, k_block, gate_factor) * scale
            kk = torch.einsum("hik,hjk,hijk->hij", k_block, k_block, gate_factor)
            aqk_block = torch.where(causal[None], qk, 0.0)
            strict_kk = torch.where(strict_causal[None], kk * beta_block[:, :, None], 0.0)
            akk_block = torch.linalg.solve_triangular(eye + strict_kk, eye, upper=False)
            w_block = torch.bmm(
                akk_block,
                k_block * beta_block[:, :, None] * torch.exp2(gk_block),
            )
            u_block = torch.bmm(akk_block, v_block * beta_block[:, :, None])
            qg_block = q_block * torch.exp2(gk_block)
            kg_block = k_block * torch.exp2(gk_block[:, -1, None, :] - gk_block)
            v_new_block = u_block - torch.bmm(w_block, state_kv)

            gk[0, :, start:end] = gk_block
            aqk[0, :, start:end, :length] = aqk_block.to(dtype)
            akk[0, :, start:end, :length] = akk_block.to(dtype)
            w[0, :, start:end] = w_block.to(dtype)
            u[0, :, start:end] = u_block.to(dtype)
            qg[0, :, start:end] = qg_block.to(dtype)
            kg[0, :, start:end] = kg_block.to(dtype)
            v_new[0, :, start:end] = v_new_block.to(dtype)
            h[0, global_chunk + chunk_offset] = state_kv.transpose(-1, -2).to(dtype)
            out[0, start:end] = (
                torch.bmm(qg_block, state_kv) * scale
                + torch.bmm(aqk_block, v_new_block)
            ).transpose(0, 1).to(dtype)
            state_kv = (
                torch.exp2(gk_block[:, -1])[:, :, None] * state_kv
                + torch.bmm(kg_block.transpose(1, 2), v_new_block)
            )
            chunk_offset += 1
        final_state_vk[seq_id] = state_kv.transpose(-1, -2)
        global_chunk += chunk_offset

    return (
        out,
        final_state_vk,
        gk,
        aqk,
        akk,
        w,
        u,
        qg,
        kg,
        v_new,
        h,
        initial_state_vk,
    )


def _for_layout(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    if layout == "BSND":
        return tensor.contiguous()
    assert layout == "TND"
    return tensor.squeeze(0).contiguous()


def _expected_for_layout(outputs: tuple[torch.Tensor, ...], layout: str) -> tuple[torch.Tensor, ...]:
    if layout == "BSND":
        return outputs
    rank3_indices = {0, 2, 3, 4, 5, 6, 7, 8, 9, 10}
    return tuple(output.squeeze(0) if index in rank3_indices else output for index, output in enumerate(outputs))


def _expected_output_shapes(
    layout: str,
    tokens: int,
    heads: int,
    head_dim: int,
    cu_seqlens: tuple[int, ...],
) -> tuple[tuple[int, ...], ...]:
    seq_num = len(cu_seqlens) - 1
    total_chunks = sum(
        math.ceil((end - start) / 64)
        for start, end in zip(cu_seqlens, cu_seqlens[1:])
    )
    sequence_shape = (1, tokens, heads, head_dim)
    head_shape = (1, heads, tokens, head_dim)
    matrix_shape = (1, heads, tokens, 64)
    state_shape = (seq_num, heads, head_dim, head_dim)
    chunk_state_shape = (1, total_chunks, heads, head_dim, head_dim)
    if layout == "TND":
        sequence_shape = sequence_shape[1:]
        head_shape = head_shape[1:]
        matrix_shape = matrix_shape[1:]
        chunk_state_shape = chunk_state_shape[1:]
    return (
        sequence_shape,
        state_shape,
        head_shape,
        matrix_shape,
        matrix_shape,
        head_shape,
        head_shape,
        head_shape,
        head_shape,
        head_shape,
        chunk_state_shape,
        state_shape,
    )


def _assert_outputs(actual, expected, retained: set[int]) -> None:
    assert len(actual) == len(expected) == len(OUTPUT_NAMES)
    for index, (name, expected_output) in enumerate(zip(OUTPUT_NAMES, expected)):
        if index not in retained:
            assert actual[index] is None, f"{name} must not be retained"
            continue
        assert actual[index] is not None, f"{name} must be retained"
        torch.testing.assert_close(
            actual[index].detach().cpu(),
            expected_output,
            rtol=3e-2,
            atol=3e-2,
            msg=name,
        )


@pytest.mark.parametrize(
    ("layout", "tokens", "heads", "cu_seqlens"),
    [
        pytest.param("BSND", 131, 6, (0, 131), id="BSND-T131-H6"),
        pytest.param("TND", 191, 6, (0, 63, 131, 191), id="TND-varlen"),
    ],
)
@torch.inference_mode()
def test_chunk_kda_fwd_a5_multichunk_all_outputs(layout, tokens, heads, cu_seqlens):
    if not _is_ascend950():
        pytest.skip("requires an Ascend 950 device")

    torch.manual_seed(20260819)
    device = torch.device("npu:0")
    torch.npu.set_device(0)
    head_dim = 128
    q_bsnd = _l2norm(torch.randn(1, tokens, heads, head_dim, dtype=torch.bfloat16, device=device))
    k_bsnd = _l2norm(torch.randn_like(q_bsnd))
    v_bsnd = torch.randn_like(q_bsnd) * 0.05
    raw_gate_bsnd = torch.randn(
        1, tokens, heads, head_dim, dtype=torch.float32, device=device
    ) * 0.1
    beta_bsnd = torch.rand(1, tokens, heads, dtype=torch.float32, device=device).sigmoid()
    a_log = torch.randn(heads, dtype=torch.float32, device=device) * 0.05
    dt_bias = torch.randn(heads * head_dim, dtype=torch.float32, device=device) * 0.05
    initial_state_vk = torch.randn(
        len(cu_seqlens) - 1,
        heads,
        head_dim,
        head_dim,
        dtype=torch.float32,
        device=device,
    ) * 0.01
    indices = _chunk_indices(cu_seqlens, 64)
    args = (
        _for_layout(q_bsnd, layout),
        _for_layout(k_bsnd, layout),
        _for_layout(v_bsnd, layout).contiguous(),
        _for_layout(raw_gate_bsnd, layout).contiguous(),
        _for_layout(beta_bsnd, layout).contiguous(),
        head_dim**-0.5,
        64,
    )
    kwargs = {
        "layout": layout,
        "initial_state": initial_state_vk,
        "output_final_state": True,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": indices,
        "safe_gate": True,
        "lower_bound": -5.0,
        "use_gate_in_kernel": True,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "state_v_first": True,
    }

    retained_outputs = chunk_kda_fwd(
        *args,
        **kwargs,
        disable_recompute=True,
        return_intermediate_states=True,
    )
    torch.npu.synchronize()

    safe_gate = -5.0 * torch.sigmoid(
        (raw_gate_bsnd.float() + dt_bias.view(1, 1, heads, head_dim))
        * a_log.exp().view(1, 1, heads, 1)
    )
    expected = _expected_for_layout(
        _chunked_reference(
            q_bsnd,
            k_bsnd,
            v_bsnd,
            safe_gate,
            beta_bsnd,
            initial_state_vk,
            cu_seqlens,
        ),
        layout,
    )
    _assert_outputs(retained_outputs, expected, set(range(len(OUTPUT_NAMES))))
    assert retained_outputs[11] is initial_state_vk

    del retained_outputs, expected
    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.parametrize(
    ("layout", "cu_seqlens"),
    [
        pytest.param("BSND", (0, 8191), id="BSND-model-shape"),
        pytest.param("TND", (0, 2047, 4096, 8191), id="TND-varlen-model-shape"),
    ],
)
@torch.inference_mode()
def test_chunk_kda_fwd_a5_t8191_all_outputs_are_finite(layout, cu_seqlens):
    if not _is_ascend950():
        pytest.skip("requires an Ascend 950 device")

    device = torch.device("npu:0")
    torch.npu.set_device(0)
    tokens, heads, head_dim = 8191, 12, 128
    q_bsnd = torch.full(
        (1, tokens, heads, head_dim),
        head_dim**-0.5,
        dtype=torch.bfloat16,
        device=device,
    )
    k_bsnd = torch.full_like(q_bsnd, head_dim**-0.5)
    v_bsnd = torch.zeros_like(q_bsnd)
    raw_gate_bsnd = torch.zeros(
        (1, tokens, heads, head_dim), dtype=torch.float32, device=device
    )
    beta_bsnd = torch.full(
        (1, tokens, heads), 0.5, dtype=torch.float32, device=device
    )
    initial_state_vk = torch.zeros(
        (len(cu_seqlens) - 1, heads, head_dim, head_dim),
        dtype=torch.float32,
        device=device,
    )
    outputs = chunk_kda_fwd(
        _for_layout(q_bsnd, layout),
        _for_layout(k_bsnd, layout),
        _for_layout(v_bsnd, layout).contiguous(),
        _for_layout(raw_gate_bsnd, layout).contiguous(),
        _for_layout(beta_bsnd, layout).contiguous(),
        head_dim**-0.5,
        64,
        layout=layout,
        initial_state=initial_state_vk,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=_chunk_indices(cu_seqlens, 64),
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=True,
        A_log=torch.zeros(heads, dtype=torch.float32, device=device),
        dt_bias=torch.zeros(heads * head_dim, dtype=torch.float32, device=device),
        disable_recompute=True,
        return_intermediate_states=True,
        state_v_first=True,
    )
    torch.npu.synchronize()

    expected_shapes = _expected_output_shapes(
        layout, tokens, heads, head_dim, cu_seqlens
    )
    assert len(outputs) == len(OUTPUT_NAMES) == len(expected_shapes)
    for name, output, shape in zip(OUTPUT_NAMES, outputs, expected_shapes):
        assert output is not None, f"{name} must be retained"
        assert tuple(output.shape) == shape, name
    finite = torch.stack([torch.isfinite(output).all() for output in outputs])
    assert finite.all().item()
    assert outputs[11] is initial_state_vk

    del outputs
    gc.collect()
    torch.npu.empty_cache()
