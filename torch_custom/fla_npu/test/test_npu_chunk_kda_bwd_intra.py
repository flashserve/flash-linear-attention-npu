import os

import pytest
import torch


def _chunk_kda_bwd_intra_reference_bsnd(
    q,
    k,
    gk,
    beta,
    d_aqk,
    d_akk,
    dq,
    dk,
    db,
    dg,
    *,
    chunk_size=64,
):
    if chunk_size != 64:
        raise ValueError("reference scope is chunk_size=64")
    dq_out = dq.clone()
    dk_out = dk.clone()
    db_out = db.clone()
    dg_out = dg.clone()
    batch, seqlen, heads, _ = q.shape
    for batch_idx in range(batch):
        for chunk_begin in range(0, seqlen, chunk_size):
            chunk_end = min(chunk_begin + chunk_size, seqlen)
            valid = chunk_end - chunk_begin
            for head in range(heads):
                q_chunk = q[batch_idx, chunk_begin:chunk_end, head].float()
                k_chunk = k[batch_idx, chunk_begin:chunk_end, head].float()
                g_chunk = gk[batch_idx, chunk_begin:chunk_end, head]
                b_chunk = beta[batch_idx, chunk_begin:chunk_end, head].float()
                aqk = torch.tril(
                    d_aqk[batch_idx, chunk_begin:chunk_end, head, :valid]
                )
                akk = torch.tril(
                    d_akk[batch_idx, chunk_begin:chunk_end, head, :valid]
                )

                for row_begin in range(0, valid, 16):
                    row_end = min(row_begin + 16, valid)
                    anchor = min(row_begin + 8, row_end - 1)
                    c = g_chunk[anchor]
                    g_rows = g_chunk[row_begin:row_end]

                    lower_weight = torch.exp2(c - g_chunk[:row_end])
                    lower_b = k_chunk[:row_end] * lower_weight
                    dq_delta = aqk[row_begin:row_end, :row_end] @ lower_b
                    dk_lower = akk[row_begin:row_end, :row_end] @ lower_b
                    row_scale = torch.exp2(g_rows - c)
                    dq_delta *= row_scale
                    dk_lower *= row_scale

                    future_weight = torch.exp2(g_chunk[row_begin:] - c)
                    upper_q = q_chunk[row_begin:] * future_weight
                    upper_k = (
                        k_chunk[row_begin:]
                        * b_chunk[row_begin:, None]
                        * future_weight
                    )
                    dk_upper = (
                        aqk[row_begin:, row_begin:row_end].T @ upper_q
                        + akk[row_begin:, row_begin:row_end].T @ upper_k
                    )
                    dk_upper *= torch.exp2(c - g_rows)

                    out_slice = (
                        batch_idx,
                        slice(chunk_begin + row_begin, chunk_begin + row_end),
                        head,
                    )
                    k_rows = k_chunk[row_begin:row_end]
                    q_rows = q_chunk[row_begin:row_end]
                    beta_dk_lower = b_chunk[row_begin:row_end, None] * dk_lower
                    dq_out[out_slice] += dq_delta
                    dk_out[out_slice] += beta_dk_lower + dk_upper
                    db_out[out_slice] += (dk_lower * k_rows).sum(dim=-1)
                    dg_out[out_slice] += (
                        q_rows * dq_delta
                        + k_rows * (beta_dk_lower - dk_upper)
                    )
    return dq_out, dk_out, db_out, dg_out


def chunk_kda_bwd_intra_reference(
    q,
    k,
    gk,
    beta,
    d_aqk,
    d_akk,
    dq,
    dk,
    db,
    dg,
    *,
    chunk_size=64,
    layout="BSND",
    cu_seqlens=None,
):
    if layout == "TND":
        if cu_seqlens is None:
            raise ValueError("TND reference requires cu_seqlens")
        outputs = [[] for _ in range(4)]
        for begin, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            begin, end = int(begin), int(end)
            if begin == end:
                continue
            sequence_inputs = [
                tensor[begin:end].unsqueeze(0)
                for tensor in (q, k, gk, beta, d_aqk, d_akk, dq, dk, db, dg)
            ]
            sequence_outputs = _chunk_kda_bwd_intra_reference_bsnd(
                *sequence_inputs, chunk_size=chunk_size
            )
            for collected, tensor in zip(outputs, sequence_outputs):
                collected.append(tensor.squeeze(0))
        return tuple(torch.cat(collected, dim=0) for collected in outputs)
    if layout == "BSND":
        return _chunk_kda_bwd_intra_reference_bsnd(
            q, k, gk, beta, d_aqk, d_akk, dq, dk, db, dg,
            chunk_size=chunk_size,
        )
    if layout != "BNSD":
        raise ValueError("reference supports BSND or BNSD")

    def to_bsnd(tensor):
        if tensor.ndim == 4:
            return tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.permute(0, 2, 1).contiguous()

    outputs = _chunk_kda_bwd_intra_reference_bsnd(
        *(to_bsnd(tensor) for tensor in
          (q, k, gk, beta, d_aqk, d_akk, dq, dk, db, dg)),
        chunk_size=chunk_size,
    )
    return tuple(to_bsnd(tensor) for tensor in outputs)


@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("heads", [1, 2, 3])
@pytest.mark.parametrize("seqlen", [1, 15, 16, 17, 64, 65])
@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_reference_is_finite(head_dim, heads, seqlen, beta_dtype):
    torch.manual_seed(1)
    shape = (1, seqlen, heads, head_dim)
    q = (torch.randn(shape, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    k = (torch.randn(shape, dtype=torch.float32) * 0.1).to(torch.bfloat16)
    # KdaGateCumsum(safe_gate=True) produces non-increasing chunk-local
    # log2 gates.  Small negative increments exercise both scale directions.
    increments = -torch.rand(shape, dtype=torch.float32) * 0.05
    gk = increments.clone()
    for begin in range(0, seqlen, 64):
        gk[:, begin:begin + 64] = increments[:, begin:begin + 64].cumsum(dim=1)
    beta = torch.rand((1, seqlen, heads), dtype=torch.float32).to(beta_dtype)
    d_aqk = torch.randn((1, seqlen, heads, 64), dtype=torch.float32) * 0.1
    d_akk = torch.randn_like(d_aqk) * 0.1
    dq = torch.randn(shape, dtype=torch.float32) * 0.01
    dk = torch.randn_like(dq) * 0.01
    db = torch.randn((1, seqlen, heads), dtype=torch.float32) * 0.01
    dg = torch.randn_like(dq) * 0.01
    outputs = chunk_kda_bwd_intra_reference(
        q, k, gk, beta, d_aqk, d_akk, dq, dk, db, dg
    )
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_varlen_tnd_reference_is_finite(beta_dtype):
    torch.manual_seed(23)
    cu_seqlens = [0, 17, 64, 129]
    seqlen, heads, head_dim = cu_seqlens[-1], 3, 128
    shape = (seqlen, heads, head_dim)
    q = (torch.randn(shape) * 0.1).to(torch.bfloat16)
    k = (torch.randn(shape) * 0.1).to(torch.bfloat16)
    gk = torch.zeros(shape)
    beta = torch.rand((seqlen, heads)).to(beta_dtype)
    d_aqk = torch.randn((seqlen, heads, 64)) * 0.1
    inputs = (
        q, k, gk, beta, d_aqk, torch.randn_like(d_aqk) * 0.1,
        torch.randn(shape) * 0.01, torch.randn(shape) * 0.01,
        torch.randn((seqlen, heads)) * 0.01,
        torch.randn(shape) * 0.01,
    )
    outputs = chunk_kda_bwd_intra_reference(
        *inputs, layout="TND", cu_seqlens=cu_seqlens
    )
    assert [tuple(output.shape) for output in outputs] == [
        shape, shape, (seqlen, heads), shape
    ]
    assert all(torch.isfinite(output).all() for output in outputs)


@pytest.mark.skipif(
    os.getenv("FLA_NPU_RUN_CHUNK_KDA_BWD_INTRA", "0") != "1",
    reason="requires a built OPP and an NPU runtime",
)
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("heads", [1, 2, 3, 8])
@pytest.mark.parametrize("seqlen", [1, 15, 16, 17, 63, 64, 65, 129])
def test_npu_against_reference(head_dim, heads, seqlen):
    _check_npu_against_reference(
        head_dim, heads, seqlen, beta_dtype=torch.bfloat16
    )


def _check_npu_against_reference(head_dim, heads, seqlen, *, beta_dtype):
    from fla_npu.ops.ascendc import chunk_kda_bwd_intra

    torch.manual_seed(7)
    device = torch.device("npu:0")
    shape = (1, heads, seqlen, head_dim)
    q = (torch.randn(shape, dtype=torch.float32) * 0.1).to(torch.bfloat16).to(device)
    k = (torch.randn(shape, dtype=torch.float32) * 0.1).to(torch.bfloat16).to(device)
    increments = (-torch.rand(shape, dtype=torch.float32) * 0.05).to(device)
    gk = increments.clone()
    for begin in range(0, seqlen, 64):
        gk[:, :, begin:begin + 64] = increments[:, :, begin:begin + 64].cumsum(dim=2)
    beta = torch.rand(
        (1, heads, seqlen), dtype=torch.float32
    ).to(beta_dtype).to(device)
    d_aqk = (torch.randn((1, heads, seqlen, 64), dtype=torch.float32) * 0.1).to(device)
    d_akk = (torch.randn_like(d_aqk) * 0.1).to(device)
    dq = (torch.randn(shape, dtype=torch.float32) * 0.01).to(device)
    dk = (torch.randn_like(dq) * 0.01).to(device)
    db = (torch.randn((1, heads, seqlen), dtype=torch.float32) * 0.01).to(device)
    dg = (torch.randn_like(dq) * 0.01).to(device)

    expected = chunk_kda_bwd_intra_reference(
        q.cpu(), k.cpu(), gk.cpu(), beta.cpu(), d_aqk.cpu(), d_akk.cpu(),
        dq.cpu(), dk.cpu(), db.cpu(), dg.cpu(), layout="BNSD"
    )
    actual = chunk_kda_bwd_intra(
        q, k, gk, beta, d_aqk, d_akk, dq, dk, db, dg,
        chunk_size=64, safe_gate=True, layout="BNSD"
    )
    torch.npu.synchronize()
    for name, got, want in zip(("dq", "dk", "db", "dg"), actual, expected):
        torch.testing.assert_close(
            got.cpu(), want, rtol=2e-4, atol=2e-4, msg=f"{name} mismatch"
        )


@pytest.mark.skipif(
    os.getenv("FLA_NPU_RUN_CHUNK_KDA_BWD_INTRA", "0") != "1",
    reason="requires a built OPP and an NPU runtime",
)
@pytest.mark.parametrize(
    "head_dim,heads,seqlen",
    [(128, 3, 63), (128, 8, 65)],
)
def test_npu_fp32_beta_against_reference(head_dim, heads, seqlen):
    _check_npu_against_reference(
        head_dim, heads, seqlen, beta_dtype=torch.float32
    )


def _check_npu_bsnd_boundary_against_reference(
    *, force_segmented=False, beta_dtype=torch.bfloat16
):
    from fla_npu.ops.ascendc import chunk_kda_bwd_intra
    from fla_npu.ops.ascendc import _aclnn_ctypes as ascendc_ctypes

    torch.manual_seed(11)
    device = torch.device("npu:0")
    batch, seqlen, heads, head_dim = 1, 65, 2, 128
    shape = (batch, seqlen, heads, head_dim)
    q = (torch.randn(shape) * 0.1).to(torch.bfloat16).to(device)
    k = (torch.randn(shape) * 0.1).to(torch.bfloat16).to(device)
    increments = (-torch.rand(shape) * 0.05).to(device)
    gk = increments.clone()
    for begin in range(0, seqlen, 64):
        gk[:, begin:begin + 64] = increments[:, begin:begin + 64].cumsum(dim=1)
    beta = torch.rand((batch, seqlen, heads)).to(beta_dtype).to(device)
    d_aqk = (torch.randn((batch, seqlen, heads, 64)) * 0.1).to(device)
    d_akk = (torch.randn_like(d_aqk) * 0.1).to(device)
    dq = (torch.randn(shape) * 0.01).to(device)
    dk = (torch.randn_like(dq) * 0.01).to(device)
    db = (torch.randn((batch, seqlen, heads)) * 0.01).to(device)
    dg = (torch.randn_like(dq) * 0.01).to(device)

    expected = chunk_kda_bwd_intra_reference(
        q.cpu(), k.cpu(), gk.cpu(), beta.cpu(), d_aqk.cpu(), d_akk.cpu(),
        dq.cpu(), dk.cpu(), db.cpu(), dg.cpu(), layout="BSND"
    )
    old_budget = ascendc_ctypes._KDA_BSND_TRANSPOSE_WORKSPACE_BUDGET_BYTES
    if force_segmented:
        ascendc_ctypes._KDA_BSND_TRANSPOSE_WORKSPACE_BUDGET_BYTES = 1
    try:
        actual = chunk_kda_bwd_intra(
            q, k, gk, beta, d_aqk, d_akk, dq, dk, db, dg,
            chunk_size=64, safe_gate=True, layout="BSND"
        )
    finally:
        ascendc_ctypes._KDA_BSND_TRANSPOSE_WORKSPACE_BUDGET_BYTES = old_budget
    torch.npu.synchronize()
    for name, got, want in zip(("dq", "dk", "db", "dg"), actual, expected):
        torch.testing.assert_close(
            got.cpu(), want, rtol=2e-4, atol=2e-4, msg=f"BSND {name} mismatch"
        )


@pytest.mark.skipif(
    os.getenv("FLA_NPU_RUN_CHUNK_KDA_BWD_INTRA", "0") != "1",
    reason="requires a built OPP and an NPU runtime",
)
def test_npu_bsnd_boundary_against_reference():
    _check_npu_bsnd_boundary_against_reference()


@pytest.mark.skipif(
    os.getenv("FLA_NPU_RUN_CHUNK_KDA_BWD_INTRA", "0") != "1",
    reason="requires a built OPP and an NPU runtime",
)
def test_npu_bsnd_segmented_views_against_reference():
    _check_npu_bsnd_boundary_against_reference(force_segmented=True)


@pytest.mark.skipif(
    os.getenv("FLA_NPU_RUN_CHUNK_KDA_BWD_INTRA", "0") != "1",
    reason="requires a built OPP and an NPU runtime",
)
def test_npu_bsnd_segmented_fp32_beta_against_reference():
    _check_npu_bsnd_boundary_against_reference(
        force_segmented=True, beta_dtype=torch.float32
    )


@pytest.mark.skipif(
    os.getenv("FLA_NPU_RUN_CHUNK_KDA_BWD_INTRA", "0") != "1",
    reason="requires a built OPP and an NPU runtime",
)
@pytest.mark.parametrize("layout", ["NTD"])
def test_npu_rejects_unsupported_layout(layout):
    from fla_npu.ops.ascendc import chunk_kda_bwd_intra

    with pytest.raises(RuntimeError, match="dense BSND/BNSD or varlen TND"):
        chunk_kda_bwd_intra(
            *([None] * 10), chunk_size=64, safe_gate=True, layout=layout
        )


@pytest.mark.skipif(
    os.getenv("FLA_NPU_RUN_CHUNK_KDA_BWD_INTRA", "0") != "1",
    reason="requires a built OPP and an NPU runtime",
)
@pytest.mark.parametrize(
    "cu_seqlens",
    [
        [0, 17, 80, 145],
        [0, 1, 64, 65, 129],
        [0, 0, 63, 128, 130],
    ],
)
@pytest.mark.parametrize("provide_chunk_indices", [False, True])
@pytest.mark.parametrize("storage_layout", ["TND", "BSND"])
@pytest.mark.parametrize("beta_dtype", [torch.bfloat16, torch.float32])
def test_npu_varlen_tnd_against_reference(
    cu_seqlens, provide_chunk_indices, storage_layout, beta_dtype
):
    from fla_npu.ops.ascendc import chunk_kda_bwd_intra

    torch.manual_seed(19)
    device = torch.device("npu:0")
    seqlen, heads, head_dim = cu_seqlens[-1], 3, 128
    shape = (seqlen, heads, head_dim)
    q_cpu = (torch.randn(shape) * 0.1).to(torch.bfloat16)
    k_cpu = (torch.randn(shape) * 0.1).to(torch.bfloat16)
    increments = -torch.rand(shape, dtype=torch.float32) * 0.05
    gk_cpu = torch.empty_like(increments)
    for seq_begin, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        for chunk_begin in range(seq_begin, seq_end, 64):
            chunk_end = min(chunk_begin + 64, seq_end)
            gk_cpu[chunk_begin:chunk_end] = increments[
                chunk_begin:chunk_end
            ].cumsum(dim=0)
    beta_cpu = torch.rand((seqlen, heads)).to(beta_dtype)
    d_aqk_cpu = torch.randn((seqlen, heads, 64)) * 0.1
    d_akk_cpu = torch.randn_like(d_aqk_cpu) * 0.1
    dq_cpu = torch.randn(shape) * 0.01
    dk_cpu = torch.randn_like(dq_cpu) * 0.01
    db_cpu = torch.randn((seqlen, heads)) * 0.01
    dg_cpu = torch.randn_like(dq_cpu) * 0.01
    cpu_inputs = (
        q_cpu, k_cpu, gk_cpu, beta_cpu, d_aqk_cpu,
        d_akk_cpu, dq_cpu, dk_cpu, db_cpu, dg_cpu,
    )
    expected = chunk_kda_bwd_intra_reference(
        *cpu_inputs, chunk_size=64, layout="TND", cu_seqlens=cu_seqlens
    )
    chunk_indices = None
    if provide_chunk_indices:
        chunk_indices = []
        for seq, (begin, end) in enumerate(
            zip(cu_seqlens[:-1], cu_seqlens[1:])
        ):
            for local_chunk in range((end - begin + 63) // 64):
                chunk_indices.extend((seq, local_chunk))
    npu_inputs = cpu_inputs
    npu_expected = expected
    if storage_layout == "BSND":
        npu_inputs = tuple(tensor.unsqueeze(0) for tensor in cpu_inputs)
        npu_expected = tuple(tensor.unsqueeze(0) for tensor in expected)
    actual = chunk_kda_bwd_intra(
        *(tensor.to(device) for tensor in npu_inputs),
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=64,
        safe_gate=True,
        layout=storage_layout,
    )
    torch.npu.synchronize()
    for name, got, want in zip(
        ("dq", "dk", "db", "dg"), actual, npu_expected
    ):
        torch.testing.assert_close(
            got.cpu(), want, rtol=2e-4, atol=2e-4,
            msg=f"varlen TND {name} mismatch",
        )
