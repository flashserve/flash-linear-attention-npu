import os

import torch
import torch_npu

import fla_npu  # noqa: F401


torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", "0")))


def _make_gk(batch, heads, seqlen, kdim, chunk_size):
    raw = torch.randn(batch, heads, seqlen, kdim, dtype=torch.float32) * 0.01 - 0.02
    gk = torch.empty_like(raw)
    for start in range(0, seqlen, chunk_size):
        end = min(start + chunk_size, seqlen)
        gk[:, :, start:end, :] = raw[:, :, start:end, :].cumsum(dim=2)
    return gk


def _reference_bwd_dhu_gk(qg, kg, w, do, dv, gk, scale, chunk_size):
    batch, heads, seqlen, kdim = qg.shape
    vdim = dv.shape[-1]
    num_chunks = (seqlen + chunk_size - 1) // chunk_size
    dh = torch.zeros(batch, heads, num_chunks, kdim, vdim, dtype=torch.float32)
    dv2 = torch.empty(batch, heads, seqlen, vdim, dtype=torch.float32)

    qg = qg.float()
    kg = kg.float()
    w = w.float()
    do = do.float()
    dv = dv.float()
    gk = gk.float()
    for b_idx in range(batch):
        for h_idx in range(heads):
            dh_running = torch.zeros(kdim, vdim, dtype=torch.float32)
            for chunk_idx in range(num_chunks - 1, -1, -1):
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, seqlen)
                dh[b_idx, h_idx, chunk_idx] = dh_running
                cur_dv2 = kg[b_idx, h_idx, start:end] @ dh_running + dv[b_idx, h_idx, start:end]
                dv2[b_idx, h_idx, start:end] = cur_dv2
                if chunk_idx > 0:
                    decay = torch.exp2(gk[b_idx, h_idx, end - 1])[:, None]
                    dh_running = dh_running * decay
                    dh_running = dh_running + qg[b_idx, h_idx, start:end].transpose(-1, -2) @ do[b_idx, h_idx, start:end] * scale
                    dh_running = dh_running - w[b_idx, h_idx, start:end].transpose(-1, -2) @ cur_dv2
    return dh.to(qg.dtype), dv2.to(qg.dtype)


def test_chunk_gated_delta_rule_bwd_dhu_gk_matches_reference():
    torch.manual_seed(13)
    batch, heads, seqlen, kdim, vdim, chunk_size = 1, 1, 128, 128, 128, 64
    dtype = torch.float16
    qg = torch.randn(batch, heads, seqlen, kdim, dtype=dtype) * 0.02
    kg = torch.randn(batch, heads, seqlen, kdim, dtype=dtype) * 0.02
    w = torch.randn(batch, heads, seqlen, kdim, dtype=dtype) * 0.02
    do = torch.randn(batch, heads, seqlen, vdim, dtype=dtype) * 0.02
    dv = torch.randn(batch, heads, seqlen, vdim, dtype=dtype) * 0.02
    gk = _make_gk(batch, heads, seqlen, kdim, chunk_size)
    scale = kdim ** -0.5

    ref_dh, ref_dv2 = _reference_bwd_dhu_gk(qg, kg, w, do, dv, gk, scale, chunk_size)
    dh, dh0, dv2 = torch.ops.npu.npu_chunk_gated_delta_rule_bwd_dhu(
        qg.npu(),
        kg.npu(),
        w.npu(),
        do.npu(),
        dv.npu(),
        scale=scale,
        chunk_size=chunk_size,
        g=None,
        gK=gk.npu(),
        h0=None,
        dht=None,
        cu_seqlens=None,
        chunk_indices=None,
        use_exp2=False,
        transpose_state_layout=False,
    )
    torch_npu._C._npu_synchronize()

    assert dh0 is None
    torch.testing.assert_close(dh.cpu().float(), ref_dh.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(dv2.cpu().float(), ref_dv2.float(), atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    test_chunk_gated_delta_rule_bwd_dhu_gk_matches_reference()
