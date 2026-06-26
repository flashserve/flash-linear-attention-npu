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


def _make_kg(k, gk, chunk_size):
    kg = torch.empty_like(k)
    for start in range(0, k.shape[2], chunk_size):
        end = min(start + chunk_size, k.shape[2])
        last = gk[:, :, end - 1 : end, :]
        kg[:, :, start:end, :] = k[:, :, start:end, :] * torch.exp2(last - gk[:, :, start:end, :]).to(k.dtype)
    return kg.contiguous()


def _reference_fwd_h_gk(kg, w, u, gk, initial_state, chunk_size):
    batch, k_heads, seqlen, kdim = kg.shape
    v_heads, vdim = u.shape[1], u.shape[-1]
    num_chunks = (seqlen + chunk_size - 1) // chunk_size
    head_ratio = v_heads // k_heads

    h = torch.zeros(batch, v_heads, num_chunks, kdim, vdim, dtype=torch.float32)
    v_new = torch.zeros(batch, v_heads, seqlen, vdim, dtype=torch.float32)
    final_state = torch.zeros(batch, v_heads, kdim, vdim, dtype=torch.float32)
    if initial_state is not None:
        h[:, :, 0] = initial_state.reshape(batch, v_heads, kdim, vdim).float()

    kg = kg.float()
    w = w.float()
    u = u.float()
    gk = gk.float()
    for b_idx in range(batch):
        for vh_idx in range(v_heads):
            kh_idx = vh_idx // head_ratio
            for chunk_idx, start in enumerate(range(0, seqlen, chunk_size)):
                end = min(start + chunk_size, seqlen)
                cur_v = u[b_idx, vh_idx, start:end] - w[b_idx, vh_idx, start:end] @ h[b_idx, vh_idx, chunk_idx]
                next_state = h[b_idx, vh_idx, chunk_idx] * torch.exp2(gk[b_idx, vh_idx, end - 1])[:, None]
                next_state = next_state + kg[b_idx, kh_idx, start:end].transpose(-1, -2) @ cur_v
                if chunk_idx + 1 < num_chunks:
                    h[b_idx, vh_idx, chunk_idx + 1] = next_state
                else:
                    final_state[b_idx, vh_idx] = next_state
                v_new[b_idx, vh_idx, start:end] = cur_v

    state_dtype = initial_state.dtype if initial_state is not None else torch.float32
    return h.to(kg.dtype), v_new.to(kg.dtype), final_state.to(state_dtype)


def test_chunk_gated_delta_rule_fwd_h_gk_matches_reference():
    torch.manual_seed(7)
    batch, heads, seqlen, kdim, vdim, chunk_size = 1, 1, 128, 128, 128, 64
    dtype = torch.float16
    k = torch.randn(batch, heads, seqlen, kdim, dtype=dtype) * 0.02
    w = torch.randn(batch, heads, seqlen, kdim, dtype=dtype) * 0.02
    u = torch.randn(batch, heads, seqlen, vdim, dtype=dtype) * 0.02
    gk = _make_gk(batch, heads, seqlen, kdim, chunk_size)
    kg = _make_kg(k, gk, chunk_size)
    initial_state = torch.randn(batch, heads, 1, kdim, vdim, dtype=torch.float32) * 0.02

    ref = _reference_fwd_h_gk(kg, w, u, gk, initial_state, chunk_size)
    got = torch.ops.npu.npu_chunk_gated_delta_rule_fwd_h(
        kg.npu(),
        w.npu(),
        u.npu(),
        g=None,
        gk=gk.npu(),
        initial_state=initial_state.npu(),
        output_final_state=True,
        chunk_size=chunk_size,
        save_new_value=True,
        cu_seqlens=None,
        chunk_indices=None,
        use_exp2=False,
        transpose_state_layout=False,
    )
    torch_npu._C._npu_synchronize()

    for name, actual, expected in zip(("h", "v_new", "final_state"), got, ref):
        torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=5e-2, rtol=5e-2, msg=name)


if __name__ == "__main__":
    test_chunk_gated_delta_rule_fwd_h_gk_matches_reference()
