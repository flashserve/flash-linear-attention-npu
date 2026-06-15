#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Dump realistic chunk_gated_delta_rule_fwd_h INPUT fixtures from the model pipeline.
#
# The fwd_h operator only behaves numerically well when its inputs (k, w, u, g) follow the distribution
# produced by the upstream GDN pipeline (l2-normalised key, cumsum gate, WY representation). Feeding random
# tensors makes the chunk recurrence diverge and the CPU golden becomes unreliable.
#
# This script reproduces the exact prefix of `flash_chunk_gated_delta_rule_fwd`:
#   g = chunk_local_cumsum(g)          (triton)
#   A = chunk_scaled_dot_kkt_fwd(...)  (triton)
#   A = solve_tril(A)                  (triton)
#   w, u = WY representation           (computed here with the repo's recompute_w_u CPU golden)
# and saves model-distributed k / w / u / g (+ optional initial_state) plus the varlen metadata.
#
# The golden h / v_new / final_state are NOT produced by any custom op here; the consuming pytest computes
# them from these realistic inputs with the fp32 CPU reference (forward_h_trans_cpu), which is the repo's
# accepted golden for fwd_h. That keeps this dump free of the custom-op / vendor toolchain and lets the
# fixtures be consumed from a different python/torch env.
#
# Run inside the README env (python3.10 + torch2.7.1 + torch_npu + triton-ascend):
#   ASCEND_RT_VISIBLE_DEVICES=4 python dump_model_data.py
# -----------------------------------------------------------------------------------------------------------

import sys
import math
from pathlib import Path

import torch
import torch_npu  # noqa: F401

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from examples.flash_gated_delta_rule import (  # noqa: E402
    _make_gate,
    _chunk_list,
    _chunk_tensor,
    _ensure_varlen_metadata,
    chunk_local_cumsum,
    chunk_scaled_dot_kkt_fwd,
    l2norm_fwd,
    solve_tril,
)

DATA_DIR = Path(__file__).resolve().parent / "data"


def _get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices):
    if cu_seqlens is not None:
        seq_idx = chunk_indices[idx * 2]
        chunk_idx = chunk_indices[idx * 2 + 1]
        bos = cu_seqlens[seq_idx] + chunk_idx * chunk_size
        eos = min(bos + chunk_size, cu_seqlens[seq_idx + 1])
    else:
        bos = idx * chunk_size
        eos = min(bos + chunk_size, T)
    return bos, eos


def compute_w_golden(k, v, beta, A, g, cu_seqlens, chunk_indices, B, H, T, D, chunk_size, NT, Hk):
    """Repo CPU golden for w: w_chunk = A_chunk @ (k_chunk * beta * exp(g)). Layouts:
    k [B,Hk,T,D], A [B,H,T,cs], beta [B,H,T], g [B,H,T]."""
    hv_per_hk = H // Hk
    w = torch.zeros(B, H, T, D, dtype=v.dtype)
    for i_b in range(B):
        for idx in range(NT):
            bos, eos = _get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            for i_h in range(H):
                hk = i_h // hv_per_hk
                A_chunk = A[i_b, i_h, bos:eos, : eos - bos].float()
                k_chunk = k[i_b, hk, bos:eos, :].float()
                beta_chunk = beta[i_b, i_h, bos:eos].float()
                g_exp = torch.exp(g[i_b, i_h, bos:eos].float())
                kbg = k_chunk * (beta_chunk * g_exp)[:, None]
                w[i_b, i_h, bos:eos, :] = torch.matmul(A_chunk, kbg).to(w.dtype)
    return w


def compute_u_golden(v, beta, A, cu_seqlens, chunk_indices, B, Hv, T, chunk_size, NT):
    """Repo CPU golden for u: u_chunk = A_chunk @ (v_chunk * beta). Layouts as above."""
    u = torch.zeros_like(v)
    for i_b in range(B):
        for idx in range(NT):
            bos, eos = _get_bos_eos(idx, T, chunk_size, cu_seqlens, chunk_indices)
            for i_h in range(Hv):
                A_chunk = A[i_b, i_h, bos:eos, : eos - bos].float()
                v_chunk = v[i_b, i_h, bos:eos, :].float()
                beta_chunk = beta[i_b, i_h, bos:eos].float()
                vb = v_chunk * beta_chunk[:, None]
                u[i_b, i_h, bos:eos, :] = torch.matmul(A_chunk, vb).to(u.dtype)
    return u


def _gen_cu_seqlens(total_tokens: int, batch: int, device) -> torch.Tensor:
    import random
    random.seed(total_tokens * 131 + batch)
    cu = [0]
    avg = total_tokens // batch
    for _ in range(batch - 1):
        diff = random.randint(max(1, avg // 2), max(1, avg * 3 // 2))
        nxt = min(cu[-1] + diff, total_tokens - (batch - len(cu)))
        cu.append(max(cu[-1] + 1, nxt))
    cu.append(total_tokens)
    return torch.tensor(cu, dtype=torch.int64, device=device)


def build_case(name, *, HV, T, K, V, chunk_size, dtype, use_init, varlen, batch, seed, device):
    torch.manual_seed(seed)
    B = 1

    # After the upstream GQA repeat_interleave, the key reaching fwd_h always has value_heads,
    # so kNumHead == vNumHead == HV here (head_ratio == 1), matching the real model distribution.
    k = torch.randn(B, HV, T, K, dtype=dtype, device=device)
    v = torch.randn(B, HV, T, V, dtype=dtype, device=device)
    beta = torch.rand(B, T, HV, dtype=dtype, device=device).sigmoid()
    g = _make_gate((B, T, HV), dtype, device, "logsigmoid")

    # use_qk_l2norm_in_kernel=True -> key is l2-normalised before the chunk pipeline.
    k, _ = l2norm_fwd(k)

    cu_seqlens = None
    cu_list = None
    chunk_indices = None
    chunk_list = None
    if varlen:
        cu_seqlens = _gen_cu_seqlens(T, batch, device)
        cu_seqlens, cu_list, chunk_indices, chunk_list = _ensure_varlen_metadata(
            g=g, cu_seqlens=cu_seqlens, cu_seqlens_list=None,
            chunk_indices=None, chunk_indices_list=None, chunk_size=chunk_size,
        )

    g = chunk_local_cumsum(
        g, chunk_size=chunk_size, cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices, head_first=False,
    )
    A = chunk_scaled_dot_kkt_fwd(
        k=k, g=g, beta=beta, cu_seqlens=cu_seqlens,
        chunk_indices=_chunk_tensor(chunk_indices, chunk_size),
        chunk_size=chunk_size, output_dtype=torch.float32,
    )
    A = solve_tril(A=A, cu_seqlens=cu_seqlens, chunk_indices_out=chunk_indices, output_dtype=k.dtype)

    # Match the flash pipeline layout before the WY recompute.
    g = g.transpose(1, 2).contiguous()        # [B, HV, T]
    beta_t = beta.transpose(1, 2).contiguous()  # [B, HV, T]
    A = A.transpose(1, 2).contiguous()        # [B, HV, T, chunk_size]

    # Move to CPU and compute the WY representation with the repo golden (no custom op needed).
    k_c, v_c, beta_c, A_c, g_c = (t.detach().cpu() for t in (k, v, beta_t, A, g))
    flat_chunk_indices = _chunk_list(chunk_list, chunk_size)
    NT = (len(flat_chunk_indices) // 2) if flat_chunk_indices is not None else (T + chunk_size - 1) // chunk_size
    w_c = compute_w_golden(k_c, v_c, beta_c, A_c, g_c, cu_list, flat_chunk_indices, B, HV, T, K, chunk_size, NT, Hk=HV)
    u_c = compute_u_golden(v_c, beta_c, A_c, cu_list, flat_chunk_indices, B, HV, T, chunk_size, NT)

    initial_state = None
    if use_init:
        N = (len(cu_list) - 1) if cu_list is not None else B
        initial_state = torch.randn(N, HV, K, V, dtype=dtype)

    payload = {
        "name": name,
        "chunk_size": int(chunk_size),
        "dtype": str(dtype).replace("torch.", ""),
        "cu_seqlens": cu_list,
        "chunk_indices": flat_chunk_indices,
        "k": k_c,
        "w": w_c,
        "u": u_c,
        "g": g_c,
        "initial_state": initial_state,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{name}.pt"
    torch.save(payload, out_path)
    print(
        f"[dump] {name}: k={tuple(k_c.shape)} w={tuple(w_c.shape)} u={tuple(u_c.shape)} g={tuple(g_c.shape)} "
        f"init={None if initial_state is None else tuple(initial_state.shape)} "
        f"cu={cu_list} -> {out_path.name}",
        flush=True,
    )


CASES = [
    dict(name="fix_bf16_noinit", HV=4, T=256, K=128, V=128, chunk_size=64, dtype=torch.bfloat16, use_init=False, varlen=False, batch=1, seed=0),
    dict(name="fix_fp16_noinit", HV=4, T=256, K=128, V=128, chunk_size=64, dtype=torch.float16, use_init=False, varlen=False, batch=1, seed=1),
    dict(name="fix_bf16_init", HV=4, T=256, K=128, V=128, chunk_size=64, dtype=torch.bfloat16, use_init=True, varlen=False, batch=1, seed=2),
    dict(name="fix_fp16_init", HV=4, T=256, K=128, V=128, chunk_size=64, dtype=torch.float16, use_init=True, varlen=False, batch=1, seed=3),
    dict(name="var_bf16_noinit", HV=4, T=320, K=128, V=128, chunk_size=64, dtype=torch.bfloat16, use_init=False, varlen=True, batch=3, seed=4),
    dict(name="var_fp16_init", HV=4, T=320, K=128, V=128, chunk_size=64, dtype=torch.float16, use_init=True, varlen=True, batch=3, seed=5),
]


def main():
    device = "npu"
    torch.npu.set_compile_mode(jit_compile=False)
    for cfg in CASES:
        build_case(device=device, **cfg)
    print("[dump] done. files in", DATA_DIR)


if __name__ == "__main__":
    main()
