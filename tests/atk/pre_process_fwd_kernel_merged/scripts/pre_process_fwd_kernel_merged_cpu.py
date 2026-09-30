"""pre_process_fwd_kernel_merged 的唯一可编辑 PyTorch 标杆源码。

语义来源: fla-org/flash-linear-attention @ e52dbc0e
          fla/ops/cp/chunk_delta_h.py::pre_process_fwd_kernel_merged
实现为设备无关的纯 PyTorch 算子, 可在 CPU(正式验收)或其它设备上运行。

计算约定(与上游 kernel 逐条对应)
---------------------------------
* 张量按 token-major 布局: k[T,HK,K] / w[T,HV,K](DPLR 为 [T,HK,K]) /
  v[T,HV,V] / u[T,HV,V] / g[T,HV] / gk[T,HV,K] / bg[T,HK,K]
* gate 是 base-2 的 chunk 内累积对数衰减, 衰减一律用 exp2
* 一次调用处理一个窗口(一个 part): bos/eos 由 cu_seqlens 给出, 缺省为整个 T
* k 的 head 维按 i_h // (HV // HK) 展开到 HV 个 value head
* 输出 hm[HV, K, V+K]: 左 [0,V) 是 h, 右 [V, V+K) 是 m

契约的数值精度(改变它们会显著改变结果, 因此属于接口契约的一部分)
----------------------------------------------------------------
1. 输入为 BF16; 累加为 **FP32** —— 上游 kernel 的 h/m 累加器都是
   `tl.zeros(..., dtype=tl.float32)`, 所有 `tl.dot` 都往 FP32 累加器里累加,
   因此 `accum_dtype` 默认取 `float32`, 与 kernel 声明的精度一致
2. `h` 在进入 `w @ h` 之前先降到输入 dtype(BF16)
3. `v_new` 在进入 `k^T @ v_new` 之前先降到输入 dtype(BF16)
4. `m` 的链式乘 `M_c @ m` 每个 chunk 更新后回落到 FP32(上游 `input_precision="ieee"`)

把 `accum_dtype` 换成 `float64` 并把三个开关都关掉, 就得到"纯数学"版本
(`calibrate_reference.py` 用它做灵敏度对照)。**只有"契约"版本用于精度验收**:
kernel 的 h 累加器是 FP32, 用 FP64 基准会让任何忠实的实现都平白多出 ~9e-3 的
绝对偏差(2026-09-20 与 H20 `ieee` 实测对齐时测得 9.371e-03)。
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

__all__ = ["pre_process_fwd_kernel_merged", "REFERENCE_CONTRACT"]


REFERENCE_CONTRACT = {
    "source": "fla-org/flash-linear-attention@e52dbc0e "
              "fla/ops/cp/chunk_delta_h.py::pre_process_fwd_kernel_merged",
    "layout": "token-major [T, H, D]",
    "gate_units": "log2 (exp2 衰减, chunk 内累积)",
    "rounding_points": [
        "accumulation in float32 (kernel's h/m accumulators are tl.float32)",
        "h -> input dtype before w @ h",
        "v_new -> input dtype before k^T @ v_new",
        "M_c @ m rounded back to float32 per chunk",
    ],
    "outputs": ["hm[HV, K, V+K] = [h | m]"],
}


def _check(cond: bool, message: str) -> None:
    if not cond:
        raise ValueError(message)


@torch.no_grad()
def pre_process_fwd_kernel_merged(
    k: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    *,
    g: Optional[torch.Tensor] = None,
    gk: Optional[torch.Tensor] = None,
    bg: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    chunk_size: int = 64,
    cu_seqlens: Optional[Sequence[int]] = None,
    accum_dtype: torch.dtype = torch.float32,
    round_h_to_input_dtype: bool = True,
    round_v_new_to_input_dtype: bool = True,
    round_affine_chain_to_float32: bool = True,
) -> torch.Tensor:
    """返回 hm, shape [HV, K, V+K], dtype = accum_dtype。

    参数含义见模块文档。`u` 缺省等于 `v`(GDN/KDA 路径下两者是同一个张量);
    DPLR 路径(bg 存在)必须显式给出 `u`。
    """

    _check(k.dim() == 3, "k must be [T, HK, K]")
    _check(v.dim() == 3, "v must be [T, HV, V]")
    _check(w.dim() == 3, "w must be [T, H, K]")
    _check((g is None) != (gk is None), "exactly one of g / gk must be given")
    use_bg = bg is not None
    if use_bg:
        _check(gk is not None, "DPLR (bg given) requires gk")
        _check(bg.dim() == 3, "bg must be [T, HK, K]")
        _check(u is not None, "DPLR requires u")
    if u is None:
        u = v

    T, HK, K = k.shape
    _, HV, V = v.shape
    _check(HV % HK == 0, "HV must be a multiple of HK")
    _check(w.shape[2] == K, "w's last dim must equal K")
    _check(v.shape[0] == T and w.shape[0] == T, "k/v/w must share T")
    _check(u.shape == v.shape, "u must have v's shape")
    _check(0 < chunk_size <= T, "chunk_size must be in (0, T]")

    BT = int(chunk_size)
    ratio = HV // HK
    head_of_k = torch.arange(HV) // ratio

    if cu_seqlens is None:
        bos, eos = 0, T
    else:
        _check(len(cu_seqlens) == 2, "this operator consumes one window: cu_seqlens must be [bos, eos]")
        bos, eos = int(cu_seqlens[0]), int(cu_seqlens[1])
        _check(0 <= bos < eos <= T, "cu_seqlens must satisfy 0 <= bos < eos <= T")
    length = eos - bos

    dt = accum_dtype
    src = k.dtype

    def cast_operand(x: torch.Tensor, enabled: bool) -> torch.Tensor:
        """按契约把操作数降到输入 dtype 后再升回累加 dtype。"""
        if not enabled:
            return x.to(dt)
        return x.to(src).to(dt)

    h = torch.zeros(HV, K, V, dtype=dt, device=k.device)
    m = torch.eye(K, dtype=dt, device=k.device).repeat(HV, 1, 1)

    n_chunks = -(-length // BT)
    for c in range(n_chunks):
        lo = bos + c * BT
        hi = min(bos + (c + 1) * BT, eos)
        idx = torch.arange(lo, hi, device=k.device)
        last = hi - 1

        kc = k[idx][:, head_of_k, :].to(dt)                    # (bt,HV,K)
        wc = w[idx].to(dt)
        if use_bg:
            wc = wc[:, head_of_k, :]                           # (bt,HV,K)
        vc = v[idx].to(dt)                                     # (bt,HV,V)
        uc = u[idx].to(dt)

        # ------------------------------- h 半边
        h_dot = cast_operand(h, round_h_to_input_dtype)
        v_decay = torch.einsum("thk,hkv->thv", wc, h_dot)
        v_new = (v_decay + uc) if use_bg else (vc - v_decay)

        if g is not None:
            gl = g[last].to(dt)                                # (HV,)
            gs = g[idx].to(dt)                                 # (bt,HV)
            v_new = v_new * torch.exp2(gl[None, :] - gs).unsqueeze(-1)
            h = h * torch.exp2(gl)[:, None, None]
        if gk is not None:
            h = h * torch.exp2(gk[last].to(dt))[:, :, None]

        v_dot = cast_operand(v_new, round_v_new_to_input_dtype)
        h = h + torch.einsum("thk,thv->hkv", kc, v_dot)
        if use_bg:
            bgc = bg[idx][:, head_of_k, :].to(dt)
            h = h + torch.einsum("thk,thv->hkv", bgc,
                                 cast_operand(vc, round_v_new_to_input_dtype))

        # ------------------------------- m 半边
        left = bg[idx][:, head_of_k, :].to(dt) if use_bg else kc
        if g is not None:
            gl = g[last].to(dt)
            gs = g[idx].to(dt)
            left = left * torch.exp2(gl[None, :] - gs).unsqueeze(-1)
        kw = torch.einsum("thk,thj->hkj",
                          cast_operand(left, round_v_new_to_input_dtype),
                          cast_operand(wc, round_v_new_to_input_dtype))
        if g is not None:
            diag = torch.eye(K, dtype=dt, device=k.device) \
                * torch.exp2(g[last].to(dt))[:, None, None]
        elif gk is not None:
            diag = torch.diag_embed(torch.exp2(gk[last].to(dt)))
        else:  # pragma: no cover - g/gk 二选一, 不会走到
            diag = torch.eye(K, dtype=dt, device=k.device).repeat(HV, 1, 1)
        M = (diag + kw) if use_bg else (diag - kw)
        m = M @ m
        if round_affine_chain_to_float32:
            m = m.to(torch.float32).to(dt)

    hm = torch.cat([h, m], dim=-1)
    return hm


def _self_test() -> None:
    gen = torch.Generator().manual_seed(0)
    T, HK, HV, K, V, BT = 256, 4, 4, 128, 128, 64
    k = torch.nn.functional.normalize(torch.randn(T, HK, K, generator=gen), dim=-1).bfloat16()
    beta = torch.rand(T, HV, 1, generator=gen) * 0.02
    w = (beta * k[:, torch.arange(HV)]) .bfloat16()
    v = torch.randn(T, HV, V, generator=gen).bfloat16()
    n = -(-T // BT)
    g = (-0.013 / BT * (1 + torch.rand(T, HV, generator=gen) * 0.5))
    g = g.view(n, BT, HV).cumsum(1).reshape(-1, HV)[:T].contiguous()
    hm = pre_process_fwd_kernel_merged(k, v, w, g=g)
    assert hm.shape == (HV, K, V + K), hm.shape
    assert torch.isfinite(hm).all(), "self test produced non-finite values"
    assert hm.dtype == torch.float32, "default accum_dtype must match the kernel (fp32)"
    print(f"self test OK: {tuple(hm.shape)} {hm.dtype} "
          f"|h|max={hm[:, :, :V].abs().max():.4e} |m|max={hm[:, :, V:].abs().max():.4e}")


if __name__ == "__main__":
    _self_test()
