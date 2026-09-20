#!/usr/bin/env python3
"""ChunkKDA forward/backward 看护脚本（Example/ST）。

覆盖 `fla_npu.ops.ascendc.chunk_kda_fwd` / `chunk_kda_bwd` 的正反向链路：

- 前向：raw gate（`A_log` / `dt_bias` + `safe_gate` + `use_exp2`）走 `disable_recompute=True`
  路径，取回反向需要的保存量；`use_qk_l2norm_in_kernel=True` 时同时校验导出的
  `q_hat/k_hat/q_rstd/k_rstd`（与 `rsqrt(sum(q^2)+eps)` 对比）。
- 反向：用前向保存量调 `chunk_kda_bwd`，得到 `dq/dk/dv/dbeta/dg`。
- 精度：CPU fp32 recurrent 参考（`o`）+ autograd 参考梯度，逐 tensor 打印看护指标，
  格式与 `ci/run_example_st_cases.py` 解析的一致（`o/dq/dk/dv/dbeta/dg`）。

CLI 与 `ci/example_st_cases.json` 的字段映射保持一致（`--batch/--tokens/--chunk-size/
--query-heads/--value-heads/--key-dim/--value-dim/--dtype/--device/--case-name` 等），
同时在 CI 传入 `--accuracy-check` 时按阈值判定并通过退出码反馈。
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover - 非 NPU 环境
    torch_npu = None

from fla_npu.ops.ascendc import chunk_kda_bwd, chunk_kda_fwd  # noqa: E402


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
ACCURACY_TENSORS = ("o", "dq", "dk", "dv", "dbeta", "dg")
DEFAULT_TOLERANCES = {
    "output_tol": 5e-3,
    "grad_tol": 8e-3,
    "beta_grad_tol": 2e-2,
    "gate_grad_tol": 2e-2,
}
DEFAULT_COS_MIN = {
    "output_cos_min": 0.999,
    "grad_cos_min": 0.999,
    "beta_grad_cos_min": 0.99,
    "gate_grad_cos_min": 0.99,
}
METRIC_TOL = {
    "o": "output_tol",
    "dq": "grad_tol",
    "dk": "grad_tol",
    "dv": "grad_tol",
    "dbeta": "beta_grad_tol",
    "dg": "gate_grad_tol",
}
METRIC_COS = {
    "o": "output_cos_min",
    "dq": "grad_cos_min",
    "dk": "grad_cos_min",
    "dv": "grad_cos_min",
    "dbeta": "beta_grad_cos_min",
    "dg": "gate_grad_cos_min",
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--query-heads", type=int, default=32)
    parser.add_argument("--value-heads", type=int, default=32)
    parser.add_argument("--key-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="bf16")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--layout", default="BSND")
    parser.add_argument("--case-name", default="chunk_kda_fwd_bwd")
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--cu-seqlens", default="")
    parser.add_argument("--varlen", dest="varlen", action="store_true", default=False)
    parser.add_argument("--no-varlen", dest="varlen", action="store_false")
    parser.add_argument("--qk-l2norm", dest="qk_l2norm", action="store_true", default=False)
    parser.add_argument("--no-qk-l2norm", dest="qk_l2norm", action="store_false")
    parser.add_argument("--gate-source", default="g")
    parser.add_argument("--gate-function", default="safe")
    parser.add_argument("--initial-state", default="none")
    parser.add_argument("--output-final-state", action="store_true", default=False)
    # CI 侧可能传入、对 KDA 无意义的开关：接受但忽略，避免参数解析失败。
    parser.add_argument("--dim", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--mean-len", type=float, default=None)
    parser.add_argument("--conv-kernel", type=int, default=None)
    parser.add_argument("--demo-model", action="store_true", default=False)
    parser.add_argument("--legacy-unfused-core", action="store_true", default=False)
    parser.add_argument("--accuracy-check", action="store_true", default=False)
    parser.add_argument("--accuracy-tensors", default=",".join(ACCURACY_TENSORS))
    parser.add_argument("--accuracy-output-tol", type=float,
                        default=DEFAULT_TOLERANCES["output_tol"])
    parser.add_argument("--accuracy-grad-tol", type=float,
                        default=DEFAULT_TOLERANCES["grad_tol"])
    parser.add_argument("--accuracy-beta-grad-tol", type=float,
                        default=DEFAULT_TOLERANCES["beta_grad_tol"])
    parser.add_argument("--accuracy-gate-grad-tol", type=float,
                        default=DEFAULT_TOLERANCES["gate_grad_tol"])
    parser.add_argument("--accuracy-output-cos-min", type=float,
                        default=DEFAULT_COS_MIN["output_cos_min"])
    parser.add_argument("--accuracy-grad-cos-min", type=float,
                        default=DEFAULT_COS_MIN["grad_cos_min"])
    parser.add_argument("--accuracy-beta-grad-cos-min", type=float,
                        default=DEFAULT_COS_MIN["beta_grad_cos_min"])
    parser.add_argument("--accuracy-gate-grad-cos-min", type=float,
                        default=DEFAULT_COS_MIN["gate_grad_cos_min"])
    return parser.parse_args(argv)


def resolve_device(name: str) -> torch.device:
    if torch_npu is not None and torch.npu.is_available():
        return torch.device(name if ":" in name else f"{name}:0")
    if name.startswith("npu"):
        raise RuntimeError("torch_npu 不可用：ChunkKDA 看护脚本需要 NPU 环境")
    return torch.device(name)


def parse_cu_seqlens(raw: str, tokens: int, batch: int) -> Optional[Tuple[int, ...]]:
    raw = (raw or "").strip()
    if not raw:
        return None
    values = tuple(int(item) for item in raw.split(",") if item.strip())
    if values[0] != 0 or values[-1] != tokens * batch:
        raise ValueError(f"--cu-seqlens must start at 0 and end at B*T={tokens * batch}")
    return values


def make_inputs(args: argparse.Namespace, dtype: torch.dtype,
                device: torch.device) -> Dict[str, torch.Tensor]:
    """构造模型侧 BSND 输入：q/k 已 L2 归一化、raw gate + A_log/dt_bias、beta∈(0,1)。"""

    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    batch, tokens = args.batch, args.tokens
    h, hv, kd, vd = args.query_heads, args.value_heads, args.key_dim, args.value_dim

    q = torch.randn(batch, tokens, h, kd, generator=gen, dtype=torch.float32)
    k = torch.randn(batch, tokens, h, kd, generator=gen, dtype=torch.float32)
    v = torch.randn(batch, tokens, hv, vd, generator=gen, dtype=torch.float32) * 0.5
    if not args.qk_l2norm:
        # use_qk_l2norm_in_kernel=False 时由调用方预先归一化 q/k；
        # 打开该开关时喂原始 q/k，让 kernel 内的归一化与反向回代真正被覆盖。
        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)
    a_log = torch.empty(hv, dtype=torch.float32).uniform_(0.0, 0.6, generator=gen)
    dt_bias = torch.empty(hv * kd, dtype=torch.float32).uniform_(-3.0, 0.0, generator=gen)
    raw_gate = torch.randn(batch, tokens, hv, kd, generator=gen, dtype=torch.float32)
    beta = torch.sigmoid(
        torch.randn(batch, tokens, hv, generator=gen, dtype=torch.float32) + 1.5)

    to_dev = lambda x: x.to(dtype=dtype, device=device) if x.is_floating_point() else x.to(device)
    tensors = {
        "q": to_dev(q),
        "k": to_dev(k),
        "v": to_dev(v),
        "g": raw_gate.to(device),
        "beta": to_dev(beta),
        "A_log": a_log.to(device),
        "dt_bias": dt_bias.to(device),
    }
    return tensors


def recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_gate: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    scale: float,
    lower_bound: float,
    cu_seqlens: Optional[Tuple[int, ...]],
    qk_l2norm: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CPU fp32 recurrent 参考：返回 (o, dq/dk/dv/dbeta/dg 的梯度元组, 输入叶子表)。

    与 chunk 形式数学等价；梯度由 autograd 在参考上反传得到。
    """

    q_raw = q.float()
    k_raw = k.float()
    v = v.float()
    gate_input = raw_gate.float() + dt_bias.float().reshape(a_log.numel(), -1).unsqueeze(0).unsqueeze(0)
    decay = torch.exp(a_log.float()).view(1, 1, -1, 1)

    leaves = {
        "q": q_raw.clone().requires_grad_(True),
        "k": k_raw.clone().requires_grad_(True),
        "v": v.clone().requires_grad_(True),
        "raw_gate": gate_input.detach().clone().requires_grad_(True),
        "beta": beta.float().clone().requires_grad_(True),
    }
    # 参考内部做归一化（与 kernel 的 use_qk_l2norm_in_kernel 口径一致），
    # 这样 autograd 给出的 dq/dk 也是对**原始** q/k 的梯度，可与算子输出直接对比。
    if qk_l2norm:
        q_norm = leaves["q"] * torch.rsqrt(
            leaves["q"].square().sum(-1, keepdim=True) + 1e-6)
        k_norm = leaves["k"] * torch.rsqrt(
            leaves["k"].square().sum(-1, keepdim=True) + 1e-6)
    else:
        q_norm, k_norm = leaves["q"], leaves["k"]
    # gate 必须由叶子张量算出，否则 raw_gate 不在反传图上。
    gate = lower_bound * torch.sigmoid(decay * leaves["raw_gate"])
    batch, tokens = q_norm.shape[0], q_norm.shape[1]
    heads, kd = q_norm.shape[2], q_norm.shape[3]
    vd = leaves["v"].shape[-1]
    ranges = [(0, tokens * batch)] if cu_seqlens is None else list(zip(cu_seqlens, cu_seqlens[1:]))

    out = torch.zeros_like(leaves["v"])
    for start, end in ranges:
        state = torch.zeros(heads, vd, kd, dtype=torch.float32)
        for token in range(start, end):
            batch_id, token_id = token // tokens, token % tokens
            # decay 按 key 维广播：state [H,V,K] × gate [H,K] → unsqueeze(1)。
            state = state * torch.exp(gate[batch_id, token_id]).unsqueeze(1)
            key = k_norm[batch_id, token_id]
            delta = (leaves["v"][batch_id, token_id]
                     - torch.einsum("hvk,hk->hv", state, key)) \
                * leaves["beta"][batch_id, token_id].unsqueeze(-1)
            state = state + delta.unsqueeze(-1) * key.unsqueeze(1)
            out[batch_id, token_id] = torch.einsum(
                "hvk,hk->hv", state, q_norm[batch_id, token_id] * scale)
    return out, gate, leaves


def cosine_similarity(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual = actual.float().reshape(-1)
    expected = expected.float().reshape(-1)
    denom = (actual.norm() * expected.norm()).clamp_min(1e-12)
    return float(torch.clamp(torch.dot(actual, expected) / denom, -1.0, 1.0).item())


def accuracy_metric(name: str, actual: torch.Tensor, expected: torch.Tensor,
                    tol: float, cos_min: float) -> Tuple[bool, str]:
    actual = actual.float()
    expected = expected.float()
    diff = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item())
    allclose = bool(torch.allclose(actual, expected, rtol=tol, atol=tol))
    cosine = cosine_similarity(actual, expected) if finite else float("nan")
    cosine_ok = bool(cosine >= cos_min)
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rmse = float(torch.sqrt((diff * diff).mean()).item())
    bad = diff > (tol + tol * expected.abs())
    bad_frac = float(bad.float().mean().item())
    ok = bool(finite and allclose and cosine_ok)
    line = (
        f"{name}: finite={finite} allclose={allclose} tol={tol:g} "
        f"cosine={cosine:.9g} cos_min={cos_min:g} cosine_ok={cosine_ok} "
        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g} rmse={rmse:.6g} "
        f"bad_frac={bad_frac:.6g}"
    )
    return ok, line


def tolerance_for(args: argparse.Namespace, name: str) -> Tuple[float, float]:
    return (
        float(getattr(args, f"accuracy_{METRIC_TOL[name]}")),
        float(getattr(args, f"accuracy_{METRIC_COS[name]}")),
    )


def run_once(args: argparse.Namespace) -> int:
    dtype = DTYPES[args.dtype]
    device = resolve_device(args.device)
    if device.type == "npu" and torch_npu is not None:
        torch.npu.set_device(device)
    batch, tokens = args.batch, args.tokens
    h, hv, kd, vd = args.query_heads, args.value_heads, args.key_dim, args.value_dim
    scale = args.scale if args.scale is not None else kd ** -0.5
    cu = parse_cu_seqlens(args.cu_seqlens, tokens, batch)
    packed = cu is not None
    # 打包场景本脚本按 head-major 拼写构造（[H,T,D] / beta [H,T]），即 NTD；
    # 与 MindSpeed 入口 _bsnd_to_head_major 的口径一致。
    layout = "NTD" if packed else args.layout
    if padded_layout_unsupported(layout, packed):
        raise ValueError(f"unsupported layout {layout} for packed={packed}")

    tensors = make_inputs(args, dtype, device)
    q, k, v, g, beta = (tensors[name] for name in ("q", "k", "v", "g", "beta"))

    def to_op_layout(x: torch.Tensor, is_beta: bool = False) -> torch.Tensor:
        if not packed:
            if layout == "BNSD":
                return x.permute(0, 2, 1) if is_beta else x.permute(0, 2, 1, 3)
            return x.contiguous()
        flat = x.reshape(batch * tokens, *x.shape[2:])
        return flat.permute(1, 0) if is_beta else flat.permute(1, 0, 2)

    q_op = to_op_layout(q).contiguous()
    k_op = to_op_layout(k).contiguous()
    v_op = to_op_layout(v).contiguous()
    g_op = to_op_layout(g).contiguous()
    beta_op = to_op_layout(beta, is_beta=True).contiguous()

    saved = {}
    if args.qk_l2norm:
        # q_hat/k_hat 是 head-major（dense [B,H,T,D] / packed [H,T,D]），
        # 与前向公开输入布局无关；反向直接吃这两个张量。
        qk_head_shape = ((h, tokens * batch, kd) if packed else (batch, h, tokens, kd))
        saved["q_hat"] = torch.empty(qk_head_shape, dtype=dtype, device=device)
        saved["k_hat"] = torch.empty(qk_head_shape, dtype=dtype, device=device)
        rstd_shape = (h, tokens * batch) if packed else (batch, h, tokens)
        beta_eff_shape = (hv, tokens * batch) if packed else (batch, hv, tokens)
        saved["q_rstd"] = torch.empty(rstd_shape, dtype=torch.float32, device=device)
        saved["k_rstd"] = torch.empty(rstd_shape, dtype=torch.float32, device=device)
        saved["beta_eff"] = torch.empty(beta_eff_shape, dtype=torch.float32, device=device)

    fwd_kwargs = dict(
        layout=layout,
        cu_seqlens=cu,
        safe_gate=True,
        lower_bound=args.lower_bound,
        use_gate_in_kernel=True,
        A_log=tensors["A_log"],
        dt_bias=tensors["dt_bias"],
        disable_recompute=True,
        return_intermediate_states=False,
        state_v_first=False,
        use_qk_l2norm_in_kernel=bool(args.qk_l2norm),
        use_beta_sigmoid_in_kernel=False,
        use_exp2=True,
    )
    if args.qk_l2norm:
        fwd_kwargs.update(
            q_hat_out=saved["q_hat"], k_hat_out=saved["k_hat"],
            q_rstd_out=saved["q_rstd"], k_rstd_out=saved["k_rstd"],
            beta_eff_out=saved["beta_eff"],
        )

    if os.getenv("CHUNK_KDA_DEBUG_ARGS", "").strip() in ("1", "true", "yes", "on"):
        for tag, tensor in (("q", q_op), ("k", k_op), ("v", v_op), ("g", g_op),
                            ("beta", beta_op), ("A_log", tensors["A_log"]),
                            ("dt_bias", tensors["dt_bias"])):
            print(f"[args] {tag}: shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                  f"contig={tensor.is_contiguous()}")
        print(f"[args] layout={layout} cu={cu} scale={scale} chunk={args.chunk_size} "
              f"l2norm={bool(args.qk_l2norm)}")
    outputs = chunk_kda_fwd(q_op, k_op, v_op, g_op, beta_op, scale, args.chunk_size, **fwd_kwargs)
    if torch_npu is not None and device.type == "npu":
        torch.npu.synchronize()
    (o, final_state, gk, aqk, akk, w, u, qg, kg, v_new, hstate, _initial) = outputs

    if args.qk_l2norm:
        q_ref = q.float()
        expect_bsnd = torch.rsqrt(q_ref.square().sum(-1) + 1e-6)  # [B,T,H]
        expect_rstd = (expect_bsnd[0].permute(1, 0) if packed
                       else expect_bsnd.permute(0, 2, 1))  # packed [H,T] / dense [B,H,T]
        got_rstd = saved["q_rstd"].float()
        diff = (got_rstd - expect_rstd).abs().max().item()
        print(f"q_rstd: max_abs={diff:.6g} (vs rsqrt(sum(q^2)+1e-6))")
        if diff > 1e-3:
            print("q_rstd 校验失败", file=sys.stderr)
            return 2

    # 反向：用前向保存量算梯度。反向的 q/k/v/beta 一律是 head-major
    # （dense 为 [B,H,T,D]、packed 为 [H,T,D]），与前向的公开输入布局不同。
    def to_head_major(x: torch.Tensor, is_beta: bool = False) -> torch.Tensor:
        if packed:
            return x.contiguous()
        # 反向要求连续张量（permute 视图会被 optimized 路径拒绝）。
        return (x.permute(0, 2, 1) if is_beta else x.permute(0, 2, 1, 3)).contiguous()

    grad_out = torch.ones_like(o)
    # 反向 d_o 恒为 head-major：dense [B,H,T,V]、packed [H,T,V]。
    grad_out = (grad_out.permute(1, 0, 2) if packed
                else grad_out.permute(0, 2, 1, 3)).contiguous()
    bwd_kwargs = dict(
        raw_g=to_head_major(g_op),
        A_log=tensors["A_log"],
        dt_bias=tensors["dt_bias"].reshape(hv, kd),
        initial_state=None,
        dht=None,
        cu_seqlens=cu,
        chunk_indices=None,
        chunk_size=args.chunk_size,
        safe_gate=True,
        lower_bound=args.lower_bound,
        use_gate_in_kernel=True,
        disable_recompute=True,
        use_exp2=True,
        state_v_first=False,
    )
    if args.qk_l2norm:
        bwd_kwargs.update(q_rstd=saved["q_rstd"], k_rstd=saved["k_rstd"])
        bwd_q, bwd_k = saved["q_hat"], saved["k_hat"]
    else:
        bwd_q, bwd_k = to_head_major(q_op), to_head_major(k_op)
    bwd_v = to_head_major(v_op)
    bwd_beta = to_head_major(beta_op, is_beta=True)
    dq, dk, dv, dbeta, dg, _dh0, _da, _dbias = chunk_kda_bwd(
        bwd_q, bwd_k, bwd_v, bwd_beta, gk, aqk, akk, w, qg, kg, v_new, hstate,
        grad_out, scale, **bwd_kwargs)
    if torch_npu is not None and device.type == "npu":
        torch.npu.synchronize()

    if not args.accuracy_check:
        print(f"{args.case_name}: forward/backward smoke OK "
              f"(o absmax={o.detach().float().abs().max().item():.6g}, "
              f"dq absmax={dq.detach().float().abs().max().item():.6g})")
        return 0

    # CPU 参考：同分布 fp32 recurrent + autograd。
    ref_o, _ref_gate, leaves = recurrent_reference(
        q.cpu(), k.cpu(), v.cpu(), g.cpu(), beta.cpu(), tensors["A_log"].cpu(),
        tensors["dt_bias"].cpu(), scale=scale, lower_bound=args.lower_bound,
        cu_seqlens=cu, qk_l2norm=bool(args.qk_l2norm))
    loss = ref_o.sum()
    grads = torch.autograd.grad(loss, [leaves[name] for name in ("q", "k", "v", "beta", "raw_gate")])
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dg_raw = grads

    def to_bsnd(tensor: torch.Tensor, is_beta: bool = False) -> torch.Tensor:
        # 反向输出恒为 head-major（dense [B,H,T,D] / packed [H,T,D]），
        # 与参考的 BSND 口径对比时要转回来。
        if not packed:
            return (tensor.permute(0, 2, 1) if is_beta
                    else tensor.permute(0, 2, 1, 3))
        unflat = tensor.permute(1, 0) if is_beta else tensor.permute(1, 0, 2)
        return unflat.reshape(batch, tokens, *unflat.shape[1:])

    # 前向 attn_out 恒为 sequence-major（dense BSND / packed TND），无需转置；
    # 反向输出是 head-major，需要用 to_bsnd 转回 BSND。
    o_bsnd = (o.detach().reshape(batch, tokens, *o.shape[1:]) if packed
              else o.detach())
    dq_bsnd = to_bsnd(dq.detach())
    dk_bsnd = to_bsnd(dk.detach())
    dv_bsnd = to_bsnd(dv.detach())
    dbeta_bsnd = to_bsnd(dbeta.detach(), is_beta=True)
    dg_bsnd = to_bsnd(dg.detach())

    if args.qk_l2norm:
        # 反向返回的是对归一化 q/k 的梯度，参考也按同样口径（leaves 即归一化后的 q/k）。
        ref_dg = ref_dg_raw
    else:
        ref_dg = ref_dg_raw

    selected = {item.strip() for item in args.accuracy_tensors.split(",") if item.strip()}
    unknown = selected.difference(ACCURACY_TENSORS)
    if unknown:
        raise ValueError(f"Unsupported accuracy tensor(s): {', '.join(sorted(unknown))}")

    pairs = {
        "o": (o_bsnd, ref_o),
        "dq": (dq_bsnd, ref_dq),
        "dk": (dk_bsnd, ref_dk),
        "dv": (dv_bsnd, ref_dv),
        "dbeta": (dbeta_bsnd, ref_dbeta),
        "dg": (dg_bsnd, ref_dg),
    }
    failed = []
    for name in ACCURACY_TENSORS:
        if name not in selected:
            continue
        actual, expected = pairs[name]
        tol, cos_min = tolerance_for(args, name)
        ok, line = accuracy_metric(name, actual.cpu(), expected.cpu(), tol, cos_min)
        print(line, flush=True)
        if not ok:
            failed.append(name)
    if failed:
        print(f"chunk_kda accuracy failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


def padded_layout_unsupported(layout: str, packed: bool) -> bool:
    if packed:
        return layout not in {"TND", "NTD"}
    return layout not in {"BSND", "BNSD"}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.initial_state not in ("none", "None", None) and args.initial_state != "":
        print(f"[warn] --initial-state={args.initial_state} 暂不支持，按 none 执行", flush=True)
    if args.output_final_state:
        print("[warn] --output-final-state 暂不参与看护，按 false 执行", flush=True)
    if args.demo_model:
        print("[warn] --demo-model 不适用于 ChunkKDA 看护脚本，忽略", flush=True)
    os.environ.setdefault("PYTORCH_NPU_ALLOC_CONF", "expandable_segments:True")
    return run_once(args)


if __name__ == "__main__":
    sys.exit(main())
