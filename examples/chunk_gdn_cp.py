#!/usr/bin/env python3
"""A2/A3/A5 GDN CP 正反向示例与精度检查（保存中间量，不重计算）。

本文件独立包含：参数解析、输入生成、完整序列 CPU 参考、CP 通信、
分阶段前后向调用和精度检查，不依赖另一个 CP 示例 或公共示例文件。

支持范围：连续等长切分，B=1、Hk=Hv、K=V=128、BF16、chunk_size=64。
暂不覆盖变长打包序列、交错切分、外部初始状态或末状态损失；显式调用反向，
不是自动求导封装。完整 CPU 输入仅用于对照，每个进程只把自己的切片放到 NPU。

前向：Prepare -> CP pre_fwd -> all_gather/merge -> FwdH -> FwdO。
反向：DvLocal -> CP pre_bwd -> all_gather/merge -> 边界适配 -> Dhu -> Finalize。
当前 Dhu 忽略 dht，因此非末尾进程追加两个虚拟分块 注入边界梯度，随后裁剪输出。

A2/A3 将 Prepare 和反向 Finalize 展开为独立小算子，普通 GDN 阶段使用
自然对数门控，仅 CP 预处理使用转换后的 log2 门控。

运行（需要包含 CP 和对应 GDN 小算子的本机架构算子包）：
    torchrun --standalone --nproc_per_node=2 examples/chunk_gdn_cp.py --accuracy-check
    python examples/chunk_gdn_cp.py --device 0 --accuracy-check
"""
from __future__ import annotations

import argparse
import math
from datetime import timedelta
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=256, help="切分前的全局序列长度")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--device", type=int, default=None, help="单进程使用的设备编号；torchrun 按 LOCAL_RANK 分配设备")
    parser.add_argument("--accuracy-check", action="store_true")
    # 沿用 chunk_kda.py 的逐元素容差和余弦相似度判据，额外检查 CP 边界状态。
    parser.add_argument("--output-tol", type=float, default=0.005)
    parser.add_argument("--grad-tol", type=float, default=0.008)
    parser.add_argument("--gate-tol", type=float, default=0.02)
    parser.add_argument("--boundary-tol", type=float, default=0.02)
    parser.add_argument("--cpu-reference-only", action="store_true", help="仅检查 CPU 参考，不执行 NPU 算子或跨卡通信")
    args = parser.parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if args.tokens <= 0 or args.tokens % (64 * world):
        parser.error("--tokens must be a positive multiple of 64 * WORLD_SIZE")
    if not 1 <= args.heads <= 128:
        parser.error("--heads must be in [1,128]")
    if min(args.output_tol, args.grad_tol, args.gate_tol, args.boundary_tol) <= 0:
        parser.error("tolerances must be positive")
    if args.cpu_reference_only and world != 1:
        parser.error("--cpu-reference-only must run without torchrun")
    if world > 1 and args.device is not None:
        parser.error("use ASCEND_RT_VISIBLE_DEVICES with torchrun, not --device")
    return args


def make_inputs(tokens, heads, seed):
    """与 chunk_kda.py 一致，先将输入转换到实际计算精度，再计算参考结果。

    采用较缓的衰减以保留跨进程影响；衰减过强可能掩盖 CP 状态未正确传递的问题。
    """
    gen = torch.Generator().manual_seed(seed)
    shape = (1, heads, tokens, 128)
    rand = lambda s: torch.randn(s, generator=gen)
    values = {"q": F.normalize(rand(shape), dim=-1).bfloat16(),
              "k": F.normalize(rand(shape), dim=-1).bfloat16(),
              "v": (0.5 * rand(shape)).bfloat16(),
              "beta": torch.sigmoid(rand(shape[:-1])).float(),
              "do": rand(shape).bfloat16()}
    # GDN 输入 g 是自然对数域的逐词元门控；Prepare 输出以 2 为底的分块累积值。
    values["g"] = (-0.01 - 0.01 * torch.rand(shape[:-1], generator=gen)).float()
    return values


def recurrent_reference(inputs, scale, local_tokens):
    """对未切分的完整序列执行独立的逐词元递推，并用自动求导计算梯度。

    梯度包含整个序列的损失，因此前面切片也会收到后面切片传来的梯度。
    保留边界状态及其梯度，用于检查前缀、后缀的合并顺序。
    """
    names = ["q", "k", "v", "beta", "g"]
    leaves = {name: inputs[name].float().detach().requires_grad_() for name in names}
    q, k, v, beta = (leaves[n] for n in ("q", "k", "v", "beta"))
    gate = leaves["g"]
    state = torch.zeros(1, q.shape[1], 128, 128)
    boundaries, outputs = [state], []
    for t in range(q.shape[2]):
        decay = gate[:, :, t].exp()
        state = state * decay[..., None, None]
        key = k[:, :, t]
        delta = (v[:, :, t] - (key.unsqueeze(-1) * state).sum(-2)) * beta[:, :, t, None]
        state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((q[:, :, t, :, None] * state).sum(-2) * scale)
        if (t + 1) % local_tokens == 0:
            # 在当前词元输出之后新增计算图边，使此处梯度只包含后续进程的损失，
            # 与传入本进程的末状态梯度 dht 保持一致。
            state = state + 0.0
            state.retain_grad()
            boundaries.append(state)
    out = torch.stack(outputs, dim=2)
    (out * inputs["do"].float()).sum().backward()
    result = {"o": out.detach()}
    result.update({"d" + n: leaf.grad.detach() for n, leaf in leaves.items()})
    # 全局末状态不参与额外损失，其后也没有词元，因此末状态梯度为零。
    dh = [torch.zeros_like(boundaries[0])]
    dh += [s.grad.detach() if s.grad is not None else torch.zeros_like(s)
           for s in boundaries[1:]]
    return result, [s.detach() for s in boundaries], dh


def metric(actual, expected, tol, cos_min=0.999):
    actual, expected = actual.detach().float().cpu(), expected.detach().float().cpu()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
    error = float(torch.linalg.vector_norm(actual - expected)
                  / torch.linalg.vector_norm(expected).clamp_min(1e-8))
    close = bool(torch.allclose(actual, expected, rtol=tol, atol=tol))
    if expected.count_nonzero().item() == 0:
        cosine = 1.0 if close else 0.0
    else:
        cosine = float(F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0))
    return finite and close and cosine >= cos_min, error, cosine


def merge_boundary(summary, *, forward, rank, world):
    """所有进程都参与通信，包括无需接收前缀或后缀状态的首尾进程。

    进程编号顺序就是序列顺序。通信和合并使用 FP32，再将 [H,K,V] 转成
    FwdH/Dhu 所需的 BF16 [1,H,K,V]。合并算子不接受 N=0，
    因此前向首进程、反向末进程直接使用零边界状态。
    """
    from fla_npu.ops.ascendc import merge_fwd_bwd_kernel

    summary = summary.reshape(-1, 128, 256).float().contiguous()
    gathered = [torch.empty_like(summary) for _ in range(world)]
    if world > 1:
        dist.all_gather(gathered, summary)
    else:
        gathered[0].copy_(summary)
    count = rank if forward else world - rank - 1
    boundary = torch.zeros(summary.shape[0], 128, 128,
                           device=summary.device, dtype=torch.float32)
    if count:
        merge_fwd_bwd_kernel(boundary, torch.stack(gathered), count, rank,
                             forward=forward, state_v_first=False)
    return boundary.unsqueeze(0).to(torch.bfloat16)


def append_terminal_boundary(q, k, w, do, dv, gate, dht, scale):
    """用两个虚拟分块向当前从零末状态开始的 Dhu 注入 dht。

    当前仓库中的 Dhu 内核忽略 dht 参数。在 K=128、BT=64 时追加 128 个词元，
    设置 Q=I、dO=dht/scale、K=W=dV=gate=0。反向扫描先经过这些词元，
    在进入真实序列之前累加 scale * I.T @ (dht/scale) = dht。
    此适配会增加两个分块的计算和 BF16 舍入，但能复用现有 NPU 内核传递边界梯度。
    """
    if scale == 0:
        raise ValueError("terminal boundary adapter requires nonzero scale")
    batch, heads, _, dim = q.shape
    if batch != 1 or dim != 128 or k.shape != q.shape or w.shape != q.shape:
        raise ValueError("terminal adapter requires B=1, Hk=Hv, K=128")
    identity = torch.eye(128, device=q.device, dtype=q.dtype)[None, None].expand(1, heads, -1, -1)
    zero_key = torch.zeros_like(identity)
    zero_value = torch.zeros(1, heads, 128, dv.shape[-1], device=dv.device, dtype=dv.dtype)
    zero_gate = torch.zeros(*gate.shape[:2], 128, *gate.shape[3:], device=gate.device, dtype=gate.dtype)
    tails = (identity, zero_key, zero_key, (dht.float() / scale).to(do.dtype), zero_value, zero_gate)
    return tuple(torch.cat((x, tail), dim=2).contiguous()
                 for x, tail in zip((q, k, w, do, dv, gate), tails))


def dhu_with_boundary(q, k, w, do, dv, gate, dht, *, scale, has_future, use_exp2=True):
    from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd_dhu

    tokens = q.shape[2]
    if has_future:
        q, k, w, do, dv, gate = append_terminal_boundary(q, k, w, do, dv, gate, dht, scale)
    dh, _, dv_scan = chunk_gated_delta_rule_bwd_dhu(
        q, k, w, do, dv, scale, 64, g=gate, h0=None, dht=None, use_exp2=use_exp2)
    return dh[:, :tokens // 64].contiguous(), dv_scan[:, :, :tokens].contiguous()


def run_gdn_a5(x, *, scale, rank, world):
    from fla_npu.ops.ascendc import (
        chunk_gated_delta_rule_fwd_prepare, chunk_fwd_h, chunk_fwd_o,
        pre_process_fwd_kernel_merged, chunk_delta_h_bwd_preprocess,
        chunk_bwd_dv_local,
        chunk_gated_delta_rule_bwd_finalize,
    )

    q, k, v, g, beta, do = (x[n] for n in ("q", "k", "v", "g", "beta", "do"))
    q, k, _, _, beta_eff, gc, w, u, a = chunk_gated_delta_rule_fwd_prepare(
        q, k, v, g, beta, use_exp2=True, output_a=True)
    hm = pre_process_fwd_kernel_merged(k, w, u, g=gc,
                                      cu_seqlens=[0, q.shape[2]], chunk_size=64)
    h0 = merge_boundary(hm[0], forward=True, rank=rank, world=world)
    h, v_new, _ = chunk_fwd_h(k, w, u, g=gc, initial_state=h0,
                              use_exp2=True, save_new_value=True)
    # A5 的 exp2 FwdO 路径要求输出 BSND/TND，此处再转回统一使用的 BNSD。
    o = chunk_fwd_o(q, k, v_new, h, scale, g=gc, use_exp2=True,
                    output_layout="BSND").permute(0, 2, 1, 3).contiguous()

    # DvLocal 没有 use_exp2 开关，内部使用 exp()，因此只将它的门控输入转回
    # 自然对数域；其他阶段仍使用以 2 为底的累积门控 gc。
    dv_local = chunk_bwd_dv_local(q, k, do, gc * math.log(2.0), scale, 64)
    dhm = chunk_delta_h_bwd_preprocess(q, k, w, do, dv_local, scale, 64,
                                      g=gc, cu_seqlens=[0, q.shape[2]])
    dht = merge_boundary(dhm, forward=False, rank=rank, world=world)
    dh, du = dhu_with_boundary(q, k, w, do, dv_local, gc, dht,
                              scale=scale, has_future=rank + 1 < world)
    gradients = chunk_gated_delta_rule_bwd_finalize(
        q, k, v, v_new, do, du, gc, beta_eff, h, dh, a,
        scale=scale, use_exp2=True)
    return dict(zip(("dq", "dk", "dv", "dbeta", "dg"), gradients), o=o, h0=h0, dht=dht)


def run_gdn_a2_a3(x, *, scale, rank, world):
    """复用 A2/A3 独立小算子，在状态递推之前插入 CP 边界通信。

    前向保存 A、w、h、v_new，反向直接读取，不重算前向中间量。
    gc 是自然对数域的分块累积门控，gc_cp 仅供两个 CP 预处理使用。
    """
    from fla_npu.ops.ascendc import (
        chunk_local_cumsum, chunk_scaled_dot_kkt, solve_tri, recompute_w_u_fwd,
        chunk_gated_delta_rule_fwd_h, chunk_fwd_o, chunk_bwd_dv_local,
        pre_process_fwd_kernel_merged, chunk_delta_h_bwd_preprocess,
        chunk_bwd_dqkwg, prepare_wy_repr_bwd_da, prepare_wy_repr_bwd_full,
    )

    q, k, v, g, beta, do = (x[n] for n in ("q", "k", "v", "g", "beta", "do"))
    gc = chunk_local_cumsum(g, 64, head_first=True)
    gc_cp = (gc / math.log(2.0)).contiguous()
    kkt = chunk_scaled_dot_kkt(k, gc, beta, chunk_size=64)
    # SolveTri 接收 BSND，其他 WY 阶段使用 BNSD；输入输出均转为 BF16。
    a = solve_tri(kkt.permute(0, 2, 1, 3).to(q.dtype).contiguous(), layout="bsnd")
    a = a.permute(0, 2, 1, 3).contiguous()
    w, u = recompute_w_u_fwd(k, v, beta, a, 64, g=gc)
    hm = pre_process_fwd_kernel_merged(k, w, u, g=gc_cp,
                                      cu_seqlens=[0, q.shape[2]], chunk_size=64)
    h0 = merge_boundary(hm[0], forward=True, rank=rank, world=world)
    h, v_new, _ = chunk_gated_delta_rule_fwd_h(k, w, u, g=gc, initial_state=h0,
                                               chunk_size=64)
    o = chunk_fwd_o(q, k, v_new, h, scale, g=gc, use_exp2=False)

    dv_local = chunk_bwd_dv_local(q, k, do, gc, scale, 64)
    dhm = chunk_delta_h_bwd_preprocess(q, k, w, do, dv_local, scale, 64,
                                      g=gc_cp, cu_seqlens=[0, q.shape[2]])
    dht = merge_boundary(dhm, forward=False, rank=rank, world=world)
    dh, du = dhu_with_boundary(q, k, w, do, dv_local, gc, dht,
                              scale=scale, has_future=rank + 1 < world, use_exp2=False)
    dq, dk, dw, dg = chunk_bwd_dqkwg(q, k, v_new, gc, h, do, dh, du, 64,
                                    scale=scale, use_exp2=False)
    da = prepare_wy_repr_bwd_da(k, v, beta, a, dw, du, gc, chunk_size=64)
    dk_wy, dv, dbeta, dg_wy = prepare_wy_repr_bwd_full(k, v, beta, a, da, dw, du, gc, 64)
    # 两路贡献先相加，再对分块累积门控做反向累加，得到原始逐词元 g 的梯度。
    dg = chunk_local_cumsum((dg + dg_wy).contiguous(), 64, reverse=True, head_first=True)
    return dict(o=o, dq=dq, dk=dk + dk_wy, dv=dv, dbeta=dbeta, dg=dg, h0=h0, dht=dht)


def run_gdn(x, *, scale, rank, world, architecture="a5"):
    runner = run_gdn_a5 if architecture == "a5" else run_gdn_a2_a3
    return runner(x, scale=scale, rank=rank, world=world)



def main(argv=None):
    args = parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    torch.set_num_threads(1)
    local_tokens = args.tokens // world
    source = make_inputs(args.tokens, args.heads, args.seed)
    scale = 128 ** -0.5
    if args.cpu_reference_only:
        outputs, _, _ = recurrent_reference(source, scale, local_tokens)
        if not all(bool(torch.isfinite(x).all()) for x in outputs.values()):
            raise RuntimeError("nonfinite CPU reference")
        print("gdn: CPU reference only OK; no NPU/CP operators executed")
        return 0

    # 先加载包内 OPP，再显式初始化 NPU 运行环境。
    import fla_npu  # noqa: F401
    import torch_npu  # noqa: F401

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_id = local_rank if args.device is None else args.device
    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    device_name = torch.npu.get_device_name(device_id)
    if "Ascend950" in device_name:
        architecture = "a5"
    elif "Ascend910B" in device_name or "Ascend910_93" in device_name:
        architecture = "a2_a3"
    else:
        raise RuntimeError(f"此示例仅支持 A2/A3/A5，当前设备为 {device_name}")
    if rank == 0:
        print(f"GDN CP: device={device_name}, architecture={architecture}, world={world}", flush=True)
    if world > 1:
        dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        lo, hi = rank * local_tokens, (rank + 1) * local_tokens
        local = {name: x[:, :, lo:hi].contiguous().to(device) for name, x in source.items()}
        with torch.no_grad():
            actual = run_gdn(local, scale=scale, rank=rank, world=world, architecture=architecture)
        torch.npu.synchronize()
        passed = all(bool(torch.isfinite(x).all()) for x in actual.values())
        if args.accuracy_check:
            ref, states, dstates = recurrent_reference(source, scale, local_tokens)
            ref = {name: x[:, :, lo:hi]
                   for name, x in ref.items()}
            ref.update(h0=states[rank], dht=dstates[rank + 1])
            for name, expected in ref.items():
                gate_grad = name in ("dbeta", "dg")
                tol = (args.boundary_tol if name in ("h0", "dht") else
                       args.gate_tol if gate_grad else
                       args.output_tol if name == "o" else args.grad_tol)
                ok, error, cosine = metric(actual[name], expected, tol, 0.99 if gate_grad else 0.999)
                passed &= ok
                print(f"rank={rank} {name}: rel_l2={error:.6g} cosine={cosine:.6g} "
                      f"tol={tol:g} {'PASS' if ok else 'FAIL'}", flush=True)
        flag = torch.tensor([int(passed)], device=device, dtype=torch.int32)
        if world > 1:
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        passed = bool(flag.item())
        if rank == 0:
            mode = "accuracy" if args.accuracy_check else "finite smoke (accuracy unchecked)"
            print(f"gdn CP world={world} {mode}: {'PASS' if passed else 'FAIL'}", flush=True)
        return 0 if passed else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
