#!/usr/bin/env python3
"""A5 KDA CP 正反向示例与精度检查（保存中间量，不重计算）。

本文件独立包含：参数解析、输入生成、完整序列 CPU 参考、CP 通信、
分阶段前后向调用和精度检查，不依赖另一个 CP 示例 或公共示例文件。

支持范围：连续等长切分，B=1、Hk=Hv、K=V=128、BF16、chunk_size=64。
暂不覆盖变长打包序列、交错切分、外部初始状态或末状态损失；显式调用反向，
不是自动求导封装。完整 CPU 输入仅用于对照，每个进程只把自己的切片放到 NPU。

前向：Prepare -> CP pre_fwd -> all_gather/merge -> FwdH -> Finalize。
反向：Bwd Prepare -> CP pre_bwd -> all_gather/merge -> 边界适配 -> Dhu -> Finalize。
当前 Dhu 忽略 dht，因此非末尾进程追加两个虚拟分块 注入边界梯度，随后裁剪输出。

运行（需要包含 CP 算子的 A5 包）：
    torchrun --standalone --nproc_per_node=2 examples/chunk_kda_cp.py --accuracy-check
    python examples/chunk_kda_cp.py --device 0 --accuracy-check
"""
from __future__ import annotations

import argparse
import ctypes as C
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
    values.update(g=(rand(shape) * 0.1 - 5.0).float(),
                  A_log=torch.zeros(heads), dt_bias=torch.zeros(heads, 128))
    return values


def recurrent_reference(inputs, scale, local_tokens):
    """对未切分的完整序列执行独立的逐词元递推，并用自动求导计算梯度。

    梯度包含所有进程对应的损失，以及共享门控参数的梯度。
    保留边界状态及其梯度，用于检查前缀、后缀的合并顺序。
    """
    names = ["q", "k", "v", "beta", "g"]
    names += ["A_log", "dt_bias"]
    leaves = {name: inputs[name].float().detach().requires_grad_() for name in names}
    q, k, v, beta = (leaves[n] for n in ("q", "k", "v", "beta"))
    gate = leaves["g"]
    gate = -5.0 * torch.sigmoid(
        leaves["A_log"].exp()[None, :, None, None]
        * (gate + leaves["dt_bias"][None, :, None, :]))
    state = torch.zeros(1, q.shape[1], 128, 128)
    boundaries, outputs = [state], []
    for t in range(q.shape[2]):
        decay = gate[:, :, t].exp()
        state = state * decay[..., None]
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


def dhu_with_boundary(q, k, w, do, dv, gate, dht, *, scale, has_future):
    from fla_npu.ops.ascendc import chunk_gated_delta_rule_bwd_dhu

    tokens = q.shape[2]
    if has_future:
        q, k, w, do, dv, gate = append_terminal_boundary(q, k, w, do, dv, gate, dht, scale)
    dh, _, dv_scan = chunk_gated_delta_rule_bwd_dhu(
        q, k, w, do, dv, scale, 64, gK=gate, h0=None, dht=None, use_exp2=True)
    return dh[:, :tokens // 64].contiguous(), dv_scan[:, :, :tokens].contiguous()


def _nd_tensor(context, tensor, name):
    from fla_npu.ops.ascendc._runtime import ACL_FORMAT_ND

    return context.tensor(tensor, name, acl_format_override=ACL_FORMAT_ND,
                          storage_shape_override=tuple(tensor.shape) if tensor is not None else None)


def backward_prepare(aqk, v_new, do, h, scale):
    from fla_npu.ops.ascendc._runtime import call_aclnn

    outputs = (torch.empty_like(aqk, dtype=torch.float32), torch.empty_like(do),
               torch.empty_like(do, dtype=torch.float32))
    return call_aclnn(
        "aclnnChunkKdaBwdPrepare",
        lambda c: [*(_nd_tensor(c, t, n) for t, n in zip((aqk, v_new, do, h),
                                                    ("aqk", "v_new", "do", "h"))),
                   c.int_array(None), c.int_array(None), C.c_double(scale),
                   C.c_int64(64), C.c_bool(False),
                   *(_nd_tensor(c, t, f"out{i}") for i, t in enumerate(outputs))], outputs,
        get_workspace_argtypes=[C.c_void_p] * 6 + [C.c_double, C.c_int64, C.c_bool]
        + [C.c_void_p] * 3 + [C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)])


def backward_finalize(x, saved, h, v_new, dh, dv_scan, d_aqk, dq_raw, scale):
    from fla_npu.ops.ascendc._runtime import call_aclnn

    gk, _, akk = saved[:3]
    outputs = tuple(torch.empty_like(x[n]) for n in
                    ("q", "k", "v", "beta", "g", "A_log", "dt_bias"))
    tensors = (x["q"], x["k"], x["v"], gk, x["g"], x["beta"],
               x["A_log"], x["dt_bias"], akk, v_new, h, dh, dv_scan, d_aqk, dq_raw,
               None, None)
    return call_aclnn(
        "aclnnChunkKdaBwdFinalize",
        lambda c: [*(_nd_tensor(c, t, f"in{i}") for i, t in enumerate(tensors)),
                   c.int_array(None), c.int_array(None), C.c_double(scale),
                   C.c_double(-5.0), C.c_int64(64), C.c_bool(True), C.c_bool(True),
                   C.c_bool(True), C.c_bool(False),
                   *(_nd_tensor(c, t, f"out{i}") for i, t in enumerate(outputs))], outputs,
        get_workspace_argtypes=[C.c_void_p] * 19 + [C.c_double, C.c_double, C.c_int64]
        + [C.c_bool] * 4 + [C.c_void_p] * 7
        + [C.POINTER(C.c_uint64), C.POINTER(C.c_void_p)])


def run_kda(x, *, scale, rank, world):
    from fla_npu.ops.ascendc import (
        chunk_kda_fwd_prepare, chunk_fwd_h, chunk_kda_fwd_finalize,
        pre_process_fwd_kernel_merged, chunk_delta_h_bwd_preprocess,
    )

    q, k, v, g, beta, do = (x[n] for n in ("q", "k", "v", "g", "beta", "do"))
    saved = chunk_kda_fwd_prepare(
        q, k, v, g, beta, scale, layout="BNSD", safe_gate=True,
        lower_bound=-5.0, use_gate_in_kernel=True, A_log=x["A_log"],
        dt_bias=x["dt_bias"].flatten(), use_qk_l2norm_in_kernel=False,
        use_exp2=True, backward_mode="save")
    gk, aqk, akk, w, u, qg, kg, qg_scaled = saved[:8]
    # CP 的 gk 分支直接计算 kg.T @ (u-w@h)，不会再给键乘逐词元门控。
    # 因此这里传入 Prepare 生成的 kg，而不是原始 k。
    hm = pre_process_fwd_kernel_merged(kg, w, u, gk=gk,
                                      cu_seqlens=[0, q.shape[2]], chunk_size=64)
    h0 = merge_boundary(hm[0], forward=True, rank=rank, world=world)
    h, v_new, _ = chunk_fwd_h(kg, w, u, gk=gk, initial_state=h0,
                              use_exp2=True, save_new_value=True)
    o = chunk_kda_fwd_finalize(qg_scaled, aqk, v_new, h, output_layout="BNSD")

    d_aqk, dv_local, dq_raw = backward_prepare(aqk, v_new, do, h, scale)
    # 此处使用未乘 scale 的 qg；qg_scaled 仅供前向 Finalize 使用。
    # CP 反向接口当前要求 gk 为 BF16 或 FP16。
    dhm = chunk_delta_h_bwd_preprocess(qg, kg, w, do, dv_local, scale, 64,
                                      gk=gk.to(q.dtype), cu_seqlens=[0, q.shape[2]])
    dht = merge_boundary(dhm, forward=False, rank=rank, world=world)
    dh, dv_scan = dhu_with_boundary(
        qg, kg, w, do, dv_local, gk, dht, scale=scale,
        has_future=rank + 1 < world)
    gradients = backward_finalize(x, saved, h, v_new, dh, dv_scan, d_aqk, dq_raw, scale)
    return dict(zip(("dq", "dk", "dv", "dbeta", "dg", "dA_log", "ddt_bias"), gradients),
                o=o, h0=h0, dht=dht)



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
        print("kda: CPU reference only OK; no NPU/CP operators executed")
        return 0

    # 先加载包内 OPP，再显式初始化 NPU 运行环境。
    import fla_npu  # noqa: F401
    import torch_npu  # noqa: F401

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device_id = local_rank if args.device is None else args.device
    torch.npu.set_device(device_id)
    device = torch.device(f"npu:{device_id}")
    if "Ascend950" not in torch.npu.get_device_name(device_id):
        raise RuntimeError("This staged example currently requires A5 / Ascend950")
    if world > 1:
        dist.init_process_group("hccl", timeout=timedelta(seconds=180))
    try:
        lo, hi = rank * local_tokens, (rank + 1) * local_tokens
        local = {name: (x[:, :, lo:hi].contiguous() if name not in ("A_log", "dt_bias") else x)
                 .to(device) for name, x in source.items()}
        with torch.no_grad():
            actual = run_kda(local, scale=scale, rank=rank, world=world)
        # 共享门控参数的梯度需要跨序列切片求和，不取平均值。
        if world > 1:
            for name in ("dA_log", "ddt_bias"):
                if name in actual:
                    dist.all_reduce(actual[name], op=dist.ReduceOp.SUM)
        torch.npu.synchronize()
        passed = all(bool(torch.isfinite(x).all()) for x in actual.values())
        if args.accuracy_check:
            ref, states, dstates = recurrent_reference(source, scale, local_tokens)
            ref = {name: (x[:, :, lo:hi] if name not in ("dA_log", "ddt_bias") else x)
                   for name, x in ref.items()}
            ref.update(h0=states[rank], dht=dstates[rank + 1])
            for name, expected in ref.items():
                gate_grad = name in ("dbeta", "dg", "dA_log", "ddt_bias")
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
            print(f"kda CP world={world} {mode}: {'PASS' if passed else 'FAIL'}", flush=True)
        return 0 if passed else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
