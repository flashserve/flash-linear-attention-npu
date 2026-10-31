#!/usr/bin/env python3
"""GDN forward/backward: no CP, AllGather, or PTP (recursive doubling).

A2/A3/A5, B=1, Hv/Hk=1..4, K=V=128, BF16 QKV, FP32 gate/beta.
Equal contiguous or zigzag token shards; packed sequences reset at global boundaries.
Q/K are pre-normalized; beta is already activated; g is a natural-log token gate.
Saved intermediates (no recompute), chunk_size=64; token outputs are BSND/BSN.
Explicit backward calls, no autograd wrapper or external state/state loss.

python examples/cp/chunk_gdn_cp.py --communication none --accuracy-check
torchrun --standalone --nproc_per_node=4 examples/cp/chunk_gdn_cp.py \
    --communication allgather --accuracy-check
Replace allgather with ptp to run the ordered P/E scan from MindSpeed issue #4.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--communication", choices=("none", "allgather", "ptp"), default="allgather")
    parser.add_argument("--tokens", type=int, default=256, help="global token count")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--partition", choices=("contiguous", "zigzag"), default="contiguous")
    parser.add_argument("--value-heads", type=int, help="value heads; defaults to --heads, Hv/Hk=1..4")
    parser.add_argument("--cu-seqlens", help="global packed offsets, e.g. 0,130,256")
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--device", type=int)
    parser.add_argument("--accuracy-check", action="store_true", help="compare with CPU recurrence (small inputs)")
    parser.add_argument("--cpu-reference-only", action="store_true")
    parser.add_argument("--reference", type=Path, help="compare with a saved no-CP NPU result")
    parser.add_argument("--save-result", type=Path, help="save outputs/gradients from a single-rank run")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--profile", action="store_true", help="save CANN op_summary and timeline after timing")
    parser.add_argument("--output-tol", type=float, default=0.005)
    parser.add_argument("--grad-tol", type=float, default=0.008)
    parser.add_argument("--gate-tol", type=float, default=0.02)
    parser.add_argument("--boundary-tol", type=float, default=0.02)
    args = parser.parse_args(argv)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    parts = world * (2 if args.partition == "zigzag" else 1)
    if world < 1 or args.tokens < parts or args.tokens % parts:
        parser.error("tokens must be at least and divisible by WORLD_SIZE (2*WORLD_SIZE for zigzag)")
    if args.heads < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("heads must be positive, warmup >= 0, iterations >= 1")
    args.value_heads = args.heads if args.value_heads is None else args.value_heads
    if args.value_heads % args.heads or not 1 <= args.value_heads // args.heads <= 4:
        parser.error("value-heads must be heads times an integer in [1,4]")
    if min(args.output_tol, args.grad_tol, args.gate_tol, args.boundary_tol) <= 0:
        parser.error("tolerances must be positive")
    if world > 1 and (args.communication == "none" or args.device is not None
                      or args.cpu_reference_only or args.save_result is not None):
        parser.error("none/device/cpu-reference-only/save-result require a single process")
    if args.communication == "none" and args.partition != "contiguous":
        parser.error("--partition applies to CP; use contiguous for the no-CP reference")
    if args.profile and args.output_dir is None:
        parser.error("--profile requires --output-dir")
    if (args.accuracy_check or args.cpu_reference_only) and args.tokens * args.value_heads > 8192:
        parser.error("CPU autograd reference is limited to tokens*value_heads <= 8192; use --reference for large cases")
    try:
        args.cu = ([int(s) for s in args.cu_seqlens.split(",")]
                   if args.cu_seqlens else [0, args.tokens])
    except ValueError:
        parser.error("cu-seqlens must contain comma-separated integers")
    if (len(args.cu) < 2 or args.cu[0] != 0 or args.cu[-1] != args.tokens
            or any(b < a for a, b in zip(args.cu, args.cu[1:]))):
        parser.error("cu-seqlens must be nondecreasing from 0 to tokens")
    return args


def build_partition(cu, tokens, rank, world, partition):
    """Match FLA's equal intervals and local [front; back] order for zigzag."""
    num_parts = world * (2 if partition == "zigzag" else 1)
    part_len = tokens // num_parts
    slots = [rank] if partition == "contiguous" else [rank, num_parts - 1 - rank]
    plan = dict(rank=rank, world=world, num_parts=num_parts, part_len=part_len, cu=[0], parts=[])
    for i, slot in enumerate(slots):
        lo, hi = slot * part_len, (slot + 1) * part_len
        segments = [(a, b) for a, b in zip(cu, cu[1:]) if a < hi and b > lo and b > a]
        first = len(plan["cu"]) - 1
        plan["cu"].extend(i * part_len + min(b, hi) - lo for a, b in segments)
        plan["parts"].append(dict(slot=slot, lo=lo, hi=hi, first=first, last=len(plan["cu"]) - 2,
                                  prefix=slot - segments[0][0] // part_len,
                                  suffix=(segments[-1][1] - 1) // part_len - slot))
    plan["indices"] = [value for seq, (a, b) in enumerate(zip(plan["cu"], plan["cu"][1:]))
                       for chunk in range((b - a + 63) // 64) for value in (seq, chunk)]
    return plan


def make_inputs(tokens, heads, seed, lo=0, hi=None, device="cpu", value_heads=None):
    """Preserve the original GDN input distribution; move only this shard to NPU."""
    gen = torch.Generator().manual_seed(seed)
    value_heads = heads if value_heads is None else value_heads
    result = {}
    for name in ("q", "k", "v", "beta", "do", "g"):
        h = heads if name in ("q", "k") else value_heads
        shape = (1, h, tokens) if name in ("beta", "g") else (1, h, tokens, 128)
        if name == "g":
            value = -0.01 - 0.01 * torch.rand(shape, generator=gen)
        else:
            value = torch.randn(shape, generator=gen)
            if name in ("q", "k"):
                value = F.normalize(value, dim=-1)
            elif name == "v":
                value *= 0.5
            elif name == "beta":
                value = value.sigmoid()
            if name != "beta":
                value = value.bfloat16()
        result[name] = value[:, :, lo:hi].contiguous().to(device)
    return result


@torch.enable_grad()
def recurrent_reference(inputs, cu, local_tokens):
    """对未切分的完整序列执行独立的逐词元递推，并用自动求导计算梯度。

    梯度包含整个序列的损失，因此前面切片也会收到后面切片传来的梯度。
    保留边界状态及其梯度，用于检查前缀、后缀的合并顺序。
    """
    names = ["q", "k", "v", "beta", "g"]
    leaves = {name: inputs[name].float().detach().requires_grad_() for name in names}
    q, k, v, beta = (leaves[n] for n in ("q", "k", "v", "beta"))
    gate = leaves["g"]
    # 同一组 value head 共享一个 Q/K head；自动求导会将组内贡献求和回原始叶子。
    ratio = v.shape[1] // q.shape[1]
    head_index = torch.arange(v.shape[1]) // ratio
    q = q.index_select(1, head_index)
    k = k.index_select(1, head_index)
    state = torch.zeros(1, v.shape[1], 128, 128)
    boundaries, outputs = [state], []
    for t in range(q.shape[2]):
        if t in cu:
            state = torch.zeros_like(state)
        decay = gate[:, :, t].exp()
        state = state * decay[..., None, None]
        key = k[:, :, t]
        delta = (v[:, :, t] - (key.unsqueeze(-1) * state).sum(-2)) * beta[:, :, t, None]
        state = state + key.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((q[:, :, t, :, None] * state).sum(-2) * 128 ** -0.5)
        if (t + 1) % local_tokens == 0:
            # 在当前词元输出之后新增计算图边，使此处梯度只包含后续进程的损失，
            # 与传入本进程的末状态梯度 dht 保持一致。
            state = state + 0.0
            state.retain_grad()
            boundaries.append(state)
    out = torch.stack(outputs, dim=2)
    (out * inputs["do"].float()).sum().backward()
    result = {"o": out.detach().transpose(1, 2).contiguous()}
    result.update({"d" + n: leaf.grad.detach().transpose(1, 2).contiguous() for n, leaf in leaves.items()})
    # 全局末状态不参与额外损失，其后也没有词元，因此末状态梯度为零。
    dh = [torch.zeros_like(boundaries[0])]
    dh += [s.grad.detach() if s.grad is not None else torch.zeros_like(s)
           for s in boundaries[1:]]
    return result, [s.detach() for s in boundaries], dh


def exchange_boundary(summaries, plan, *, forward, communication):
    """Summary [E | M] maps H -> M @ H + E. Return only same-sequence state.

    PTP follows issue #4 section 4.6: inclusive P is sent, exclusive E is local.
    Every round sends the OLD P; compositions use fresh buffers. Reverse backward
    interval order, but keep the same matrix multiplication order. A packed reset
    zeroes outgoing M, so earlier sequences cannot leak through the scan.
    """
    world, num_parts = plan["world"], plan["num_parts"]
    parts = plan["parts"]
    states = summaries.new_zeros((len(plan["cu"]) - 1, summaries.shape[1], 128, 128))
    if communication == "allgather":
        from fla_npu.ops.ascendc import merge_fwd_bwd_kernel
        gathered = [torch.empty_like(summaries) for _ in range(world)]
        dist.all_gather(gathered, summaries.contiguous())
        gathered = torch.stack(gathered)
        # Global sequence order is fronts ascending, then backs descending.
        chain = (gathered[:, 0].contiguous() if len(parts) == 1
                 else torch.cat((gathered[:, 0], gathered[:, 1].flip(0))))
        for part in parts:
            count = part["prefix" if forward else "suffix"]
            if count:
                merge_fwd_bwd_kernel(
                    states[part["first" if forward else "last"]], chain, count, part["slot"],
                    forward=forward, state_v_first=False,
                )
    elif communication == "ptp":
        # Logical order follows the scan direction; zigzag puts two nodes on each rank.
        slots = [p["slot"] if forward else num_parts - 1 - p["slot"] for p in parts]
        owners = list(range(world)) if len(parts) == 1 else list(range(world)) + list(range(world - 1, -1, -1))
        if not forward:
            owners.reverse()
        inclusive = {slot: summaries[i].clone() for i, slot in enumerate(slots)}
        exclusive = {slot: None for slot in slots}
        for slot, part in zip(slots, parts):
            if part["first"] != part["last"] or part["prefix" if forward else "suffix"] == 0:
                inclusive[slot][..., 128:].zero_()
        distance = 1
        while distance < num_parts:
            received = {}
            operations = []
            # Sort by global edge to match repeated messages between the same peers
            # without transport tags. Same-rank edges use the OLD local snapshot.
            edges = {(slot, slot + distance) for slot in slots if slot + distance < num_parts}
            edges.update((slot - distance, slot) for slot in slots if slot >= distance)
            for source, target in sorted(edges):
                if source in inclusive and target in inclusive:
                    received[target] = inclusive[source]
                elif source in inclusive:
                    operations.append(dist.P2POp(dist.isend, inclusive[source], owners[target]))
                else:
                    received[target] = torch.empty_like(inclusive[target])
                    operations.append(dist.P2POp(dist.irecv, received[target], owners[source]))
            if operations:
                for request in dist.batch_isend_irecv(operations):
                    request.wait()
            updated = inclusive.copy()
            for slot, earlier in received.items():
                # compose(earlier, later) = [M_later E_earlier + E_later | M_later M_earlier]
                if slot + 2 * distance < num_parts:
                    merged = torch.bmm(inclusive[slot][..., 128:].contiguous(), earlier)
                    merged[..., :128].add_(inclusive[slot][..., :128])
                    updated[slot] = merged
                if exclusive[slot] is None:
                    exclusive[slot] = earlier
                else:
                    merged = torch.bmm(exclusive[slot][..., 128:].contiguous(), earlier)
                    merged[..., :128].add_(exclusive[slot][..., :128])
                    exclusive[slot] = merged
            inclusive = updated
            distance *= 2
        for slot, part in zip(slots, parts):
            if part["prefix" if forward else "suffix"]:
                states[part["first" if forward else "last"]].copy_(exclusive[slot][..., :128])
    else:
        raise ValueError("boundary exchange requires allgather or ptp")
    # FwdH selects recurrent-state precision from h0: preserve FP32.
    # Dhu's terminal gradient ABI instead requires BF16.
    return states if forward else states.bfloat16()


def run_gdn(x, plan, communication, *, is_a5=False):
    """Keep all forward/backward stages and both boundary exchanges visible."""
    from fla_npu.ops import ascendc as op
    q, k, v, g, beta, do = (x[n] for n in ("q", "k", "v", "g", "beta", "do"))
    cu, indices, scale = plan["cu"], plan["indices"], 128 ** -0.5
    q, k, _, _, beta_eff, gc, w, u, a = op.chunk_gated_delta_rule_fwd_prepare(
        q, k, v, g, beta, cu_seqlens=cu, chunk_indices=indices,
        use_exp2=True, output_a=True,
    )
    # Preserve FP32 recurrent state in both the no-CP baseline and CP runs.
    h0 = q.new_zeros((len(cu) - 1, v.shape[1], 128, 128), dtype=torch.float32)
    if communication != "none":
        summaries = q.new_zeros((len(plan["parts"]), v.shape[1], 128, 256), dtype=torch.float32)
        for i, part in enumerate(plan["parts"]):
            if part["suffix"]:
                summaries[i] = op.pre_process_fwd_kernel_merged(
                    k, w, u, g=gc, cu_seqlens=cu[part["last"]:part["last"] + 2], chunk_size=64,
                )[0]
        h0 = exchange_boundary(summaries, plan, forward=True, communication=communication)
    h, vn, _ = op.chunk_fwd_h(
        k, w, u, g=gc, initial_state=h0, cu_seqlens=cu, chunk_indices=indices,
        save_new_value=True, output_final_state=False, use_exp2=True,
    )
    # A2/A3 FwdO requires exp() and BNSD; A5 supports exp2() and direct BSND.
    gc_ln = gc * math.log(2.0)
    o = op.chunk_fwd_o(q, k, vn, h, scale, g=gc if is_a5 else gc_ln,
                       cu_seqlens=cu, chunk_indices=indices, use_exp2=is_a5,
                       output_layout="BSND" if is_a5 else "BNSD")
    if not is_a5:
        o = o.transpose(1, 2)
    # DvLocal always uses exp(); other stages retain Prepare's log2 gate.
    dv_local = op.chunk_bwd_dv_local(
        q, k, do, gc_ln, scale, 64, cu_seqlens=cu, chunk_indices=indices,
    )
    dht = None
    if communication != "none":
        summaries = q.new_zeros((len(plan["parts"]), v.shape[1], 128, 256), dtype=torch.float32)
        for i, part in enumerate(plan["parts"]):
            if part["prefix"]:
                summaries[i] = op.pre_process_bwd_kernel_merged(
                    q, k, w, do, dv_local, scale, 64,
                    g=gc, cu_seqlens=cu[part["first"]:part["first"] + 2],
                )
        dht = exchange_boundary(summaries, plan, forward=False, communication=communication)
    dh, _, du = op.chunk_gated_delta_rule_bwd_dhu(
        q, k, w, do, dv_local, scale, 64, g=gc, dht=dht,
        cu_seqlens=cu, chunk_indices=indices, use_exp2=True,
    )
    dq, dk, dv, db, dg = op.chunk_gated_delta_rule_bwd_finalize(
        q, k, v, vn, do, du, gc, beta_eff, h, dh, a,
        cu_seqlens=cu, chunk_indices=indices, scale=scale, use_exp2=True,
    )
    result = dict(o=o, dq=dq.transpose(1, 2), dk=dk.transpose(1, 2), dv=dv.transpose(1, 2),
                  dbeta=db.transpose(1, 2), dg=dg.transpose(1, 2))
    if communication != "none":
        result.update(h0=h0, dht=dht)
    return result


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(1)
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    plan = build_partition(args.cu, args.tokens, rank, world, args.partition)
    first = plan["parts"][0]
    lo, hi = first["lo"], first["hi"]
    if args.cpu_reference_only:
        result, _, _ = recurrent_reference(make_inputs(args.tokens, args.heads, args.seed, value_heads=args.value_heads), args.cu, plan["part_len"])
        assert all(torch.isfinite(t).all() for t in result.values())
        print("CPU reference PASS; no NPU operators executed")
        return 0
    import fla_npu
    import torch_npu
    device_id = int(os.environ.get("LOCAL_RANK", "0")) if args.device is None else args.device
    torch.npu.set_device(device_id)
    device_name = torch.npu.get_device_name(device_id)
    is_a5 = "Ascend950" in device_name
    if not is_a5 and not device_name.startswith(("Ascend910B", "Ascend910C", "Ascend910_93")):
        raise RuntimeError(f"This GDN example supports A2/A3/A5, got {device_name}")
    if args.communication != "none":
        # Establish a collective before the first P2P round (HCCL initialization).
        if world == 1:
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("hccl", rank=rank, world_size=world, timeout=timedelta(seconds=180))
        dist.barrier()
    output_dir = args.output_dir / f"rank{rank}" if args.output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    try:
        x = make_inputs(args.tokens, args.heads, args.seed, lo, hi, f"npu:{device_id}", value_heads=args.value_heads)
        if args.partition == "zigzag":
            back = plan["parts"][1]
            y = make_inputs(args.tokens, args.heads, args.seed, back["lo"], back["hi"], f"npu:{device_id}", value_heads=args.value_heads)
            for name in x:
                x[name] = torch.cat((x[name], y[name]), dim=2)
            del y
        metadata = dict(communication=args.communication, rank=rank, world=world, tokens=args.tokens,
                        partition=args.partition,
                        heads=args.heads, value_heads=args.value_heads, cu_seqlens=args.cu, seed=args.seed, disable_recompute=True,
                        device=device_name, torch=str(torch.__version__), torch_npu=str(torch_npu.__version__),
                        package=fla_npu.__file__, local_plan={k: v for k, v in plan.items() if k != "indices"})
        print(json.dumps(metadata), flush=True)
        with torch.no_grad():
            actual = run_gdn(x, plan, args.communication, is_a5=is_a5)
            torch.npu.synchronize()
            passed = all(bool(torch.isfinite(t).all()) for t in actual.values())
            checks = {}
            reference = None
            if args.accuracy_check:
                reference, states, dstates = recurrent_reference(
                    make_inputs(args.tokens, args.heads, args.seed, value_heads=args.value_heads), args.cu, plan["part_len"],
                )
            elif args.reference:
                payload = torch.load(args.reference, map_location="cpu", weights_only=True, mmap=True)
                for name in ("tokens", "heads", "value_heads", "cu_seqlens", "seed"):
                    if payload["metadata"][name] != metadata[name]:
                        raise ValueError(f"reference input mismatch: {name}")
                reference = payload["outputs"]
            if reference is not None:
                reference = {n: torch.cat([t[:, p["lo"]:p["hi"]] for p in plan["parts"]], dim=1)
                             for n, t in reference.items() if n not in ("h0", "dht")}
                if args.accuracy_check and args.communication != "none":
                    boundary = torch.zeros_like(actual["h0"], device="cpu")
                    terminal = torch.zeros_like(actual["dht"], device="cpu")
                    for part in plan["parts"]:
                        if part["prefix"]:
                            boundary[part["first"]] = states[part["slot"]][0]
                        if part["suffix"]:
                            terminal[part["last"]] = dstates[part["slot"] + 1][0]
                    reference.update(h0=boundary, dht=terminal)
                for name, expected in reference.items():
                    value, expected = actual[name].float().cpu(), expected.float().contiguous()
                    if value.shape != expected.shape:
                        raise RuntimeError(f"{name}: {value.shape} != {expected.shape}")
                    tol = (args.boundary_tol if name in ("h0", "dht") else args.output_tol if name == "o"
                           else args.gate_tol if name in ("dbeta", "dg") else args.grad_tol)
                    delta = (value - expected).abs()
                    bad = int((delta > tol + tol * expected.abs()).sum())
                    finite = bool(torch.isfinite(value).all() and torch.isfinite(expected).all())
                    # Long model vectors need FP64 accumulation: FP32 norm/dot
                    # reductions can even yield a spurious cosine above one.
                    actual_norm = torch.linalg.vector_norm(value, dtype=torch.float64)
                    expected_norm = torch.linalg.vector_norm(expected, dtype=torch.float64)
                    rel_l2 = float(torch.linalg.vector_norm(delta, dtype=torch.float64)
                                   / expected_norm.clamp_min(1e-8))
                    cosine = (float(((value * expected).sum(dtype=torch.float64)
                                     / (actual_norm * expected_norm).clamp_min(1e-16)).clamp(-1, 1))
                              if expected_norm > 0 else float(bad == 0))
                    cos_min = 0.99 if name in ("dbeta", "dg") else 0.999
                    ok = finite and bad == 0 and cosine >= cos_min
                    checks[name] = dict(passed=ok, mismatches=bad, rel_l2=rel_l2,
                                        max_abs=float(delta.max()), cosine=cosine, tol=tol, cos_min=cos_min)
                    passed &= ok
                    print(f"rank={rank} {name}: {json.dumps(checks[name])}", flush=True)
            if args.save_result:
                args.save_result.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(metadata=metadata, outputs={n: t.cpu() for n, t in actual.items()}), args.save_result)
            del actual, reference
            flag = torch.tensor([int(passed)], dtype=torch.int32, device=f"npu:{device_id}")
            if dist.is_initialized():
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            passed = bool(flag.item())
            # Validation and CPU copies are excluded from timing/profiling.
            for _ in range(args.warmup):
                run_gdn(x, plan, args.communication, is_a5=is_a5)
            torch.npu.synchronize()
            if dist.is_initialized():
                dist.barrier()
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                run_gdn(x, plan, args.communication, is_a5=is_a5)
            torch.npu.synchronize()
            elapsed = (time.perf_counter() - start) * 1000 / args.iterations
            times = torch.tensor([elapsed], device=f"npu:{device_id}", dtype=torch.float32)
            if dist.is_initialized():
                dist.all_reduce(times, op=dist.ReduceOp.MAX)
            if args.profile:
                with torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(output_dir / "profiling")),
                    record_shapes=True,
                ) as prof:
                    run_gdn(x, plan, args.communication, is_a5=is_a5)
                    torch.npu.synchronize()
                    prof.step()
            metadata.update(passed=passed, accuracy_checked=bool(checks), checks=checks,
                            warmup=args.warmup, iterations=args.iterations,
                            rank_ms=elapsed, max_rank_ms=float(times.item()))
            if output_dir:
                (output_dir / "result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"RESULT {json.dumps(metadata)}", flush=True)
            return 0 if passed else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
