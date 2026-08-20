"""Precision/performance canary for the single-launch A/B/C backward op."""

from __future__ import annotations

import argparse
import ctypes
import math
import os
import statistics

import torch
import torch_npu  # noqa: F401

from a2_canary import ACL_SUCCESS, CanaryRuntime, _error


def configure(runtime):
    query = runtime.op.aclnnChunkKdaBwdGetWorkspaceSize
    query.restype = ctypes.c_int
    query.argtypes = (
        [ctypes.c_void_p] * 20
        + [ctypes.c_double, ctypes.c_int64, ctypes.c_bool,
           ctypes.c_bool, ctypes.c_double, ctypes.c_bool, ctypes.c_bool,
           ctypes.c_bool]
        + [ctypes.c_void_p] * 8
        + [ctypes.POINTER(ctypes.c_uint64),
           ctypes.POINTER(ctypes.c_void_p)]
    )
    launch = runtime.op.aclnnChunkKdaBwd
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                       ctypes.c_void_p, ctypes.c_void_p]
    return query, launch


def run_b_only(args):
    """Isolate PR291 Kernel B from A/C and the composite executor."""
    torch.manual_seed(20260811)
    data = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    bsz, heads, seqlen, key_dim, value_dim = (
        1, args.heads, args.seqlen, 128, args.value_dim)
    chunks = seqlen // 64
    make = lambda shape: (torch.randn(shape) * 0.02).to(data).to("npu")
    qg = make((bsz, heads, seqlen, key_dim))
    kg = make((bsz, heads, seqlen, key_dim))
    w = make((bsz, heads, seqlen, key_dim))
    do = make((bsz, heads, seqlen, value_dim))
    dv0 = make((bsz, heads, seqlen, value_dim))
    gk = torch.zeros((bsz, heads, seqlen, key_dim),
                     dtype=torch.float32, device="npu")
    dh = torch.empty((bsz, heads, chunks, key_dim, value_dim),
                     dtype=data, device="npu")
    dv2 = torch.empty_like(dv0)
    runtime = CanaryRuntime(args.op_api)
    query = runtime.op.aclnnChunkGatedDeltaRuleBwdDhuGetWorkspaceSize
    query.restype = ctypes.c_int
    query.argtypes = ([ctypes.c_void_p] * 11 +
                      [ctypes.c_double, ctypes.c_int64] +
                      [ctypes.c_void_p] * 3 +
                      [ctypes.POINTER(ctypes.c_uint64),
                       ctypes.POINTER(ctypes.c_void_p)])
    launch = runtime.op.aclnnChunkGatedDeltaRuleBwdDhu
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64,
                       ctypes.c_void_p, ctypes.c_void_p]
    values = [qg, kg, w, do, dv0, gk, dh, dv2]
    descriptors = [runtime.tensor(x) for x in values]
    qgd, kgd, wd, dod, dvd, gkd, dhd, dv2d = [x.handle for x in descriptors]
    size = ctypes.c_uint64()
    executor = ctypes.c_void_p()
    status = query(qgd, kgd, wd, dod, dvd, None, gkd, None, None,
                   None, None, args.scale, 64, dhd, None, dv2d,
                   ctypes.byref(size), ctypes.byref(executor))
    if status != ACL_SUCCESS:
        raise RuntimeError(f"B GetWorkspaceSize failed: {status}")
    workspace = torch.empty(size.value, dtype=torch.uint8, device="npu")
    stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
    status = launch(ctypes.c_void_p(workspace.data_ptr()) if size.value else None,
                    size.value, executor, stream)
    if status != ACL_SUCCESS:
        raise RuntimeError(f"B launch failed: {status}")
    torch.npu.synchronize()
    print("B_ONLY_OK", {"dh_max": float(dh.float().abs().max().cpu()),
                        "dv2_max": float(dv2.float().abs().max().cpu()),
                        "workspace_bytes": size.value}, flush=True)
    runtime.destroy(*descriptors)


def reference(aqk, qg, kg, w, vnew, h, do, gk, scale, data,
              sequence_chunks=None, raw_g=None, a_log=None,
              lower_bound=-5.0, safe_gate=True, dt_bias=None):
    bsz, heads, seqlen, chunk = aqk.shape
    chunks, key_dim, value_dim = seqlen // chunk, qg.shape[-1], do.shape[-1]
    dv0 = torch.empty_like(do)
    dq_raw = torch.empty((bsz, heads, seqlen, key_dim))
    dh = torch.empty((bsz, heads, chunks, key_dim, value_dim))
    dv_scan = torch.empty_like(do)
    if sequence_chunks is None:
        sequence_chunks = [list(range(chunks))]
    for b in range(bsz):
        for head in range(heads):
            for ci in range(chunks):
                begin, end = ci * chunk, (ci + 1) * chunk
                do_c = do[b, head, begin:end]
                dv0[b, head, begin:end] = aqk[b, head, begin:end].T @ do_c
                dq_raw[b, head, begin:end] = do_c @ h[b, ci, head].T
            for chunk_group in sequence_chunks:
                state = torch.zeros((key_dim, value_dim), dtype=torch.float32)
                for ci in reversed(chunk_group):
                    begin, end = ci * chunk, (ci + 1) * chunk
                    dh_dt = state.to(data).float()
                    dh[b, head, ci] = dh_dt
                    dv2 = (kg[b, head, begin:end] @ dh_dt +
                           dv0[b, head, begin:end]).to(data).float()
                    dv_scan[b, head, begin:end] = dv2
                    term_q = (qg[b, head, begin:end].T @
                              do[b, head, begin:end]).to(data).float()
                    term_w = (w[b, head, begin:end].T @ dv2).to(data).float()
                    decay = torch.exp2(gk[b, head, end - 1]).unsqueeze(-1)
                    state = state * decay + term_q * scale - term_w

    # The precision case zeros q/k/v/beta/Akk. These are the remaining
    # nonzero final outputs of Kernel C's complete graph.
    dq = dq_raw * scale
    dk = torch.empty_like(dq)
    dg = torch.empty_like(dq)
    for b in range(bsz):
        for head in range(heads):
            for ci in range(chunks):
                begin, end = ci * chunk, (ci + 1) * chunk
                dk[b, head, begin:end] = (
                    vnew[b, head, begin:end] @ dh[b, head, ci].T)
                dg[b, head, begin:end] = (
                    h[b, ci, head] * dh[b, head, ci]).sum(dim=-1)
    d_a = None
    d_bias = None
    if raw_g is None:
        # The dense aligned A5 Intra merge directly writes the accumulated
        # gate gradient, so the standalone Gate pass is skipped.
        pass
    else:
        d_a = torch.zeros((heads,), dtype=torch.float32)
        if dt_bias is not None:
            d_bias = torch.zeros((heads, key_dim), dtype=torch.float32)
        for b in range(bsz):
            for head in range(heads):
                exp_a = torch.exp(a_log[head])
                for ci in range(chunks):
                    begin, end = ci * chunk, (ci + 1) * chunk
                    upstream = torch.flip(
                        torch.cumsum(torch.flip(
                            dg[b, head, begin:end], dims=(0,)), dim=0),
                        dims=(0,))
                    raw = raw_g[b, head, begin:end]
                    x = (raw + dt_bias[head]
                         if dt_bias is not None else raw)
                    if safe_gate:
                        sigmoid = torch.sigmoid(exp_a * x)
                        grad = (upstream * lower_bound * exp_a * sigmoid *
                                (1.0 - sigmoid))
                        d_a[head] += (grad * x).sum()
                    else:
                        a = -exp_a
                        grad = upstream * a * torch.sigmoid(x)
                        d_a[head] += (
                            upstream * a * torch.nn.functional.softplus(x)
                        ).sum()
                    dg[b, head, begin:end] = grad
                    if d_bias is not None:
                        d_bias[head] += grad.sum(dim=0)
    return dq, dk, dg, d_a, d_bias


def run(args):
    if args.seqlen % 64:
        raise ValueError("dense canary requires T divisible by 64")
    torch.set_num_threads(1)
    torch.manual_seed(20260811)
    data = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    bsz, heads, seqlen, key_dim, value_dim = (
        1, args.heads, args.seqlen, 128, args.value_dim)
    chunk, chunks = 64, seqlen // 64
    token_prefix = ((heads, seqlen) if args.varlen else
                    (bsz, heads, seqlen))
    state_prefix = ((chunks, heads) if args.varlen else
                    (bsz, chunks, heads))

    def rand(shape, gain=0.03, dtype=data):
        if not args.check:
            return torch.empty(shape, dtype=dtype, device="npu")
        return (torch.randn(shape) * gain).to(dtype).to("npu")

    def zero(shape, dtype=data):
        return torch.zeros(shape, dtype=dtype, device="npu")

    aqk = rand((*token_prefix, chunk))
    qg = rand((*token_prefix, key_dim))
    kg = rand((*token_prefix, key_dim))
    w = rand((*token_prefix, key_dim))
    vnew = rand((*token_prefix, value_dim))
    h = rand((*state_prefix, key_dim, value_dim), gain=0.02)
    do = rand((*token_prefix, value_dim))
    gk = zero((*token_prefix, key_dim), torch.float32)
    raw_g = (rand((*token_prefix, key_dim), gain=0.02,
                  dtype=torch.float32)
             if args.use_gate_in_kernel else None)
    a_log = (torch.full((heads,), -1.0, dtype=torch.float32,
                        device="npu")
             if args.use_gate_in_kernel else None)
    dt_bias = (rand((heads, key_dim), gain=0.01, dtype=torch.float32)
               if args.use_gate_in_kernel and args.with_dt_bias else None)
    if args.check:
        q = zero((*token_prefix, key_dim))
        k = zero((*token_prefix, key_dim))
        v = zero((*token_prefix, value_dim))
        beta = zero(token_prefix, torch.float32)
        akk = zero((*token_prefix, chunk))
    else:
        q, k, v, akk = qg, kg, vnew, aqk
        beta = torch.empty(token_prefix, dtype=torch.float32,
                           device="npu")

    dq = torch.empty_like(q, dtype=torch.float32)
    dk = torch.empty_like(k, dtype=torch.float32)
    dv = torch.empty_like(v)
    db = torch.empty(token_prefix, dtype=torch.float32, device="npu")
    dg = torch.empty_like(gk)
    d_a = (torch.empty((heads,), dtype=torch.float32, device="npu")
           if args.use_gate_in_kernel else None)
    d_bias = (torch.empty((heads, key_dim), dtype=torch.float32,
                          device="npu")
              if dt_bias is not None else None)

    runtime = CanaryRuntime(args.op_api)
    query, launch = configure(runtime)
    values = [q, k, v, beta, gk, aqk, akk, w, qg, kg, vnew, h, do,
              dq, dk, dv, db, dg]
    values += [x for x in (raw_g, a_log, dt_bias, d_a, d_bias)
               if x is not None]
    descriptors = {id(x): runtime.tensor(x) for x in values}
    handle = lambda x: descriptors[id(x)].handle
    cu_handle = None
    chunk_handle = None
    cu_backing = None
    chunk_backing = None
    if args.varlen:
        if chunks < 2:
            raise ValueError("varlen canary requires at least two chunks")
        if chunks == 2:
            # Keep one two-chunk sequence so dh/dg are nonzero; splitting it
            # into two one-chunk sequences makes the reverse state zero and
            # cannot validate the Gate path.
            cu_values = [0, seqlen]
            chunk_values = [0, 0, 0, 1]
        else:
            split_chunk = chunks // 2
            cu_values = [0, split_chunk * chunk, seqlen]
            chunk_values = [
                item
                for seq_idx, chunk_range in enumerate(
                    (range(split_chunk), range(split_chunk, chunks)))
                for global_chunk in chunk_range
                for item in (seq_idx, global_chunk - chunk_range.start)
            ]
        cu_handle, cu_backing = runtime.int_array(cu_values)
        chunk_handle, chunk_backing = runtime.int_array(chunk_values)
    query_args = [
        handle(q), handle(k), handle(v), handle(beta), handle(gk),
        handle(aqk), handle(akk), handle(w), handle(qg), handle(kg),
        handle(vnew), handle(h), handle(do),
        handle(raw_g) if raw_g is not None else None,
        handle(a_log) if a_log is not None else None,
        handle(dt_bias) if dt_bias is not None else None,
        None, None, cu_handle, chunk_handle,
        args.scale, chunk, args.safe_gate, args.use_gate_in_kernel,
        args.lower_bound, True, True, False,
        handle(dq), handle(dk), handle(dv), handle(db), handle(dg),
        None, handle(d_a) if d_a is not None else None,
        handle(d_bias) if d_bias is not None else None,
    ]
    stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
    workspace = None

    def invoke():
        nonlocal workspace
        size = ctypes.c_uint64()
        executor = ctypes.c_void_p()
        status = query(*query_args, ctypes.byref(size), ctypes.byref(executor))
        if status != ACL_SUCCESS:
            raise RuntimeError(f"GetWorkspaceSize failed: {status}")
        if workspace is None or workspace.numel() < size.value:
            workspace = torch.empty(size.value, dtype=torch.uint8, device="npu")
        workspace_ptr = (ctypes.c_void_p(workspace.data_ptr())
                         if size.value else None)
        status = launch(workspace_ptr, size.value, executor, stream)
        if status != ACL_SUCCESS:
            raise RuntimeError(f"launch failed: {status}")

    invoke()
    torch.npu.synchronize()
    if args.check:
        cpu = [x.detach().cpu().float() for x in
               (aqk, qg, kg, w, vnew, h, do, gk)]
        if args.varlen:
            cpu = [x.unsqueeze(0) for x in cpu[:5]] + [
                cpu[5].unsqueeze(0)] + [x.unsqueeze(0) for x in cpu[6:]]
            sequence_chunks = (
                [list(range(chunks))] if chunks == 2 else
                [list(range(0, chunks // 2)),
                 list(range(chunks // 2, chunks))]
            )
        else:
            sequence_chunks = None
        raw_cpu = (raw_g.detach().cpu().float().unsqueeze(0)
                   if args.varlen and raw_g is not None else
                   raw_g.detach().cpu().float()
                   if raw_g is not None else None)
        a_log_cpu = (a_log.detach().cpu().float()
                     if a_log is not None else None)
        dt_bias_cpu = (dt_bias.detach().cpu().float()
                       if dt_bias is not None else None)
        dq_ref, dk_ref, dg_ref, d_a_ref, d_bias_ref = reference(
            *cpu, args.scale, data, sequence_chunks, raw_cpu, a_log_cpu,
            args.lower_bound, args.safe_gate, dt_bias_cpu)
        if args.varlen:
            dq_ref = dq_ref.squeeze(0)
            dk_ref = dk_ref.squeeze(0)
            dg_ref = dg_ref.squeeze(0)
        reports = {
            "dq": _error("dq", dq, dq_ref),
            "dk": _error("dk", dk, dk_ref),
            "dg": _error("dg", dg, dg_ref),
        }
        if d_a is not None:
            reports["dA"] = _error("dA", d_a, d_a_ref)
        if d_bias is not None:
            reports["dbias"] = _error("dbias", d_bias, d_bias_ref)
        if args.debug_values:
            rows = (0, 1, chunk - 2, chunk - 1,
                    chunk, chunk + 1, seqlen - 2, seqlen - 1)
            def dg_value(tensor, head, row):
                return (tensor[head, row, 0] if args.varlen else
                        tensor[0, head, row, 0])
            print("DG_DEBUG", {
                (head, row): (float(dg_value(dg, head, row).cpu()),
                              float(dg_value(dg_ref, head, row)))
                for head in range(min(heads, 2)) for row in rows
                if row < seqlen
            }, flush=True)
        zeros = {name: float(x.float().abs().max().cpu()) for name, x in
                 (("dv", dv), ("db", db))}
        print("ABC_PRECISION", reports, zeros, flush=True)
        failed = [name for name, report in reports.items()
                  if not all(math.isfinite(report[key])
                             for key in ("max_abs", "mean_abs", "cos")) or
                  (report["max_abs"] > 1e-6 and
                   (report["cos"] < 0.99 or
                    report["mean_abs"] > 3e-3))]
        failed += [name for name, maximum in zeros.items()
                   if not math.isfinite(maximum) or maximum > 1e-6]
        if failed:
            raise AssertionError(f"precision failed: {failed}")

    for _ in range(args.warmup):
        invoke()
    torch.npu.synchronize()
    samples = []
    for _ in range(args.repeat):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        invoke()
        end.record()
        torch.npu.synchronize()
        samples.append(start.elapsed_time(end))
    print("ABC_PERF", {"shape": [bsz, heads, seqlen, key_dim, value_dim],
          "median_ms": statistics.median(samples), "min_ms": min(samples),
          "max_ms": max(samples), "workspace_bytes": workspace.numel(),
          "varlen": args.varlen, "samples_ms": samples},
          flush=True)
    runtime.destroy(*descriptors.values())
    if cu_handle is not None:
        runtime.destroy_int_array(cu_handle)
        runtime.destroy_int_array(chunk_handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-api", default=os.environ.get(
        "KDA_BWD_OP_API", "libcust_opapi.so"))
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=128)
    parser.add_argument("--value-dim", type=int, choices=(128, 256), default=128)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16")
    parser.add_argument("--scale", type=float, default=0.08838834764831845)
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--use-gate-in-kernel", action="store_true")
    parser.add_argument("--unsafe-gate", dest="safe_gate",
                        action="store_false")
    parser.add_argument("--with-dt-bias", action="store_true")
    parser.set_defaults(safe_gate=True)
    parser.add_argument("--debug-values", action="store_true")
    parser.add_argument("--b-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=9)
    args = parser.parse_args()
    if args.b_only:
        run_b_only(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
