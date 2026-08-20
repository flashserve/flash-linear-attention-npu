"""A2 runtime/precision canary for the internal Kernel A/C aclnn APIs.

This file deliberately calls the two-stage aclnn ABI directly.  It is kept
outside the public Python wrapper because Kernel B is not connected yet.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import statistics
import time
from dataclasses import dataclass

import torch
import torch_npu  # noqa: F401


ACL_FLOAT = 0
ACL_FLOAT16 = 1
ACL_INT64 = 9
ACL_BF16 = 27
ACL_FORMAT_ND = 2
ACL_SUCCESS = 0


@dataclass
class AclTensor:
    handle: ctypes.c_void_p
    shape: object
    strides: object


class CanaryRuntime:
    def __init__(self, op_api: str):
        # aclTensor descriptors belong to libnnopbase; libascendcl only owns
        # the runtime/memory APIs and does not export aclCreateTensor.
        self.acl = ctypes.CDLL("libnnopbase.so", mode=ctypes.RTLD_GLOBAL)
        self.op = ctypes.CDLL(op_api, mode=ctypes.RTLD_GLOBAL)
        self.acl.aclCreateTensor.restype = ctypes.c_void_p
        self.acl.aclCreateTensor.argtypes = [
            ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_int64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64, ctypes.c_void_p,
        ]
        self.acl.aclDestroyTensor.restype = ctypes.c_int
        self.acl.aclDestroyTensor.argtypes = [ctypes.c_void_p]
        self.acl.aclCreateIntArray.restype = ctypes.c_void_p
        self.acl.aclCreateIntArray.argtypes = [
            ctypes.POINTER(ctypes.c_int64), ctypes.c_uint64]
        self.acl.aclDestroyIntArray.restype = ctypes.c_int
        self.acl.aclDestroyIntArray.argtypes = [ctypes.c_void_p]

    def tensor(self, value: torch.Tensor) -> AclTensor:
        assert value.is_npu and value.is_contiguous()
        dtype = {
            torch.float16: ACL_FLOAT16,
            torch.bfloat16: ACL_BF16,
            torch.float32: ACL_FLOAT,
            torch.int64: ACL_INT64,
        }[value.dtype]
        dims = (ctypes.c_int64 * value.ndim)(*value.shape)
        strides = (ctypes.c_int64 * value.ndim)(*value.stride())
        handle = self.acl.aclCreateTensor(
            dims, value.ndim, dtype, strides, 0, ACL_FORMAT_ND,
            dims, value.ndim, ctypes.c_void_p(value.data_ptr()),
        )
        if not handle:
            raise RuntimeError("aclCreateTensor returned nullptr")
        return AclTensor(ctypes.c_void_p(handle), dims, strides)

    def destroy(self, *values: AclTensor) -> None:
        for value in values:
            status = self.acl.aclDestroyTensor(value.handle)
            if status != ACL_SUCCESS:
                raise RuntimeError(f"aclDestroyTensor failed: {status}")

    def int_array(self, values):
        backing = (ctypes.c_int64 * len(values))(*values)
        handle = self.acl.aclCreateIntArray(backing, len(values))
        if not handle:
            raise RuntimeError("aclCreateIntArray returned nullptr")
        return ctypes.c_void_p(handle), backing

    def destroy_int_array(self, handle) -> None:
        status = self.acl.aclDestroyIntArray(handle)
        if status != ACL_SUCCESS:
            raise RuntimeError(f"aclDestroyIntArray failed: {status}")


def _error(name: str, actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    a = actual.detach().cpu().float()
    e = expected.detach().cpu().float()
    diff = (a - e).abs()
    denom = e.abs().clamp_min(1e-6)
    flat_index = int(diff.argmax())
    max_index = []
    remainder = flat_index
    for size in reversed(diff.shape):
        max_index.append(remainder % size)
        remainder //= size
    max_index.reverse()
    index = tuple(max_index)
    return {
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "max_rel": (diff / denom).max().item(),
        "cos": torch.nn.functional.cosine_similarity(a.flatten(), e.flatten(), dim=0).item(),
        "max_index": max_index,
        "actual_at_max": a[index].item(),
        "expected_at_max": e[index].item(),
    }


def run_a(rt: CanaryRuntime, *, heads: int, seqlen: int, value_dim: int,
          scale: float, warmup: int, repeat: int, dtype: str,
          perf_only: bool) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(20260808)
    batch, chunk, key_dim = 1, 64, 128
    chunks = (seqlen + chunk - 1) // chunk
    device = "npu"
    data = torch.bfloat16 if dtype == "bf16" else torch.float16

    def make(shape, gain=0.1):
        print(f"MAKE {shape}", flush=True)
        if perf_only:
            # Input values do not affect the fixed instruction path.  Avoid
            # launching a separate device memset for every large benchmark
            # tensor before the custom operator is measured.
            result = torch.empty(shape, dtype=data, device=device)
            print(f"MADE {shape}", flush=True)
            return None, result
        cpu = torch.randn(shape, dtype=torch.float32) * gain
        result = cpu.to(data).to(device)
        print(f"MADE {shape}", flush=True)
        return cpu, result

    aqk_cpu, aqk = make((batch, heads, seqlen, chunk))
    vnew_cpu, vnew = make((batch, heads, seqlen, value_dim))
    h_cpu, h = make((batch, chunks, heads, key_dim, value_dim))
    do_cpu, d_o = make((batch, heads, seqlen, value_dim))
    del aqk_cpu

    dv0 = torch.empty_like(d_o)
    dq_raw = torch.empty((batch, heads, seqlen, key_dim), dtype=torch.float32, device=device)
    daqk = torch.empty((batch, heads, seqlen, chunk), dtype=torch.float32, device=device)
    torch.npu.synchronize()
    print("INPUTS_READY", flush=True)

    tensors = [rt.tensor(x) for x in (aqk, vnew, h, d_o, dv0, dq_raw, daqk)]
    print("ACL_TENSORS_READY", flush=True)
    get_ws = rt.op.aclnnChunkKdaBwdAGetWorkspaceSize
    get_ws.restype = ctypes.c_int
    get_ws.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_int64] +
                       [ctypes.c_void_p] * 3 +
                       [ctypes.POINTER(ctypes.c_uint64),
                        ctypes.POINTER(ctypes.c_void_p)])
    launch = rt.op.aclnnChunkKdaBwdA
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]

    def enqueue():
        workspace_size = ctypes.c_uint64()
        executor = ctypes.c_void_p()
        ins = [x.handle for x in tensors[:4]]
        outs = [x.handle for x in tensors[4:]]
        status = get_ws(*ins, None, None, chunk, *outs,
                        ctypes.byref(workspace_size), ctypes.byref(executor))
        print(f"WORKSPACE_READY {workspace_size.value}", flush=True)
        if status != ACL_SUCCESS:
            raise RuntimeError(f"Kernel A GetWorkspaceSize failed: {status}")
        workspace = torch.empty(workspace_size.value, dtype=torch.uint8, device=device)
        stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        status = launch(ctypes.c_void_p(workspace.data_ptr()) if workspace_size.value else None,
                        workspace_size.value, executor, stream)
        print("LAUNCH_ENQUEUED", flush=True)
        if status != ACL_SUCCESS:
            raise RuntimeError(f"Kernel A launch failed: {status}")
        return workspace

    live = enqueue()
    torch.npu.synchronize()

    if perf_only:
        for _ in range(warmup):
            live = enqueue()
        torch.npu.synchronize()
        samples = []
        for _ in range(repeat):
            t0 = time.perf_counter_ns()
            live = enqueue()
            torch.npu.synchronize()
            samples.append((time.perf_counter_ns() - t0) / 1e3)
        print("KERNEL_A_PERF_US", {
            "median": statistics.median(samples), "min": min(samples),
            "max": max(samples), "repeat": repeat,
            "shape": [batch, heads, seqlen, key_dim, value_dim],
            "dtype": dtype, "perf_only": True,
        }, flush=True)
        del live
        rt.destroy(*tensors)
        return

    dv_ref = torch.empty_like(do_cpu)
    dq_ref = torch.empty((batch, heads, seqlen, key_dim))
    da_ref = torch.zeros((batch, heads, seqlen, chunk))
    for c in range(chunks):
        begin, end = c * chunk, min((c + 1) * chunk, seqlen)
        valid = end - begin
        for head in range(heads):
            do_c = do_cpu[0, head, begin:end]
            # Cube operands are BF16/FP16 views, so reference the quantized inputs.
            aqk_c = aqk[0, head, begin:end].cpu().float()
            vn_c = vnew[0, head, begin:end].cpu().float()
            h_c = h[0, c, head].cpu().float()
            do_q = d_o[0, head, begin:end].cpu().float()
            # dv0 is [C,V]; Aqk is the saved [C,C] triangular matrix.
            dv_ref[0, head, begin:end] = aqk_c.transpose(0, 1).matmul(do_q)[:valid]
            dq_ref[0, head, begin:end] = do_q.matmul(h_c.transpose(0, 1))
            # Kernel A publishes the full raw product.  Kernel C's intra pack
            # applies scale and the causal lower mask when consuming it.
            da_ref[0, head, begin:end, :valid] = do_q.matmul(vn_c.transpose(0, 1))

    reports = {
        "dv0": _error("dv0", dv0, dv_ref.to(data)),
        "dq_raw": _error("dq_raw", dq_raw, dq_ref),
        "dAqk": _error("dAqk", daqk, da_ref),
    }
    print("KERNEL_A_PRECISION", reports, flush=True)
    da_actual_rows = daqk.detach().cpu().float().abs().sum(dim=-1)
    da_expected_rows = da_ref.abs().sum(dim=-1)
    missing_rows = (da_actual_rows == 0) & (da_expected_rows > 1e-6)
    missing_row_count = int(missing_rows.sum())
    print("DAQK_ROW_DIAGNOSTIC", {
        "missing_row_count": missing_row_count,
        "missing_rows": missing_rows.nonzero()[:16].tolist(),
    }, flush=True)
    if (missing_row_count != 0 or
            any(v["cos"] < 0.99 or v["mean_abs"] > 2e-3
                for v in reports.values())):
        raise AssertionError(f"Kernel A precision smoke failed: {reports}")

    for _ in range(warmup):
        live = enqueue()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        live = enqueue()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e3)
    post_daqk = _error("dAqk", daqk, da_ref)
    post_rows = daqk.detach().cpu().float().abs().sum(dim=-1)
    post_missing = (post_rows == 0) & (da_expected_rows > 1e-6)
    post_missing_count = int(post_missing.sum())
    print("KERNEL_A_POST_REPEAT_PRECISION", {
        "dAqk": post_daqk,
        "missing_row_count": post_missing_count,
        "missing_rows": post_missing.nonzero()[:16].tolist(),
    }, flush=True)
    if (post_missing_count != 0 or post_daqk["cos"] < 0.99 or
            post_daqk["mean_abs"] > 2e-3):
        raise AssertionError(
            f"Kernel A repeated-launch precision failed: {post_daqk}")
    print("KERNEL_A_PERF_US", {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "repeat": repeat,
        "shape": [batch, heads, seqlen, key_dim, value_dim],
    }, flush=True)
    del live
    rt.destroy(*tensors)


def run_c(rt: CanaryRuntime, *, heads: int, seqlen: int, value_dim: int,
          scale: float, warmup: int, repeat: int, lower_bound: float,
          varlen: bool, nonzero_daqk: bool, daqk_impulse: bool,
          dtype: str, perf_only: bool, dh_head_major: bool,
          use_raw_gate: bool, use_dt_bias: bool) -> None:
    """Independent nonzero smoke for Kernel C while Kernel B is not connected."""
    batch, chunk, key_dim = 1, 64, 128
    chunks = (seqlen + chunk - 1) // chunk
    device = "npu"
    data = torch.bfloat16 if dtype == "bf16" else torch.float16
    torch.manual_seed(20260808)

    def zeros(shape, dtype=data):
        return torch.zeros(shape, dtype=dtype, device=device)

    def random_fp16(shape, gain=0.05):
        return (torch.randn(shape, dtype=torch.float32) * gain).to(data).to(device)

    token_prefix = (heads, seqlen) if varlen else (batch, heads, seqlen)
    state_prefix = (chunks, heads) if varlen else (batch, chunks, heads)
    q = random_fp16((*token_prefix, key_dim)) if nonzero_daqk else zeros((*token_prefix, key_dim))
    k = random_fp16((*token_prefix, key_dim)) if nonzero_daqk else zeros((*token_prefix, key_dim))
    if daqk_impulse:
        q.zero_()
        k.zero_()
        for token in range(min(seqlen, key_dim)):
            if varlen:
                q[0, token, token] = 1
                k[0, token, token] = 1
            else:
                q[0, 0, token, token] = 1
                k[0, 0, token, token] = 1
    v = zeros((*token_prefix, value_dim))
    vnew = zeros((*token_prefix, value_dim)) if nonzero_daqk else random_fp16((*token_prefix, value_dim))
    gk = zeros((*token_prefix, key_dim), torch.float32)
    beta = zeros(token_prefix, torch.float32)
    akk = zeros((*token_prefix, chunk))
    h = zeros((*state_prefix, key_dim, value_dim))
    dh_shape = ((batch, heads, chunks, key_dim, value_dim)
                if dh_head_major and not varlen else
                (*state_prefix, key_dim, value_dim))
    dh = (zeros(dh_shape) if nonzero_daqk else random_fp16(dh_shape))
    dv_scan = zeros((*token_prefix, value_dim))
    dq_raw = (zeros((*token_prefix, key_dim), torch.float32) if nonzero_daqk else
              (torch.randn((*token_prefix, key_dim), dtype=torch.float32) * 0.05).to(device))
    if nonzero_daqk:
        daqk = (torch.randn((*token_prefix, chunk), dtype=torch.float32) * 0.02).to(device)
        if daqk_impulse:
            daqk.zero_()
            if varlen:
                daqk[0, 31, 7] = 1
            else:
                daqk[0, 0, 31, 7] = 1
    else:
        daqk = zeros((*token_prefix, chunk), torch.float32)
    raw_g = zeros((*token_prefix, key_dim), torch.float32)
    a_log = zeros((heads,), torch.float32)
    dt_bias = zeros((heads, key_dim), torch.float32)

    dq = torch.full_like(q, float("nan"), dtype=torch.float32)
    dk = torch.full_like(k, float("nan"), dtype=torch.float32)
    dv = torch.full_like(v, float("nan"))
    db = torch.full(token_prefix, float("nan"), dtype=torch.float32, device=device)
    dg = torch.full_like(gk, float("nan"))
    dakk = torch.full_like(daqk, float("nan"))
    da = torch.full((heads,), float("nan"), dtype=torch.float32, device=device)
    dbias = torch.full((heads, key_dim), float("nan"), dtype=torch.float32, device=device)
    required = [q, k, v, vnew, gk, beta, akk, h, dh, dv_scan, dq_raw, daqk]
    if varlen:
        cu_seqlens = torch.tensor([0, seqlen], dtype=torch.int64, device=device)
        chunk_indices = torch.tensor(
            [item for local_chunk in range(chunks) for item in (0, local_chunk)],
            dtype=torch.int64, device=device)
        metadata = [rt.tensor(cu_seqlens), rt.tensor(chunk_indices)]
    else:
        metadata = [None, None]
    optional_inputs = [raw_g, a_log, dt_bias]
    outputs = [dq, dk, dv, db, dg, dakk, da, dbias]
    required_desc = [rt.tensor(x) for x in required]
    optional_desc = [rt.tensor(x) for x in optional_inputs]
    output_desc = [rt.tensor(x) for x in outputs]
    tensors = required_desc + [x for x in metadata if x is not None] + optional_desc + output_desc

    get_ws = rt.op.aclnnChunkKdaBwdCGetWorkspaceSize
    get_ws.restype = ctypes.c_int
    get_ws.argtypes = (
        [ctypes.c_void_p] * 17
        + [ctypes.c_float, ctypes.c_int64, ctypes.c_bool, ctypes.c_bool,
           ctypes.c_float, ctypes.c_bool]
        + [ctypes.c_void_p] * 8
        + [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_void_p)]
    )
    launch = rt.op.aclnnChunkKdaBwdC
    launch.restype = ctypes.c_int
    launch.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]

    def enqueue():
        workspace_size = ctypes.c_uint64()
        executor = ctypes.c_void_p()
        status = get_ws(
            *[x.handle for x in required_desc],
            *([optional_desc[0].handle, optional_desc[1].handle,
               optional_desc[2].handle if use_dt_bias else None]
              if use_raw_gate else [None, None, None]),
            *[x.handle if x is not None else None for x in metadata],
            scale, chunk, True, use_raw_gate, lower_bound, dh_head_major,
            *([x.handle for x in output_desc[:7]] +
              [output_desc[7].handle if use_dt_bias else None]
              if use_raw_gate else
              [x.handle for x in output_desc[:6]] + [None, None]),
            ctypes.byref(workspace_size), ctypes.byref(executor),
        )
        print(f"KERNEL_C_WORKSPACE_READY {workspace_size.value}", flush=True)
        if status != ACL_SUCCESS:
            raise RuntimeError(f"Kernel C GetWorkspaceSize failed: {status}")
        workspace = torch.empty(workspace_size.value, dtype=torch.uint8, device=device)
        stream = ctypes.c_void_p(torch.npu.current_stream().npu_stream)
        status = launch(
            ctypes.c_void_p(workspace.data_ptr()) if workspace_size.value else None,
            workspace_size.value, executor, stream,
        )
        print("KERNEL_C_LAUNCH_ENQUEUED", flush=True)
        if status != ACL_SUCCESS:
            raise RuntimeError(f"Kernel C launch failed: {status}")
        return workspace

    live = enqueue()
    torch.npu.synchronize()

    if perf_only:
        for _ in range(warmup):
            live = enqueue()
        torch.npu.synchronize()
        samples = []
        for _ in range(repeat):
            t0 = time.perf_counter_ns()
            live = enqueue()
            torch.npu.synchronize()
            samples.append((time.perf_counter_ns() - t0) / 1e3)
        print("KERNEL_C_PERF_US", {
            "median": statistics.median(samples), "min": min(samples),
            "max": max(samples), "repeat": repeat,
            "shape": [batch, heads, seqlen, key_dim, value_dim],
            "varlen": varlen, "dtype": dtype, "perf_only": True,
        }, flush=True)
        del live
        rt.destroy(*tensors)
        return

    # With q/k/v/gk/beta/Akk/h/dv_scan/dAqk all zero, the complete fused
    # graph reduces to two independent, nonzero paths.  This validates both
    # the AIV dq postprocess and the AIC v_new @ dh^T contraction without
    # depending on the not-yet-connected Kernel B implementation.
    dq_expected = dq_raw.detach().cpu().float() * scale
    dk_expected = torch.zeros_like(dq_expected)
    q_cpu = q.detach().cpu().float()
    k_cpu = k.detach().cpu().float()
    daqk_cpu = daqk.detach().cpu().float()
    vnew_cpu = vnew.detach().cpu().float()
    dh_cpu = dh.detach().cpu().float()
    for c in range(chunks):
        begin, end = c * chunk, min((c + 1) * chunk, seqlen)
        for head in range(heads):
            if nonzero_daqk:
                if varlen:
                    raw = daqk_cpu[head, begin:end, :end - begin]
                    causal = torch.tril(raw) * scale
                    dq_expected[head, begin:end] = causal @ k_cpu[head, begin:end]
                    dk_expected[head, begin:end] = causal.transpose(0, 1) @ q_cpu[head, begin:end]
                else:
                    raw = daqk_cpu[0, head, begin:end, :end - begin]
                    causal = torch.tril(raw) * scale
                    dq_expected[0, head, begin:end] = causal @ k_cpu[0, head, begin:end]
                    dk_expected[0, head, begin:end] = causal.transpose(0, 1) @ q_cpu[0, head, begin:end]
                continue
            if varlen:
                dk_expected[head, begin:end] = (
                    vnew_cpu[head, begin:end] @ dh_cpu[c, head].transpose(0, 1))
            else:
                dk_expected[0, head, begin:end] = (
                    vnew_cpu[0, head, begin:end] @
                    (dh_cpu[0, head, c] if dh_head_major else
                     dh_cpu[0, c, head]).transpose(0, 1))

    reports = {
        "dq": _error("dq", dq, dq_expected),
        "dk": _error("dk", dk, dk_expected),
    }
    zero_reports = {}
    zero_names = (("dv", "db", "dg", "dAkk", "dA", "dbias")
                  if use_raw_gate and use_dt_bias else
                  ("dv", "db", "dg", "dAkk", "dA")
                  if use_raw_gate else
                  ("dv", "db", "dg", "dAkk"))
    for name, value in zip(zero_names, outputs[2:]):
        cpu = value.detach().cpu().float()
        zero_reports[name] = {
            "finite": bool(torch.isfinite(cpu).all()),
            "max_abs": float(torch.nan_to_num(cpu).abs().max()),
        }
    print("KERNEL_C_PRECISION", reports, zero_reports, flush=True)
    if daqk_impulse:
        for name, value in (("dq", dq), ("dk", dk)):
            flat = value.detach().cpu().float().reshape(-1)
            top = torch.topk(flat.abs(), 16)
            print(f"KERNEL_C_IMPULSE_{name.upper()}", [
                {"flat": int(index), "value": float(flat[index])}
                for index in top.indices
            ], flush=True)
    if nonzero_daqk and not varlen:
        actual_dq0 = dq.detach().cpu().float()[0, 0, :chunk]
        actual_dk0 = dk.detach().cpu().float()[0, 0, :chunk]
        raw0 = daqk_cpu[0, 0, :chunk, :chunk]
        d0 = torch.tril(raw0) * scale
        q0 = q_cpu[0, 0, :chunk]
        k0 = k_cpu[0, 0, :chunk]
        candidates = {
            "D@k": d0 @ k0,
            "DT@k": d0.transpose(0, 1) @ k0,
            "D@q": d0 @ q0,
            "DT@q": d0.transpose(0, 1) @ q0,
        }
        print("KERNEL_C_DAQK_HYPOTHESES", {
            name: {
                "dq_cos": torch.nn.functional.cosine_similarity(
                    actual_dq0.flatten(), value.flatten(), dim=0).item(),
                "dq_mean_abs": (actual_dq0 - value).abs().mean().item(),
                "dk_cos": torch.nn.functional.cosine_similarity(
                    actual_dk0.flatten(), value.flatten(), dim=0).item(),
                "dk_mean_abs": (actual_dk0 - value).abs().mean().item(),
            }
            for name, value in candidates.items()
        }, flush=True)
    nonzero_failed = any(
        item["cos"] < 0.99 or item["mean_abs"] > 2e-3
        for item in reports.values())
    zero_failed = (not nonzero_daqk) and any(
        not item["finite"] or item["max_abs"] > 1e-6
        for item in zero_reports.values())
    if nonzero_failed or zero_failed:
        dq_cpu = dq.detach().cpu().float()
        dq_diff = (dq_cpu - dq_expected).abs()
        tail = dq_cpu.reshape(-1, key_dim)[-1]
        expected_rows = dq_expected.reshape(-1, key_dim)
        nearest = int(((expected_rows - tail) ** 2).sum(dim=-1).argmin())
        print("KERNEL_C_DQ_ROW_DIAGNOSTIC", {
            "row_max_abs": dq_diff.amax(dim=-1).tolist(),
            "tail_actual": tail[:16].tolist(),
            "tail_expected": expected_rows[-1, :16].tolist(),
            "nearest_expected_flat_row": nearest,
            "nearest_expected_l2": float(
                ((expected_rows[nearest] - tail) ** 2).sum()),
            "actual_l2": float((tail ** 2).sum()),
        }, flush=True)
    if nonzero_failed:
        raise AssertionError(f"Kernel C nonzero smoke failed: {reports}")
    if zero_failed:
        raise AssertionError(f"Kernel C zero branches failed: {zero_reports}")

    for _ in range(warmup):
        live = enqueue()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter_ns()
        live = enqueue()
        torch.npu.synchronize()
        samples.append((time.perf_counter_ns() - t0) / 1e3)
    print("KERNEL_C_PERF_US", {
        "median": statistics.median(samples), "min": min(samples),
        "max": max(samples), "repeat": repeat,
        "shape": [batch, heads, seqlen, key_dim, value_dim],
        "varlen": varlen,
    }, flush=True)
    del live
    rt.destroy(*tensors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--op-api", default=os.environ.get("KDA_BWD_OP_API", "libcust_opapi.so"))
    parser.add_argument("--kernel", choices=("a", "c"), default="a")
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--seqlen", type=int, default=64)
    parser.add_argument("--value-dim", type=int, choices=(128, 256), default=128)
    parser.add_argument("--scale", type=float, default=0.125)
    parser.add_argument("--lower-bound", type=float, default=-5.0)
    parser.add_argument("--varlen", action="store_true")
    parser.add_argument("--c-nonzero-daqk", action="store_true")
    parser.add_argument("--c-daqk-impulse", action="store_true")
    parser.add_argument("--c-dh-head-major", action="store_true")
    parser.add_argument("--c-no-raw-gate", action="store_true")
    parser.add_argument("--c-no-dt-bias", action="store_true")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--perf-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    runtime = CanaryRuntime(args.op_api)
    if args.kernel == "a":
        run_a(runtime, heads=args.heads, seqlen=args.seqlen,
              value_dim=args.value_dim, scale=args.scale, warmup=args.warmup,
              repeat=args.repeat, dtype=args.dtype,
              perf_only=args.perf_only)
    else:
        run_c(runtime, heads=args.heads, seqlen=args.seqlen,
              value_dim=args.value_dim, scale=args.scale, warmup=args.warmup,
              repeat=args.repeat, lower_bound=args.lower_bound,
              varlen=args.varlen,
              nonzero_daqk=args.c_nonzero_daqk or args.c_daqk_impulse,
              daqk_impulse=args.c_daqk_impulse,
              dtype=args.dtype, perf_only=args.perf_only,
              dh_head_major=args.c_dh_head_major,
              use_raw_gate=not args.c_no_raw_gate,
              use_dt_bias=not args.c_no_dt_bias)


if __name__ == "__main__":
    main()
