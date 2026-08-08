#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Full-tensor accuracy checks for chunk_gated_delta_rule_bwd_dhu.

The comparison covers every element of dh, dv2, and dh0 when h0 is provided.
Large tensors are compared in flat slices only to limit host memory peak; no
sampling or masking is used.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import io
import importlib.util
import math
import os
import random
import selectors
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable, Optional

import torch


os.environ["TBE_PARALLEL_COMPILE_ENABLE"] = "0"
os.environ["PARALLEL_COMPILE"] = "0"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location("test_bwd_dhu_golden", os.path.join(_SCRIPT_DIR, "test_bwd_dhu.py"))
_GOLDEN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_GOLDEN)

chunk_gated_delta_rule_bwd_dhu_cpu = _GOLDEN.chunk_gated_delta_rule_bwd_dhu_cpu
create_bwd_dhu_random_inputs = _GOLDEN.create_bwd_dhu_random_inputs
effective_scale = _GOLDEN.effective_scale
prepare_chunk_indices = _GOLDEN.prepare_chunk_indices
scale_for_compute_dtype = _GOLDEN.scale_for_compute_dtype

KERNEL_STARTUP_TIMEOUT_S = 300
KERNEL_SYNC_TIMEOUT_S = 60
GOLDEN_CACHE_VERSION = 1
GOLDEN_SEMANTIC_VERSION_G = "dhu_cpu_dual_g_exp_v2"
GOLDEN_SEMANTIC_VERSION_GK = "dhu_cpu_dual_exp2_v1"
GOLDEN_OUTPUT_NAMES = ("dh", "dh0", "dv2")
DEFAULT_GOLDEN_CACHE_DIR = os.path.join(_SCRIPT_DIR, "test_output", "dhu_cpu_golden_cache")


@dataclass(frozen=True)
class Case:
    name: str
    B: int
    Hk: int
    Hv: int
    T: int
    K: int
    V: int
    chunk_size: int
    q_dtype: torch.dtype
    gate_dtype: torch.dtype
    gate_kind: str
    cu_seqlens: Optional[list[int]] = None
    with_h0: bool = False


def random_seq_split(total: int, parts: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    cuts = sorted(rng.sample(range(1, total), parts - 1))
    return [0, *cuts, total]


def random_three_way_split(total: int, seed: int) -> list[int]:
    return random_seq_split(total, 3, seed)


def rand_symmetric_uniform(shape: Iterable[int], dtype: torch.dtype, half_range: float) -> torch.Tensor:
    x = torch.rand(tuple(shape), dtype=torch.float32)
    x = (x * 2.0 - 1.0) * float(half_range)
    return x.to(dtype=dtype)


def create_gate_gk(B: int, Hv: int, T: int, K: int, dtype: torch.dtype) -> torch.Tensor:
    base = torch.linspace(-1e-6, -1e-2, T, dtype=torch.float32).view(1, 1, T, 1)
    k_jitter = rand_symmetric_uniform((1, 1, 1, K), torch.float32, 5e-4)
    h_jitter = rand_symmetric_uniform((1, Hv, 1, 1), torch.float32, 5e-4)
    gk = base + k_jitter + h_jitter
    return gk.expand(B, Hv, T, K).contiguous().to(dtype)


def make_inputs(case: Case, seed: int):
    torch.manual_seed(seed)
    q, k, w, d_o, dv, g = create_bwd_dhu_random_inputs(
        case.B, case.Hk, case.Hv, case.T, case.K, case.V, case.q_dtype, case.gate_dtype,
    )
    if case.gate_kind == "g":
        return q, k, w, d_o, dv, g, None
    gk = create_gate_gk(case.B, case.Hv, case.T, case.K, case.gate_dtype)
    return q, k, w, d_o, dv, None, gk


_NPU_RUNTIME_READY = False


def configure_npu_runtime() -> None:
    global _NPU_RUNTIME_READY
    if _NPU_RUNTIME_READY:
        return
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch.npu.set_compile_mode(jit_compile=False)
    _NPU_RUNTIME_READY = True


def build_case_inputs(case: Case, seed: int):
    if case.Hv % case.Hk != 0:
        raise ValueError(f"Hv must be divisible by Hk, got Hk={case.Hk}, Hv={case.Hv}")

    q, k, w, d_o, dv, g, gk = make_inputs(case, seed)
    cu_seqlens = case.cu_seqlens
    chunk_indices = prepare_chunk_indices(cu_seqlens, case.chunk_size) if cu_seqlens is not None else None
    chunk_count = len(chunk_indices) // 2 if chunk_indices is not None else (case.T + case.chunk_size - 1) // case.chunk_size
    h0 = torch.zeros((case.B, case.Hv, chunk_count, case.K, case.V), dtype=case.q_dtype) if case.with_h0 else None
    scale = scale_for_compute_dtype(effective_scale(1.0 / math.sqrt(float(case.K)), case.K), case.q_dtype)
    return q, k, w, d_o, dv, g, gk, cu_seqlens, chunk_indices, h0, scale


def ct_success(result) -> bool:
    if isinstance(result, dict):
        return bool(result.get("success", False))
    return bool(result)


def ct_failure_summary(result) -> str:
    if not isinstance(result, dict):
        return "no detail"
    metrics = result.get("metrics")
    if metrics is None:
        return f"checks={result.get('checks', {})} ratios={result.get('ratios', {})}"
    return (
        f"fail_count={getattr(metrics, 'fail_count', 'NA')}, "
        f"pass_rate={getattr(metrics, 'pass_rate', 'NA')}, "
        f"max_diff={getattr(metrics, 'max_diff', 'NA')}, "
        f"max_re={getattr(metrics, 'max_re_calc', 'NA')}"
    )


def ct_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype).removeprefix("torch.")


def safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)


def golden_cache_path(cache_dir: str, case_index: int, case: Case, seed: int) -> str:
    return os.path.join(cache_dir, f"{case_index:03d}_{safe_filename(case.name)}_seed{seed}_cpu_dual.pt")


def tensor_meta(tensor: Optional[torch.Tensor]):
    if tensor is None:
        return None
    return {"shape": list(tensor.shape), "dtype": ct_dtype(tensor.dtype)}


def update_tensor_signature(digest: "hashlib._Hash", name: str, tensor: Optional[torch.Tensor]) -> None:
    digest.update(name.encode("utf-8"))
    if tensor is None:
        digest.update(b":none;")
        return
    cpu_tensor = tensor.detach().cpu().contiguous()
    digest.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
    digest.update(ct_dtype(cpu_tensor.dtype).encode("utf-8"))
    digest.update(str(cpu_tensor.stride()).encode("utf-8"))
    digest.update(memoryview(cpu_tensor.view(torch.uint8).numpy()))


def input_signature(
    case: Case,
    seed: int,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    d_o: torch.Tensor,
    dv: torch.Tensor,
    g: Optional[torch.Tensor],
    gk: Optional[torch.Tensor],
    h0: Optional[torch.Tensor],
    cu_seqlens: Optional[list[int]],
    chunk_indices: Optional[list[int]],
    scale: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(case.name.encode("utf-8"))
    digest.update(str(seed).encode("utf-8"))
    digest.update(str(cu_seqlens).encode("utf-8"))
    digest.update(str(chunk_indices).encode("utf-8"))
    digest.update(repr(float(scale)).encode("utf-8"))
    for name, tensor in (
        ("q", q),
        ("k", k),
        ("w", w),
        ("d_o", d_o),
        ("dv", dv),
        ("g", g),
        ("gk", gk),
        ("h0", h0),
    ):
        update_tensor_signature(digest, name, tensor)
    return digest.hexdigest()


def case_meta(case: Case) -> dict:
    return {
        "name": case.name,
        "B": case.B,
        "Hk": case.Hk,
        "Hv": case.Hv,
        "T": case.T,
        "K": case.K,
        "V": case.V,
        "chunk_size": case.chunk_size,
        "q_dtype": ct_dtype(case.q_dtype),
        "gate_dtype": ct_dtype(case.gate_dtype),
        "gate_kind": case.gate_kind,
        "cu_seqlens": case.cu_seqlens,
        "with_h0": case.with_h0,
    }


def expected_golden_meta(
    case_index: int,
    case: Case,
    seed: int,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    d_o: torch.Tensor,
    dv: torch.Tensor,
    g: Optional[torch.Tensor],
    gk: Optional[torch.Tensor],
    h0: Optional[torch.Tensor],
    cu_seqlens: Optional[list[int]],
    chunk_indices: Optional[list[int]],
    scale: float,
) -> dict:
    return {
        "cache_version": GOLDEN_CACHE_VERSION,
        "golden_semantic_version": (
            GOLDEN_SEMANTIC_VERSION_G if case.gate_kind == "g" else GOLDEN_SEMANTIC_VERSION_GK
        ),
        "case_index": case_index,
        "case": case_meta(case),
        "seed": seed,
        "scale": repr(float(scale)),
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "input_signature": input_signature(case, seed, q, k, w, d_o, dv, g, gk, h0, cu_seqlens, chunk_indices, scale),
        "input_meta": {
            "q": tensor_meta(q),
            "k": tensor_meta(k),
            "w": tensor_meta(w),
            "d_o": tensor_meta(d_o),
            "dv": tensor_meta(dv),
            "g": tensor_meta(g),
            "gk": tensor_meta(gk),
            "h0": tensor_meta(h0),
        },
        "output_names": list(GOLDEN_OUTPUT_NAMES),
    }


def output_dict(outputs) -> dict:
    dh, dh0, dv2 = outputs
    return {
        "dh": dh.detach().cpu(),
        "dh0": dh0.detach().cpu() if dh0 is not None else None,
        "dv2": dv2.detach().cpu(),
    }


def outputs_from_dict(outputs: dict):
    return outputs["dh"], outputs["dh0"], outputs["dv2"]


def validate_golden_payload(payload, expected_meta: dict) -> tuple[bool, str]:
    if not isinstance(payload, dict):
        return False, "payload is not a dict"
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        return False, "missing meta"
    for key, expected_value in expected_meta.items():
        if meta.get(key) != expected_value:
            return False, f"meta mismatch: {key}"
    for group_name in ("golden_fp64", "bench_cpu"):
        outputs = payload.get(group_name)
        if not isinstance(outputs, dict):
            return False, f"missing {group_name}"
        for name in GOLDEN_OUTPUT_NAMES:
            if name not in outputs:
                return False, f"missing {group_name}.{name}"
        dh0 = outputs["dh0"]
        if expected_meta["case"]["with_h0"] and dh0 is None:
            return False, f"{group_name}.dh0 expected tensor"
        if not expected_meta["case"]["with_h0"] and dh0 is not None:
            return False, f"{group_name}.dh0 expected None"
    return True, "ok"


def try_load_golden_cache(path: str, expected_meta: dict):
    if not path or not os.path.exists(path):
        return None, "missing"
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return None, f"load error: {exc}"
    ok, reason = validate_golden_payload(payload, expected_meta)
    if not ok:
        return None, reason
    return (outputs_from_dict(payload["golden_fp64"]), outputs_from_dict(payload["bench_cpu"])), "loaded"


def save_golden_cache(path: str, expected_meta: dict, golden_fp64, bench_cpu) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = dict(expected_meta)
    meta["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "meta": meta,
        "golden_fp64": output_dict(golden_fp64),
        "bench_cpu": output_dict(bench_cpu),
    }
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def compute_cpu_dual_golden(
    case: Case,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    d_o: torch.Tensor,
    dv: torch.Tensor,
    g: Optional[torch.Tensor],
    gk: Optional[torch.Tensor],
    h0: Optional[torch.Tensor],
    cu_seqlens: Optional[list[int]],
    chunk_indices: Optional[list[int]],
    scale: float,
):
    print("  phase golden_start mode=cpu-dual source=compute", flush=True)
    golden_t0 = time.perf_counter()
    golden_fp64 = chunk_gated_delta_rule_bwd_dhu_cpu(
        q, k, w, d_o, dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        g=g,
        gK=gk,
        h0=h0,
        dht=None,
        scale=scale,
        chunk_size=case.chunk_size,
        golden_mode="fp64",
    )
    bench_cpu = chunk_gated_delta_rule_bwd_dhu_cpu(
        q, k, w, d_o, dv,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        g=g,
        gK=gk,
        h0=h0,
        dht=None,
        scale=scale,
        chunk_size=case.chunk_size,
        golden_mode="npu",
    )
    golden_s = time.perf_counter() - golden_t0
    print(f"  phase_time golden_cpu_dual={golden_s:.6f}s", flush=True)
    return golden_fp64, bench_cpu


def get_cpu_dual_golden(
    case_index: int,
    case: Case,
    seed: int,
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    d_o: torch.Tensor,
    dv: torch.Tensor,
    g: Optional[torch.Tensor],
    gk: Optional[torch.Tensor],
    h0: Optional[torch.Tensor],
    cu_seqlens: Optional[list[int]],
    chunk_indices: Optional[list[int]],
    scale: float,
    cache_dir: str,
    cache_mode: str,
):
    expected_meta = expected_golden_meta(
        case_index, case, seed, q, k, w, d_o, dv, g, gk, h0, cu_seqlens, chunk_indices, scale,
    )
    path = golden_cache_path(cache_dir, case_index, case, seed) if cache_dir else ""

    if cache_mode in ("auto", "load"):
        cached, reason = try_load_golden_cache(path, expected_meta)
        if cached is not None:
            print(f"  phase golden_cache_load path={path}", flush=True)
            return cached
        print(f"  phase golden_cache_miss path={path} reason={reason}", flush=True)
        if cache_mode == "load":
            print("  result=ERROR golden cache required but not usable", flush=True)
            return None

    golden_fp64, bench_cpu = compute_cpu_dual_golden(
        case, q, k, w, d_o, dv, g, gk, h0, cu_seqlens, chunk_indices, scale,
    )
    if cache_mode != "none" and path:
        save_t0 = time.perf_counter()
        save_golden_cache(path, expected_meta, golden_fp64, bench_cpu)
        save_s = time.perf_counter() - save_t0
        print(f"  phase_time golden_cache_save={save_s:.6f}s path={path}", flush=True)
    return golden_fp64, bench_cpu


def compare_with_ct_dual(
    name: str,
    actual: torch.Tensor,
    golden_fp64: torch.Tensor,
    bench_cpu: torch.Tensor,
    verbose_ct: bool,
    ct_level: str,
) -> bool:
    if tuple(actual.shape) != tuple(golden_fp64.shape) or tuple(actual.shape) != tuple(bench_cpu.shape):
        print(
            f"  {name}: shape mismatch actual={tuple(actual.shape)} "
            f"fp64={tuple(golden_fp64.shape)} bench={tuple(bench_cpu.shape)}",
            flush=True,
        )
        return False
    import ct

    dtype_name = ct_dtype(actual.dtype)
    actual_cpu = actual.detach().cpu().float()
    golden_cpu = golden_fp64.detach().cpu().float()
    bench_cpu = bench_cpu.detach().cpu().float()
    ct_kwargs = {"dtype": dtype_name}
    if ct_level:
        ct_kwargs["level"] = ct_level
    try:
        if verbose_ct:
            result = ct.dual(actual_cpu, golden_cpu, bench_cpu, **ct_kwargs)
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = ct.dual(actual_cpu, golden_cpu, bench_cpu, **ct_kwargs)
    except Exception as exc:
        print(f"  ct.dual_cpu {name}: ERROR {exc}", flush=True)
        return False
    if ct_success(result):
        print(f"  ct.dual_cpu {name}: PASS", flush=True)
        return True
    print(f"  ct.dual_cpu {name}: FAIL ({ct_failure_summary(result)})", flush=True)
    return False


def print_case_header(case: Case, cu_seqlens: Optional[list[int]], chunk_indices: Optional[list[int]]) -> None:
    print(f"\n===== {case.name} =====", flush=True)
    print(
        f"shape: B={case.B} Hk={case.Hk} Hv={case.Hv} T={case.T} "
        f"K={case.K} V={case.V} chunk={case.chunk_size} gate={case.gate_kind} "
        f"q_dtype={case.q_dtype} gate_dtype={case.gate_dtype} h0={case.with_h0}",
        flush=True,
    )
    if cu_seqlens is not None:
        print(f"cu_seqlens={cu_seqlens} chunk_pairs={len(chunk_indices) // 2}", flush=True)


def run_kernel(case: Case, seed: int, device: int, artifact_path: Optional[str] = None):
    configure_npu_runtime()
    from fla_npu.ops import ascendc as ascendc_ops
    from fla_npu.ops.ascendc import _runtime as ascendc_runtime

    torch.npu.set_device(device)
    q, k, w, d_o, dv, g, gk, cu_seqlens, chunk_indices, h0, scale = build_case_inputs(case, seed)

    print("  phase kernel_launch_start", flush=True)
    kernel_t0 = time.perf_counter()
    dh, dh0, dv2 = ascendc_ops.npu_chunk_gated_delta_rule_bwd_dhu(
        q.npu(),
        k.npu(),
        w.npu(),
        d_o.npu(),
        dv.npu(),
        scale=scale,
        chunk_size=case.chunk_size,
        g=g.npu() if g is not None else None,
        gK=gk.npu() if gk is not None else None,
        h0=h0.npu() if h0 is not None else None,
        dht=None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    torch.npu.synchronize()
    kernel_s = time.perf_counter() - kernel_t0
    print(f"  phase_time kernel_sync={kernel_s:.6f}s", flush=True)

    if artifact_path is not None:
        payload = {
            "dh": dh.detach().cpu(),
            "dh0": dh0.detach().cpu() if dh0 is not None else None,
            "dv2": dv2.detach().cpu(),
            "dh0_is_none": dh0 is None,
        }
        torch.npu.synchronize()
        recent_launch_storage = getattr(ascendc_runtime, "_RECENT_LAUNCH_STORAGE", None)
        if recent_launch_storage is not None:
            recent_launch_storage.clear()
        del dh, dh0, dv2
        gc.collect()
        torch.npu.empty_cache()
        torch.npu.synchronize()
        torch.save(payload, artifact_path)
        return True

    return dh, dh0, dv2


def run_kernel_subprocess(
    case: Case,
    seed: int,
    device: int,
    artifact_path: str,
    required_only: bool,
    large_suite: bool,
) -> bool:
    env = os.environ.copy()
    if not env.get("FLA_NPU_OPP_PATH"):
        for search_path in sys.path:
            if "site-packages" not in search_path and "dist-packages" not in search_path:
                continue
            candidate = os.path.join(search_path, "fla_npu", "opp")
            vendor_opapi = os.path.join(
                candidate, "vendors", "fla_npu_transformer", "op_api", "lib", "libcust_opapi.so"
            )
            if os.path.exists(vendor_opapi):
                env["FLA_NPU_OPP_PATH"] = candidate
                break

    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--device",
        str(device),
        "--seed",
        str(seed),
        "--case",
        case.name,
        "--kernel-only",
        "--kernel-artifact",
        artifact_path,
        "--final-seed",
    ]
    if required_only:
        cmd.append("--required-only")
    if large_suite:
        cmd.append("--large-suite")
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    start_t = time.perf_counter()
    launch_t = None
    kernel_synced = False

    while True:
        for key, _ in selector.select(timeout=1.0):
            line = key.fileobj.readline()
            if not line:
                continue
            print(line, end="", flush=True)
            if "phase kernel_launch_start" in line:
                launch_t = time.perf_counter()
            if "phase_time kernel_sync=" in line:
                kernel_synced = True

        if process.poll() is not None:
            for line in process.stdout:
                print(line, end="", flush=True)
                if "phase kernel_launch_start" in line and launch_t is None:
                    launch_t = time.perf_counter()
                if "phase_time kernel_sync=" in line:
                    kernel_synced = True
            selector.unregister(process.stdout)
            process.stdout.close()
            return process.returncode == 0

        now = time.perf_counter()
        if launch_t is None and now - start_t > KERNEL_STARTUP_TIMEOUT_S:
            process.kill()
            process.wait()
            print(
                f"  result=ERROR kernel subprocess startup timeout before launch_start "
                f"after {KERNEL_STARTUP_TIMEOUT_S}s",
                flush=True,
            )
            selector.unregister(process.stdout)
            process.stdout.close()
            return False
        if launch_t is not None and not kernel_synced and now - launch_t > KERNEL_SYNC_TIMEOUT_S:
            process.kill()
            process.wait()
            print(
                f"  result=ERROR kernel_sync timeout after launch_start "
                f"after {KERNEL_SYNC_TIMEOUT_S}s",
                flush=True,
            )
            selector.unregister(process.stdout)
            process.stdout.close()
            return False


def load_kernel_payload(artifact_path: str):
    kernel_payload = torch.load(artifact_path, map_location="cpu")
    return (
        kernel_payload["dh"],
        kernel_payload["dh0"],
        kernel_payload["dv2"],
        bool(kernel_payload["dh0_is_none"]),
    )


def run_case(
    case_index: int,
    case: Case,
    seed: int,
    device: int,
    verbose_ct: bool,
    ct_level: str,
    required_only: bool,
    large_suite: bool,
    golden_cache_dir: str,
    golden_cache_mode: str,
    actual_artifact_path: str = "",
) -> bool:
    q, k, w, d_o, dv, g, gk, cu_seqlens, chunk_indices, h0, scale = build_case_inputs(case, seed)

    print_case_header(case, cu_seqlens, chunk_indices)
    if actual_artifact_path:
        print(f"  phase actual_load_start path={actual_artifact_path}", flush=True)
        dh, dh0, dv2, dh0_is_none = load_kernel_payload(actual_artifact_path)
    else:
        with tempfile.TemporaryDirectory(prefix="dhu_kernel_") as tmp_dir:
            artifact_path = os.path.join(tmp_dir, f"{case.name}.pt")
            if not run_kernel_subprocess(case, seed, device, artifact_path, required_only, large_suite):
                return False
            dh, dh0, dv2, dh0_is_none = load_kernel_payload(artifact_path)

    if case.with_h0 and dh0_is_none:
        print("  dh0: expected tensor, got None", flush=True)
        return False
    if not case.with_h0 and not dh0_is_none:
        print("  dh0: expected None, got tensor", flush=True)
        return False

    golden_payload = get_cpu_dual_golden(
        case_index,
        case,
        seed,
        q,
        k,
        w,
        d_o,
        dv,
        g,
        gk,
        h0,
        cu_seqlens,
        chunk_indices,
        scale,
        golden_cache_dir,
        golden_cache_mode,
    )
    if golden_payload is None:
        return False
    (dh_fp64, dh0_fp64, dv2_fp64), (dh_bench, dh0_bench, dv2_bench) = golden_payload

    ok_dh = compare_with_ct_dual("dh", dh, dh_fp64, dh_bench, verbose_ct, ct_level)
    ok_dh0 = True
    if case.with_h0:
        ok_dh0 = compare_with_ct_dual("dh0", dh0, dh0_fp64, dh0_bench, verbose_ct, ct_level)
    ok_dv2 = compare_with_ct_dual("dv2", dv2, dv2_fp64, dv2_bench, verbose_ct, ct_level)
    ok = ok_dh and ok_dh0 and ok_dv2
    print(f"  result={'PASS' if ok else 'FAIL'}", flush=True)
    return ok


def run_golden_only_case(
    case_index: int,
    case: Case,
    seed: int,
    golden_cache_dir: str,
    golden_cache_mode: str,
) -> bool:
    q, k, w, d_o, dv, g, gk, cu_seqlens, chunk_indices, h0, scale = build_case_inputs(case, seed)
    print_case_header(case, cu_seqlens, chunk_indices)
    golden_payload = get_cpu_dual_golden(
        case_index,
        case,
        seed,
        q,
        k,
        w,
        d_o,
        dv,
        g,
        gk,
        h0,
        cu_seqlens,
        chunk_indices,
        scale,
        golden_cache_dir,
        golden_cache_mode,
    )
    ok = golden_payload is not None
    print(f"  result={'GOLDEN_READY' if ok else 'FAIL'}", flush=True)
    return ok


def build_cases(required_only: bool, large_suite: bool) -> list[Case]:
    split = random_three_way_split(8192, seed=20260805)
    required = [
        Case("required_varlen_g_fp16", 1, 96, 96, 8192, 128, 128, 64, torch.float16, torch.float32, "g", split),
        Case("required_varlen_gK_fp16", 1, 96, 96, 8192, 128, 128, 64, torch.float16, torch.float32, "gK", split),
    ]
    if required_only:
        return required
    non_gva_split = random_seq_split(4096, 3, seed=20260805)
    gva_split = random_seq_split(4096, 5, seed=20260807)
    large = [
        Case("large_nongva_varlen_v128_no_dh0_g_bf16", 1, 16, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "g", non_gva_split),
        Case("large_nongva_varlen_v128_no_dh0_gK_bf16", 1, 16, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "gK", non_gva_split),
        Case("large_nongva_varlen_v128_dh0_g_bf16", 1, 16, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "g", non_gva_split, with_h0=True),
        Case("large_nongva_varlen_v128_dh0_gK_bf16", 1, 16, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "gK", non_gva_split, with_h0=True),
        Case("large_nongva_varlen_v256_no_dh0_g_bf16", 1, 16, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "g", non_gva_split),
        Case("large_nongva_varlen_v256_no_dh0_gK_bf16", 1, 16, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "gK", non_gva_split),
        Case("large_nongva_varlen_v256_dh0_g_bf16", 1, 16, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "g", non_gva_split, with_h0=True),
        Case("large_nongva_varlen_v256_dh0_gK_bf16", 1, 16, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "gK", non_gva_split, with_h0=True),
        Case("large_gva_varlen_v128_no_dh0_g_bf16", 1, 8, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "g", gva_split),
        Case("large_gva_varlen_v128_no_dh0_gK_bf16", 1, 8, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "gK", gva_split),
        Case("large_gva_varlen_v128_dh0_g_bf16", 1, 8, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "g", gva_split, with_h0=True),
        Case("large_gva_varlen_v128_dh0_gK_bf16", 1, 8, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "gK", gva_split, with_h0=True),
        Case("large_gva_varlen_v256_no_dh0_g_bf16", 1, 8, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "g", gva_split),
        Case("large_gva_varlen_v256_no_dh0_gK_bf16", 1, 8, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "gK", gva_split),
        Case("large_gva_varlen_v256_dh0_g_bf16", 1, 8, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "g", gva_split, with_h0=True),
        Case("large_gva_varlen_v256_dh0_gK_bf16", 1, 8, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "gK", gva_split, with_h0=True),
        Case("large_nongva_fixed_v128_no_dh0_g_bf16", 1, 16, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "g"),
        Case("large_nongva_fixed_v128_no_dh0_gK_bf16", 1, 16, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "gK"),
        Case("large_nongva_fixed_v128_dh0_g_bf16", 1, 16, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "g", with_h0=True),
        Case("large_nongva_fixed_v128_dh0_gK_bf16", 1, 16, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "gK", with_h0=True),
        Case("large_nongva_fixed_v256_no_dh0_g_bf16", 1, 16, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "g"),
        Case("large_nongva_fixed_v256_no_dh0_gK_bf16", 1, 16, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "gK"),
        Case("large_nongva_fixed_v256_dh0_g_bf16", 1, 16, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "g", with_h0=True),
        Case("large_nongva_fixed_v256_dh0_gK_bf16", 1, 16, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "gK", with_h0=True),
        Case("large_gva_fixed_v128_no_dh0_g_bf16", 1, 8, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "g"),
        Case("large_gva_fixed_v128_no_dh0_gK_bf16", 1, 8, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "gK"),
        Case("large_gva_fixed_v128_dh0_g_bf16", 1, 8, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "g", with_h0=True),
        Case("large_gva_fixed_v128_dh0_gK_bf16", 1, 8, 16, 4096, 128, 128, 64,
             torch.bfloat16, torch.float32, "gK", with_h0=True),
        Case("large_gva_fixed_v256_no_dh0_g_bf16", 1, 8, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "g"),
        Case("large_gva_fixed_v256_no_dh0_gK_bf16", 1, 8, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "gK"),
        Case("large_gva_fixed_v256_dh0_g_bf16", 1, 8, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "g", with_h0=True),
        Case("large_gva_fixed_v256_dh0_gK_bf16", 1, 8, 16, 4096, 128, 256, 64,
             torch.bfloat16, torch.float32, "gK", with_h0=True),
        Case("large_nongva_fixed_v128_no_dh0_g_fp16", 1, 16, 16, 4096, 128, 128, 64,
             torch.float16, torch.float32, "g"),
        Case("large_gva_varlen_v256_dh0_gK_fp16", 1, 8, 16, 4096, 128, 256, 64,
             torch.float16, torch.float32, "gK", gva_split, with_h0=True),
    ]
    if large_suite:
        return large
    return [
        Case("fixed_small_g_fp16", 1, 4, 4, 128, 128, 128, 64, torch.float16, torch.float32, "g"),
        Case("fixed_small_gK_fp16", 1, 4, 4, 128, 128, 128, 64, torch.float16, torch.float32, "gK"),
        Case("fixed_small_g_dh0_fp16", 1, 4, 4, 128, 128, 128, 64, torch.float16, torch.float32, "g",
             with_h0=True),
        Case("fixed_small_gK_dh0_fp16", 1, 4, 4, 128, 128, 128, 64, torch.float16, torch.float32, "gK",
             with_h0=True),
        Case("gva_varlen_g_bf16_gate_fp32", 1, 2, 4, 257, 128, 128, 64, torch.bfloat16, torch.float32,
             "g", [0, 63, 151, 257]),
        Case("gva_varlen_gK_bf16_gate_fp32", 1, 2, 4, 257, 128, 128, 64, torch.bfloat16, torch.float32,
             "gK", [0, 63, 151, 257]),
        Case("gva_varlen_g_dh0_bf16_gate_fp32", 1, 2, 4, 257, 128, 128, 64, torch.bfloat16, torch.float32,
             "g", [0, 63, 151, 257], with_h0=True),
        Case("gva_varlen_gK_dh0_bf16_gate_fp32", 1, 2, 4, 257, 128, 128, 64, torch.bfloat16, torch.float32,
             "gK", [0, 63, 151, 257], with_h0=True),
        *required,
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="DHU full-tensor accuracy test for g and gK branches.")
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", "0")))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--required-only", action="store_true")
    parser.add_argument("--large-suite", action="store_true", help="Run the DHU large accuracy suite.")
    parser.add_argument(
        "--gate-kind", choices=("all", "g", "gK"), default="all", help="Filter cases by gate branch."
    )
    parser.add_argument("--case", default="", help="Run one case by name.")
    parser.add_argument("--golden-mode", choices=("cpu-dual",), default="cpu-dual")
    parser.add_argument("--golden-device", choices=("npu", "cpu"), default="", help=argparse.SUPPRESS)
    parser.add_argument("--golden-only", action="store_true", help="Build or validate CPU dual golden cache only.")
    parser.add_argument("--golden-cache-dir", default=DEFAULT_GOLDEN_CACHE_DIR, help="Per-case CPU dual golden cache dir.")
    parser.add_argument(
        "--golden-cache-mode",
        choices=("auto", "load", "refresh", "none"),
        default="auto",
        help="auto: load valid cache or compute/save; load: require cache; refresh: recompute/save; none: do not cache.",
    )
    parser.add_argument("--ct-level", default="", help="Optional ct.dual level override; default matches prepare.")
    parser.add_argument("--verbose-ct", action="store_true", help="Print every ct report.")
    parser.add_argument("--actual-artifact", default="", help="Load one kernel actual artifact instead of running kernel.")
    parser.add_argument("--actual-artifact-dir", default="", help="Load per-case kernel actual artifacts from this dir.")
    parser.add_argument("--kernel-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kernel-artifact", default="", help=argparse.SUPPRESS)
    parser.add_argument("--final-seed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.golden_device:
        print("deprecated --golden-device ignored; using --golden-mode cpu-dual", flush=True)
    if args.golden_only and (args.actual_artifact or args.actual_artifact_dir):
        raise SystemExit("--golden-only cannot be combined with actual artifact loading")

    cases = build_cases(args.required_only, args.large_suite)
    indexed_cases = list(enumerate(cases))
    if args.gate_kind != "all":
        indexed_cases = [(index, case) for index, case in indexed_cases if case.gate_kind == args.gate_kind]
        cases = [case for _, case in indexed_cases]
    if args.case:
        indexed_cases = [(index, case) for index, case in indexed_cases if case.name == args.case]
        if not indexed_cases:
            known_cases = build_cases(False, False) + build_cases(False, True)
            indexed_cases = [(index, case) for index, case in enumerate(known_cases) if case.name == args.case]
        if not indexed_cases:
            raise SystemExit(f"unknown case: {args.case}")
        cases = [case for _, case in indexed_cases]

    if args.actual_artifact and len(cases) != 1:
        raise SystemExit("--actual-artifact requires exactly one --case")

    if args.large_suite and not args.case and not args.case_worker:
        all_ok = True
        passed = 0
        for case in cases:
            cmd = [
                sys.executable,
                os.path.abspath(__file__),
                "--device",
                str(args.device),
                "--seed",
                str(args.seed),
                "--large-suite",
                "--case",
                case.name,
                "--case-worker",
                "--golden-cache-dir",
                args.golden_cache_dir,
                "--golden-cache-mode",
                args.golden_cache_mode,
            ]
            if args.golden_only:
                cmd.append("--golden-only")
            if args.ct_level:
                cmd.extend(("--ct-level", args.ct_level))
            if args.verbose_ct:
                cmd.append("--verbose-ct")
            if args.actual_artifact_dir:
                cmd.extend(("--actual-artifact-dir", args.actual_artifact_dir))
            return_code = subprocess.run(cmd, env=os.environ.copy()).returncode
            all_ok = return_code == 0 and all_ok
            passed += int(return_code == 0)
        print(f"\nLARGE_SUITE {passed}/{len(cases)} PASS", flush=True)
        return 0 if all_ok else 1

    if args.kernel_only:
        if len(cases) != 1:
            raise SystemExit("--kernel-only requires exactly one --case")
        if not args.kernel_artifact:
            raise SystemExit("--kernel-only requires --kernel-artifact")
        case_index, case = indexed_cases[0]
        kernel_seed = args.seed if args.final_seed else args.seed + case_index
        return 0 if run_kernel(case, kernel_seed, args.device, artifact_path=args.kernel_artifact) else 1

    all_ok = True
    if args.golden_only:
        for index, case in indexed_cases:
            all_ok = run_golden_only_case(
                index,
                case,
                seed=args.seed + index,
                golden_cache_dir=args.golden_cache_dir,
                golden_cache_mode=args.golden_cache_mode,
            ) and all_ok
        print(f"\nOVERALL {'GOLDEN_READY' if all_ok else 'FAIL'}", flush=True)
        return 0 if all_ok else 1

    for index, case in indexed_cases:
        actual_artifact_path = args.actual_artifact
        if args.actual_artifact_dir:
            actual_artifact_path = os.path.join(args.actual_artifact_dir, f"{case.name}.pt")
        all_ok = run_case(
            index,
            case,
            seed=args.seed + index,
            device=args.device,
            verbose_ct=args.verbose_ct,
            ct_level=args.ct_level,
            required_only=args.required_only,
            large_suite=args.large_suite,
            golden_cache_dir=args.golden_cache_dir,
            golden_cache_mode=args.golden_cache_mode,
            actual_artifact_path=actual_artifact_path,
        ) and all_ok
        gc.collect()
    print(f"\nOVERALL {'PASS' if all_ok else 'FAIL'}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
