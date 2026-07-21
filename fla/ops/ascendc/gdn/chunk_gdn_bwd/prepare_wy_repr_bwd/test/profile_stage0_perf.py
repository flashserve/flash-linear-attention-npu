#!/usr/bin/env python3
import gc
import importlib
import os
import random
import time
from dataclasses import dataclass

import torch
import torch_npu

from fla_npu.ops import ascendc as fla_ascendc


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class PerfCase:
    name: str
    B: int
    H: int
    T: int
    K: int
    V: int
    chunk_size: int
    ktype: str = "bf16"
    gtype: str = "fp32"
    seq_num: int = 64


PERF_CASES = (
    PerfCase("P1_V128_C64", 1, 32, 65536, 128, 128, 64),
    PerfCase("P2_V128_C128", 1, 32, 65536, 128, 128, 128),
    PerfCase("P3_V256_C64", 1, 32, 65536, 128, 256, 64),
    PerfCase("P4_V256_C128", 1, 32, 65536, 128, 256, 128),
)


def release_aclnn_keepalive():
    try:
        runtime_mod = importlib.import_module("fla_npu.ops.ascendc._runtime")
        runtime_mod._RECENT_LAUNCH_STORAGE.clear()
    except Exception:
        pass


def rand_symmetric(shape: tuple[int, ...], device: str, dtype: torch.dtype):
    return torch.rand(shape, device=device, dtype=dtype) * 2.0 - 1.0


def prepare_cu_seqlens(total_tokens: int, seq_num: int, seed: int) -> list[int]:
    random.seed(seed)
    middle_points = random.sample(range(1, total_tokens), seq_num - 1)
    middle_points.sort()
    return [0] + middle_points + [total_tokens]


def prepare_chunk_indices(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    indices: list[int] = []
    for seq_idx in range(len(cu_seqlens) - 1):
        length = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        if length <= 0:
            continue
        for chunk_idx in range((length + chunk_size - 1) // chunk_size):
            indices.extend([seq_idx, chunk_idx])
    return indices


def make_inputs(case: PerfCase, device: str, seed: int):
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    ktype = DTYPES[case.ktype]
    gtype = DTYPES[case.gtype]
    k = rand_symmetric((case.B, case.H, case.T, case.K), device, ktype)
    v = rand_symmetric((case.B, case.H, case.T, case.V), device, ktype)
    beta = rand_symmetric((case.B, case.H, case.T), device, gtype)
    A = rand_symmetric((case.B, case.H, case.T, case.chunk_size), device, ktype)
    dw = rand_symmetric((case.B, case.H, case.T, case.K), device, ktype)
    du = rand_symmetric((case.B, case.H, case.T, case.V), device, ktype)
    g = rand_symmetric((case.B, case.H, case.T), device, gtype)
    cu_seqlens = prepare_cu_seqlens(case.T, case.seq_num, seed)
    chunk_indices = prepare_chunk_indices(cu_seqlens, case.chunk_size)
    return k, v, beta, A, dw, du, g, cu_seqlens, chunk_indices


def run_once(case: PerfCase, device: str, seed: int, tag: str):
    k, v, beta, A, dw, du, g, cu_seqlens, chunk_indices = make_inputs(case, device, seed)
    torch.npu.synchronize()
    task_num = len(chunk_indices) // 2
    print(
        f"{tag} {case.name}: B={case.B}, H={case.H}, T={case.T}, K={case.K}, V={case.V}, "
        f"chunk={case.chunk_size}, seq_num={case.seq_num}, tasks={task_num}, "
        f"ktype={case.ktype}, gtype={case.gtype}",
        flush=True,
    )
    start = time.perf_counter()
    outputs = fla_ascendc.prepare_wy_repr_bwd_stage0_debug(
        k, v, beta, A, dw, du, g, case.chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    torch.npu.synchronize()
    seconds = time.perf_counter() - start
    print(f"{tag} {case.name}: kernel_sync={seconds * 1000000.0:.3f} us", flush=True)
    release_aclnn_keepalive()
    del k, v, beta, A, dw, du, g, outputs
    gc.collect()
    torch.npu.empty_cache()


def selected_cases() -> list[PerfCase]:
    wanted = os.environ.get("PREPARE_WY_STAGE0_PERF_CASES", "").strip()
    if not wanted:
        return list(PERF_CASES)
    names = {name.strip() for name in wanted.split(",") if name.strip()}
    return [case for case in PERF_CASES if case.name in names]


def main() -> int:
    device = os.environ.get("PREPARE_WY_STAGE0_PERF_DEVICE", "npu")
    seed = int(os.environ.get("PREPARE_WY_STAGE0_PERF_SEED", "2026"))
    warmup = int(os.environ.get("PREPARE_WY_STAGE0_PERF_WARMUP", "1"))
    repeat = int(os.environ.get("PREPARE_WY_STAGE0_PERF_REPEAT", "1"))
    cases = selected_cases()
    if not cases:
        raise RuntimeError("No perf cases selected.")
    with torch.no_grad():
        for case in cases:
            for idx in range(warmup):
                run_once(case, device, seed + idx, "WARMUP")
            for idx in range(repeat):
                run_once(case, device, seed + warmup + idx, "MEASURE")
    print("ALL PERF CASES DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
