import argparse
import gc
import importlib
import random
import time
from dataclasses import dataclass
from typing import Iterable

import ct
import torch
import torch_npu

from fla_npu.ops import ascendc as fla_ascendc


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def release_aclnn_keepalive():
    try:
        runtime_mod = importlib.import_module("fla_npu.ops.ascendc._runtime")
        runtime_mod._RECENT_LAUNCH_STORAGE.clear()
    except Exception:
        pass


@dataclass(frozen=True)
class Stage0Case:
    name: str
    B: int
    KH: int
    VH: int
    T: int
    K: int
    V: int
    chunk_size: int
    ktype: str
    gtype: str
    cu_seqlens_len: int | None = None


def prepare_cu_seqlens(T: int, L: int, seed: int = 42) -> list[int]:
    if T < 1:
        raise ValueError("T must be at least 1.")
    if L < 2 or L > T + 1:
        raise ValueError(f"L must satisfy 2 <= L <= T + 1, got L={L}, T={T}.")
    random.seed(seed)
    if L == 2:
        return [0, T]
    middle_points = random.sample(range(1, T), L - 2)
    middle_points.sort()
    return [0] + middle_points + [T]


def prepare_chunk_indices(cu_seqlens: list[int], chunk_size: int) -> list[int]:
    indices: list[int] = []
    for seq_idx in range(len(cu_seqlens) - 1):
        length = cu_seqlens[seq_idx + 1] - cu_seqlens[seq_idx]
        if length <= 0:
            continue
        for chunk_idx in range((length + chunk_size - 1) // chunk_size):
            indices.extend([seq_idx, chunk_idx])
    return indices


FULL_GDN_CASES: tuple[Stage0Case, ...] = (
    Stage0Case("F1", 64, 8, 8, 1024, 128, 128, 64, "fp16", "fp16"),
    Stage0Case("F2", 32, 16, 16, 2048, 128, 128, 64, "bf16", "bf16"),
    Stage0Case("F3", 16, 32, 32, 4096, 128, 128, 128, "fp16", "fp32"),
    Stage0Case("F4", 8, 32, 32, 8192, 128, 128, 128, "bf16", "bf16"),
    Stage0Case("F5", 128, 4, 4, 1024, 128, 128, 64, "fp16", "fp16"),
    Stage0Case("F6", 64, 8, 8, 2048, 128, 128, 64, "bf16", "fp32"),
    Stage0Case("F7", 32, 16, 16, 4096, 128, 128, 128, "fp16", "fp16"),
    Stage0Case("F8", 16, 32, 32, 8192, 128, 128, 128, "bf16", "bf16"),
    Stage0Case("F9", 64, 8, 8, 4096, 128, 128, 128, "fp16", "fp16"),
    Stage0Case("F10", 32, 16, 16, 8192, 128, 128, 128, "bf16", "bf16"),
    Stage0Case("F11", 16, 32, 32, 16384, 128, 128, 64, "fp16", "fp32"),
    Stage0Case("F12", 8, 32, 32, 32768, 128, 128, 64, "bf16", "bf16"),
    Stage0Case("F13", 64, 8, 8, 1024, 128, 128, 64, "fp16", "fp16"),
    Stage0Case("F14", 32, 16, 16, 2048, 128, 128, 64, "bf16", "bf16"),
    Stage0Case("F15", 16, 32, 32, 4096, 128, 128, 128, "fp16", "fp32"),
    Stage0Case("F16", 8, 32, 32, 8192, 128, 128, 128, "bf16", "bf16"),
    Stage0Case("F17", 64, 8, 8, 2048, 128, 128, 64, "bf16", "bf16"),
    Stage0Case("F18", 32, 16, 16, 4096, 128, 128, 128, "fp16", "fp16"),
    Stage0Case("L1", 1, 32, 32, 65536, 128, 128, 64, "bf16", "bf16", 64),
    Stage0Case("L2", 1, 16, 16, 65536, 128, 128, 64, "fp16", "fp16", 33),
    Stage0Case("L3", 1, 8, 8, 131072, 128, 128, 64, "bf16", "bf16", 333),
    Stage0Case("L4", 1, 4, 4, 262144, 128, 128, 64, "fp16", "fp32", 567),
    Stage0Case("L5", 1, 16, 16, 32768, 128, 128, 64, "bf16", "bf16", 7),
    Stage0Case("L6", 1, 8, 8, 65536, 128, 128, 64, "fp16", "fp16", 25),
    Stage0Case("L7", 1, 16, 32, 16384, 128, 256, 64, "fp16", "fp32", 128),
    Stage0Case("L8", 1, 21, 63, 16384, 128, 256, 64, "bf16", "fp32", 2),
    Stage0Case("L9", 1, 8, 32, 65536, 128, 256, 128, "bf16", "fp32", 172),
    Stage0Case("L10", 1, 16, 32, 65536, 128, 128, 64, "fp16", "fp32", 668),
    Stage0Case("L11", 1, 4, 32, 65536, 128, 128, 128, "bf16", "fp32", 17),
    Stage0Case("L12", 1, 2, 64, 262144, 128, 256, 64, "fp16", "fp32", 32),
    Stage0Case("F19", 1, 16, 32, 4096, 128, 256, 64, "fp16", "fp32"),
    Stage0Case("F20", 16, 21, 63, 2048, 128, 256, 64, "bf16", "fp32"),
    Stage0Case("F21", 711, 4, 32, 196, 128, 128, 128, "fp16", "fp32"),
    Stage0Case("F22", 176, 2, 64, 24, 128, 256, 64, "bf16", "fp32"),
)

SMOKE_CASE_NAMES = {"F1", "F3", "F19", "F22", "L1", "L7", "L8", "L9"}


def rand_symmetric(shape: tuple[int, ...], device: str, dtype: torch.dtype, input_scale: float):
    return (torch.rand(shape, device=device, dtype=dtype) * 2.0 - 1.0) * input_scale


def make_inputs(case: Stage0Case, device: str, seed: int, input_scale: float):
    torch.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
    ktype = DTYPES[case.ktype]
    gtype = DTYPES[case.gtype]
    k = rand_symmetric((case.B, case.KH, case.T, case.K), device, ktype, input_scale)
    v = rand_symmetric((case.B, case.VH, case.T, case.V), device, ktype, input_scale)
    beta = rand_symmetric((case.B, case.VH, case.T), device, gtype, input_scale)
    A = rand_symmetric((case.B, case.VH, case.T, case.chunk_size), device, ktype, input_scale)
    dw = rand_symmetric((case.B, case.VH, case.T, case.K), device, ktype, input_scale)
    du = rand_symmetric((case.B, case.VH, case.T, case.V), device, ktype, input_scale)
    g = rand_symmetric((case.B, case.VH, case.T), device, gtype, input_scale)
    return k, v, beta, A, dw, du, g


def run_stage0_golden(case: Stage0Case, device: str, seed: int, input_scale: float, kernel_only: bool):
    k, v, beta, A, dw, du, g = make_inputs(case, device, seed, input_scale)
    cu_seqlens = None
    chunk_indices = None
    if case.cu_seqlens_len is not None:
        cu_seqlens = prepare_cu_seqlens(case.T, case.cu_seqlens_len)
        chunk_indices = prepare_chunk_indices(cu_seqlens, case.chunk_size)

    actual_start = time.perf_counter()
    actual = fla_ascendc.prepare_wy_repr_bwd(
        k, v, beta, A, dw, du, g, case.chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    torch.npu.synchronize()
    release_aclnn_keepalive()
    print(f"  phase_time actual_kernel_sync={time.perf_counter() - actual_start:.6f}s", flush=True)
    if kernel_only:
        return actual, None

    golden_start = time.perf_counter()
    dA = fla_ascendc.prepare_wy_repr_bwd_da(
        k, v, beta, A, dw, du, g, chunk_size=case.chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    torch.npu.synchronize()
    release_aclnn_keepalive()
    print(f"  phase_time golden_da_kernel_sync={time.perf_counter() - golden_start:.6f}s", flush=True)
    full_start = time.perf_counter()
    expected = fla_ascendc.prepare_wy_repr_bwd_full(
        k, v, beta, A, dA, dw, du, g, case.chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    torch.npu.synchronize()
    release_aclnn_keepalive()
    print(f"  phase_time golden_full_kernel_sync={time.perf_counter() - full_start:.6f}s", flush=True)
    return actual, expected


def ct_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype).removeprefix("torch.")


def ct_success(result) -> bool:
    if isinstance(result, dict):
        return bool(result.get("success", False))
    return bool(result)


def ct_failure_summary(result) -> str:
    if not isinstance(result, dict):
        return "no detail"
    metrics = result.get("metrics")
    if metrics is None:
        return "no metrics"
    return (
        f"fail_count={getattr(metrics, 'fail_count', 'NA')}, "
        f"pass_rate={getattr(metrics, 'pass_rate', 'NA')}, "
        f"max_diff={getattr(metrics, 'max_diff', 'NA')}, "
        f"max_re={getattr(metrics, 'max_re_calc', 'NA')}"
    )


def compare_outputs(case: Stage0Case, actual, expected) -> list[str]:
    output_names = ("dk", "dv", "dbeta", "dg")
    failures: list[str] = []
    for name, act, exp in zip(output_names, actual, expected):
        print(f"  {name}: shape={tuple(act.shape)} dtype={act.dtype}", flush=True)
        compare_start = time.perf_counter()
        try:
            result = ct.single(act.detach().cpu(), exp.detach().cpu(), dtype=ct_dtype(act.dtype))
        except Exception as exc:
            failures.append(f"{case.name}.{name}: ct.single failed: {exc}")
        else:
            if not ct_success(result):
                failures.append(f"{case.name}.{name}: ct.single failed ({ct_failure_summary(result)})")
        print(f"  phase_time ct_single_{name}={time.perf_counter() - compare_start:.6f}s", flush=True)
    return failures


def select_cases(args) -> list[Stage0Case]:
    if args.cases:
        wanted = {case.strip() for case in args.cases.split(",") if case.strip()}
        return [case for case in FULL_GDN_CASES if case.name in wanted]
    if args.suite == "smoke":
        return [case for case in FULL_GDN_CASES if case.name in SMOKE_CASE_NAMES]
    if args.suite == "fixed":
        return [case for case in FULL_GDN_CASES if case.cu_seqlens_len is None]
    if args.suite == "varlen":
        return [case for case in FULL_GDN_CASES if case.cu_seqlens_len is not None]
    return list(FULL_GDN_CASES)


def run_cases(cases: Iterable[Stage0Case], args) -> int:
    all_failures: list[str] = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx}] {case.name}: B={case.B}, KH={case.KH}, VH={case.VH}, T={case.T}, "
            f"K={case.K}, V={case.V}, chunk={case.chunk_size}, ktype={case.ktype}, gtype={case.gtype}, "
            f"varlen_L={case.cu_seqlens_len}",
            flush=True,
        )
        actual = expected = None
        try:
            actual, expected = run_stage0_golden(case, args.device, args.seed, args.input_scale, args.kernel_only)
            failures = [] if args.kernel_only else compare_outputs(case, actual, expected)
        finally:
            del actual, expected
            gc.collect()
            torch.npu.empty_cache()
        if failures:
            all_failures.extend(failures)
            print(f"  RESULT: FAIL ({len(failures)} outputs)")
            if args.stop_on_fail:
                break
        elif args.kernel_only:
            print("  RESULT: KERNEL_ONLY_PASS (precision not checked)")
        else:
            print("  RESULT: PASS")

    if all_failures:
        print("FAILED CASES:")
        for failure in all_failures:
            print(f"  {failure}")
        return 1
    if args.kernel_only:
        print("ALL CASES KERNEL RETURNED")
        return 0
    print("ALL CASES PASS")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="prepare_wy_repr_bwd stage0 golden precision test")
    parser.add_argument("--suite", choices=("smoke", "fixed", "varlen", "all"), default="smoke")
    parser.add_argument("--cases", default="", help="Comma-separated case names, e.g. F1,L1,F22.")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Half-width of the symmetric random input range. Default 1.0 generates values in [-1, 1].",
    )
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--kernel-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    raise SystemExit(run_cases(select_cases(cli_args), cli_args))
