import argparse
import contextlib
import gc
import importlib
import io
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
class Stage1Case:
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


@dataclass(frozen=True)
class TaskRange:
    batch: int
    begin: int
    end: int


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


FULL_GDN_CASES: tuple[Stage1Case, ...] = (
    Stage1Case("F1", 64, 8, 8, 1024, 128, 128, 64, "fp16", "fp16"),
    Stage1Case("F2", 32, 16, 16, 2048, 128, 128, 64, "bf16", "bf16"),
    Stage1Case("F3", 16, 32, 32, 4096, 128, 128, 128, "fp16", "fp32"),
    Stage1Case("F4", 8, 32, 32, 8192, 128, 128, 128, "bf16", "bf16"),
    Stage1Case("F5", 128, 4, 4, 1024, 128, 128, 64, "fp16", "fp16"),
    Stage1Case("F6", 64, 8, 8, 2048, 128, 128, 64, "bf16", "fp32"),
    Stage1Case("F7", 32, 16, 16, 4096, 128, 128, 128, "fp16", "fp16"),
    Stage1Case("F8", 16, 32, 32, 8192, 128, 128, 128, "bf16", "bf16"),
    Stage1Case("F9", 64, 8, 8, 4096, 128, 128, 128, "fp16", "fp16"),
    Stage1Case("F10", 32, 16, 16, 8192, 128, 128, 128, "bf16", "bf16"),
    Stage1Case("F11", 16, 32, 32, 16384, 128, 128, 64, "fp16", "fp32"),
    Stage1Case("F12", 8, 32, 32, 32768, 128, 128, 64, "bf16", "bf16"),
    Stage1Case("F13", 64, 8, 8, 1024, 128, 128, 64, "fp16", "fp16"),
    Stage1Case("F14", 32, 16, 16, 2048, 128, 128, 64, "bf16", "bf16"),
    Stage1Case("F15", 16, 32, 32, 4096, 128, 128, 128, "fp16", "fp32"),
    Stage1Case("F16", 8, 32, 32, 8192, 128, 128, 128, "bf16", "bf16"),
    Stage1Case("F17", 64, 8, 8, 2048, 128, 128, 64, "bf16", "bf16"),
    Stage1Case("F18", 32, 16, 16, 4096, 128, 128, 128, "fp16", "fp16"),
    Stage1Case("L1", 1, 32, 32, 65536, 128, 128, 64, "bf16", "bf16", 64),
    Stage1Case("L2", 1, 16, 16, 65536, 128, 128, 64, "fp16", "fp16", 33),
    Stage1Case("L3", 1, 8, 8, 131072, 128, 128, 64, "bf16", "bf16", 333),
    Stage1Case("L4", 1, 4, 4, 262144, 128, 128, 64, "fp16", "fp32", 567),
    Stage1Case("L5", 1, 16, 16, 32768, 128, 128, 64, "bf16", "bf16", 7),
    Stage1Case("L6", 1, 8, 8, 65536, 128, 128, 64, "fp16", "fp16", 25),
    Stage1Case("L7", 1, 16, 32, 16384, 128, 256, 64, "fp16", "fp32", 128),
    Stage1Case("L8", 1, 21, 63, 16384, 128, 256, 64, "bf16", "fp32", 2),
    Stage1Case("L9", 1, 8, 32, 65536, 128, 256, 128, "bf16", "fp32", 172),
    Stage1Case("L10", 1, 16, 32, 65536, 128, 128, 64, "fp16", "fp32", 668),
    Stage1Case("L11", 1, 4, 32, 65536, 128, 128, 128, "bf16", "fp32", 17),
    Stage1Case("L12", 1, 2, 64, 262144, 128, 256, 64, "fp16", "fp32", 32),
    Stage1Case("F19", 1, 16, 32, 4096, 128, 256, 64, "fp16", "fp32"),
    Stage1Case("F20", 16, 21, 63, 2048, 128, 256, 64, "bf16", "fp32"),
    Stage1Case("F21", 711, 4, 32, 196, 128, 128, 128, "fp16", "fp32"),
    Stage1Case("F22", 176, 2, 64, 24, 128, 256, 64, "bf16", "fp32"),
)


SMOKE_CASE_NAMES = {"F1", "F3", "F19", "F20", "L9"}


def rand_symmetric(shape: tuple[int, ...], device: str, dtype: torch.dtype, input_scale: float):
    return (torch.rand(shape, device=device, dtype=dtype) * 2.0 - 1.0) * input_scale


def make_inputs(case: Stage1Case, device: str, seed: int, input_scale: float):
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


def build_tasks(case: Stage1Case, cu_seqlens: list[int] | None, chunk_indices: list[int] | None) -> list[TaskRange]:
    tasks: list[TaskRange] = []
    if cu_seqlens is None:
        chunk_num_per_b = (case.T + case.chunk_size - 1) // case.chunk_size
        for batch in range(case.B):
            for chunk_idx in range(chunk_num_per_b):
                begin = chunk_idx * case.chunk_size
                end = min(begin + case.chunk_size, case.T)
                tasks.append(TaskRange(batch, begin, end))
        return tasks

    assert chunk_indices is not None
    for idx in range(0, len(chunk_indices), 2):
        seq_idx = chunk_indices[idx]
        chunk_idx = chunk_indices[idx + 1]
        seq_begin = cu_seqlens[seq_idx]
        seq_end = cu_seqlens[seq_idx + 1]
        begin = seq_begin + chunk_idx * case.chunk_size
        end = min(begin + case.chunk_size, seq_end)
        tasks.append(TaskRange(0, begin, end))
    return tasks


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


def compare_with_ct(name: str, actual: torch.Tensor, expected: torch.Tensor, failures: list[str], verbose_ct: bool):
    try:
        if verbose_ct:
            result = ct.single(actual.detach().cpu(), expected.detach().cpu(), dtype=ct_dtype(actual.dtype))
        else:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = ct.single(actual.detach().cpu(), expected.detach().cpu(), dtype=ct_dtype(actual.dtype))
    except Exception as exc:
        failures.append(f"{name}: ct.single failed: {exc}")
    else:
        if not ct_success(result):
            failures.append(f"{name}: ct.single failed ({ct_failure_summary(result)})")


def run_case(case: Stage1Case, args) -> list[str]:
    k, v, beta, A, dw, du, g = make_inputs(case, args.device, args.seed, args.input_scale)
    cu_seqlens = None
    chunk_indices = None
    if case.cu_seqlens_len is not None:
        cu_seqlens = prepare_cu_seqlens(case.T, case.cu_seqlens_len)
        chunk_indices = prepare_chunk_indices(cu_seqlens, case.chunk_size)
    tasks = build_tasks(case, cu_seqlens, chunk_indices)
    group_size = case.VH // case.KH

    with torch.no_grad():
        case_start = time.perf_counter()
        kernel_start = time.perf_counter()
        outputs = fla_ascendc.prepare_wy_repr_bwd_stage1_debug(
            k, v, beta, A, dw, du, g, case.chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )
        torch.npu.synchronize()
        release_aclnn_keepalive()
        kernel_seconds = time.perf_counter() - kernel_start
        print(f"  phase_time kernel_sync={kernel_seconds:.6f}s", flush=True)
        if args.kernel_only:
            del k, v, beta, A, dw, du, g, outputs
            gc.collect()
            torch.npu.empty_cache()
            return []

        compare_start = time.perf_counter()
        debug_da4 = outputs[7]
        failures: list[str] = []
        checked_lines = 0

        for task_idx, task in enumerate(tasks):
            cur = task.end - task.begin
            expect_da4_all = torch.empty((case.VH, cur, cur), device=args.device, dtype=k.dtype)
            mask = torch.tril(torch.ones((cur, cur), device=args.device, dtype=torch.bool), diagonal=-1)
            for hv in range(case.VH):
                hk = hv // group_size
                k_chunk = k[task.batch, hk, task.begin:task.end, :]
                v_chunk = v[task.batch, hv, task.begin:task.end, :]
                beta_chunk = beta[task.batch, hv, task.begin:task.end].float()
                g_chunk = g[task.batch, hv, task.begin:task.end].float()
                dw_chunk = dw[task.batch, hv, task.begin:task.end, :]
                du_chunk = du[task.batch, hv, task.begin:task.end, :]

                kbg = (k_chunk.float() * (beta_chunk * torch.exp(g_chunk))[:, None]).to(k.dtype)
                vb = (v_chunk.float() * beta_chunk[:, None]).to(k.dtype)
                da1 = torch.matmul(dw_chunk, kbg.transpose(0, 1)).to(k.dtype)
                da2 = torch.matmul(du_chunk, vb.transpose(0, 1)).to(k.dtype)
                da4 = torch.where(mask, da1.float() + da2.float(), 0.0)
                expect_da4_all[hv].copy_(da4.to(k.dtype))
            checked_lines += case.VH

            compare_with_ct(
                f"{case.name}.task{task_idx}.DA4",
                debug_da4[task_idx, :, :cur, :cur],
                expect_da4_all,
                failures,
                args.verbose_ct,
            )
            del expect_da4_all, mask
            if args.progress_interval > 0 and (
                (task_idx + 1) % args.progress_interval == 0 or task_idx + 1 == len(tasks)
            ):
                print(f"  compare_progress tasks={task_idx + 1}/{len(tasks)}", flush=True)

        torch.npu.synchronize()
        compare_seconds = time.perf_counter() - compare_start
        total_seconds = time.perf_counter() - case_start
        print(f"  phase_time golden_compare={compare_seconds:.6f}s total={total_seconds:.6f}s", flush=True)

    print(f"  checked_chunks={len(tasks)} checked_lines={checked_lines}")

    del k, v, beta, A, dw, du, g, outputs
    gc.collect()
    torch.npu.empty_cache()
    return failures


def select_cases(args) -> list[Stage1Case]:
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


def run_cases(cases: Iterable[Stage1Case], args) -> int:
    all_failures: list[str] = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"[{idx}] {case.name}: B={case.B}, KH={case.KH}, VH={case.VH}, T={case.T}, "
            f"K={case.K}, V={case.V}, chunk={case.chunk_size}, ktype={case.ktype}, gtype={case.gtype}, "
            f"varlen_L={case.cu_seqlens_len}",
            flush=True,
        )
        failures = run_case(case, args)
        if failures:
            all_failures.extend(failures)
            print(f"  RESULT: FAIL ({len(failures)} outputs)", flush=True)
            if args.stop_on_fail:
                break
        elif args.kernel_only:
            print("  RESULT: KERNEL_ONLY_PASS (precision not checked)", flush=True)
        else:
            print("  RESULT: PASS", flush=True)

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


def main() -> int:
    parser = argparse.ArgumentParser(description="prepare_wy_repr_bwd stage1 DA4 golden precision test")
    parser.add_argument("--suite", choices=("smoke", "fixed", "varlen", "all"), default="smoke")
    parser.add_argument("--cases", default="")
    parser.add_argument("--device", default="npu")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
        help="Half-width of the symmetric random input range. Default 1.0 generates values in [-1, 1].",
    )
    parser.add_argument("--stop-on-fail", action="store_true")
    parser.add_argument("--kernel-only", action="store_true")
    parser.add_argument("--verbose-ct", action="store_true", help="Print every ct.single report.")
    parser.add_argument("--progress-interval", type=int, default=64, help="Task interval for compare progress logs.")
    args = parser.parse_args()
    selected = select_cases(args)
    if not selected:
        raise RuntimeError("No cases selected.")
    return run_cases(selected, args)


if __name__ == "__main__":
    raise SystemExit(main())
