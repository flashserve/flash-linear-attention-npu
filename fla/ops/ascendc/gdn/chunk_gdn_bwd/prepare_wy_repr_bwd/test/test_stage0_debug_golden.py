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


SMOKE_CASE_NAMES = {"F1", "F19", "F22", "L2", "L7", "L8"}


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


def build_tasks(case: Stage0Case, cu_seqlens: list[int] | None, chunk_indices: list[int] | None) -> list[TaskRange]:
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


def compare_with_ct(name: str, actual: torch.Tensor, expected: torch.Tensor, failures: list[str]):
    try:
        result = ct.single(actual.detach().cpu(), expected.detach().cpu(), dtype=ct_dtype(actual.dtype))
    except Exception as exc:
        failures.append(f"{name}: ct.single failed: {exc}")
    else:
        if not ct_success(result):
            failures.append(f"{name}: ct.single failed ({ct_failure_summary(result)})")


def run_case(case: Stage0Case, args) -> list[str]:
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
        outputs = fla_ascendc.prepare_wy_repr_bwd_stage0_debug(
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
        debug_kbg, debug_vb, debug_kbeta, debug_dkbg, debug_dvb, debug_kkt = outputs[4:]
        failures: list[str] = []
        checked_lines = 0

        for task_idx, task in enumerate(tasks):
            cur = task.end - task.begin
            expect_kbg_all = torch.empty((case.VH, cur, case.K), device=args.device, dtype=k.dtype)
            expect_vb_all = torch.empty((case.VH, cur, case.V), device=args.device, dtype=k.dtype)
            expect_kbeta_all = torch.empty((case.VH, cur, case.K), device=args.device, dtype=k.dtype)
            expect_dkbg_all = torch.empty((case.VH, cur, case.K), device=args.device, dtype=k.dtype)
            expect_dvb_all = torch.empty((case.VH, cur, case.V), device=args.device, dtype=k.dtype)
            expect_kkt_all = torch.empty((case.KH, cur, cur), device=args.device, dtype=k.dtype)
            for hk in range(case.KH):
                k_chunk = k[task.batch, hk, task.begin:task.end, :]
                expect_kkt_all[hk].copy_(torch.matmul(k_chunk.float(), k_chunk.transpose(0, 1).float()).to(k.dtype))
            for hv in range(case.VH):
                hk = hv // group_size
                k_chunk = k[task.batch, hk, task.begin:task.end, :]
                v_chunk = v[task.batch, hv, task.begin:task.end, :]
                beta_chunk = beta[task.batch, hv, task.begin:task.end].float()
                g_chunk = g[task.batch, hv, task.begin:task.end].float()
                A_chunk = A[task.batch, hv, task.begin:task.end, :cur]
                dw_chunk = dw[task.batch, hv, task.begin:task.end, :]
                du_chunk = du[task.batch, hv, task.begin:task.end, :]

                scale_bg = beta_chunk * torch.exp(g_chunk)
                expect_kbg = (k_chunk.float() * scale_bg[:, None]).to(k.dtype)
                expect_kbeta = (k_chunk.float() * beta_chunk[:, None]).to(k.dtype)
                expect_vb = (v_chunk.float() * beta_chunk[:, None]).to(k.dtype)
                expect_dkbg = torch.matmul(A_chunk.transpose(0, 1).float(), dw_chunk.float()).to(k.dtype)
                expect_dvb = torch.matmul(A_chunk.transpose(0, 1).float(), du_chunk.float()).to(k.dtype)

                expect_kbg_all[hv].copy_(expect_kbg)
                expect_kbeta_all[hv].copy_(expect_kbeta)
                expect_vb_all[hv].copy_(expect_vb)
                expect_dkbg_all[hv].copy_(expect_dkbg)
                expect_dvb_all[hv].copy_(expect_dvb)
            checked_lines += case.VH + case.KH

            compare_with_ct(f"{case.name}.task{task_idx}.Kbg", debug_kbg[task_idx, :, :cur, :], expect_kbg_all, failures)
            compare_with_ct(f"{case.name}.task{task_idx}.Kbeta", debug_kbeta[task_idx, :, :cur, :], expect_kbeta_all, failures)
            compare_with_ct(f"{case.name}.task{task_idx}.Vb", debug_vb[task_idx, :, :cur, :], expect_vb_all, failures)
            compare_with_ct(f"{case.name}.task{task_idx}.Dkbg", debug_dkbg[task_idx, :, :cur, :], expect_dkbg_all, failures)
            compare_with_ct(f"{case.name}.task{task_idx}.Dvb", debug_dvb[task_idx, :, :cur, :], expect_dvb_all, failures)
            compare_with_ct(f"{case.name}.task{task_idx}.Kkt", debug_kkt[task_idx, :, :cur, :cur], expect_kkt_all, failures)
            del expect_kbg_all, expect_kbeta_all, expect_vb_all, expect_dkbg_all, expect_dvb_all, expect_kkt_all

        torch.npu.synchronize()
        compare_seconds = time.perf_counter() - compare_start
        total_seconds = time.perf_counter() - case_start
        print(f"  phase_time golden_compare={compare_seconds:.6f}s total={total_seconds:.6f}s", flush=True)

    print(f"  checked_chunks={len(tasks)} checked_lines={checked_lines}")

    del k, v, beta, A, dw, du, g, outputs
    gc.collect()
    torch.npu.empty_cache()
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
        failures = run_case(case, args)
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


def main() -> int:
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    selected = select_cases(args)
    if not selected:
        raise RuntimeError("No cases selected.")
    return run_cases(selected, args)


if __name__ == "__main__":
    raise SystemExit(main())
