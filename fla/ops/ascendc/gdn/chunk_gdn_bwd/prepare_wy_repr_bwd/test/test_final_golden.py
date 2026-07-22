import argparse
import contextlib
import gc
import importlib
import io
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ct
import torch
import torch_npu

from fla_npu.ops import ascendc as fla_ascendc
from case_utils import (
    FULL_GDN_CASES,
    GdnCase,
    compare_with_ct,
    ct_failure_summary,
    ct_success,
    make_inputs,
    prepare_chunk_indices,
    prepare_cu_seqlens,
    release_aclnn_keepalive,
)


@dataclass(frozen=True)
class FinalCase:
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


FINAL_SMOKE_CASES: tuple[FinalCase, ...] = (
    FinalCase("S1_V128_C64", 2, 2, 2, 192, 128, 128, 64, "fp16", "fp16"),
    FinalCase("S2_V128_C128", 1, 4, 4, 256, 128, 128, 128, "bf16", "fp32"),
    FinalCase("S3_GVA12_V256_C64", 1, 2, 4, 192, 128, 256, 64, "fp16", "fp32"),
    FinalCase("S4_GVA13_V256_C64", 1, 2, 6, 192, 128, 256, 64, "bf16", "fp32"),
    FinalCase("S5_VARLEN_TAIL_V256_C128", 1, 2, 6, 320, 128, 256, 128, "bf16", "fp32", 7),
)


def import_golden_helpers():
    current = Path(__file__).resolve()
    for parent in current.parents:
        helper_dir = parent / "torch_custom" / "fla_npu" / "test"
        full_helper_file = helper_dir / "test_npu_prepare_wy_repr_bwd_full.py"
        da_helper_file = helper_dir / "test_npu_prepare_wy_repr_bwd_da.py"
        if full_helper_file.exists() and da_helper_file.exists():
            helper_dir_str = str(helper_dir)
            if helper_dir_str not in sys.path:
                sys.path.insert(0, helper_dir_str)
            return (
                importlib.import_module("test_npu_prepare_wy_repr_bwd_da"),
                importlib.import_module("test_npu_prepare_wy_repr_bwd_full"),
            )
    raise RuntimeError("Cannot find torch_custom/fla_npu/test prepare_wy_repr_bwd da/full helpers")


def convert_case(case: GdnCase | FinalCase) -> FinalCase:
    if isinstance(case, FinalCase):
        return case
    return FinalCase(
        case.name,
        case.B,
        case.KH,
        case.VH,
        case.T,
        case.K,
        case.V,
        case.chunk_size,
        case.ktype,
        case.gtype,
        case.cu_seqlens_len,
    )


def make_case_inputs(case: FinalCase, args):
    gdn_case = GdnCase(
        case.name,
        case.B,
        case.KH,
        case.VH,
        case.T,
        case.K,
        case.V,
        case.chunk_size,
        case.ktype,
        case.gtype,
        case.cu_seqlens_len,
    )
    return make_inputs(gdn_case, args.device, args.seed, args.input_scale)


def to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu()


def chunk_count(case: FinalCase, chunk_indices: list[int] | None) -> int:
    if chunk_indices is not None:
        return len(chunk_indices) // 2
    return (case.T + case.chunk_size - 1) // case.chunk_size


def expand_k_to_v_heads(k: torch.Tensor, case: FinalCase) -> torch.Tensor:
    if case.VH % case.KH != 0:
        raise ValueError(f"VH ({case.VH}) must be divisible by KH ({case.KH}).")
    if case.VH == case.KH:
        return k
    heads_per_kv = case.VH // case.KH
    return k[:, :, None, :, :].expand(case.B, case.KH, heads_per_kv, case.T, case.K).reshape(
        case.B, case.VH, case.T, case.K
    )


def build_cpu_small_op_chain(
    case: FinalCase,
    inputs: tuple[torch.Tensor, ...],
    cu_seqlens: list[int] | None,
    chunk_indices: list[int] | None,
    high_precision: bool,
) -> tuple[torch.Tensor, ...]:
    da_helpers, full_helpers = import_golden_helpers()
    k, v, beta, A, dw, du, g = inputs
    NT = chunk_count(case, chunk_indices)
    k_for_da = expand_k_to_v_heads(k, case)

    if high_precision:
        dA = da_helpers.compute_dA_cpu_high_precision(
            A,
            dw,
            g,
            beta,
            k_for_da,
            v,
            du,
            chunk_indices,
            cu_seqlens,
            case.B,
            case.VH,
            case.T,
            case.K,
            case.chunk_size,
            NT,
        )
        return (
            full_helpers.compute_dk_golden_high_precision(
                A,
                dw,
                g,
                beta,
                dA,
                k,
                cu_seqlens,
                chunk_indices,
                case.B,
                case.KH,
                case.VH,
                case.T,
                case.K,
                case.chunk_size,
                NT,
            ),
            full_helpers.compute_dv_golden_high_precision(
                A,
                du,
                beta,
                cu_seqlens,
                chunk_indices,
                case.B,
                case.VH,
                case.T,
                case.V,
                case.chunk_size,
                NT,
            ),
            full_helpers.compute_dbeta_golden_high_precision(
                A,
                dw,
                g,
                beta,
                dA,
                k,
                v,
                du,
                cu_seqlens,
                chunk_indices,
                case.B,
                case.KH,
                case.VH,
                case.T,
                case.K,
                case.chunk_size,
                NT,
            ),
            full_helpers.compute_dg_golden_high_precision(
                A,
                dw,
                g,
                beta,
                dA,
                k,
                cu_seqlens,
                chunk_indices,
                case.B,
                case.KH,
                case.VH,
                case.T,
                case.K,
                case.chunk_size,
                NT,
            ),
        )

    dA = da_helpers.compute_dA_cpu(
        A,
        dw,
        g,
        beta,
        k_for_da,
        v,
        du,
        chunk_indices,
        cu_seqlens,
        case.B,
        case.VH,
        case.T,
        case.K,
        case.chunk_size,
        NT,
    )
    return (
        full_helpers.compute_dk_golden(
            A,
            dw,
            g,
            beta,
            dA,
            k,
            cu_seqlens,
            chunk_indices,
            case.B,
            case.KH,
            case.VH,
            case.T,
            case.K,
            case.chunk_size,
            NT,
        ),
        full_helpers.compute_dv_golden(
            A,
            du,
            beta,
            cu_seqlens,
            chunk_indices,
            case.B,
            case.VH,
            case.T,
            case.V,
            case.chunk_size,
            NT,
        ),
        full_helpers.compute_dbeta_golden(
            A,
            dw,
            g,
            beta,
            dA,
            k,
            v,
            du,
            cu_seqlens,
            chunk_indices,
            case.B,
            case.KH,
            case.VH,
            case.T,
            case.K,
            case.chunk_size,
            NT,
        ),
        full_helpers.compute_dg_golden(
            A,
            dw,
            g,
            beta,
            dA,
            k,
            cu_seqlens,
            chunk_indices,
            case.B,
            case.KH,
            case.VH,
            case.T,
            case.K,
            case.chunk_size,
            NT,
        ),
    )


def compare_with_dual_small_op(
    case: FinalCase,
    actual: tuple[torch.Tensor, ...],
    inputs: tuple[torch.Tensor, ...],
    cu_seqlens: list[int] | None,
    chunk_indices: list[int] | None,
    failures: list[str],
    verbose_ct: bool,
):
    cpu_inputs = tuple(to_cpu(tensor) for tensor in inputs)
    bench = build_cpu_small_op_chain(case, cpu_inputs, cu_seqlens, chunk_indices, high_precision=False)
    high_precision = build_cpu_small_op_chain(case, cpu_inputs, cu_seqlens, chunk_indices, high_precision=True)

    for name, actual_tensor, golden_tensor, bench_tensor in zip(
        ("dk", "dv", "dbeta", "dg"), actual, high_precision, bench, strict=True
    ):
        try:
            if verbose_ct:
                result = ct.dual(to_cpu(actual_tensor), golden_tensor, to_cpu(bench_tensor))
            else:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    result = ct.dual(to_cpu(actual_tensor), golden_tensor, to_cpu(bench_tensor))
        except Exception as exc:
            failures.append(f"{case.name}.{name}: ct.dual failed: {exc}")
            print(f"  ct.dual_small_op {case.name}.{name}: ERROR {exc}", flush=True)
        else:
            if ct_success(result):
                print(f"  ct.dual_small_op {case.name}.{name}: PASS", flush=True)
            else:
                failures.append(f"{case.name}.{name}: ct.dual failed ({ct_failure_summary(result)})")
                print(f"  ct.dual_small_op {case.name}.{name}: FAIL ({ct_failure_summary(result)})", flush=True)


def run_case(case: FinalCase, args) -> list[str]:
    k, v, beta, A, dw, du, g = make_case_inputs(case, args)
    cu_seqlens = None
    chunk_indices = None
    if case.cu_seqlens_len is not None:
        cu_seqlens = prepare_cu_seqlens(case.T, case.cu_seqlens_len)
        chunk_indices = prepare_chunk_indices(cu_seqlens, case.chunk_size)

    with torch.no_grad():
        case_start = time.perf_counter()
        kernel_start = time.perf_counter()
        actual = fla_ascendc.prepare_wy_repr_bwd(
            k, v, beta, A, dw, du, g, case.chunk_size, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )
        torch.npu.synchronize()
        release_aclnn_keepalive()
        kernel_seconds = time.perf_counter() - kernel_start
        print(f"  phase_time kernel_sync={kernel_seconds:.6f}s", flush=True)
        if args.kernel_only:
            del k, v, beta, A, dw, du, g, actual
            gc.collect()
            torch.npu.empty_cache()
            return []

        failures: list[str] = []
        if args.dual_small_op:
            compare_start = time.perf_counter()
            compare_with_dual_small_op(
                case,
                actual,
                (k, v, beta, A, dw, du, g),
                cu_seqlens,
                chunk_indices,
                failures,
                args.verbose_ct,
            )
            compare_seconds = time.perf_counter() - compare_start
            total_seconds = time.perf_counter() - case_start
            print(f"  phase_time golden_cpu_dual_compare={compare_seconds:.6f}s total={total_seconds:.6f}s", flush=True)
        else:
            golden_start = time.perf_counter()
            dA = fla_ascendc.prepare_wy_repr_bwd_da(
                k,
                v,
                beta,
                A,
                dw,
                du,
                g,
                chunk_size=case.chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
            expected = fla_ascendc.prepare_wy_repr_bwd_full(
                k,
                v,
                beta,
                A,
                dA,
                dw,
                du,
                g,
                chunk_size=case.chunk_size,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
            )
            torch.npu.synchronize()
            release_aclnn_keepalive()
            golden_seconds = time.perf_counter() - golden_start
            print(f"  phase_time golden_npu={golden_seconds:.6f}s", flush=True)

            compare_start = time.perf_counter()
            for name, actual_tensor, expected_tensor in zip(("dk", "dv", "dbeta", "dg"), actual, expected, strict=True):
                compare_with_ct(f"{case.name}.{name}", actual_tensor, expected_tensor, failures, args.verbose_ct)
            compare_seconds = time.perf_counter() - compare_start
            total_seconds = time.perf_counter() - case_start
            print(f"  phase_time golden_compare={compare_seconds:.6f}s total={total_seconds:.6f}s", flush=True)

    del k, v, beta, A, dw, du, g, actual
    gc.collect()
    torch.npu.empty_cache()
    return failures


def select_cases(args) -> list[FinalCase]:
    if args.cases:
        wanted = {case.strip() for case in args.cases.split(",") if case.strip()}
        all_cases = list(FINAL_SMOKE_CASES) + [convert_case(case) for case in FULL_GDN_CASES]
        return [case for case in all_cases if case.name in wanted]
    if args.suite == "smoke":
        return list(FINAL_SMOKE_CASES)
    if args.suite == "fixed":
        return [case for case in FINAL_SMOKE_CASES if case.cu_seqlens_len is None]
    if args.suite == "varlen":
        return [case for case in FINAL_SMOKE_CASES if case.cu_seqlens_len is not None]
    return [convert_case(case) for case in FULL_GDN_CASES]


def run_cases(cases: Iterable[FinalCase], args) -> int:
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
    parser = argparse.ArgumentParser(description="prepare_wy_repr_bwd final output golden precision test")
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
    parser.add_argument("--verbose-ct", action="store_true", help="Print every ct report.")
    parser.add_argument(
        "--dual-small-op",
        action="store_true",
        help="Use ct.dual with fused output as test, CPU high precision da+full as gt, and CPU normal da+full as bench.",
    )
    args = parser.parse_args()
    selected = select_cases(args)
    if not selected:
        raise RuntimeError("No cases selected.")
    return run_cases(selected, args)


if __name__ == "__main__":
    raise SystemExit(main())
