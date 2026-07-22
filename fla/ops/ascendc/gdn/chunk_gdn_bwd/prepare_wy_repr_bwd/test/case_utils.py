import contextlib
import importlib
import io
import random
from dataclasses import dataclass

import ct
import torch
import torch_npu


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass(frozen=True)
class GdnCase:
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


FULL_GDN_CASES: tuple[GdnCase, ...] = (
    GdnCase("F1", 64, 8, 8, 1024, 128, 128, 64, "fp16", "fp16"),
    GdnCase("F2", 32, 16, 16, 2048, 128, 128, 64, "bf16", "bf16"),
    GdnCase("F3", 16, 32, 32, 4096, 128, 128, 128, "fp16", "fp32"),
    GdnCase("F4", 8, 32, 32, 8192, 128, 128, 128, "bf16", "bf16"),
    GdnCase("F5", 128, 4, 4, 1024, 128, 128, 64, "fp16", "fp16"),
    GdnCase("F6", 64, 8, 8, 2048, 128, 128, 64, "bf16", "fp32"),
    GdnCase("F7", 32, 16, 16, 4096, 128, 128, 128, "fp16", "fp16"),
    GdnCase("F8", 16, 32, 32, 8192, 128, 128, 128, "bf16", "bf16"),
    GdnCase("F9", 64, 8, 8, 4096, 128, 128, 128, "fp16", "fp16"),
    GdnCase("F10", 32, 16, 16, 8192, 128, 128, 128, "bf16", "bf16"),
    GdnCase("F11", 16, 32, 32, 16384, 128, 128, 64, "fp16", "fp32"),
    GdnCase("F12", 8, 32, 32, 32768, 128, 128, 64, "bf16", "bf16"),
    GdnCase("F13", 64, 8, 8, 1024, 128, 128, 64, "fp16", "fp16"),
    GdnCase("F14", 32, 16, 16, 2048, 128, 128, 64, "bf16", "bf16"),
    GdnCase("F15", 16, 32, 32, 4096, 128, 128, 128, "fp16", "fp32"),
    GdnCase("F16", 8, 32, 32, 8192, 128, 128, 128, "bf16", "bf16"),
    GdnCase("F17", 64, 8, 8, 2048, 128, 128, 64, "bf16", "bf16"),
    GdnCase("F18", 32, 16, 16, 4096, 128, 128, 128, "fp16", "fp16"),
    GdnCase("L1", 1, 32, 32, 65536, 128, 128, 64, "bf16", "bf16", 64),
    GdnCase("L2", 1, 16, 16, 65536, 128, 128, 64, "fp16", "fp16", 33),
    GdnCase("L3", 1, 8, 8, 131072, 128, 128, 64, "bf16", "bf16", 333),
    GdnCase("L4", 1, 4, 4, 262144, 128, 128, 64, "fp16", "fp32", 567),
    GdnCase("L5", 1, 16, 16, 32768, 128, 128, 64, "bf16", "bf16", 7),
    GdnCase("L6", 1, 8, 8, 65536, 128, 128, 64, "fp16", "fp16", 25),
    GdnCase("L7", 1, 16, 32, 16384, 128, 256, 64, "fp16", "fp32", 128),
    GdnCase("L8", 1, 21, 63, 16384, 128, 256, 64, "bf16", "fp32", 2),
    GdnCase("L9", 1, 8, 32, 65536, 128, 256, 128, "bf16", "fp32", 172),
    GdnCase("L10", 1, 16, 32, 65536, 128, 128, 64, "fp16", "fp32", 668),
    GdnCase("L11", 1, 4, 32, 65536, 128, 128, 128, "bf16", "fp32", 17),
    GdnCase("L12", 1, 2, 64, 262144, 128, 256, 64, "fp16", "fp32", 32),
    GdnCase("F19", 1, 16, 32, 4096, 128, 256, 64, "fp16", "fp32"),
    GdnCase("F20", 16, 21, 63, 2048, 128, 256, 64, "bf16", "fp32"),
    GdnCase("F21", 711, 4, 32, 196, 128, 128, 128, "fp16", "fp32"),
    GdnCase("F22", 176, 2, 64, 24, 128, 256, 64, "bf16", "fp32"),
)


def release_aclnn_keepalive():
    try:
        runtime_mod = importlib.import_module("fla_npu.ops.ascendc._runtime")
        runtime_mod._RECENT_LAUNCH_STORAGE.clear()
    except Exception:
        pass


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


def rand_symmetric(shape: tuple[int, ...], device: str, dtype: torch.dtype, input_scale: float):
    return (torch.rand(shape, device=device, dtype=dtype) * 2.0 - 1.0) * input_scale


def make_inputs(case: GdnCase, device: str, seed: int, input_scale: float):
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
