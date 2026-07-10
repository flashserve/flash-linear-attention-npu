#!/usr/bin/env python3
"""Dual-reference dump test for torch.ops.npu.npu_chunk_scaled_dot_kkt.

The dump format follows the GDN local dump convention:
  storage layout : [B, T, H, *]
  NPU layout     : [B, H, T, *]

This test compares the AscendC output against:
  1. the dumped GPU/framework output with the same dtype, and
  2. an independent CPU fp64 implementation with fp64 accumulation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch
import ct


DEFAULT_DUMP_ROOT = Path("/home/m00913889/codex04/gva_fix_1")
DEFAULT_ATOL = 5e-3
DEFAULT_RTOL = 5e-3
ZERO_TOL = 1e-6

_BTH_TO_BHT_NAMES = frozenset({"k", "g", "beta", "A"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify chunk_scaled_dot_kkt dump cases on NPU.")
    parser.add_argument(
        "--dump-root",
        default=os.environ.get("CHUNK_SCALED_DOT_KKT_DUMP_ROOT", str(DEFAULT_DUMP_ROOT)),
        help="Dump directory or a single *_chunk_scaled_dot_kkt_fwd.pt file.",
    )
    parser.add_argument(
        "--device",
        default=os.environ.get("CHUNK_SCALED_DOT_KKT_DEVICE", "npu:0"),
        help="NPU device, for example npu:0.",
    )
    parser.add_argument("--atol", type=float, default=float(os.environ.get("CHUNK_SCALED_DOT_KKT_ATOL", DEFAULT_ATOL)))
    parser.add_argument("--rtol", type=float, default=float(os.environ.get("CHUNK_SCALED_DOT_KKT_RTOL", DEFAULT_RTOL)))
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N cases.",
    )
    return parser.parse_args()


def bth_to_bht(t: torch.Tensor) -> torch.Tensor:
    if t.ndim < 3:
        return t.detach().cpu()
    return t.transpose(1, 2).contiguous().detach().cpu()


def to_npu_tensor(name: str, value: Any) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    out = bth_to_bht(value) if name in _BTH_TO_BHT_NAMES else value.detach().cpu()
    if name == "beta" and out.is_floating_point():
        out = out.float()
    return out


def to_npu_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not mapping:
        return {}
    return {name: to_npu_tensor(name, value) for name, value in mapping.items() if value is not None}


def load_dump_for_npu(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dump = torch.load(path, map_location="cpu", weights_only=False)
    meta = dict(dump.get("meta") or {})
    if "inputs_npu" in dump:
        inputs = dump["inputs_npu"]
        outputs = dump.get("outputs_npu") or {}
    else:
        inputs = to_npu_mapping(dump.get("inputs") or {})
        outputs = to_npu_mapping(dump.get("outputs") or {})
    return inputs, meta, outputs


def inspect_case(path: Path) -> dict[str, Any]:
    meta_path = path.parent / "case_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        return {
            "dtype": meta.get("dtype"),
            "gtype": meta.get("gtype"),
            "varlen": meta.get("varlen"),
            "B": meta.get("B"),
            "T": meta.get("T"),
            "Hk": meta.get("Hk"),
            "Hv": meta.get("Hv"),
            "K": meta.get("K"),
            "chunk_size": meta.get("chunk_size"),
        }

    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            dump = torch.load(path, map_location="cpu", weights_only=False)
        k = dump.get("inputs", {}).get("k")
        g = dump.get("inputs", {}).get("g")
        meta = dict(dump.get("meta") or {})
        dtype = str(k.dtype).replace("torch.", "") if isinstance(k, torch.Tensor) else None
        return {
            "dtype": {"float16": "fp16", "bfloat16": "bf16"}.get(dtype, dtype),
            "gtype": str(g.dtype).replace("torch.", "") if isinstance(g, torch.Tensor) else None,
            "varlen": None,
            "B": k.shape[0] if isinstance(k, torch.Tensor) else None,
            "T": k.shape[1] if isinstance(k, torch.Tensor) and k.ndim >= 2 else None,
            "Hk": k.shape[2] if isinstance(k, torch.Tensor) and k.ndim >= 3 else None,
            "Hv": g.shape[2] if isinstance(g, torch.Tensor) and g.ndim >= 3 else None,
            "K": k.shape[3] if isinstance(k, torch.Tensor) and k.ndim >= 4 else None,
            "chunk_size": meta.get("chunk_size"),
        }
    except Exception:
        return {}


def find_cases(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(root.rglob("*_chunk_scaled_dot_kkt_fwd.pt"))


def select_current_op_heads(
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    gpu_ref: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, str]:
    """Match dump heads to the current AscendC API.

    Current npu_chunk_scaled_dot_kkt accepts k/g/beta with the same head count
    and returns [B,H,T,BT]. The gva_fix_1 dump stores g/beta with 32 value heads
    but k/A with 16 key heads; the dumped A matches the prefix 16 gate/beta
    heads, so this helper selects that prefix.
    """

    h = k.shape[1]
    if gpu_ref.shape[1] != h:
        raise ValueError(
            "Current AscendC op returns [B,Hk,T,BT], but dump output has a different head count: "
            f"k={tuple(k.shape)} A={tuple(gpu_ref.shape)}"
        )
    if g.shape[:1] != k.shape[:1] or g.shape[2] != k.shape[2]:
        raise ValueError(f"g prefix dimensions must match k [B,*,T], got k={tuple(k.shape)} g={tuple(g.shape)}")
    if beta.shape[:1] != k.shape[:1] or beta.shape[2] != k.shape[2]:
        raise ValueError(f"beta prefix dimensions must match k [B,*,T], got k={tuple(k.shape)} beta={tuple(beta.shape)}")
    if g.shape[1] < h or beta.shape[1] < h:
        raise ValueError(f"g/beta head count must be >= k head count, got k={h} g={g.shape[1]} beta={beta.shape[1]}")
    if g.shape[1] == h and beta.shape[1] == h:
        return g.contiguous(), beta.contiguous(), "exact"
    return g[:, :h, :].contiguous(), beta[:, :h, :].contiguous(), f"prefix:{h}"


def cpu_fp64_reference(k: torch.Tensor, g: torch.Tensor, beta: torch.Tensor, chunk_size: int) -> torch.Tensor:
    k_f = k.to(torch.float64).cpu()
    g_f = g.to(torch.float64).cpu()
    beta_f = beta.to(torch.float64).cpu()
    bsz, heads, seq_len, _ = k_f.shape
    out = torch.zeros((bsz, heads, seq_len, chunk_size), dtype=torch.float64)

    mask_cache: dict[int, torch.Tensor] = {}
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        valid = end - start
        if valid not in mask_cache:
            mask_cache[valid] = torch.tril(torch.ones((valid, valid), dtype=torch.bool), diagonal=-1)
        mask = mask_cache[valid]

        k_block = k_f[:, :, start:end, :]
        scores = torch.matmul(k_block, k_block.transpose(-1, -2))
        gate = torch.exp(torch.clamp(g_f[:, :, start:end, None] - g_f[:, :, None, start:end], -50.0, 50.0))
        scaled = scores * gate * beta_f[:, :, start:end, None]
        out[:, :, start:end, :valid] = torch.where(mask, scaled, torch.zeros_like(scaled))

    return out


def summarize_dual_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(result.get("success")),
        "checks": result.get("checks", {}),
        "ratios": result.get("ratios", {}),
    }


def compare_chunk(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict[str, Any]:
    a = actual.float().cpu()
    b = expected.float().cpu()
    diff = (a - b).abs()
    close = diff <= (atol + rtol * b.abs())
    return {
        "ok": bool(close.all().item()),
        "max_err": diff.max().item(),
        "sum_err": diff.sum().item(),
        "count": diff.numel(),
    }


def finalize_compare(parts: list[dict[str, Any]]) -> dict[str, Any]:
    if not parts:
        return {"ok": False, "max_err": float("nan"), "mean_err": float("nan")}
    total = sum(part["count"] for part in parts)
    return {
        "ok": all(part["ok"] for part in parts),
        "max_err": max(part["max_err"] for part in parts),
        "mean_err": sum(part["sum_err"] for part in parts) / total,
    }


def compare_npu_gpu_stream(npu_out: torch.Tensor, gpu_ref: torch.Tensor, chunk_size: int, atol: float, rtol: float) -> dict[str, Any]:
    parts = []
    seq_len = npu_out.shape[2]
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        parts.append(compare_chunk(npu_out[:, :, start:end, :], gpu_ref[:, :, start:end, :], atol, rtol))
    return finalize_compare(parts)


def compare_npu_cpu_stream(
    npu_out: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    k_f = k.float().cpu()
    g_f = g.float().cpu()
    beta_f = beta.float().cpu()
    parts = []
    seq_len = k_f.shape[2]
    mask_cache: dict[int, torch.Tensor] = {}
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        valid = end - start
        if valid not in mask_cache:
            mask_cache[valid] = torch.tril(torch.ones((valid, valid), dtype=torch.bool), diagonal=-1)
        k_block = k_f[:, :, start:end, :]
        scores = torch.matmul(k_block, k_block.transpose(-1, -2))
        gate = torch.exp(torch.clamp(g_f[:, :, start:end, None] - g_f[:, :, None, start:end], -50.0, 50.0))
        ref = torch.zeros((k_f.shape[0], k_f.shape[1], valid, chunk_size), dtype=torch.float32)
        scaled = scores * gate * beta_f[:, :, start:end, None]
        ref[:, :, :, :valid] = torch.where(mask_cache[valid], scaled, torch.zeros_like(scaled))
        parts.append(compare_chunk(npu_out[:, :, start:end, :], ref, atol, rtol))
    return finalize_compare(parts)


def check_zero_regions(out: torch.Tensor, chunk_size: int) -> float:
    _, _, seq_len, _ = out.shape
    max_zero = 0.0
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        valid = end - start
        block = out[:, :, start:end, :].float().cpu()
        upper = torch.triu(block[:, :, :, :valid], diagonal=0)
        max_zero = max(max_zero, upper.abs().max().item())
        if valid < chunk_size:
            max_zero = max(max_zero, block[:, :, :, valid:].abs().max().item())
    return max_zero


def require_npu(device: str) -> torch.device:
    try:
        import torch_npu  # noqa: F401
        import fla_npu  # noqa: F401
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("NPU dump test requires torch_npu and fla_npu to be importable") from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU device is not available")
    if not hasattr(torch.ops.npu, "npu_chunk_scaled_dot_kkt"):
        raise RuntimeError("torch.ops.npu.npu_chunk_scaled_dot_kkt is not registered")

    dev = torch.device(device)
    if dev.type == "npu":
        torch.npu.set_device(dev.index or 0)
    return dev


def verify_case(path: Path, device: torch.device, atol: float, rtol: float) -> dict[str, Any]:
    inputs, meta, outputs = load_dump_for_npu(path)
    if not {"k", "g", "beta"}.issubset(inputs):
        raise KeyError(f"{path} is missing one of k/g/beta inputs")
    if "A" not in outputs:
        raise KeyError(f"{path} is missing dumped A output")

    chunk_size = int(meta["chunk_size"])
    k = inputs["k"].contiguous()
    gpu_out_from_dump = outputs["A"].contiguous()
    g, beta, head_map = select_current_op_heads(k, inputs["g"].float(), inputs["beta"].float(), gpu_out_from_dump)

    npu_out = torch.ops.npu.npu_chunk_scaled_dot_kkt(
        k.to(device),
        g.to(device),
        beta.to(device),
        chunk_size=chunk_size,
    ).cpu()

    cpu_fp64_golden = cpu_fp64_reference(k, g, beta, chunk_size)
    print("       running ct.dual(npu_out, cpu_fp64_golden, gpu_out_from_dump, level=\"L1\")", flush=True)
    dual_result = ct.dual(npu_out, cpu_fp64_golden, gpu_out_from_dump, level="L1")
    gpu_result = compare_npu_gpu_stream(npu_out, gpu_out_from_dump, chunk_size, atol, rtol)
    cpu_result = compare_npu_gpu_stream(npu_out, cpu_fp64_golden, chunk_size, atol, rtol)
    max_zero = check_zero_regions(npu_out, chunk_size)
    shape_ok = tuple(npu_out.shape) == tuple(gpu_out_from_dump.shape) == tuple(cpu_fp64_golden.shape)
    dtype_ok = npu_out.dtype == gpu_out_from_dump.dtype

    return {
        "case": str(path),
        "shape": tuple(npu_out.shape),
        "input_shape": tuple(k.shape),
        "g_shape": tuple(inputs["g"].shape),
        "beta_shape": tuple(inputs["beta"].shape),
        "chunk_size": chunk_size,
        "head_map": head_map,
        "dtype": str(npu_out.dtype),
        "gpu_dump_dtype": str(gpu_out_from_dump.dtype),
        "cpu_golden_dtype": str(cpu_fp64_golden.dtype),
        "shape_ok": shape_ok,
        "dtype_ok": dtype_ok,
        "max_zero": max_zero,
        "gpu": gpu_result,
        "cpu": cpu_result,
        "dual": summarize_dual_result(dual_result),
        "ok": shape_ok and dtype_ok and max_zero <= ZERO_TOL and bool(dual_result.get("success")),
    }


def print_result(index: int, total: int, result: dict[str, Any]) -> None:
    status = "UNSUPPORTED" if "unsupported" in result else ("PASS" if result["ok"] else "FAIL")
    print(
        f"[{status}] {index:03d}/{total:03d} {result['case']} "
        f"k={result['input_shape']} g={result['g_shape']} beta={result['beta_shape']} "
        f"out={result['shape']} dtype={result['dtype']} gpu_dtype={result['gpu_dump_dtype']} "
        f"cpu_dtype={result['cpu_golden_dtype']} BT={result['chunk_size']} head_map={result['head_map']}"
    )
    if "unsupported" in result:
        print(f"       reason={result['unsupported']}")
        return
    print(
        f"       npu_vs_gpu max_err={result['gpu']['max_err']:.9f} "
        f"mean_err={result['gpu']['mean_err']:.9f} allclose={result['gpu']['ok']}"
    )
    print(
        f"       npu_vs_cpu max_err={result['cpu']['max_err']:.9f} "
        f"mean_err={result['cpu']['mean_err']:.9f} allclose={result['cpu']['ok']} "
        f"shape_ok={result['shape_ok']} dtype_ok={result['dtype_ok']} max_zero={result['max_zero']:.3e}"
    )
    print(
        f"       ct.dual L1 success={result['dual']['success']} "
        f"checks={result['dual']['checks']} ratios={result['dual']['ratios']}"
    )


def cleanup_case(device: torch.device) -> None:
    gc.collect()
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.empty_cache()


def main() -> int:
    args = parse_args()
    root = Path(args.dump_root)
    cases = find_cases(root)
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        print(f"[ERROR] no chunk_scaled_dot_kkt dump cases found under {root}")
        return 1

    device = require_npu(args.device)
    print(f"chunk_scaled_dot_kkt dump root={root} cases={len(cases)} device={device} atol={args.atol} rtol={args.rtol}")

    passed = 0
    unsupported = 0
    failures: list[str] = []
    for idx, path in enumerate(cases, start=1):
        try:
            result = verify_case(path, device, args.atol, args.rtol)
            print_result(idx, len(cases), result)
            if "unsupported" in result:
                unsupported += 1
            elif result["ok"]:
                passed += 1
            else:
                failures.append(result["case"])
        except Exception as exc:
            failures.append(str(path))
            print(f"[FAIL] {idx:03d}/{len(cases):03d} {path} error={type(exc).__name__}: {exc}")
        finally:
            cleanup_case(device)

    failed = len(cases) - passed - unsupported
    print("=" * 90)
    print(f"Summary: PASS={passed} FAIL={failed} UNSUPPORTED={unsupported} TOTAL={len(cases)}")
    if failures:
        print("Failed cases:")
        for case in failures:
            print(f"  {case}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
