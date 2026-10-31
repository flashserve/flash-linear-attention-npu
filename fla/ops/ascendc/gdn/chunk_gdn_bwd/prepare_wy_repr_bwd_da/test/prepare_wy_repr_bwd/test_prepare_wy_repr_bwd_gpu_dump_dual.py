#!/usr/bin/env python3
"""prepare_wy_repr_bwd NPU vs GPU dump and CPU fp64 dual benchmark.

The GPU dump file is expected to be 008_prepare_wy_repr_bwd.pt and to contain:
  inputs:  k, v, beta, g, A, dw, du
  outputs: dk2, dv, db, dg2

dA is intentionally recomputed by npu_prepare_wy_repr_bwd_da instead of being
loaded from the dump.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import ct
import fla_npu  # noqa: F401
import torch
import torch_npu  # noqa: F401

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

from test_prepare_wy_repr_bwd import (  # noqa: E402
    compute_dbeta_golden_high_precision,
    compute_dg_golden_high_precision,
    compute_dk_golden_high_precision,
    compute_dv_golden_high_precision,
)

torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)

OP_NAME = "prepare_wy_repr_bwd"
DEFAULT_PT_NAME = "008_prepare_wy_repr_bwd.pt"

_BTH_TO_BHT_NAMES = frozenset({
    "q", "k", "v", "w", "u", "g", "beta", "do", "dv", "dv2", "du",
    "dq", "dk", "dw", "dg", "db", "dg2", "dk2", "v_new", "o", "A",
})
_BNTH_TO_BHNT_NAMES = frozenset({"h", "dh"})
_PASSTHROUGH = frozenset({"initial_state", "final_state", "h0", "dh0", "dht"})


def _to_int_list(x: Any) -> list[int] | None:
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return [int(v) for v in x.detach().cpu().reshape(-1).tolist()]
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return None


def _to_npu_tensor(name: str, value: Any, *, beta_fp32: bool = True) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    if name in _PASSTHROUGH:
        out = value
    elif name in _BNTH_TO_BHNT_NAMES and value.ndim == 5:
        out = value.permute(0, 2, 1, 3, 4).contiguous()
    elif name in _BTH_TO_BHT_NAMES and value.ndim >= 3:
        out = value.transpose(1, 2).contiguous()
    else:
        out = value
    out = out.detach().cpu()
    if beta_fp32 and name == "beta" and out.is_floating_point():
        out = out.float()
    return out


def _to_npu_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    if not mapping:
        return {}
    return {k: _to_npu_tensor(k, v) for k, v in mapping.items() if v is not None}


def load_dump_for_npu(pt_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    meta = dict(data.get("meta") or {})
    meta["op"] = str(data.get("op") or "")
    if "inputs_npu" in data:
        inputs = data["inputs_npu"]
        outputs = data.get("outputs_npu") or {}
    else:
        inputs = _to_npu_mapping(data.get("inputs") or {})
        outputs = _to_npu_mapping(data.get("outputs") or {})
    if "chunk_indices_npu" not in meta and "chunk_indices" in meta:
        meta["chunk_indices_npu"] = meta["chunk_indices"]
    return inputs, meta, outputs


def load_case_meta(case_dir: Path) -> dict[str, Any]:
    path = case_dir / "case_meta.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def resolve_seq_meta(
    inputs: dict[str, Any],
    meta: dict[str, Any],
    case_meta: dict[str, Any],
) -> tuple[list[int] | None, list[int] | None, int]:
    cu_seqlens = _to_int_list(meta.get("cu_seqlens"))
    if cu_seqlens is None:
        cu_seqlens = _to_int_list(case_meta.get("cu_seqlens"))
    if cu_seqlens is not None and len(cu_seqlens) == 0:
        cu_seqlens = None

    chunk_indices = _to_int_list(meta.get("chunk_indices_npu"))
    if chunk_indices is None:
        chunk_indices = _to_int_list(meta.get("chunk_indices"))

    chunk_size = meta.get("chunk_size") or case_meta.get("chunk_size")
    if chunk_size is None:
        chunk_size = int(inputs["A"].shape[-1])
    return cu_seqlens, chunk_indices, int(chunk_size)


def find_prepare_wy_repr_bwd_pt(case_dir: Path) -> Path:
    exact = case_dir / DEFAULT_PT_NAME
    if exact.is_file():
        return exact
    candidates = sorted(case_dir.glob(f"*_{OP_NAME}.pt"))
    if not candidates:
        raise FileNotFoundError(f"no {DEFAULT_PT_NAME} or *_{OP_NAME}.pt under {case_dir}")
    return candidates[-1]


def list_case_dirs(dump_root: Path) -> list[Path]:
    if not dump_root.is_dir():
        raise FileNotFoundError(f"dump root not found: {dump_root}")
    if (dump_root / DEFAULT_PT_NAME).is_file():
        return [dump_root]
    dirs = []
    for path in sorted(dump_root.iterdir()):
        if not path.is_dir():
            continue
        if (path / DEFAULT_PT_NAME).is_file() or (path / "case_meta.json").is_file():
            dirs.append(path)
    return dirs


def dual_then_viz(
    tensor_name: str,
    npu_out: torch.Tensor,
    cpu_fp64: torch.Tensor,
    gpu_out: torch.Tensor,
    *,
    enable_viz: bool,
    viz_dir: Path | None,
    sample_count: int,
) -> None:
    print(f"  [{tensor_name}] ct.dual(npu, cpu_fp64, gpu)", flush=True)
    ct.dual(npu_out.cpu(), cpu_fp64, gpu_out, level="L1")
    if not enable_viz or viz_dir is None:
        return
    viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [{tensor_name}] ct.viz(npu, cpu_fp64)", flush=True)
    ct.viz(npu_out.cpu(), cpu_fp64, name=tensor_name, out_dir=str(viz_dir), sample_count=sample_count)


def _require_keys(mapping: dict[str, Any], keys: tuple[str, ...], *, label: str, pt_path: Path) -> None:
    missing = [k for k in keys if k not in mapping]
    if missing:
        raise KeyError(f"{pt_path}: missing {label} keys {missing}, available={sorted(mapping)}")


def run_one_pt(
    pt_path: Path,
    *,
    case_meta: dict[str, Any] | None = None,
    label: str | None = None,
    enable_viz: bool = False,
    sample_count: int = 200_000,
    viz_dir: Path | None = None,
) -> dict[str, Any]:
    pt_path = pt_path.resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f"dump .pt not found: {pt_path}")
    if case_meta is None:
        case_meta = load_case_meta(pt_path.parent)

    inputs, meta, gpu_outputs = load_dump_for_npu(pt_path)
    if meta.get("op") and meta["op"] != OP_NAME:
        raise ValueError(f"{pt_path}: expected op={OP_NAME!r}, got {meta['op']!r}")

    _require_keys(inputs, ("k", "v", "beta", "g", "A", "dw", "du"), label="input", pt_path=pt_path)
    _require_keys(gpu_outputs, ("dk2", "dv", "db", "dg2"), label="output", pt_path=pt_path)

    k = inputs["k"]
    v = inputs["v"]
    beta = inputs["beta"]
    g = inputs["g"]
    A = inputs["A"]
    dw = inputs["dw"]
    du = inputs["du"]
    gpu_dk = gpu_outputs["dk2"]
    gpu_dv = gpu_outputs["dv"]
    gpu_dbeta = gpu_outputs["db"]
    gpu_dg = gpu_outputs["dg2"]

    B, KH, T, K = k.shape
    VH = v.shape[1]
    V = v.shape[-1]
    cu_seqlens, chunk_indices, chunk_size = resolve_seq_meta(inputs, meta, case_meta)
    NT = len(chunk_indices) // 2 if chunk_indices is not None else (T + chunk_size - 1) // chunk_size

    case_name = label or pt_path.name
    print(
        f"\n=== {case_name} ===\n"
        f"  pt: {pt_path} phase={meta.get('phase')} "
        f"B={B} KH={KH} VH={VH} T={T} K={K} V={V} cs={chunk_size} "
        f"varlen={cu_seqlens is not None} NT={NT}",
        flush=True,
    )

    t0 = time.time()
    k_npu = k.npu()
    v_npu = v.npu()
    beta_npu = beta.npu()
    g_npu = g.npu()
    A_npu = A.npu()
    dw_npu = dw.npu()
    du_npu = du.npu()
    dA = torch.ops.npu.npu_prepare_wy_repr_bwd_da(
        k_npu,
        v_npu,
        beta_npu,
        A_npu,
        dw_npu,
        du_npu,
        g_npu,
        chunk_size=chunk_size,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    dk, dv, dbeta, dg = torch.ops.npu.npu_prepare_wy_repr_bwd_full(
        k_npu,
        v_npu,
        beta_npu,
        A_npu,
        dA,
        dw_npu,
        du_npu,
        g_npu,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_size=chunk_size,
    )
    torch.npu.synchronize()
    npu_elapsed = time.time() - t0

    dA_cpu = dA.cpu()
    cpu_dv = compute_dv_golden_high_precision(A, du, beta, cu_seqlens, chunk_indices, B, VH, T, V, chunk_size, NT)
    cpu_dk = compute_dk_golden_high_precision(A, dw, g, beta, dA_cpu, k, cu_seqlens, chunk_indices, B, KH, VH, T, K, chunk_size, NT)
    cpu_dg = compute_dg_golden_high_precision(A, dw, g, beta, dA_cpu, k, cu_seqlens, chunk_indices, B, KH, VH, T, K, chunk_size, NT)
    cpu_dbeta = compute_dbeta_golden_high_precision(A, dw, g, beta, dA_cpu, k, v, du, cu_seqlens, chunk_indices, B, KH, VH, T, K, chunk_size, NT)

    tensor_viz_dir = None
    if enable_viz:
        base_viz_dir = viz_dir or pt_path.parent / "viz"
        tensor_viz_dir = base_viz_dir / case_name.replace("/", "_")

    dual_then_viz("dk", dk, cpu_dk, gpu_dk, enable_viz=enable_viz, viz_dir=tensor_viz_dir, sample_count=sample_count)
    dual_then_viz("dv", dv, cpu_dv, gpu_dv, enable_viz=enable_viz, viz_dir=tensor_viz_dir, sample_count=sample_count)
    dual_then_viz("dbeta", dbeta, cpu_dbeta, gpu_dbeta, enable_viz=enable_viz, viz_dir=tensor_viz_dir, sample_count=sample_count)
    dual_then_viz("dg", dg, cpu_dg, gpu_dg, enable_viz=enable_viz, viz_dir=tensor_viz_dir, sample_count=sample_count)

    return {
        "case": case_name,
        "status": "pass",
        "pt": str(pt_path),
        "phase": meta.get("phase"),
        "npu_elapsed_s": round(npu_elapsed, 4),
        "shapes": {
            "B": B,
            "KH": KH,
            "VH": VH,
            "T": T,
            "K": K,
            "V": V,
            "chunk_size": chunk_size,
        },
    }


def run_one_case(
    case_dir: Path,
    *,
    enable_viz: bool = True,
    sample_count: int = 200_000,
    viz_dir: Path | None = None,
) -> dict[str, Any]:
    case_dir = case_dir.resolve()
    pt_path = find_prepare_wy_repr_bwd_pt(case_dir)
    return run_one_pt(
        pt_path,
        case_meta=load_case_meta(case_dir),
        label=case_dir.name,
        enable_viz=enable_viz,
        sample_count=sample_count,
        viz_dir=viz_dir or case_dir / "viz",
    )


def collect_pt_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.pt is not None:
        paths.append(args.pt)
    if args.pts.strip():
        paths.extend(Path(p.strip()) for p in args.pts.split(",") if p.strip())
    return paths


def select_cases(dump_root: Path, args: argparse.Namespace) -> list[Path]:
    all_dirs = list_case_dirs(dump_root)
    if args.case:
        case_dir = dump_root / args.case
        if not case_dir.is_dir():
            raise FileNotFoundError(f"case dir not found: {case_dir}")
        return [case_dir]
    if args.cases.strip():
        names = [name.strip() for name in args.cases.split(",") if name.strip()]
        by_name = {path.name: path for path in all_dirs}
        missing = [name for name in names if name not in by_name]
        if missing:
            raise ValueError(f"unknown case(s): {', '.join(missing)}")
        return [by_name[name] for name in names]

    phase = args.phase.strip().lower()
    if phase in ("", "all"):
        return all_dirs
    if phase.startswith("prefix:"):
        prefix = phase.split(":", 1)[1]
        return [path for path in all_dirs if path.name.startswith(prefix)]
    raise ValueError(f"unknown --phase {args.phase!r}; use all or prefix:<name_prefix>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="prepare_wy_repr_bwd NPU vs GPU dump and CPU fp64 dual benchmark")
    parser.add_argument("--dump-root", type=Path, default=None, help="GPU dump root for batch mode")
    parser.add_argument("--pt", type=Path, default=None, help=f"single dump .pt file, usually {DEFAULT_PT_NAME}")
    parser.add_argument("--pts", default="", help="comma-separated dump .pt files")
    parser.add_argument("--case", default="", help="single case directory name under dump-root")
    parser.add_argument("--cases", default="", help="comma-separated case names")
    parser.add_argument("--phase", default="all", help="filter case dirs: all | prefix:<name_prefix>")
    parser.add_argument("--report", type=Path, default=None, help="write JSON report path")
    parser.add_argument("--device", type=int, default=int(os.environ.get("TEST_DEVICE_ID", 0)), help="NPU device id")
    parser.add_argument("--no-viz", action="store_true", help="skip ct.viz after ct.dual")
    parser.add_argument("-sc", "--sample-count", type=int, default=200_000, help="ct.viz sample count")
    parser.add_argument("--viz-dir", type=Path, default=None, help="ct.viz output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.npu.set_device(args.device)
    pt_paths = collect_pt_paths(args)
    enable_viz = not args.no_viz
    results: list[dict[str, Any]] = []
    failed = 0

    if pt_paths:
        for pt_path in pt_paths:
            try:
                results.append(run_one_pt(
                    pt_path,
                    label=pt_path.name,
                    enable_viz=enable_viz,
                    sample_count=args.sample_count,
                    viz_dir=args.viz_dir,
                ))
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"\n=== {pt_path} FAILED ===\n{exc}", flush=True)
                traceback.print_exc()
                results.append({"case": pt_path.name, "status": "fail", "pt": str(pt_path), "error": str(exc)})
        report_dir = pt_paths[0].resolve().parent
        mode = "pt"
    else:
        dump_root = args.dump_root or Path("./GPU_DUMP")
        selected = select_cases(dump_root, args)
        if not selected:
            print("No cases selected.", file=sys.stderr)
            return 1
        for case_dir in selected:
            try:
                results.append(run_one_case(
                    case_dir,
                    enable_viz=enable_viz,
                    sample_count=args.sample_count,
                    viz_dir=args.viz_dir,
                ))
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"\n=== {case_dir.name} FAILED ===\n{exc}", flush=True)
                traceback.print_exc()
                results.append({"case": case_dir.name, "status": "fail", "error": str(exc)})
        report_dir = dump_root
        mode = "case_dir"

    report_path = args.report or (report_dir / "prepare_wy_repr_bwd_gpu_dump_dual_report.json")
    report = {
        "op": OP_NAME,
        "mode": mode,
        "total": len(results),
        "passed": sum(1 for item in results if item.get("status") == "pass"),
        "failed": sum(1 for item in results if item.get("status") == "fail"),
        "results": results,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nDone: {report['passed']} passed, {report['failed']} failed / {report['total']} total", flush=True)
    print(f"report -> {report_path}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
