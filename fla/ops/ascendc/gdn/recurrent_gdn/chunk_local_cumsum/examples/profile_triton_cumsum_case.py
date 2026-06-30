#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401


def find_repo_root(start: Path) -> Path:
    for path in (start, *start.parents):
        if (path / ".git").exists():
            return path
    return Path(__file__).resolve().parents[7]


REPO_ROOT = find_repo_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(REPO_ROOT))

try:
    from fla.ops.triton.triton_core.cumsum import chunk_local_cumsum  # noqa: E402
except ModuleNotFoundError:
    from triton_ops.triton_core.cumsum import chunk_local_cumsum  # noqa: E402


def load_case(case_dir: Path, case_name: str):
    summary_path = case_dir / "summary.json"
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    for case in summary["cases"]:
        if case["name"] == case_name:
            return case
    raise RuntimeError(f"case {case_name} not found in {summary_path}")


def load_float_tensor(path: Path, shape):
    data = np.fromfile(path, dtype=np.float32)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise RuntimeError(f"{path} has {data.size} elements, expected {expected}")
    return torch.from_numpy(data.reshape(shape)).to("npu")


def load_int64_tensor(path: Path, shape):
    data = np.fromfile(path, dtype=np.int64)
    shape = tuple(shape)
    infer_dims = [idx for idx, dim in enumerate(shape) if dim == -1]
    if len(infer_dims) > 1:
        raise RuntimeError(f"{path} shape {shape} has more than one inferred dimension")
    if infer_dims:
        known = int(np.prod([dim for dim in shape if dim != -1]))
        if known == 0 or data.size % known != 0:
            raise RuntimeError(f"{path} has {data.size} elements, cannot infer shape {shape}")
        shape = tuple((data.size // known) if dim == -1 else dim for dim in shape)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise RuntimeError(f"{path} has {data.size} elements, expected {expected}")
    return torch.from_numpy(data.reshape(shape)).to("npu")


def main():
    parser = argparse.ArgumentParser(description="Run one Triton ChunkLocalCumsum case for msprof op.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--case-dir", default=str(Path(__file__).resolve().parent / "testdata"))
    parser.add_argument("--check", action="store_true", help="Compare Triton output with saved golden after profiling run.")
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    case = load_case(case_dir, args.case)
    shape = (case["B"], case["T"], case["H"])
    g = load_float_tensor(case_dir / case["g_file"], shape)

    cu_seqlens = None
    chunk_indices_out = None
    if case["is_varlen"]:
        cu_host = np.fromfile(case_dir / case["cu_file"], dtype=np.int64)
        cu_seqlens = torch.from_numpy(cu_host).to("npu")
        chunk_indices = load_int64_tensor(case_dir / case["chunk_indices_file"], (-1, 2))
        chunk_indices_out = {str(case["block_t"]): chunk_indices}

    scale = None if float(case["scale"]) == 1.0 else float(case["scale"])
    out = chunk_local_cumsum(
        g,
        chunk_size=int(case["chunk_size"]),
        reverse=bool(case["reverse"]),
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices_out=chunk_indices_out,
        head_first=False,
        output_dtype=torch.float32,
    )
    torch.npu.synchronize()

    if args.check:
        golden = np.fromfile(case_dir / case["golden_file"], dtype=np.float32).reshape(shape)
        actual = out.detach().cpu().numpy()
        max_abs = float(np.max(np.abs(actual - golden)))
        if not np.allclose(actual, golden, atol=2e-5, rtol=2e-5):
            raise RuntimeError(f"{case['name']} mismatch against Triton golden, max_abs={max_abs}")
        print(f"{case['name']} max_abs={max_abs:.9f}")


if __name__ == "__main__":
    main()
