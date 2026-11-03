#!/usr/bin/env python3
# Adapted from the CATLASS Python quick-start matmul example.
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0, AS IS without warranties.
# See LICENSES/CANN-Open-Software-License-Agreement-Version-2.0.txt in the repository root.
"""CATLASS DSL implementation of the same 16x16 FP16 -> FP32 GEMM (Ascend 950)."""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch_npu  # noqa: F401: registers the NPU backend
from catlass import tla
from catlass.tla.runtime import from_dlpack

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixtures import CASES, read_inputs  # noqa: E402

M = K = N = 16


@tla.kernel
def matmul_kernel(a: tla.Tensor, b: tla.Tensor, c: tla.Tensor) -> None:
    # Each event connects a producer pipe to its consumer pipe.
    input_ready = tla.flag("input_ready", tla.arch.MTE2, tla.arch.MTE1)
    operands_ready = tla.flag("operands_ready", tla.arch.MTE1, tla.arch.CUBE)
    result_ready = tla.flag("result_ready", tla.arch.CUBE, tla.arch.FIX)

    a_l1_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l1, 512)
    b_l1_ptr = tla.allocate(K * N, tla.Float16, tla.AddressSpace.l1, 512)
    a_l0_ptr = tla.allocate(M * K, tla.Float16, tla.AddressSpace.l0a, 512)
    b_l0_ptr = tla.allocate(K * N, tla.Float16, tla.AddressSpace.l0b, 512)
    c_l0_ptr = tla.allocate(M * N, tla.Float32, tla.AddressSpace.l0c, 512)

    with tla.cube():
        # 1. GM -> L1.
        a_l1 = tla.make_tensor_like(a_l1_ptr, a, tla.arch.zN)
        b_l1 = tla.make_tensor_like(b_l1_ptr, b, tla.arch.zN)
        tla.copy(a_l1, a)
        tla.copy(b_l1, b)
        tla.set_flag(input_ready)
        tla.wait_flag(input_ready)

        # 2. L1 -> the Cube input buffers L0A/L0B.
        a_l0 = tla.make_tensor_like(a_l0_ptr, a_l1, tla.arch.zN)
        b_l0 = tla.make_tensor_like(b_l0_ptr, b_l1, tla.arch.nZ)
        c_l0 = tla.make_tensor_like(c_l0_ptr, c, tla.arch.L0Clayout)
        tla.copy(a_l0, a_l1)
        tla.copy(b_l0, b_l1)
        tla.set_flag(operands_ready)
        tla.wait_flag(operands_ready)

        # 3. Actual NPU matrix multiplication, initializing the FP32 accumulator.
        tla.mmad(c_l0, a_l0, b_l0, init_c=True)

        # 4. L0C -> GM.
        tla.set_flag(result_ready)
        tla.wait_flag(result_ready)
        tla.copy(c, c_l0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--artifacts", type=Path, default=Path("build/dsl"))
    args = parser.parse_args()
    torch.npu.set_device(args.device)
    name = torch.npu.get_device_name(args.device)
    if "950" not in name:
        raise RuntimeError(f"CATLASS DSL requires Ascend 950; detected {name}")
    args.artifacts.mkdir(parents=True, exist_ok=True)
    compiled = None
    for case in CASES:
        directory = args.fixtures / case
        a, b = read_inputs(directory)
        device = f"npu:{args.device}"
        tensors = [
            torch.from_numpy(a).to(device),
            torch.from_numpy(b).to(device),
            torch.full((M, N), float("nan"), dtype=torch.float32, device=device),
        ]
        host_args = [
            from_dlpack(tensor, layout_tag=tla.arch.RowMajor, origin_shape=tuple(tensor.shape))
            for tensor in tensors
        ]
        if compiled is None:
            # Compilation is separate from launch. Static shapes are identical for all cases.
            compiled = tla.compile(matmul_kernel, *host_args, options="--npu-arch 3510")
            frontend_path = args.artifacts / "01-tla.mlir"
            backend_path = args.artifacts / "02-hivm.mlir"
            frontend_path.write_text(compiled.artifacts.MLIR, encoding="utf-8")
            # In CATLASS 2.2.0 the legacy attribute name LLVM contains lowered
            # MLIR (func/memref/hivm), NOT textual LLVM IR. Keep the .mlir suffix.
            backend_path.write_text(compiled.artifacts.LLVM, encoding="utf-8")
            details = {
                "frontend_mlir": str(frontend_path),
                "backend_mlir": str(backend_path),
                "kernel_binary_path": str(compiled.kernel_binary_path),
                "cache_dir": str(compiled.artifacts.cache_dir),
                "backend_compiler": str(compiled.artifacts.hivmc_path),
            }
            (args.artifacts / "artifacts.json").write_text(
                json.dumps(details, indent=2) + "\n", encoding="utf-8"
            )
            print(json.dumps(details, indent=2))
        compiled(*host_args, block_num=1)
        torch.npu.synchronize()
        tensors[2].cpu().numpy().astype("<f4").tofile(directory / "dsl_output.bin")
        print(f"CATLASS DSL: wrote {directory / 'dsl_output.bin'}")


if __name__ == "__main__":
    main()
