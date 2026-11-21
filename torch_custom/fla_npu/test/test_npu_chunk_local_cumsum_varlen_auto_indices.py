"""Regression: varlen chunk_local_cumsum must work without chunk_indices_out.

``npu_chunk_local_cumsum`` takes ``cu_seqlens`` plus ``chunk_indices_out``.
Every other varlen wrapper in ``_stable.py`` derives the canonical
sequence-major ``(seq, block)`` list when the caller leaves it out; this one
used to forward ``None`` and the aclnn host check failed with 161001 before
the kernel was even launched.  The cases below call with boundaries only and
compare against both a torch reference and the explicit-index call.
"""

import math
import os
import warnings

import torch
import torch_npu

from fla_npu.ops.ascendc import chunk_local_cumsum


torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))

DTYPE_LABELS = {torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}


def prepare_chunk_indices(cu_seqlens, chunk_size):
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        for block_idx in range(math.ceil((end - start) / chunk_size)):
            rows.extend((seq_idx, block_idx))
    return rows


def reference_impl(g, chunk_size, cu_seqlens, reverse, scale):
    out = torch.empty_like(g, dtype=torch.float32)
    for head in range(g.size(1)):
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            for chunk_start in range(start, end, chunk_size):
                chunk_end = min(chunk_start + chunk_size, end)
                segment = g[0, head, chunk_start:chunk_end].to(torch.float32)
                if reverse:
                    value = torch.flip(torch.cumsum(torch.flip(segment, dims=[0]), dim=0), dims=[0])
                else:
                    value = torch.cumsum(segment, dim=0)
                out[0, head, chunk_start:chunk_end] = value * scale
    return out


def tolerances(dtype):
    return (1e-6, 1e-6) if dtype == torch.float32 else (5e-3, 5e-3)


def run_case(name, shape, chunk_size, cu_seqlens_values, reverse=False, scale=1.0,
             dtype=torch.float32, output_dtype="float32"):
    torch.manual_seed(sum(ord(ch) for ch in name))
    g_cpu = torch.randn(shape, dtype=torch.float32).to(dtype)
    g_npu = g_cpu.npu()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        # Boundaries only: the wrapper has to build chunk_indices_out itself.
        actual = chunk_local_cumsum(
            g_npu, chunk_size, cu_seqlens=cu_seqlens_values, reverse=reverse,
            scale=scale, head_first=True, output_dtype=output_dtype)
        expected_indices = prepare_chunk_indices(cu_seqlens_values, chunk_size)
        explicit = chunk_local_cumsum(
            g_npu, chunk_size, cu_seqlens=cu_seqlens_values,
            chunk_indices_out=expected_indices, reverse=reverse, scale=scale,
            head_first=True, output_dtype=output_dtype)
    torch.npu.synchronize()

    expected = reference_impl(g_cpu, chunk_size, list(cu_seqlens_values), reverse, scale)
    rtol, atol = tolerances(dtype)
    torch.testing.assert_close(actual.float().cpu(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(actual, explicit, rtol=0, atol=0)
    print(f"[PASS] {name}: {DTYPE_LABELS[dtype]}->{output_dtype}, shape={shape}, "
          f"chunk_size={chunk_size}, cu_seqlens={list(cu_seqlens_values)}, "
          f"reverse={reverse}, scale={scale}")


def run_cpu_tensor_case():
    """cu_seqlens may arrive as a CPU tensor; the wrapper still derives."""
    torch.manual_seed(7)
    g_cpu = torch.randn((1, 2, 9), dtype=torch.float32)
    cu = torch.tensor([0, 3, 9], dtype=torch.int64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        actual = chunk_local_cumsum(
            g_cpu.npu(), 4, cu_seqlens=cu, head_first=True, output_dtype="float32")
        explicit = chunk_local_cumsum(
            g_cpu.npu(), 4, cu_seqlens=[0, 3, 9],
            chunk_indices_out=prepare_chunk_indices([0, 3, 9], 4),
            head_first=True, output_dtype="float32")
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), explicit.cpu(), rtol=0, atol=0)
    print("[PASS] cpu-tensor cu_seqlens")


def main():
    run_case("varlen_fp32_forward", (1, 2, 9), 4, [0, 3, 9])
    run_case("varlen_fp32_reverse_scale", (1, 2, 69), 16, [0, 5, 69], reverse=True, scale=0.5)
    run_case("varlen_fp32_aligned", (1, 3, 128), 64, [0, 64, 128])
    # output_dtype carries only the two codes the operator enum has today:
    # float32 (0) and bfloat16 (1).  fp16 inputs still accumulate in fp32.
    run_case("varlen_bf16_forward", (1, 2, 70), 32, [0, 17, 70], dtype=torch.bfloat16,
             output_dtype="bfloat16")
    run_case("varlen_fp16_reverse", (1, 1, 33), 16, [0, 1, 33], reverse=True,
             dtype=torch.float16, output_dtype="float32")
    run_cpu_tensor_case()
    print("all varlen auto-index cases passed")


if __name__ == "__main__":
    main()