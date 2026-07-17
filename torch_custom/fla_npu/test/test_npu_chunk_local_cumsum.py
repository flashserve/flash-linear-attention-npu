import math
import os
from typing import List, Optional, Tuple

import torch
import torch_npu
import fla_npu


torch.npu.config.allow_internal_format = False
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.set_device(int(os.environ.get("TEST_DEVICE_ID", 0)))


DTYPE_LABELS = {
    torch.float32: "fp32",
    torch.float16: "fp16",
    torch.bfloat16: "bf16",
}

OUTPUT_DTYPE_SPECS = (None, "float32", "float16", "bfloat16", "same")


def _next_power_of_two(value: int) -> int:
    value = max(value, 1)
    return 1 << (value - 1).bit_length()


def _block_t(shape: Tuple[int, ...], chunk_size: int) -> int:
    if len(shape) != 3:
        raise ValueError(f"chunk_local_cumsum only supports rank-3 [B, H, T], got shape={shape}")
    return _next_power_of_two((1 << 17) // chunk_size)


def prepare_chunk_indices(cu_seqlens: torch.Tensor, block_t: int) -> torch.Tensor:
    rows = []
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())):
        num_blocks = math.ceil((end - start) / block_t)
        for block_idx in range(num_blocks):
            rows.append((seq_idx, block_idx))
    return torch.tensor(rows, dtype=torch.long)


def reference_impl(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool,
    scale: float,
    cu_seqlens: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    out = torch.empty_like(g, dtype=torch.float32)
    for batch in range(g.size(0)):
        if cu_seqlens is None:
            ranges = [(0, g.size(2))]
        else:
            ranges = [(start, end) for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())]

        for head in range(g.size(1)):
            for seq_start, seq_end in ranges:
                seq_len = seq_end - seq_start
                for chunk_start in range(0, seq_len, chunk_size):
                    start = seq_start + chunk_start
                    end = min(start + chunk_size, seq_end)
                    segment = g[batch, head, start:end].to(torch.float32)
                    if reverse:
                        value = torch.flip(torch.cumsum(torch.flip(segment, dims=[0]), dim=0), dims=[0])
                    else:
                        value = torch.cumsum(segment, dim=0)
                    out[batch, head, start:end] = value * scale
    return out.to(output_dtype)


def resolve_output_dtype(input_dtype: torch.dtype, output_dtype: Optional[str]) -> torch.dtype:
    if output_dtype is None or output_dtype in {"", "float", "float32", "fp32", "torch.float", "torch.float32"}:
        return torch.float32
    if output_dtype in {"same", "same_as_input", "input", "none", "None", "null"}:
        return input_dtype
    if output_dtype in {"float16", "fp16", "half", "torch.float16", "torch.half"}:
        return torch.float16
    if output_dtype in {"bfloat16", "bf16", "torch.bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"unsupported output_dtype={output_dtype}")


def _tolerances(dtype: torch.dtype) -> Tuple[float, float]:
    if dtype is torch.float32:
        return 1e-4, 1e-4
    if dtype is torch.float16:
        return 1e-3, 2e-3
    return 2e-2, 5e-2


def run_case(
    name: str,
    shape: Tuple[int, ...],
    chunk_size: int,
    reverse: bool = False,
    scale: float = 1.0,
    cu_seqlens_values: Optional[List[int]] = None,
    dtype: torch.dtype = torch.float32,
    output_dtype: Optional[str] = "same",
) -> None:
    torch.manual_seed(sum(ord(ch) for ch in name))
    if len(shape) != 3:
        raise ValueError(f"{name}: chunk_local_cumsum only supports rank-3 [B, H, T], got shape={shape}")
    g_cpu = torch.randn(shape, dtype=torch.float32).to(dtype)
    g_npu = g_cpu.npu()

    cu_seqlens_arg = None
    chunk_indices_arg = None
    cu_seqlens_cpu = None
    if cu_seqlens_values is not None:
        cu_seqlens_cpu = torch.tensor(cu_seqlens_values, dtype=torch.long)
        block_t = _block_t(shape, chunk_size)
        chunk_indices_cpu = prepare_chunk_indices(cu_seqlens_cpu, block_t)
        cu_seqlens_arg = cu_seqlens_cpu.tolist()
        chunk_indices_arg = chunk_indices_cpu.reshape(-1).tolist()

    kwargs = {
        "cu_seqlens": cu_seqlens_arg,
        "chunk_indices_out": chunk_indices_arg,
        "reverse": reverse,
        "scale": scale,
        "head_first": True,
    }
    if output_dtype is not None:
        kwargs["output_dtype"] = output_dtype
    actual = torch.ops.npu.npu_chunk_local_cumsum(g_npu, chunk_size, **kwargs).cpu()
    expected_dtype = resolve_output_dtype(dtype, output_dtype)
    expected = reference_impl(g_cpu, chunk_size, reverse, scale, cu_seqlens_cpu, expected_dtype)

    if actual.dtype != expected_dtype:
        raise AssertionError(f"{name}: expected output dtype {expected_dtype}, got {actual.dtype}")
    rtol, atol = _tolerances(expected_dtype)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    print(
        f"[PASS] {name}: input={DTYPE_LABELS[dtype]}, output={DTYPE_LABELS[expected_dtype]}, "
        f"output_dtype={output_dtype}, shape={shape}, chunk_size={chunk_size}, reverse={reverse}, scale={scale}"
    )


if __name__ == "__main__":
    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        suffix = DTYPE_LABELS[dtype]
        for output_dtype in OUTPUT_DTYPE_SPECS:
            output_suffix = "default" if output_dtype is None else output_dtype.replace(".", "_")
            run_case(
                f"fixed_bht_output_{suffix}_{output_suffix}",
                (2, 3, 129),
                chunk_size=64,
                dtype=dtype,
                output_dtype=output_dtype,
            )
        run_case(f"fixed_bht_forward_{suffix}", (9, 2, 128), chunk_size=64, dtype=dtype)
        run_case(
            f"fixed_bht_reverse_scale_{suffix}",
            (9, 2, 128),
            chunk_size=64,
            reverse=True,
            scale=0.25,
            dtype=dtype,
        )
        run_case(f"fixed_bht_odd_t_forward_{suffix}", (2, 3, 129), chunk_size=64, dtype=dtype)
        run_case(
            f"varlen_bht_forward_{suffix}",
            (1, 2, 128),
            chunk_size=64,
            cu_seqlens_values=[0, 128],
            dtype=dtype,
        )
        run_case(
            f"varlen_bht_long_forward_{suffix}",
            (1, 8, 3580),
            chunk_size=64,
            cu_seqlens_values=[0, 3580],
            dtype=dtype,
        )
        run_case(
            f"varlen_bht_reverse_scale_{suffix}",
            (1, 8, 3580),
            chunk_size=64,
            reverse=True,
            scale=-0.5,
            cu_seqlens_values=[0, 1024, 2048, 3580],
            dtype=dtype,
        )
