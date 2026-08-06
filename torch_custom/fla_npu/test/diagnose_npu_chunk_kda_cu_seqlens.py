# Copyright (c) 2026 Tianjin University, Ltd.

"""Diagnose the latest low-level ChunkKdaFwd ``cu_seqlens`` contract.

The current fla-org forward path passes raw gates into ``chunk_kda_fwd`` and
controls saved activations with ``disable_recompute`` and
``return_intermediate_states``.  The low-level operator always returns the
attention output in sequence-major layout, backward intermediates in
head-major layout, and states in sequence-major chunk order.  This script
checks those semantics while printing per-sequence and per-chunk error stats.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
from typing import Iterable

import torch


ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover - permits CPU-only syntax checks
    torch_npu = None

from fla_npu.ops import ascendc as fla_ascendc  # noqa: E402
from tests.reference.chunk_kda_reference import chunk_kda_forward_reference  # noqa: E402


OUTPUT_NAMES = (
    "o",
    "final_state",
    "g_cumsum",
    "Aqk",
    "Akk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "h",
    "initial_state",
)
MODEL_SHAPE_CASE = {
    "t": 131072,
    "h": 2,
    "hv": 2,
    "kdim": 128,
    "vdim": 128,
    "chunk_size": 64,
    "cu_seqlens": [0, 31739, 55973, 78732, 97530, 115345, 130191, 131071, 131072],
    "safe_gate": True,
    "lower_bound": -5.0,
    "seed": 20260711,
}
MODEL_SHAPE_DUMP = pathlib.Path(
    os.environ.get("KDA_MODEL_SHAPE_DUMP", "/tmp/kda_model_shape_case.pt")
)
STAT_SAMPLE_COUNT = int(os.environ.get("KDA_STAT_SAMPLE_COUNT", "262144"))
RCP_LN2 = 1.4426950408889634


def _device(device_id: int | None = None) -> torch.device:
    if device_id is None:
        device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device(f"npu:{device_id}")
    return torch.device("cpu")


def _require_npu(device_id: int | None = None) -> torch.device:
    device = _device(device_id)
    if device.type == "cpu":
        raise RuntimeError(
            "ChunkKdaFwd diagnostics require torch_npu, a visible NPU, and an installed latest fla_npu package."
        )
    torch.npu.set_device(device.index or 0)
    return device


def _sync(tensor: torch.Tensor | None = None) -> None:
    if tensor is not None and not getattr(tensor, "is_npu", False):
        return
    if torch_npu is not None and hasattr(torch, "npu"):
        torch.npu.synchronize()


def _stat(tensor: torch.Tensor, name: str) -> dict[str, object]:
    flat = tensor.detach().flatten()
    total_numel = flat.numel()
    if total_numel > STAT_SAMPLE_COUNT:
        stride = (total_numel + STAT_SAMPLE_COUNT - 1) // STAT_SAMPLE_COUNT
        sample = flat[::stride][:STAT_SAMPLE_COUNT].float().cpu()
    else:
        sample = flat.float().cpu()
    finite = sample[torch.isfinite(sample)]
    if finite.numel() == 0:
        finite = torch.zeros(1, dtype=torch.float32)
    return {
        "name": name,
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": finite.min().item(),
        "max": finite.max().item(),
        "mean": finite.mean().item(),
        "std": finite.std().item(),
        "p01": torch.quantile(finite, 0.01).item(),
        "p99": torch.quantile(finite, 0.99).item(),
        "has_nan": torch.isnan(flat).any().item(),
        "has_inf": torch.isinf(flat).any().item(),
        "sample_numel": sample.numel(),
        "total_numel": total_numel,
    }


def _print_stat(tensor: torch.Tensor, name: str, prefix: str = "  ") -> None:
    stat = _stat(tensor, name)
    print(
        f"{prefix}{name:18s} shape={stat['shape']} dtype={stat['dtype']:14s} "
        f"min={stat['min']:.5e} max={stat['max']:.5e} "
        f"mean={stat['mean']:.5e} std={stat['std']:.5e} "
        f"p01={stat['p01']:.5e} p99={stat['p99']:.5e} "
        f"nan={stat['has_nan']} inf={stat['has_inf']} "
        f"sample={stat['sample_numel']}/{stat['total_numel']}"
    )


def _l2norm_fwd_torch(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_float = x.float()
    rstd = torch.rsqrt(x_float.square().sum(dim=-1, keepdim=True) + eps)
    return (x_float * rstd).to(x.dtype)


def _make_case(
    *,
    t: int,
    h: int,
    hv: int,
    kdim: int,
    vdim: int,
    cu_seqlens: Iterable[int],
    chunk_size: int,
    safe_gate: bool,
    lower_bound: float,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, object]:
    cu = tuple(int(value) for value in cu_seqlens)
    if len(cu) < 2 or cu[0] != 0 or cu[-1] != t:
        raise ValueError(f"cu_seqlens must start at 0 and end at T={t}, got {cu}")
    if any(start > end for start, end in zip(cu, cu[1:])):
        raise ValueError(f"cu_seqlens must be nondecreasing, got {cu}")

    torch.manual_seed(seed)
    q = (torch.randn(1, t, h, kdim, dtype=torch.float32) * 0.05).to(dtype)
    k = (torch.randn(1, t, h, kdim, dtype=torch.float32) * 0.05).to(dtype)
    v = (torch.randn(1, t, hv, vdim, dtype=torch.float32) * 0.05).to(dtype)
    raw_gate = (torch.randn(1, t, hv, kdim, dtype=torch.float32) * 1.25).to(dtype)
    beta_logits = torch.randn(1, t, hv, dtype=torch.float32) * 0.35 + 1.5
    beta = torch.sigmoid(beta_logits)
    a_log = torch.randn(hv, dtype=torch.float32) * 0.12
    dt_bias = torch.randn(hv * kdim, dtype=torch.float32) * 1.65 - 3.0
    initial_state = torch.randn(len(cu) - 1, hv, kdim, vdim, dtype=torch.float32) * 0.02
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": raw_gate,
        "beta": beta,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "cu_seqlens": torch.tensor(cu, dtype=torch.int64),
        "initial_state": initial_state,
        "chunk_size": chunk_size,
        "safe_gate": safe_gate,
        "lower_bound": lower_bound,
    }


def _make_model_shape_dump(dump_path: pathlib.Path) -> pathlib.Path:
    case = _make_case(**MODEL_SHAPE_CASE)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(case, dump_path)
    return dump_path


def _to_input_layout(tensor: torch.Tensor, layout: str) -> torch.Tensor:
    """Convert canonical BSND/BSN inputs to the requested public input layout."""
    if layout == "BSND":
        return tensor.contiguous()
    if layout == "BNSD":
        return tensor.permute(0, 2, 1, *range(3, tensor.ndim)).contiguous()
    squeezed = tensor.squeeze(0)
    if layout == "TND":
        return squeezed.contiguous()
    if layout == "NTD":
        return squeezed.permute(1, 0, *range(2, squeezed.ndim)).contiguous()
    raise ValueError(f"unsupported layout: {layout}")


def _to_head_major(tensor: torch.Tensor, rank3: bool) -> torch.Tensor:
    """Convert canonical BSND intermediates to fixed BNSD/NTD output layout."""
    if rank3:
        squeezed = tensor.squeeze(0)
        return squeezed.permute(1, 0, *range(2, squeezed.ndim)).contiguous()
    return tensor.permute(0, 2, 1, *range(3, tensor.ndim)).contiguous()


def _head_major_to_bsnd(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3:
        return tensor.permute(1, 0, 2).unsqueeze(0).contiguous()
    return tensor.permute(0, 2, 1, 3).contiguous()


def _chunk_indices(cu_seqlens: Iterable[int], chunk_size: int) -> tuple[int, ...]:
    indices: list[int] = []
    cu = tuple(int(value) for value in cu_seqlens)
    for seq_idx, (start, end) in enumerate(zip(cu, cu[1:])):
        chunk_count = (end - start + chunk_size - 1) // chunk_size
        for chunk_idx in range(chunk_count):
            indices.extend((seq_idx, chunk_idx))
    return tuple(indices)


def _gate_cumsum_reference(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor | None,
    cu_seqlens: Iterable[int],
    chunk_size: int,
    safe_gate: bool,
    lower_bound: float,
) -> torch.Tensor:
    """Reference for the standalone fixed head-major BNSD/NTD gate op."""
    g_float = raw_gate.float()
    hv, kdim = raw_gate.shape[-3], raw_gate.shape[-1]
    x = g_float
    if dt_bias is not None:
        bias = dt_bias.reshape(hv, kdim).float()
        x = x + (bias[None, :, None, :] if raw_gate.ndim == 4 else bias[:, None, :])
    a = torch.exp(a_log.float())
    if safe_gate:
        x = x * (a[None, :, None, None] if raw_gate.ndim == 4 else a[:, None, None])
        gate = float(lower_bound) * torch.sigmoid(x)
    else:
        scale = a[None, :, None, None] if raw_gate.ndim == 4 else a[:, None, None]
        gate = -scale * torch.nn.functional.softplus(x)

    out = torch.empty_like(gate, dtype=torch.float32)
    token_axis = 2 if raw_gate.ndim == 4 else 1
    cu = tuple(int(value) for value in cu_seqlens)
    for seq_start, seq_end in zip(cu, cu[1:]):
        for chunk_start in range(seq_start, seq_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_end)
            index = [slice(None)] * raw_gate.ndim
            index[token_axis] = slice(chunk_start, chunk_end)
            out[tuple(index)] = torch.cumsum(
                gate[tuple(index)] * RCP_LN2,
                dim=token_axis,
            )
    return out


def _assert_close(
    name: str,
    actual: torch.Tensor | None,
    expected: torch.Tensor | None,
    *,
    rtol: float,
    atol: float,
) -> None:
    if actual is None or expected is None:
        if actual is expected:
            return
        raise AssertionError(f"{name}: actual={actual} expected={expected}")
    actual_cpu = actual.detach().cpu()
    expected_cpu = expected.detach().cpu()
    try:
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=rtol, atol=atol, msg=name)
    except AssertionError:
        diff = (actual_cpu.float() - expected_cpu.float()).abs()
        flat_index = int(diff.argmax().item())
        coord = tuple(int(value) for value in torch.unravel_index(torch.tensor(flat_index), diff.shape))
        rel = diff / expected_cpu.float().abs().clamp_min(1e-12)
        print(
            f"[Mismatch] {name}: shape={tuple(actual.shape)} coord={coord} "
            f"max_abs={diff.max().item():.8e} max_rel={rel.max().item():.8e}",
            flush=True,
        )
        raise


def _chunk_kda_fwd_latest(*args, **kwargs):
    """Call the post-July-2026 low-level API with an actionable version error."""
    try:
        return fla_ascendc.chunk_kda_fwd(*args, **kwargs)
    except (TypeError, RuntimeError) as error:
        message = str(error)
        version_markers = (
            "unexpected keyword",
            "Unknown keyword",
            "disable_recompute",
            "return_intermediate_states",
            "state_v_first",
        )
        if any(marker in message for marker in version_markers):
            raise RuntimeError(
                "The installed fla_npu exposes the legacy ChunkKdaFwd API. "
                "Build/install the latest main version before running this diagnostic."
            ) from error
        raise


def _error_metrics(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, float]:
    diff = (actual.float() - expected.float()).abs()
    return (
        diff.max().item(),
        diff.mean().item(),
        (diff.sum() / expected.float().abs().sum().clamp_min(1e-12)).item(),
    )


def _slice_axis(tensor: torch.Tensor, axis: int, start: int, end: int) -> torch.Tensor:
    index = [slice(None)] * tensor.ndim
    index[axis] = slice(start, end)
    return tensor[tuple(index)]


def _print_cu_layout(cu_seqlens: Iterable[int], chunk_size: int) -> None:
    print("\n--- cu_seqlens layout ---")
    chunk_cursor = 0
    cu = tuple(int(value) for value in cu_seqlens)
    for seq_idx, (start, end) in enumerate(zip(cu, cu[1:])):
        length = end - start
        chunks = (length + chunk_size - 1) // chunk_size
        tail = length % chunk_size or (chunk_size if length else 0)
        print(
            f"  seq={seq_idx:2d} T=[{start:6d},{end:6d}) len={length:6d} "
            f"chunks=[{chunk_cursor:4d},{chunk_cursor + chunks:4d}) tail={tail:3d}"
        )
        chunk_cursor += chunks


def _print_token_errors(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    cu_seqlens: Iterable[int],
    token_axis: int,
) -> None:
    actual_cpu, expected_cpu = actual.detach().cpu(), expected.detach().cpu()
    print(f"\n--- per-sequence token errors: {name} ---")
    cu = tuple(int(value) for value in cu_seqlens)
    for seq_idx, (start, end) in enumerate(zip(cu, cu[1:])):
        actual_seq = _slice_axis(actual_cpu, token_axis, start, end)
        expected_seq = _slice_axis(expected_cpu, token_axis, start, end)
        max_abs, mean_abs, l1_rel = _error_metrics(actual_seq, expected_seq)
        print(
            f"  seq={seq_idx:2d} T=[{start:6d},{end:6d}) "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} l1_rel={l1_rel:.6e}"
        )


def _print_chunk_errors(
    actual: torch.Tensor,
    expected: torch.Tensor,
    cu_seqlens: Iterable[int],
    chunk_size: int,
    chunk_axis: int,
) -> None:
    actual_cpu, expected_cpu = actual.detach().cpu(), expected.detach().cpu()
    print("\n--- per-sequence chunk-state errors: h ---")
    cursor = 0
    cu = tuple(int(value) for value in cu_seqlens)
    for seq_idx, (start, end) in enumerate(zip(cu, cu[1:])):
        chunk_count = (end - start + chunk_size - 1) // chunk_size
        actual_seq = _slice_axis(actual_cpu, chunk_axis, cursor, cursor + chunk_count)
        expected_seq = _slice_axis(expected_cpu, chunk_axis, cursor, cursor + chunk_count)
        max_abs, mean_abs, l1_rel = _error_metrics(actual_seq, expected_seq)
        print(
            f"  seq={seq_idx:2d} chunks=[{cursor:4d},{cursor + chunk_count:4d}) "
            f"max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} l1_rel={l1_rel:.6e}"
        )
        cursor += chunk_count


def _expected_outputs(
    reference,
    gate_head_major: torch.Tensor,
    initial_state_input: torch.Tensor | None,
    *,
    rank3: bool,
    state_v_first: bool,
) -> tuple[torch.Tensor | None, ...]:
    output = reference.o.squeeze(0) if rank3 else reference.o
    final_state = reference.final_state
    h = reference.h.squeeze(0) if rank3 else reference.h
    if state_v_first:
        final_state = final_state.transpose(-1, -2).contiguous()
        h = h.transpose(-1, -2).contiguous()
    return (
        output,
        final_state,
        gate_head_major,
        _to_head_major(reference.Aqk, rank3),
        _to_head_major(reference.Akk, rank3),
        _to_head_major(reference.w, rank3),
        _to_head_major(reference.u, rank3),
        _to_head_major(reference.qg, rank3),
        _to_head_major(reference.kg, rank3),
        _to_head_major(reference.v_new, rank3),
        h,
        initial_state_input,
    )


def _check_optional_output_policy(
    inputs: tuple[torch.Tensor, ...],
    kwargs: dict[str, object],
    baseline: tuple[torch.Tensor | None, ...],
) -> None:
    minimal_kwargs = dict(kwargs)
    minimal_kwargs.update(
        output_final_state=False,
        disable_recompute=False,
        return_intermediate_states=False,
    )
    outputs = _chunk_kda_fwd_latest(*inputs, **minimal_kwargs)
    expected_visible = (
        True,
        False,
        False,  # raw gate is used and recomputation remains enabled
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
    )
    assert len(outputs) == len(expected_visible) == 12
    for index, (name, output, visible) in enumerate(zip(OUTPUT_NAMES, outputs, expected_visible)):
        if visible:
            if index == 11 and kwargs["initial_state"] is None:
                assert output is None
            else:
                assert torch.is_tensor(output), f"{name} must be visible"
                # Optional-output flags change export/recompute scheduling.  The
                # public contract is numerical equivalence, not bitwise identity.
                _assert_close(name, output, baseline[index], rtol=2e-3, atol=2e-3)
        else:
            assert output is None, f"{name} must be omitted by the latest optional-output policy"


def run_diagnostic(
    data: dict[str, object],
    *,
    layout: str,
    device_id: int | None,
    state_v_first: bool,
    use_initial_state: bool,
    skip_reference: bool,
    check_optional_policy: bool,
) -> None:
    device = _require_npu(device_id)
    rank3 = layout in {"TND", "NTD"}
    q = _l2norm_fwd_torch(data["q"].to(device, non_blocking=True))
    k = _l2norm_fwd_torch(data["k"].to(device, non_blocking=True))
    v = data["v"].to(device, non_blocking=True)
    raw_gate = data["g"].to(device, non_blocking=True)
    beta = data["beta"].to(device, non_blocking=True)
    a_log = data["A_log"].to(device, dtype=torch.float32, non_blocking=True)
    dt_bias = data["dt_bias"].to(device, dtype=torch.float32, non_blocking=True)
    cu = tuple(int(value) for value in data["cu_seqlens"].tolist())
    chunk_size = int(data["chunk_size"])
    safe_gate = bool(data["safe_gate"])
    lower_bound = float(data["lower_bound"])
    initial_state_reference = data["initial_state"] if use_initial_state else None
    initial_state_input = initial_state_reference
    if initial_state_input is not None and state_v_first:
        initial_state_input = initial_state_input.transpose(-1, -2).contiguous()
    if initial_state_input is not None:
        initial_state_input = initial_state_input.to(device, non_blocking=True)

    print("\n" + "=" * 88)
    print("Latest ChunkKdaFwd cu_seqlens diagnostic")
    print("=" * 88)
    print(
        f"device={device} layout={layout} chunk_size={chunk_size} "
        f"safe_gate={safe_gate} lower_bound={lower_bound} "
        f"state_v_first={state_v_first} initial_state={use_initial_state}"
    )
    print("q/k are L2-normalized and beta is post-sigmoid, matching the latest high-level KDA wrapper.")
    _print_cu_layout(cu, chunk_size)

    print("\n--- input statistics ---")
    for name, tensor in (
        ("q_norm", q),
        ("k_norm", k),
        ("v", v),
        ("g_raw", raw_gate),
        ("beta", beta),
        ("A_log", a_log),
        ("dt_bias", dt_bias),
    ):
        _print_stat(tensor, name)
    if initial_state_input is not None:
        _print_stat(initial_state_input, "initial_state")

    raw_gate_head_major = _to_head_major(raw_gate, rank3)
    gate_start = time.perf_counter()
    gate_npu = fla_ascendc.kda_gate_cumsum(
        raw_gate_head_major,
        chunk_size,
        A_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu,
        use_gate_in_kernel=True,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )
    _sync(gate_npu)
    print(f"\nstandalone gate cumsum: {(time.perf_counter() - gate_start) * 1000:.3f} ms")
    _print_stat(gate_npu, "g_cumsum")

    q_input = _to_input_layout(q, layout)
    k_input = _to_input_layout(k, layout)
    v_input = _to_input_layout(v, layout)
    raw_gate_input = _to_input_layout(raw_gate, layout)
    beta_input = _to_input_layout(beta, layout)
    indices = _chunk_indices(cu, chunk_size)
    op_inputs = (q_input, k_input, v_input, raw_gate_input, beta_input, q.shape[-1] ** -0.5, chunk_size)
    op_kwargs: dict[str, object] = {
        "layout": layout,
        "initial_state": initial_state_input,
        "output_final_state": True,
        "cu_seqlens": cu,
        "chunk_indices": indices,
        "safe_gate": safe_gate,
        "lower_bound": lower_bound,
        "use_gate_in_kernel": True,
        "A_log": a_log,
        "dt_bias": dt_bias,
        "disable_recompute": True,
        "return_intermediate_states": True,
        "state_v_first": state_v_first,
    }

    fwd_start = time.perf_counter()
    outputs = _chunk_kda_fwd_latest(*op_inputs, **op_kwargs)
    _sync(outputs[0])
    print(f"chunk_kda_fwd: {(time.perf_counter() - fwd_start) * 1000:.3f} ms")
    assert len(outputs) == 12, f"latest low-level contract returns 12 values, got {len(outputs)}"

    print("\n--- output statistics ---")
    for name, tensor in zip(OUTPUT_NAMES, outputs):
        if tensor is None:
            print(f"  {name:18s} None")
            continue
        _print_stat(tensor, name)
        assert torch.isfinite(tensor).all().item(), f"{name} contains NaN or Inf"
    _assert_close("integrated g_cumsum", outputs[2], gate_npu, rtol=2e-3, atol=2e-3)

    if skip_reference:
        print("\nCPU reference skipped; NPU finiteness, layout, gate integration, and output policy were checked.")
        if check_optional_policy:
            _check_optional_output_policy(op_inputs, op_kwargs, outputs)
        return

    gate_reference = _gate_cumsum_reference(
        raw_gate_head_major.detach().cpu(),
        a_log.detach().cpu(),
        dt_bias.detach().cpu(),
        cu,
        chunk_size,
        safe_gate,
        lower_bound,
    )
    _assert_close("standalone gate cumsum", gate_npu, gate_reference, rtol=2e-3, atol=2e-3)
    gate_bsnd = _head_major_to_bsnd(gate_reference)

    reference_start = time.perf_counter()
    reference = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gate_bsnd,
        beta.detach().cpu(),
        scale=q.shape[-1] ** -0.5,
        chunk_size=chunk_size,
        initial_state=None if initial_state_reference is None else initial_state_reference.detach().cpu(),
        output_final_state=True,
        cu_seqlens=torch.tensor(cu, dtype=torch.int64),
    )
    print(f"CPU reference: {time.perf_counter() - reference_start:.3f} s")
    expected = _expected_outputs(
        reference,
        gate_reference,
        initial_state_input,
        rank3=rank3,
        state_v_first=state_v_first,
    )

    for name, actual, wanted in zip(OUTPUT_NAMES, outputs, expected):
        _assert_close(name, actual, wanted, rtol=5e-2, atol=5e-2)

    output_token_axis = 0 if rank3 else 1
    intermediate_token_axis = 1 if rank3 else 2
    chunk_axis = 0 if rank3 else 1
    _print_token_errors("o", outputs[0], expected[0], cu, output_token_axis)
    for index in (2, 3, 4, 5, 6, 7, 8, 9):
        _print_token_errors(
            OUTPUT_NAMES[index],
            outputs[index],
            expected[index],
            cu,
            intermediate_token_axis,
        )
    _print_chunk_errors(outputs[10], expected[10], cu, chunk_size, chunk_axis)
    final_actual = outputs[1].detach().cpu()
    final_expected = expected[1].detach().cpu()
    print("\n--- per-sequence final-state errors ---")
    for seq_idx in range(len(cu) - 1):
        max_abs, mean_abs, l1_rel = _error_metrics(
            final_actual[seq_idx], final_expected[seq_idx]
        )
        print(
            f"  seq={seq_idx:2d} max_abs={max_abs:.6e} "
            f"mean_abs={mean_abs:.6e} l1_rel={l1_rel:.6e}"
        )

    if check_optional_policy:
        _check_optional_output_policy(op_inputs, op_kwargs, outputs)
    print("\nPASS: latest raw-gate, optional-output, layout, state, and cu_seqlens contracts match the reference.")


def _quick_case(chunk_size: int, seed: int) -> dict[str, object]:
    return _make_case(
        t=257,
        h=2,
        hv=4,
        kdim=128,
        vdim=128,
        cu_seqlens=(0, 1, 65, 130, 257),
        chunk_size=chunk_size,
        safe_gate=True,
        lower_bound=-5.0,
        seed=seed,
    )


def test_chunk_kda_fwd_model_shape_with_stats(
    dump_path: str | pathlib.Path | None = None,
    device_id: int | None = None,
) -> None:
    if _device(device_id).type == "cpu":
        print("skip model-shape ChunkKdaFwd diagnostic on CPU")
        return
    path = MODEL_SHAPE_DUMP if dump_path is None else pathlib.Path(dump_path)
    if not path.exists():
        _make_model_shape_dump(path)
    run_diagnostic(
        torch.load(path, map_location="cpu"),
        layout="NTD",
        device_id=device_id,
        state_v_first=False,
        use_initial_state=True,
        skip_reference=False,
        check_optional_policy=False,
    )


def test_chunk_kda_fwd_from_dump_with_stats(
    dump_path: str | pathlib.Path,
    device_id: int | None = None,
) -> None:
    test_chunk_kda_fwd_model_shape_with_stats(dump_path=dump_path, device_id=device_id)


test_chunk_kda_fwd_from_dump_with_stats.__test__ = False


def test_chunk_kda_fwd_cu_seqlens_latest_contract() -> None:
    if _device().type == "cpu":
        print("skip latest ChunkKdaFwd cu_seqlens test on CPU")
        return
    run_diagnostic(
        _quick_case(64, 20260801),
        layout="NTD",
        device_id=None,
        state_v_first=False,
        use_initial_state=True,
        skip_reference=False,
        check_optional_policy=True,
    )


def test_chunk_kda_fwd_cu_seqlens_state_v_first() -> None:
    if _device().type == "cpu":
        print("skip latest ChunkKdaFwd state_v_first test on CPU")
        return
    run_diagnostic(
        _quick_case(128, 20260802),
        layout="BSND",
        device_id=None,
        state_v_first=True,
        use_initial_state=True,
        skip_reference=False,
        check_optional_policy=False,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "model"), default="quick")
    parser.add_argument("--layout", choices=("BSND", "BNSD", "TND", "NTD"), default="NTD")
    parser.add_argument("--chunk-size", type=int, choices=(64, 128), default=None)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--dump-path", type=pathlib.Path, default=MODEL_SHAPE_DUMP)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--state-v-first", action="store_true")
    parser.add_argument("--no-initial-state", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--check-optional-policy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.mode == "model":
        dump_path = args.dump_path
        if not dump_path.exists():
            print(f"model-shape dump does not exist; generating {dump_path}")
            _make_model_shape_dump(dump_path)
        data = torch.load(dump_path, map_location="cpu")
        if args.chunk_size is not None:
            data["chunk_size"] = args.chunk_size
    else:
        data = _quick_case(args.chunk_size or 64, args.seed)

    run_diagnostic(
        data,
        layout=args.layout,
        device_id=args.device_id,
        state_v_first=args.state_v_first,
        use_initial_state=not args.no_initial_state,
        skip_reference=args.skip_reference,
        check_optional_policy=args.check_optional_policy,
    )


if __name__ == "__main__":
    main()
