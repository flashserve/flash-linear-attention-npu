# Copyright (c) 2026 Tianjin University, Ltd.

import json
import itertools
import math
import os
import pathlib
import subprocess
import sys
import time

import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover - CPU fallback for syntax/smoke only
    torch_npu = None

from fla_npu.ops import ascendc as fla_ascendc
from fla_npu.ops.ascendc._kda_policy import (
    FLA_ORG_KDA_FWD_ALIGNMENT_COMMIT,
    kda_fwd_optional_output_mask,
)


ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.reference.chunk_kda_reference import chunk_kda_forward_reference  # noqa: E402
from tests.operators._shared.legacy_cases import find_legacy_case, legacy_param_values  # noqa: E402


MODEL_SHAPE_CASE = find_legacy_case(
    "chunk_kda_fwd", "model_shape_131072", "model_shape"
)["legacy"]["raw"]["config"]
MODEL_SHAPE_DUMP = pathlib.Path(os.environ.get("KDA_MODEL_SHAPE_DUMP", "/tmp/kda_model_shape_case.pt"))
STAT_SAMPLE_COUNT = int(os.environ.get("KDA_STAT_SAMPLE_COUNT", "262144"))
REFERENCE_NUM_THREADS = int(os.environ.get("KDA_REFERENCE_NUM_THREADS", "16"))
FWD_H_DETERMINISM_REPEATS = int(os.environ.get("FWD_H_DETERMINISM_REPEATS", "50"))


def _device(device_id=None):
    if device_id is None:
        device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device(f"npu:{device_id}")
    return torch.device("cpu")


def _stat(tensor, name):
    flat = tensor.detach().flatten()
    nan_mask = torch.isnan(flat)
    has_nan = nan_mask.any().item()
    has_inf = torch.isinf(flat).any().item()
    nan_ratio = nan_mask.float().mean().item()

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
        "has_nan": has_nan,
        "has_inf": has_inf,
        "nan_ratio": nan_ratio,
        "percentile_1": torch.quantile(finite, 0.01).item(),
        "percentile_99": torch.quantile(finite, 0.99).item(),
        "sample_numel": sample.numel(),
        "total_numel": total_numel,
    }


def _print_stat(stat_dict, prefix=""):
    s = stat_dict
    print(
        f"{prefix}{s['name']:18s} | "
        f"shape={s['shape']} {s['dtype']:12s} | "
        f"min={s['min']:10.4f} max={s['max']:10.4f} | "
        f"mean={s['mean']:8.4f} std={s['std']:8.4f} | "
        f"p1={s['percentile_1']:8.4f} p99={s['percentile_99']:8.4f} | "
        f"nan={s['has_nan']} inf={s['has_inf']} nan_ratio={s['nan_ratio']:.4f} | "
        f"sample={s['sample_numel']}/{s['total_numel']}"
    )


def _print_first_nonfinite(tensor, name, prefix=""):
    if hasattr(tensor, "is_npu") and tensor.is_npu:
        torch.npu.synchronize()
    flat = tensor.detach().flatten().float().cpu()
    bad = ~torch.isfinite(flat)
    if not bad.any().item():
        return
    idx = int(bad.nonzero(as_tuple=False)[0].item())
    print(f"{prefix}{name}: first non-finite flat_index={idx}, value={flat[idx].item()}")


def _l2norm_fwd_torch(x, eps=1e-6):
    x_float = x.float()
    rstd = torch.rsqrt(x_float.pow(2).sum(dim=-1, keepdim=True) + eps)
    y = (x_float * rstd).to(x.dtype)
    return y, rstd.to(x.dtype)


def _make_model_shape_kda_dump(case=None, dump_path=None):
    case = dict(MODEL_SHAPE_CASE if case is None else case)
    dump_path = MODEL_SHAPE_DUMP if dump_path is None else pathlib.Path(dump_path)
    torch.manual_seed(case["seed"])

    t = case["t"]
    h = case["h"]
    hv = case["hv"]
    kdim = case["kdim"]
    vdim = case["vdim"]
    seq_num = len(case["cu_seqlens"]) - 1

    q = (torch.randn(1, t, h, kdim, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    k = (torch.randn(1, t, h, kdim, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    v = (torch.randn(1, t, hv, vdim, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    g = (torch.randn(1, t, hv, kdim, dtype=torch.float32) * 1.25).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(1, t, hv, dtype=torch.float32) * 0.35 + 1.5)
    a_log = torch.randn(hv, dtype=torch.float32) * 0.12
    dt_bias = torch.randn(hv * kdim, dtype=torch.float32) * 1.65 - 3.0
    initial_state = torch.randn(seq_num, hv, kdim, vdim, dtype=torch.float32) * 0.02

    dump_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q": q,
            "k": k,
            "v": v,
            "g": g,
            "beta": beta,
            "A_log": a_log,
            "dt_bias": dt_bias,
            "cu_seqlens": torch.tensor(case["cu_seqlens"], dtype=torch.int64),
            "initial_state": initial_state,
            "safe_gate": case["safe_gate"],
            "lower_bound": case["lower_bound"],
            "chunk_size": case["chunk_size"],
        },
        dump_path,
    )
    return dump_path


def _make_inputs(device, b=1, h=2, hv=2, t=64, kdim=128, vdim=128, dtype=torch.bfloat16):
    torch.manual_seed(1234 + b + h + hv + t + kdim + vdim)
    q = (torch.randn(b, t, h, kdim, dtype=dtype) * 0.08).to(device).requires_grad_(True)
    k = (torch.randn(b, t, h, kdim, dtype=dtype) * 0.08).to(device).requires_grad_(True)
    v = (torch.randn(b, t, hv, vdim, dtype=dtype) * 0.08).to(device).requires_grad_(True)
    gk = (torch.randn(b, t, hv, kdim, dtype=torch.float32).cumsum(dim=1) * 0.001).to(device).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(b, t, hv, dtype=torch.float32)).to(device).requires_grad_(True)
    initial_state = (torch.randn(b, hv, kdim, vdim, dtype=torch.float32) * 0.02).to(device).requires_grad_(True)
    return q, k, v, gk, beta, initial_state


def _chunk_kda_fwd_from_gk(q, k, v, gk, beta, scale, chunk_size=64, **kwargs):
    """Adapt legacy pre-cumulative test vectors to the public raw-gate contract."""
    layout = kwargs.get("layout", "BSND")
    token_axes = {"BSND": 1, "BNSD": 2, "TND": 0, "NTD": 1}
    if layout not in token_axes:
        return fla_ascendc.chunk_kda_fwd(
            q, k, v, gk, beta, scale, chunk_size, **kwargs
        )
    token_axis = token_axes[layout]
    token_count = gk.shape[token_axis]
    cu_seqlens = kwargs.get("cu_seqlens")
    boundaries = [0, token_count] if cu_seqlens is None else [int(value) for value in cu_seqlens]
    increments = torch.empty_like(gk)
    for seq_start, seq_end in zip(boundaries, boundaries[1:]):
        for chunk_start in range(seq_start, seq_end, int(chunk_size)):
            chunk_end = min(chunk_start + int(chunk_size), seq_end)
            first = [slice(None)] * gk.ndim
            first[token_axis] = chunk_start
            increments[tuple(first)] = gk[tuple(first)]
            if chunk_start + 1 < chunk_end:
                current = [slice(None)] * gk.ndim
                previous = [slice(None)] * gk.ndim
                current[token_axis] = slice(chunk_start + 1, chunk_end)
                previous[token_axis] = slice(chunk_start, chunk_end - 1)
                increments[tuple(current)] = gk[tuple(current)] - gk[tuple(previous)]
    raw_gate = increments * math.log(2.0)
    return fla_ascendc.chunk_kda_fwd(
        q, k, v, raw_gate, beta, scale, chunk_size, **kwargs
    )


def _assert_close(name, actual, expected, rtol=2e-3, atol=2e-3):
    actual_cpu = actual.cpu()
    expected_cpu = expected.cpu()
    try:
        torch.testing.assert_close(actual_cpu, expected_cpu, rtol=rtol, atol=atol, msg=name)
    except AssertionError:
        actual_float = actual_cpu.float()
        expected_float = expected_cpu.float()
        abs_diff = (actual_float - expected_float).abs()
        flat_index = int(abs_diff.argmax().item())
        max_index = tuple(
            int(index)
            for index in torch.unravel_index(
                torch.tensor(flat_index), abs_diff.shape
            )
        )
        max_abs = float(abs_diff.flatten()[flat_index].item())
        denominator = expected_float.abs().clamp_min(1e-12)
        max_rel = float((abs_diff / denominator).max().item())
        print(
            f"[Mismatch] {name}: shape={tuple(actual.shape)}, index={max_index}, "
            f"max_abs={max_abs:.8g}, "
            f"max_rel={max_rel:.8g}, actual={float(actual_float.flatten()[flat_index]):.8g}, "
            f"expected={float(expected_float.flatten()[flat_index]):.8g}",
            flush=True,
        )
        raise


def _snapshot_fwd_h_outputs(outputs):
    torch.npu.synchronize()
    snapshots = tuple(output.detach().cpu().contiguous() for output in outputs)
    for name, output in zip(("h", "v_new", "final_state"), snapshots):
        assert torch.isfinite(output.float()).all().item(), f"{name} contains NaN or Inf"
    return snapshots


def _assert_fwd_h_outputs_bitwise_equal(expected, actual, repeat):
    for name, expected_output, actual_output in zip(
        ("h", "v_new", "final_state"), expected, actual
    ):
        same_metadata = (
            expected_output.shape == actual_output.shape
            and expected_output.dtype == actual_output.dtype
        )
        same_bits = same_metadata and torch.equal(
            expected_output.view(torch.uint8), actual_output.view(torch.uint8)
        )
        assert same_bits, f"repeat={repeat} output={name} is not bitwise deterministic"


def _bsnd_intermediate_to_bnsd(tensor):
    return tensor.permute(0, 2, 1, *range(3, tensor.ndim))


def _kda_gate_cumsum_reference(g, chunk_size, A_log=None, dt_bias=None, cu_seqlens=None,
                               use_gate_in_kernel=False, safe_gate=False, lower_bound=-5.0):
    """Reference for the standalone fixed head-major BNSD/NTD gate operator."""
    rcp_ln2 = 1.4426950408889634
    g_float = g.to(torch.float32)
    if use_gate_in_kernel:
        x = g_float
        if dt_bias is not None:
            bias = dt_bias.reshape(g.shape[-3], g.shape[-1]).to(torch.float32)
            if g.dim() == 4:
                x = x + bias[None, :, None, :]
            else:
                x = x + bias[:, None, :]
        a = torch.exp(A_log.to(torch.float32))
        if safe_gate:
            if g.dim() == 4:
                x = x * a[None, :, None, None]
            else:
                x = x * a[:, None, None]
            gate = float(lower_bound) * torch.sigmoid(x)
        else:
            if g.dim() == 4:
                gate = -a[None, :, None, None] * torch.nn.functional.softplus(x)
            else:
                gate = -a[:, None, None] * torch.nn.functional.softplus(x)
    else:
        gate = g_float

    out = torch.empty_like(gate, dtype=torch.float32)
    if cu_seqlens is not None:
        cu = cu_seqlens.detach().cpu().tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
        if g.dim() == 4:
            for seq_idx in range(len(cu) - 1):
                seq_start, seq_end = int(cu[seq_idx]), int(cu[seq_idx + 1])
                for start in range(seq_start, seq_end, chunk_size):
                    end = min(start + chunk_size, seq_end)
                    out[0, :, start:end] = torch.cumsum(
                        gate[0, :, start:end] * rcp_ln2, dim=1
                    )
        else:
            for seq_idx in range(len(cu) - 1):
                seq_start, seq_end = int(cu[seq_idx]), int(cu[seq_idx + 1])
                for start in range(seq_start, seq_end, chunk_size):
                    end = min(start + chunk_size, seq_end)
                    out[:, start:end] = torch.cumsum(
                        gate[:, start:end] * rcp_ln2, dim=1
                    )
        return out

    if g.dim() == 4:
        for b in range(g.shape[0]):
            for start in range(0, g.shape[2], chunk_size):
                end = min(start + chunk_size, g.shape[2])
                out[b, :, start:end] = torch.cumsum(
                    gate[b, :, start:end] * rcp_ln2, dim=1
                )
    else:
        for start in range(0, g.shape[1], chunk_size):
            end = min(start + chunk_size, g.shape[1])
            out[:, start:end] = torch.cumsum(
                gate[:, start:end] * rcp_ln2, dim=1
            )
    return out


def _run_chunk_kda_fwd_model_shape_with_stats(
    dump_path=None,
    device_id=None,
    use_initial_state=True,
):
    device = _device(device_id)
    if device.type == "cpu":
        print("skip model-shape stats test on CPU")
        return

    dump_path = pathlib.Path(dump_path) if dump_path is not None else MODEL_SHAPE_DUMP
    if not dump_path.exists():
        dump_path = _make_model_shape_kda_dump(dump_path=dump_path)

    torch.npu.set_device(device.index or 0)
    data = torch.load(dump_path, map_location="cpu")

    q = data["q"].to(device, non_blocking=True)
    k = data["k"].to(device, non_blocking=True)
    v = data["v"].to(device, non_blocking=True)
    g = data["g"].to(device, non_blocking=True)
    beta = data["beta"].to(device, non_blocking=True)
    a_log = data["A_log"].to(device, non_blocking=True, dtype=torch.float32)
    dt_bias = data["dt_bias"].to(device, non_blocking=True, dtype=torch.float32)
    cu_seqlens_tensor = data["cu_seqlens"].to(device, non_blocking=True)
    cu_seqlens = cu_seqlens_tensor.tolist()
    initial_state = (
        data["initial_state"].to(device, non_blocking=True)
        if use_initial_state
        else None
    )

    scale = q.shape[-1] ** -0.5
    chunk_size = int(data.get("chunk_size", MODEL_SHAPE_CASE["chunk_size"]))
    safe_gate = data.get("safe_gate", True)
    lower_bound = float(data.get("lower_bound", -5.0))

    print("\n" + "=" * 80)
    initial_state_mode = "tensor" if use_initial_state else "None"
    print(
        f"=== KDA Forward Model-Shape Guard "
        f"(device={device}, initial_state={initial_state_mode}) ==="
    )
    print("=" * 80)
    print(f"[Meta] scale={scale:.6f}, chunk_size={chunk_size}")
    print(f"[Meta] cu_seqlens={cu_seqlens}")
    print(f"[Meta] safe_gate={safe_gate}, lower_bound={lower_bound}")
    print(f"[Meta] initial_state={initial_state_mode}")
    print(f"[Meta] dump_path={dump_path}")

    print("\n--- Input Statistics ---")
    input_tensors = [
        ("q", q),
        ("k", k),
        ("v", v),
        ("g_raw", g),
        ("beta", beta),
        ("A_log", a_log),
        ("dt_bias", dt_bias),
    ]
    if initial_state is not None:
        input_tensors.append(("initial_state", initial_state))
    for name, tensor in input_tensors:
        _print_stat(_stat(tensor, name), "  ")

    q, q_rstd = _l2norm_fwd_torch(q)
    k, k_rstd = _l2norm_fwd_torch(k)
    print("\n--- After L2Norm ---")
    _print_stat(_stat(q, "q_norm"), "  ")
    _print_stat(_stat(k, "k_norm"), "  ")
    _print_stat(_stat(q_rstd, "q_rstd"), "  ")
    _print_stat(_stat(k_rstd, "k_rstd"), "  ")

    print("\n--- Gate Cumsum ---")
    g_head = g.permute(0, 2, 1, 3).contiguous()
    gk = fla_ascendc.kda_gate_cumsum(
        g_head,
        chunk_size,
        A_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )
    ref_gk = _kda_gate_cumsum_reference(
        g_head.detach().cpu(),
        chunk_size,
        A_log=a_log.detach().cpu(),
        dt_bias=dt_bias.detach().cpu(),
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int64),
        use_gate_in_kernel=True,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )
    _print_stat(_stat(gk, "gk_npu"), "  ")
    _print_stat(_stat(ref_gk, "gk_ref"), "  ")
    _assert_close("model shape gate cumsum", gk, ref_gk, rtol=2e-3, atol=2e-3)

    q_ntd = q.squeeze(0).permute(1, 0, 2).contiguous()
    k_ntd = k.squeeze(0).permute(1, 0, 2).contiguous()
    v_ntd = v.squeeze(0).permute(1, 0, 2).contiguous()
    gk_ntd = gk.squeeze(0).contiguous()
    beta_nt = beta.squeeze(0).permute(1, 0).contiguous()

    print("\n--- Chunk KDA Forward ---", flush=True)
    torch.npu.synchronize()
    kda_start = time.perf_counter()
    got = _chunk_kda_fwd_from_gk(
        q_ntd,
        k_ntd,
        v_ntd,
        gk_ntd,
        beta_nt,
        scale,
        chunk_size,
        layout="NTD",
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        disable_recompute=True, return_intermediate_states=True,
        safe_gate=safe_gate,
    )
    torch.npu.synchronize()
    print(
        f"[Timing] Chunk KDA Forward synchronized wall: "
        f"{(time.perf_counter() - kda_start) * 1000.0:.3f} ms",
        flush=True,
    )
    o_npu, final_state_npu = got[0], got[1]
    print("--- Chunk KDA Output Statistics ---", flush=True)
    for name, tensor in [
        ("o_npu", o_npu),
        ("final_state_npu", final_state_npu),
        ("g_out", got[2]),
        ("Aqk_npu", got[3]),
        ("Akk_npu", got[4]),
        ("w_npu", got[5]),
        ("u_npu", got[6]),
        ("qg_npu", got[7]),
        ("kg_npu", got[8]),
        ("v_new_npu", got[9]),
        ("h_npu", got[10]),
        ("initial_state_out", got[11]),
    ]:
        if tensor is None:
            continue
        if tensor.numel() == 0:
            continue
        stat = _stat(tensor, name)
        _print_stat(stat, "  ")
        if stat["has_nan"] or stat["has_inf"]:
            _print_first_nonfinite(tensor, name, "  ")
    assert torch.isfinite(o_npu).all().item(), "model shape o contains NaN or Inf"
    assert torch.isfinite(final_state_npu).all().item(), "model shape final_state contains NaN or Inf"
    if initial_state is None:
        assert got[11] is None, "initial_state passthrough must be None when initial_state is None"
    else:
        _assert_close("model shape initial_state_out", got[11], initial_state, rtol=0, atol=0)

    previous_num_threads = torch.get_num_threads()
    reference_num_threads = max(1, min(REFERENCE_NUM_THREADS, previous_num_threads))
    torch.set_num_threads(reference_num_threads)
    print(f"\n--- CPU Reference (threads={reference_num_threads}) ---", flush=True)
    reference_start = time.perf_counter()
    try:
        ref = chunk_kda_forward_reference(
            q.detach().cpu(),
            k.detach().cpu(),
            v.detach().cpu(),
            ref_gk.permute(0, 2, 1, 3).contiguous().cpu(),
            beta.detach().cpu(),
            scale=scale,
            chunk_size=chunk_size,
            initial_state=None if initial_state is None else initial_state.detach().cpu(),
            output_final_state=True,
            cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int64),
        )
    finally:
        torch.set_num_threads(previous_num_threads)
    print(
        f"[Timing] CPU Reference wall: {time.perf_counter() - reference_start:.3f} s",
        flush=True,
    )
    ref_o_tnd = ref.o.squeeze(0).contiguous()
    _print_stat(_stat(ref_o_tnd, "o_ref"), "  ")
    _print_stat(_stat(ref.final_state, "final_state_ref"), "  ")
    assert torch.isfinite(ref_o_tnd).all().item(), "model shape reference o contains NaN or Inf"
    assert torch.isfinite(ref.final_state).all().item(), "model shape reference final_state contains NaN or Inf"

    o_diff = (o_npu.detach().cpu() - ref_o_tnd).abs()
    fs_diff = (final_state_npu.detach().cpu() - ref.final_state).abs()
    print("\n--- Diff (NPU vs CPU Ref) ---")
    _print_stat(_stat(o_diff, "o_abs_diff"), "  ")
    _print_stat(_stat(fs_diff, "fs_abs_diff"), "  ")

    _assert_close("model shape o", o_npu.detach().cpu(), ref_o_tnd, rtol=5e-2, atol=5e-2)
    _assert_close(
        "model shape final_state",
        final_state_npu.detach().cpu(),
        ref.final_state,
        rtol=5e-2,
        atol=5e-2,
    )


def test_chunk_kda_fwd_model_shape_with_stats(dump_path=None, device_id=None):
    _run_chunk_kda_fwd_model_shape_with_stats(
        dump_path=dump_path,
        device_id=device_id,
        use_initial_state=True,
    )


def test_chunk_kda_fwd_model_shape_initial_state_none_with_stats(dump_path=None, device_id=None):
    _run_chunk_kda_fwd_model_shape_with_stats(
        dump_path=dump_path,
        device_id=device_id,
        use_initial_state=False,
    )


def test_chunk_kda_fwd_from_dump_with_stats(dump_path: str, device_id=None):
    test_chunk_kda_fwd_model_shape_with_stats(dump_path=dump_path, device_id=device_id)


test_chunk_kda_fwd_from_dump_with_stats.__test__ = False


def test_chunk_kda_fwd_matches_reference():
    device = _device()
    q, k, v, gk, beta, initial_state = _make_inputs(device, h=1, hv=1, t=64)
    scale = q.shape[-1] ** -0.5

    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )

    for safe_gate in (False, True):
        mode = f"safe_gate={safe_gate}"
        got = _chunk_kda_fwd_from_gk(
            q,
            k,
            v,
            gk,
            beta,
            scale,
            64,
            layout="BSND",
            initial_state=initial_state,
            output_final_state=True,
            disable_recompute=True, return_intermediate_states=True,
            safe_gate=safe_gate,
        )
        for name, tensor in zip(
            ("o", "final_state", "g", "Aqk", "Akk", "w", "u", "qg", "kg", "v_new", "h"),
            got[:11],
        ):
            assert torch.isfinite(tensor).all().item(), f"{mode} {name} contains NaN or Inf"
        _assert_close(f"{mode} o", got[0], ref.o, rtol=2e-2, atol=2e-2)
        _assert_close(f"{mode} final_state", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
        _assert_close(f"{mode} g", got[2], _bsnd_intermediate_to_bnsd(gk))
        _assert_close(f"{mode} Aqk", got[3], _bsnd_intermediate_to_bnsd(ref.Aqk), rtol=2e-2, atol=2e-2)
        _assert_close(f"{mode} Akk", got[4], _bsnd_intermediate_to_bnsd(ref.Akk), rtol=2e-2, atol=2e-2)
        _assert_close(f"{mode} w", got[5], _bsnd_intermediate_to_bnsd(ref.w), rtol=2e-2, atol=2e-2)
        _assert_close(f"{mode} u", got[6], _bsnd_intermediate_to_bnsd(ref.u), rtol=2e-2, atol=2e-2)
        _assert_close(f"{mode} qg", got[7], _bsnd_intermediate_to_bnsd(ref.qg), rtol=2e-2, atol=2e-2)
        _assert_close(f"{mode} kg", got[8], _bsnd_intermediate_to_bnsd(ref.kg), rtol=2e-2, atol=2e-2)
        _assert_close(f"{mode} v_new", got[9], _bsnd_intermediate_to_bnsd(ref.v_new), rtol=2e-2, atol=2e-2)
        _assert_close(f"{mode} h", got[10], ref.h, rtol=2e-2, atol=2e-2)
        _assert_close(f"{mode} initial_state", got[11], initial_state)


def test_chunk_kda_fwd_upper_triangle_dirty_zero():
    device = _device()
    if device.type == "cpu":
        return

    b, t, h, hv, kdim, vdim = 1, 64, 1, 1, 128, 128
    torch.manual_seed(20260713)
    q = (torch.randn(b, t, h, kdim, dtype=torch.bfloat16) * 0.02).to(device)
    k = (torch.randn(b, t, h, kdim, dtype=torch.bfloat16) * 0.02).to(device)
    v = (torch.randn(b, t, hv, vdim, dtype=torch.bfloat16) * 0.02).to(device)
    g_step = torch.full((b, t, hv, kdim), -460.0 / t, dtype=torch.float32)
    gk = torch.cumsum(g_step, dim=1).to(device)
    beta = torch.sigmoid(torch.randn(b, t, hv, dtype=torch.float32)).to(device)
    initial_state = torch.zeros(b, hv, kdim, vdim, dtype=torch.float32).to(device)
    scale = kdim ** -0.5

    got = _chunk_kda_fwd_from_gk(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        64,
        layout="BSND",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=True, return_intermediate_states=True,
        safe_gate=True,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )

    for name, tensor in zip(("o", "final_state", "Aqk", "Akk", "w", "u", "v_new"),
                            got[:2] + got[3:7] + got[9:10]):
        assert torch.isfinite(tensor).all().item(), f"{name} contains NaN or Inf"
    for name, tensor in (
        ("ref_o", ref.o),
        ("ref_final_state", ref.final_state),
        ("ref_Aqk", ref.Aqk),
        ("ref_Akk", ref.Akk),
        ("ref_w", ref.w),
        ("ref_u", ref.u),
        ("ref_v_new", ref.v_new),
    ):
        assert torch.isfinite(tensor).all().item(), f"{name} contains NaN or Inf"

    upper = torch.triu(torch.ones(t, t, dtype=torch.bool), diagonal=1)
    diag = torch.arange(t)
    aqk_npu = got[3].detach().float().cpu()[0, 0, :, :]
    akk_npu = got[4].detach().float().cpu()[0, 0, :, :]
    aqk_ref = ref.Aqk.detach().float()[0, :, 0, :]
    akk_ref = ref.Akk.detach().float()[0, :, 0, :]

    assert (aqk_npu[upper] == 0).all().item()
    assert (akk_npu[upper] == 0).all().item()
    assert (aqk_ref[upper] == 0).all().item()
    assert (akk_ref[upper] == 0).all().item()
    torch.testing.assert_close(akk_npu[diag, diag], torch.ones(t), rtol=0, atol=0)
    torch.testing.assert_close(akk_ref[diag, diag], torch.ones(t), rtol=0, atol=0)
    for name, actual, expected in (
        ("safe gate o", got[0], ref.o),
        ("safe gate final_state", got[1], ref.final_state),
        ("safe gate Aqk", got[3], _bsnd_intermediate_to_bnsd(ref.Aqk)),
        ("safe gate Akk", got[4], _bsnd_intermediate_to_bnsd(ref.Akk)),
        ("safe gate w", got[5], _bsnd_intermediate_to_bnsd(ref.w)),
        ("safe gate u", got[6], _bsnd_intermediate_to_bnsd(ref.u)),
        ("safe gate qg", got[7], _bsnd_intermediate_to_bnsd(ref.qg)),
        ("safe gate kg", got[8], _bsnd_intermediate_to_bnsd(ref.kg)),
        ("safe gate v_new", got[9], _bsnd_intermediate_to_bnsd(ref.v_new)),
    ):
        _assert_close(name, actual, expected, rtol=2e-2, atol=2e-2)


def test_chunk_kda_fwd_vdim256_matches_reference():
    device = _device()
    if device.type == "cpu":
        return

    for dtype in (torch.bfloat16, torch.float16):
        q, k, v, gk, beta, initial_state = _make_inputs(
            device, b=1, h=1, hv=2, t=128, kdim=128, vdim=256, dtype=dtype,
        )
        scale = q.shape[-1] ** -0.5
        got = _chunk_kda_fwd_from_gk(
            q,
            k,
            v,
            gk,
            beta,
            scale,
            64,
            layout="BSND",
            initial_state=initial_state,
            output_final_state=True,
            disable_recompute=True, return_intermediate_states=True,
        )
        repeated = _chunk_kda_fwd_from_gk(
            q,
            k,
            v,
            gk,
            beta,
            scale,
            64,
            layout="BSND",
            initial_state=initial_state,
            output_final_state=True,
            disable_recompute=True, return_intermediate_states=True,
        )
        assert torch.equal(got[1], repeated[1]), f"{dtype} V256 final_state must be deterministic"
        ref = chunk_kda_forward_reference(
            q.detach().cpu(),
            k.detach().cpu(),
            v.detach().cpu(),
            gk.detach().cpu(),
            beta.detach().cpu(),
            scale=scale,
            chunk_size=64,
            initial_state=initial_state.detach().cpu(),
            output_final_state=True,
        )
        dtype_name = str(dtype).removeprefix("torch.")
        for name, actual, expected in (
            ("o", got[0], ref.o),
            ("final_state", got[1], ref.final_state),
            ("Aqk", got[3], _bsnd_intermediate_to_bnsd(ref.Aqk)),
            ("Akk", got[4], _bsnd_intermediate_to_bnsd(ref.Akk)),
            ("w", got[5], _bsnd_intermediate_to_bnsd(ref.w)),
            ("u", got[6], _bsnd_intermediate_to_bnsd(ref.u)),
            ("qg", got[7], _bsnd_intermediate_to_bnsd(ref.qg)),
            ("kg", got[8], _bsnd_intermediate_to_bnsd(ref.kg)),
            ("v_new", got[9], _bsnd_intermediate_to_bnsd(ref.v_new)),
            ("h", got[10], ref.h),
        ):
            assert torch.isfinite(actual).all().item(), f"{dtype_name} V256 {name} contains NaN or Inf"
            _assert_close(f"{dtype_name} V256 {name}", actual, expected, rtol=2e-2, atol=2e-2)


def test_chunk_kda_fwd_chunk128_matches_reference():
    device = _device()
    if device.type == "cpu":
        return

    cases = [values for _, values in legacy_param_values(
        "chunk_kda_fwd", "chunk128_backend", dtype_module=torch
    )]
    for dtype, t, vdim in cases:
        q, k, v, gk, beta, initial_state = _make_inputs(
            device, b=1, h=1, hv=2, t=t, kdim=128, vdim=vdim, dtype=dtype,
        )
        scale = q.shape[-1] ** -0.5
        got = _chunk_kda_fwd_from_gk(
            q,
            k,
            v,
            gk,
            beta,
            scale,
            128,
            layout="BSND",
            initial_state=initial_state,
            output_final_state=True,
            disable_recompute=True, return_intermediate_states=True,
        )
        ref = chunk_kda_forward_reference(
            q.detach().cpu(),
            k.detach().cpu(),
            v.detach().cpu(),
            gk.detach().cpu(),
            beta.detach().cpu(),
            scale=scale,
            chunk_size=128,
            initial_state=initial_state.detach().cpu(),
            output_final_state=True,
        )
        case_name = f"{str(dtype).removeprefix('torch.')} T{t} V{vdim} C128"
        for name, actual, expected in (
            ("o", got[0], ref.o),
            ("final_state", got[1], ref.final_state),
            ("Aqk", got[3], _bsnd_intermediate_to_bnsd(ref.Aqk)),
            ("Akk", got[4], _bsnd_intermediate_to_bnsd(ref.Akk)),
            ("w", got[5], _bsnd_intermediate_to_bnsd(ref.w)),
            ("u", got[6], _bsnd_intermediate_to_bnsd(ref.u)),
            ("qg", got[7], _bsnd_intermediate_to_bnsd(ref.qg)),
            ("kg", got[8], _bsnd_intermediate_to_bnsd(ref.kg)),
            ("v_new", got[9], _bsnd_intermediate_to_bnsd(ref.v_new)),
            ("h", got[10], ref.h),
        ):
            assert torch.isfinite(actual).all().item(), f"{case_name} {name} contains NaN or Inf"
            _assert_close(f"{case_name} {name}", actual, expected, rtol=2e-2, atol=2e-2)


def test_chunk_kda_fwd_state_v_first_modes_sequence_major_h_match_reference():
    device = _device()
    if device.type == "cpu":
        return

    q, k, v, gk, beta, initial_state = _make_inputs(
        device, b=1, h=1, hv=2, t=96, kdim=128, vdim=256, dtype=torch.float16,
    )
    scale = q.shape[-1] ** -0.5
    initial_state_v_first = initial_state.transpose(-1, -2).contiguous()
    common_args = (
        q,
        k,
        v,
        gk,
        beta,
        scale,
        64,
    )
    common_kwargs = {
        "layout": "BSND",
        "output_final_state": True,
        "disable_recompute": True,
        "return_intermediate_states": True,
    }
    got_default = _chunk_kda_fwd_from_gk(
        *common_args,
        initial_state=initial_state,
        **common_kwargs,
    )
    got_false = _chunk_kda_fwd_from_gk(
        *common_args,
        initial_state=initial_state,
        state_v_first=False,
        **common_kwargs,
    )
    got_true = _chunk_kda_fwd_from_gk(
        *common_args,
        initial_state=initial_state_v_first,
        state_v_first=True,
        **common_kwargs,
    )
    for index, (default_output, false_output) in enumerate(zip(got_default, got_false)):
        if default_output is None:
            assert false_output is None
        else:
            assert torch.equal(default_output, false_output), (
                f"default state_v_first must equal explicit false for output {index}"
            )

    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )

    assert tuple(got_false[1].shape) == (1, 2, 128, 256)
    assert tuple(got_false[10].shape) == (1, 2, 2, 128, 256)
    for name, actual, expected in (
        ("state_v_first=false o", got_false[0], ref.o),
        ("state_v_first=false final_state", got_false[1], ref.final_state),
        ("state_v_first=false h", got_false[10], ref.h),
        ("state_v_first=false initial_state", got_false[11], initial_state),
    ):
        assert torch.isfinite(actual).all().item(), f"{name} contains NaN or Inf"
        _assert_close(name, actual, expected, rtol=2e-2, atol=2e-2)

    assert tuple(got_true[1].shape) == (1, 2, 256, 128)
    assert tuple(got_true[10].shape) == (1, 2, 2, 256, 128)
    for name, actual, expected in (
        ("state_v_first=true o", got_true[0], ref.o),
        ("state_v_first=true final_state", got_true[1], ref.final_state.transpose(-1, -2)),
        ("state_v_first=true g", got_true[2], _bsnd_intermediate_to_bnsd(gk)),
        ("state_v_first=true Aqk", got_true[3], _bsnd_intermediate_to_bnsd(ref.Aqk)),
        ("state_v_first=true Akk", got_true[4], _bsnd_intermediate_to_bnsd(ref.Akk)),
        ("state_v_first=true w", got_true[5], _bsnd_intermediate_to_bnsd(ref.w)),
        ("state_v_first=true u", got_true[6], _bsnd_intermediate_to_bnsd(ref.u)),
        ("state_v_first=true qg", got_true[7], _bsnd_intermediate_to_bnsd(ref.qg)),
        ("state_v_first=true kg", got_true[8], _bsnd_intermediate_to_bnsd(ref.kg)),
        ("state_v_first=true v_new", got_true[9], _bsnd_intermediate_to_bnsd(ref.v_new)),
        ("state_v_first=true h", got_true[10], ref.h.transpose(-1, -2)),
        ("state_v_first=true initial_state", got_true[11], initial_state_v_first),
    ):
        assert torch.isfinite(actual).all().item(), f"{name} contains NaN or Inf"
        _assert_close(name, actual, expected, rtol=2e-2, atol=2e-2)


def test_chunk_kda_fwd_bsnd_export_dependency_matches_reference():
    device = _device()
    if device.type == "cpu":
        return

    q, k, v, gk, beta, initial_state = _make_inputs(
        device, b=1, h=4, hv=4, t=1024, kdim=128, vdim=128, dtype=torch.bfloat16,
    )
    scale = q.shape[-1] ** -0.5
    got = _chunk_kda_fwd_from_gk(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        64,
        layout="BSND",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=True, return_intermediate_states=True,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )
    for name, actual, expected in (
        ("o", got[0], ref.o),
        ("w", got[5], _bsnd_intermediate_to_bnsd(ref.w)),
        ("v_new", got[9], _bsnd_intermediate_to_bnsd(ref.v_new)),
        ("h", got[10], ref.h),
    ):
        assert torch.isfinite(actual).all().item(), f"BSND dependency {name} contains NaN or Inf"
        _assert_close(f"BSND dependency {name}", actual, expected, rtol=2e-2, atol=2e-2)


def test_chunk_kda_fwd_without_intermediate_matches_export_and_reference():
    device = _device()
    if device.type == "cpu":
        return

    q, k, v, gk, beta, _ = _make_inputs(
        device, b=1, h=4, hv=4, t=1024, kdim=128, vdim=128, dtype=torch.bfloat16,
    )
    scale = q.shape[-1] ** -0.5
    got_without = _chunk_kda_fwd_from_gk(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        64,
        layout="BSND",
        initial_state=None,
        output_final_state=False,
        disable_recompute=False, return_intermediate_states=False,
    )
    got_with = _chunk_kda_fwd_from_gk(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        64,
        layout="BSND",
        initial_state=None,
        output_final_state=False,
        disable_recompute=True, return_intermediate_states=True,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=None,
        output_final_state=False,
    )
    assert torch.isfinite(got_without[0]).all().item()
    assert torch.isfinite(got_with[0]).all().item()
    assert all(torch.is_tensor(value) for value in got_without[2:5])
    assert all(value is None for value in got_without[5:12])
    assert all(torch.is_tensor(value) for value in got_with[3:11])
    _assert_close("BSND no intermediate o", got_without[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("BSND exported intermediate o", got_with[0], ref.o, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(got_without[0], got_with[0], rtol=0, atol=0)


def test_chunk_kda_fwd_optional_output_matrix():
    device = _device()
    if device.type == "cpu":
        return

    manifest = json.loads(
        (ROOT / "tests/op_cases/chunk_kda_fwd.json").read_text(encoding="utf-8")
    )
    assert manifest["upstream_alignment"]["commit"] == FLA_ORG_KDA_FWD_ALIGNMENT_COMMIT
    case = next(
        item
        for item in manifest["cases"]
        if item["id"] == "chunk_kda_fwd_optional_output_matrix"
    )
    shape = case["shape"]
    attrs = case["attrs"]
    matrix = attrs["optional_output_matrix"]
    combinations = list(
        itertools.product(
            matrix["output_final_state"],
            matrix["use_gate_in_kernel"],
            matrix["disable_recompute"],
            matrix["return_intermediate_states"],
        )
    )
    assert len(combinations) == case["expect"]["matrix_size"] == 16

    b = int(shape["B"])
    h = int(shape["H_k"])
    hv = int(shape["H_v"])
    t = int(shape["T"])
    kdim = int(shape["K"])
    vdim = int(shape["V"])
    chunk_size = int(shape["chunk_size"])
    scale = float(attrs["scale"])
    torch.manual_seed(int(case["seed"]))
    q_bsnd = (torch.randn(b, t, h, kdim) * 0.04).to(torch.float16)
    k_bsnd = (torch.randn(b, t, h, kdim) * 0.04).to(torch.float16)
    v_bsnd = (torch.randn(b, t, hv, vdim) * 0.04).to(torch.float16)
    raw_bsnd = torch.randn(b, t, hv, kdim, dtype=torch.float32) * 0.2 - 1.0
    beta_bsnd = torch.sigmoid(torch.randn(b, t, hv, dtype=torch.float32))
    a_log_cpu = torch.randn(hv, dtype=torch.float32) * 0.05
    dt_bias_cpu = torch.randn(hv * kdim, dtype=torch.float32) * 0.1
    activated_bsnd = -torch.exp(a_log_cpu)[None, None, :, None] * torch.nn.functional.softplus(
        raw_bsnd + dt_bias_cpu.reshape(1, 1, hv, kdim)
    )
    gk_bsnd = torch.empty_like(activated_bsnd)
    rcp_ln2 = 1.0 / math.log(2.0)
    for start in range(0, t, chunk_size):
        end = min(start + chunk_size, t)
        gk_bsnd[:, start:end] = torch.cumsum(
            activated_bsnd[:, start:end] * rcp_ln2, dim=1
        )
    initial_cpu = torch.randn(b, hv, kdim, vdim, dtype=torch.float32) * 0.01
    reference = chunk_kda_forward_reference(
        q_bsnd,
        k_bsnd,
        v_bsnd,
        gk_bsnd,
        beta_bsnd,
        scale=scale,
        chunk_size=chunk_size,
        initial_state=initial_cpu,
        output_final_state=True,
    )

    q = q_bsnd.permute(0, 2, 1, 3).contiguous().to(device)
    k = k_bsnd.permute(0, 2, 1, 3).contiguous().to(device)
    v = v_bsnd.permute(0, 2, 1, 3).contiguous().to(device)
    raw = raw_bsnd.permute(0, 2, 1, 3).contiguous().to(device)
    activated = activated_bsnd.permute(0, 2, 1, 3).contiguous().to(device)
    beta = beta_bsnd.permute(0, 2, 1).contiguous().to(device)
    a_log = a_log_cpu.to(device)
    dt_bias = dt_bias_cpu.to(device)
    initial_state = initial_cpu.to(device)

    baselines = {}
    for use_gate_in_kernel in (False, True):
        baselines[use_gate_in_kernel] = fla_ascendc.chunk_kda_fwd(
            q,
            k,
            v,
            raw if use_gate_in_kernel else activated,
            beta,
            scale,
            chunk_size,
            layout="BNSD",
            initial_state=initial_state,
            output_final_state=True,
            safe_gate=False,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=a_log if use_gate_in_kernel else None,
            dt_bias=dt_bias if use_gate_in_kernel else None,
            disable_recompute=True,
            return_intermediate_states=True,
        )

    reference_outputs = (
        reference.o,
        reference.final_state,
        _bsnd_intermediate_to_bnsd(gk_bsnd),
        _bsnd_intermediate_to_bnsd(reference.Aqk),
        _bsnd_intermediate_to_bnsd(reference.Akk),
        _bsnd_intermediate_to_bnsd(reference.w),
        _bsnd_intermediate_to_bnsd(reference.u),
        _bsnd_intermediate_to_bnsd(reference.qg),
        _bsnd_intermediate_to_bnsd(reference.kg),
        _bsnd_intermediate_to_bnsd(reference.v_new),
        reference.h,
    )
    for use_gate_in_kernel, baseline in baselines.items():
        assert baseline[11] is initial_state
        baseline_mask = kda_fwd_optional_output_mask(
            output_final_state=True,
            use_gate_in_kernel=use_gate_in_kernel,
            disable_recompute=True,
            return_intermediate_states=True,
        )
        for index, expected in enumerate(reference_outputs):
            if baseline_mask[index]:
                _assert_close(
                    f"optional matrix baseline use_gate={use_gate_in_kernel} output={index}",
                    baseline[index],
                    expected,
                    rtol=2e-2,
                    atol=2e-2,
                )
            else:
                assert baseline[index] is None

    for output_final_state, use_gate_in_kernel, disable_recompute, return_states in combinations:
        outputs = fla_ascendc.chunk_kda_fwd(
            q,
            k,
            v,
            raw if use_gate_in_kernel else activated,
            beta,
            scale,
            chunk_size,
            layout="BNSD",
            initial_state=initial_state,
            output_final_state=output_final_state,
            safe_gate=False,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=a_log if use_gate_in_kernel else None,
            dt_bias=dt_bias if use_gate_in_kernel else None,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_states,
        )
        expected_mask = kda_fwd_optional_output_mask(
            output_final_state=output_final_state,
            use_gate_in_kernel=use_gate_in_kernel,
            disable_recompute=disable_recompute,
            return_intermediate_states=return_states,
        )
        assert len(outputs) == len(expected_mask) == 12
        assert outputs[11] is initial_state
        baseline = baselines[use_gate_in_kernel]
        for index, (output, visible) in enumerate(zip(outputs, expected_mask)):
            if not visible:
                assert output is None, (
                    output_final_state,
                    use_gate_in_kernel,
                    disable_recompute,
                    return_states,
                    index,
                )
            else:
                assert torch.is_tensor(output)
                try:
                    torch.testing.assert_close(output, baseline[index], rtol=0, atol=0)
                except AssertionError as error:
                    raise AssertionError(
                        "optional output changed with visibility policy: "
                        f"output_final_state={output_final_state}, "
                        f"use_gate_in_kernel={use_gate_in_kernel}, "
                        f"disable_recompute={disable_recompute}, "
                        f"return_intermediate_states={return_states}, "
                        f"output_index={index}"
                    ) from error


def test_chunk_kda_fwd_varlen_initial_state_shape_rejected():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, bad_initial_state = _make_inputs(device, h=1, hv=2, t=16)
    scale = q.shape[-1] ** -0.5
    cu_seqlens = [0, 6, 16]
    try:
        _chunk_kda_fwd_from_gk(
            q,
            k,
            v,
            gk,
            beta,
            scale,
            64,
            layout="BSND",
            initial_state=bad_initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
    except RuntimeError as exc:
        assert "initial_state shape/dtype does not match state_v_first" in str(exc)
    else:
        raise AssertionError("varlen initial_state with mismatched seq_num should be rejected")


def test_chunk_kda_fwd_bf16_chunk32_rejected_as_unsupported():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(device, h=1, hv=1, t=8, dtype=torch.bfloat16)
    scale = q.shape[-1] ** -0.5
    try:
        _chunk_kda_fwd_from_gk(
            q,
            k,
            v,
            gk,
            beta,
            scale,
            32,
            layout="BSND",
            initial_state=initial_state,
            output_final_state=True,
            disable_recompute=False, return_intermediate_states=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("chunk_size outside {64, 128} must be rejected as an unsupported KDA forward path")


def test_chunk_kda_fwd_bf16_gate_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(
        device, h=1, hv=1, t=64, dtype=torch.float16
    )
    gk_bf16 = gk.detach().to(torch.bfloat16).requires_grad_(True)
    beta_bf16 = beta.detach().to(torch.bfloat16).requires_grad_(True)
    scale = q.shape[-1] ** -0.5

    got = _chunk_kda_fwd_from_gk(
        q,
        k,
        v,
        gk_bf16,
        beta_bf16,
        scale,
        64,
        layout="BSND",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=False, return_intermediate_states=False,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk_bf16.detach().cpu().float(),
        beta_bf16.detach().cpu().float(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )
    _assert_close("o bf16 gate", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("final_state bf16 gate", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g bf16 gate", got[2], _bsnd_intermediate_to_bnsd(gk))
    _assert_close("initial_state bf16 gate", got[11], initial_state)


def test_chunk_kda_fwd_fp16_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(
        device, h=1, hv=1, t=8, dtype=torch.float16
    )
    scale = q.shape[-1] ** -0.5

    got = _chunk_kda_fwd_from_gk(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        64,
        layout="BSND",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=True, return_intermediate_states=True,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )
    _assert_close("o fp16", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("final_state fp16", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g fp16", got[2], _bsnd_intermediate_to_bnsd(gk))
    _assert_close("Aqk fp16", got[3], _bsnd_intermediate_to_bnsd(ref.Aqk), rtol=2e-2, atol=2e-2)
    _assert_close("Akk fp16", got[4], _bsnd_intermediate_to_bnsd(ref.Akk), rtol=2e-2, atol=2e-2)
    _assert_close("w fp16", got[5], _bsnd_intermediate_to_bnsd(ref.w), rtol=2e-2, atol=2e-2)
    _assert_close("u fp16", got[6], _bsnd_intermediate_to_bnsd(ref.u), rtol=2e-2, atol=2e-2)
    _assert_close("qg fp16", got[7], _bsnd_intermediate_to_bnsd(ref.qg), rtol=2e-2, atol=2e-2)
    _assert_close("kg fp16", got[8], _bsnd_intermediate_to_bnsd(ref.kg), rtol=2e-2, atol=2e-2)
    _assert_close("v_new fp16", got[9], _bsnd_intermediate_to_bnsd(ref.v_new), rtol=2e-2, atol=2e-2)
    _assert_close("h fp16", got[10], ref.h, rtol=2e-2, atol=2e-2)
    _assert_close("initial_state fp16", got[11], initial_state)


def test_chunk_kda_fwd_fp16_safe_gate_large_span_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    b, t, h, hv, kdim, vdim = 1, 64, 1, 1, 128, 128
    q = torch.full((b, t, h, kdim), 0.125, dtype=torch.float16, device=device)
    k = torch.full((b, t, h, kdim), 0.125, dtype=torch.float16, device=device)
    torch.manual_seed(20260730)
    v = (torch.randn(b, t, hv, vdim, dtype=torch.float32) * 0.02).half().to(device)
    gate_step = -5.0 / math.log(2.0)
    gate_rows = gate_step * torch.arange(t, dtype=torch.float32, device=device)
    gk = gate_rows.view(1, t, 1, 1).expand(b, t, hv, kdim).contiguous()
    beta = torch.full((b, t, hv), 0.5, dtype=torch.float32, device=device)
    initial_state = torch.zeros((b, hv, kdim, vdim), dtype=torch.float32, device=device)
    scale = kdim ** -0.5

    got = _chunk_kda_fwd_from_gk(
        q,
        k,
        v,
        gk,
        beta,
        scale,
        64,
        layout="BSND",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=True,
        return_intermediate_states=True,
        safe_gate=True,
        lower_bound=-5.0,
        use_gate_in_kernel=False,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )
    _assert_close(
        "Aqk fp16 safe large span",
        got[3],
        _bsnd_intermediate_to_bnsd(ref.Aqk),
        rtol=1e-2,
        atol=3e-3,
    )
    _assert_close(
        "Akk fp16 safe large span",
        got[4],
        _bsnd_intermediate_to_bnsd(ref.Akk),
        rtol=1e-2,
        atol=3e-3,
    )


def test_chunk_kda_fwd_tnd_matches_reference():
    device = _device()
    q, k, v, gk, beta, initial_state = _make_inputs(device, b=1, h=1, hv=2, t=128)
    scale = q.shape[-1] ** -0.5

    got = _chunk_kda_fwd_from_gk(
        q.squeeze(0),
        k.squeeze(0),
        v.squeeze(0),
        gk.squeeze(0),
        beta.squeeze(0),
        scale,
        64,
        layout="TND",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=True, return_intermediate_states=True,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )

    _assert_close("o tnd", got[0], ref.o.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("final_state tnd", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g tnd", got[2], gk.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("Aqk tnd", got[3], ref.Aqk.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("Akk tnd", got[4], ref.Akk.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("w tnd", got[5], ref.w.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("u tnd", got[6], ref.u.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("qg tnd", got[7], ref.qg.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("kg tnd", got[8], ref.kg.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("v_new tnd", got[9], ref.v_new.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("h tnd", got[10], ref.h.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("initial_state tnd", got[11], initial_state)


def test_chunk_kda_fwd_tnd_multi_head_supported():
    device = _device()
    if device.type == "cpu":
        return
    t, h, hv, kdim, vdim = 128, 2, 2, 128, 128
    q, k, v, gk, beta, _ = _make_inputs(
        device, b=1, h=h, hv=hv, t=t, kdim=kdim, vdim=vdim, dtype=torch.float16,
    )
    outputs = _chunk_kda_fwd_from_gk(
        q.squeeze(0),
        k.squeeze(0),
        v.squeeze(0),
        gk.squeeze(0),
        beta.squeeze(0),
        kdim ** -0.5,
        64,
        layout="TND",
        output_final_state=True,
    )
    reference = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=kdim ** -0.5,
        chunk_size=64,
        initial_state=None,
        output_final_state=True,
    )
    assert outputs[0].shape == (t, hv, vdim)
    assert outputs[1].shape == (1, hv, kdim, vdim)
    _assert_close("multi-head TND o", outputs[0], reference.o.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("multi-head TND final_state", outputs[1], reference.final_state, rtol=2e-2, atol=2e-2)


def test_chunk_kda_fwd_lowercase_layout_rejected():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, _ = _make_inputs(device, h=1, hv=1, t=8)
    try:
        _chunk_kda_fwd_from_gk(
            q,
            k,
            v,
            gk,
            beta,
            q.shape[-1] ** -0.5,
            64,
            layout="bsnd",
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "layout must be uppercase" in message
        assert "BSND, BNSD, TND, NTD" in message
    else:
        raise AssertionError("lowercase layout must be rejected")


def test_chunk_kda_fwd_head_num_gt128_rejected():
    device = _device()
    if device.type == "cpu":
        return
    h, hv, t, kdim, vdim = 129, 129, 4, 128, 128
    q = torch.randn(h, t, kdim, device=device, dtype=torch.float16)
    k = torch.randn(h, t, kdim, device=device, dtype=torch.float16)
    v = torch.randn(hv, t, vdim, device=device, dtype=torch.float16)
    gk = torch.randn(hv, t, kdim, device=device, dtype=torch.float32)
    beta = torch.rand(hv, t, device=device, dtype=torch.float32)
    try:
        _chunk_kda_fwd_from_gk(
            q,
            k,
            v,
            gk,
            beta,
            kdim ** -0.5,
            64,
            layout="NTD",
        )
    except RuntimeError as exc:
        assert "0 < H <= HV <= 128 and HV % H == 0" in str(exc)
    else:
        raise AssertionError("head counts greater than 128 must be rejected")


def test_chunk_kda_fwd_bnsd_direct_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(device, b=1, h=1, hv=2, t=16, dtype=torch.float16)
    scale = q.shape[-1] ** -0.5
    q_bnsd = q.permute(0, 2, 1, 3).contiguous()
    k_bnsd = k.permute(0, 2, 1, 3).contiguous()
    v_bnsd = v.permute(0, 2, 1, 3).contiguous()
    gk_bnsd = gk.permute(0, 2, 1, 3).contiguous()
    beta_bns = beta.permute(0, 2, 1).contiguous()

    got = _chunk_kda_fwd_from_gk(
        q_bnsd,
        k_bnsd,
        v_bnsd,
        gk_bnsd,
        beta_bns,
        scale,
        64,
        layout="BNSD",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=True, return_intermediate_states=True,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )

    _assert_close("o bnsd", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("final_state bnsd", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g bnsd", got[2], gk_bnsd, rtol=2e-2, atol=2e-2)
    _assert_close("Aqk bnsd", got[3], ref.Aqk.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("Akk bnsd", got[4], ref.Akk.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("w bnsd", got[5], ref.w.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("u bnsd", got[6], ref.u.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("qg bnsd", got[7], ref.qg.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("kg bnsd", got[8], ref.kg.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("v_new bnsd", got[9], ref.v_new.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("h bnsd", got[10], ref.h, rtol=2e-2, atol=2e-2)
    _assert_close("initial_state bnsd", got[11], initial_state)


def test_chunk_kda_fwd_ntd_direct_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(device, b=1, h=1, hv=2, t=16, dtype=torch.float16)
    scale = q.shape[-1] ** -0.5
    q_ntd = q.squeeze(0).permute(1, 0, 2).contiguous()
    k_ntd = k.squeeze(0).permute(1, 0, 2).contiguous()
    v_ntd = v.squeeze(0).permute(1, 0, 2).contiguous()
    gk_ntd = gk.squeeze(0).permute(1, 0, 2).contiguous()
    beta_nt = beta.squeeze(0).permute(1, 0).contiguous()

    got = _chunk_kda_fwd_from_gk(
        q_ntd,
        k_ntd,
        v_ntd,
        gk_ntd,
        beta_nt,
        scale,
        64,
        layout="NTD",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=True, return_intermediate_states=True,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        gk.detach().cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )

    _assert_close("o ntd", got[0], ref.o.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("final_state ntd", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g ntd", got[2], gk_ntd, rtol=2e-2, atol=2e-2)
    _assert_close("Aqk ntd", got[3], ref.Aqk.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("Akk ntd", got[4], ref.Akk.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("w ntd", got[5], ref.w.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("u ntd", got[6], ref.u.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("qg ntd", got[7], ref.qg.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("kg ntd", got[8], ref.kg.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("v_new ntd", got[9], ref.v_new.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("h ntd", got[10], ref.h.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("initial_state ntd", got[11], initial_state)


def test_kda_gate_cumsum_default_and_fwd_integration():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, _, beta, initial_state = _make_inputs(device, h=1, hv=2, t=40, dtype=torch.float16)
    g_step = (torch.randn(1, 2, 40, 128, dtype=torch.bfloat16) * 0.001).to(device)
    gk = fla_ascendc.kda_gate_cumsum(g_step, 64)
    ref_gk = _kda_gate_cumsum_reference(g_step.detach().cpu(), 64)
    _assert_close("gate cumsum default", gk, ref_gk, rtol=2e-3, atol=2e-3)
    gk_bsnd = gk.permute(0, 2, 1, 3).contiguous()

    scale = q.shape[-1] ** -0.5
    got = _chunk_kda_fwd_from_gk(
        q,
        k,
        v,
        gk_bsnd,
        beta,
        scale,
        64,
        layout="BSND",
        initial_state=initial_state,
        output_final_state=True,
        disable_recompute=False, return_intermediate_states=False,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        ref_gk.permute(0, 2, 1, 3).contiguous(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )
    _assert_close("gate cumsum fwd o", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("gate cumsum fwd state", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close(
        "gate cumsum fwd g",
        got[2],
        gk,
        rtol=2e-2,
        atol=2e-2,
    )
    _assert_close("gate cumsum fwd initial_state", got[11], initial_state)


def test_chunk_kda_fwd_small_k_rejected_as_unsupported():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, _ = _make_inputs(device, h=1, hv=1, t=8, kdim=8, vdim=128, dtype=torch.float16)
    try:
        _chunk_kda_fwd_from_gk(
            q,
            k,
            v,
            gk,
            beta,
            q.shape[-1] ** -0.5,
            64,
            layout="BSND",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("K < 16 must be rejected as an unsupported KDA forward path")


def test_chunk_kda_fwd_float_q_rejected_as_unsupported():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, _ = _make_inputs(device, h=1, hv=1, t=8, dtype=torch.float32)
    try:
        _chunk_kda_fwd_from_gk(
            q,
            k,
            v,
            gk,
            beta,
            q.shape[-1] ** -0.5,
            64,
            layout="BSND",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("float q/k/v must be rejected as an unsupported KDA forward path")


def test_kda_gate_cumsum_bnsd_direct_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(6789)
    g_bsnd = (torch.randn(1, 40, 2, 8, dtype=torch.bfloat16) * 0.001).to(device)
    g_bnsd = g_bsnd.permute(0, 2, 1, 3).contiguous()
    got = fla_ascendc.kda_gate_cumsum(g_bnsd, 32)
    ref = _kda_gate_cumsum_reference(g_bnsd.detach().cpu(), 32)
    _assert_close("gate cumsum bnsd", got, ref, rtol=2e-3, atol=2e-3)


def test_kda_gate_cumsum_ntd_direct_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(7890)
    g_bsnd = (torch.randn(1, 40, 2, 8, dtype=torch.bfloat16) * 0.001).to(device)
    g_ntd = g_bsnd.squeeze(0).permute(1, 0, 2).contiguous()
    got = fla_ascendc.kda_gate_cumsum(g_ntd, 32)
    ref = _kda_gate_cumsum_reference(g_ntd.detach().cpu(), 32)
    _assert_close("gate cumsum ntd", got, ref, rtol=2e-3, atol=2e-3)


def test_kda_gate_cumsum_safe_gate_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(5678)
    raw = (torch.randn(1, 2, 40, 8, dtype=torch.bfloat16) * 0.5).to(device)
    a_log = (torch.randn(2, dtype=torch.float32) * 0.1).to(device)
    dt_bias = (torch.randn(2 * 8, dtype=torch.float32) * 0.1).to(device)
    got = fla_ascendc.kda_gate_cumsum(
        raw,
        32,
        A_log=a_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    ref = _kda_gate_cumsum_reference(
        raw.detach().cpu(),
        32,
        A_log=a_log.detach().cpu(),
        dt_bias=dt_bias.detach().cpu(),
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    _assert_close("gate cumsum safe", got, ref, rtol=2e-3, atol=2e-3)


def test_chunk_kda_fwd_raw_gate_safe_modes_match_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(20260729)
    b, t, h, hv, kdim, vdim = 1, 65, 1, 2, 128, 128
    q = (torch.randn(b, t, h, kdim, dtype=torch.float32) * 0.02).half().to(device)
    k = (torch.randn(b, t, h, kdim, dtype=torch.float32) * 0.02).half().to(device)
    v = (torch.randn(b, t, hv, vdim, dtype=torch.float32) * 0.02).half().to(device)
    raw = (torch.randn(b, t, hv, kdim, dtype=torch.float32) * 0.5).to(device)
    beta = torch.rand(b, t, hv, dtype=torch.float32).to(device)
    a_log = (torch.randn(hv, dtype=torch.float32) * 0.2 - 0.5).to(device)
    dt_bias = (torch.randn(hv * kdim, dtype=torch.float32) * 0.1).to(device)
    scale = kdim ** -0.5

    for safe_gate in (False, True):
        lower_bound = -5.0 if safe_gate else None
        raw_head = raw.permute(0, 2, 1, 3).contiguous()
        gk_head = _kda_gate_cumsum_reference(
            raw_head.detach().cpu(),
            64,
            A_log=a_log.detach().cpu(),
            dt_bias=dt_bias.detach().cpu(),
            use_gate_in_kernel=True,
            safe_gate=safe_gate,
            lower_bound=-5.0,
        )
        ref = chunk_kda_forward_reference(
            q.detach().cpu(),
            k.detach().cpu(),
            v.detach().cpu(),
            gk_head.permute(0, 2, 1, 3).contiguous(),
            beta.detach().cpu(),
            scale=scale,
            chunk_size=64,
            initial_state=None,
            output_final_state=True,
        )
        got = fla_ascendc.chunk_kda_fwd(
            q,
            k,
            v,
            raw,
            beta,
            scale,
            64,
            layout="BSND",
            output_final_state=True,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=True,
            A_log=a_log,
            dt_bias=dt_bias,
            disable_recompute=True,
            return_intermediate_states=True,
        )
        mode = f"raw gate safe_gate={safe_gate}"
        _assert_close(f"{mode} o", got[0], ref.o, rtol=2e-2, atol=3e-2)
        _assert_close(f"{mode} final_state", got[1], ref.final_state, rtol=2e-2, atol=3e-2)
        _assert_close(
            f"{mode} gk",
            got[2],
            gk_head,
            rtol=2e-3,
            atol=2e-3,
        )
        _assert_close(f"{mode} Aqk", got[3], _bsnd_intermediate_to_bnsd(ref.Aqk), rtol=2e-2, atol=2e-2)
        _assert_close(f"{mode} Akk", got[4], _bsnd_intermediate_to_bnsd(ref.Akk), rtol=2e-2, atol=2e-2)


def test_kda_gate_cumsum_safe_gate_multitask_last_row_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(20260707)
    chunk_size = 64
    raw = torch.randn(1, 2, 1536, 128, dtype=torch.bfloat16).to(device)
    a_log = torch.log(torch.empty(2, dtype=torch.float32).uniform_(1, 16)).to(device)
    dt_bias = torch.randn(2 * 128, dtype=torch.float32).to(device)
    got = fla_ascendc.kda_gate_cumsum(
        raw,
        chunk_size,
        A_log=a_log,
        dt_bias=dt_bias,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    ref = _kda_gate_cumsum_reference(
        raw.detach().cpu(),
        chunk_size,
        A_log=a_log.detach().cpu(),
        dt_bias=dt_bias.detach().cpu(),
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=-5.0,
    )
    _assert_close("gate cumsum safe multitask", got, ref, rtol=2e-3, atol=2e-3)


def test_kda_gate_cumsum_layout_is_not_inferred_from_shape():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(20260714)
    g_bsnd = (torch.randn(1, 4, 8, 8, dtype=torch.bfloat16) * 0.001).to(device)
    g_bnsd = g_bsnd.permute(0, 2, 1, 3).contiguous()
    got_bnsd = fla_ascendc.kda_gate_cumsum(g_bnsd, 32)
    ref_bnsd = _kda_gate_cumsum_reference(g_bnsd.detach().cpu(), 32)
    _assert_close("gate cumsum BNSD T<=HV", got_bnsd, ref_bnsd, rtol=2e-3, atol=2e-3)

    g_ntd = g_bsnd.squeeze(0).permute(1, 0, 2).contiguous()
    got_ntd = fla_ascendc.kda_gate_cumsum(g_ntd, 32)
    ref_ntd = _kda_gate_cumsum_reference(g_ntd.detach().cpu(), 32)
    _assert_close("gate cumsum NTD T<=HV", got_ntd, ref_ntd, rtol=2e-3, atol=2e-3)


def test_chunk_kda_fwd_invalid_head_mapping_rejected():
    device = _device()
    if device.type == "cpu":
        return
    for h, hv in ((2, 1), (2, 3)):
        q, k, v, gk, beta, _ = _make_inputs(device, h=h, hv=hv, t=64, dtype=torch.float16)
        try:
            _chunk_kda_fwd_from_gk(q, k, v, gk, beta, q.shape[-1] ** -0.5, 64, layout="BSND")
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid H/HV mapping H={h}, HV={hv} must be rejected")


def test_chunk_kda_fwd_invalid_chunk_indices_rejected():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, _ = _make_inputs(device, h=1, hv=2, t=128, dtype=torch.float16)
    invalid_indices = ((0, 0, 1), (0, 0), (0, 0, 0, 2), (2, 0, 0, 1), (1, 0, 0, 0))
    for indices in invalid_indices:
        try:
            _chunk_kda_fwd_from_gk(
                q, k, v, gk, beta, q.shape[-1] ** -0.5, 64, layout="BSND",
                cu_seqlens=(0, 64, 128), chunk_indices=indices,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid chunk_indices={indices} must be rejected")


def test_chunk_kda_fwd_varlen_sequence_count_capacity_rejected():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, _ = _make_inputs(device, h=1, hv=1, t=1, dtype=torch.float16)
    cu_seqlens = (0,) * 1025 + (1,)
    try:
        _chunk_kda_fwd_from_gk(
            q, k, v, gk, beta, q.shape[-1] ** -0.5, 64, layout="BSND",
            cu_seqlens=cu_seqlens,
        )
    except RuntimeError as error:
        assert "at most 1024 sequences" in str(error)
    else:
        raise AssertionError("varlen input with more than 1024 sequences must be rejected")


def test_chunk_kda_fwd_varlen_large_chunk_indices_not_prechecked():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, _ = _make_inputs(device, h=1, hv=1, t=1, dtype=torch.float16)
    repeated_indices = tuple(value for _ in range(4097) for value in (0, 0))
    try:
        _chunk_kda_fwd_from_gk(
            q, k, v, gk, beta, q.shape[-1] ** -0.5, 64, layout="BSND",
            cu_seqlens=(0, 1), chunk_indices=repeated_indices,
        )
    except RuntimeError as error:
        assert "canonical sequence-major order" in str(error)
    else:
        raise AssertionError("duplicate chunk_indices should be rejected by semantic validation, not by capacity")


def test_chunk_gdn_fwd_h_gk_only_matches_neutral_g():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(20260715)
    b, h, hv, t, kdim, vdim = 1, 2, 2, 128, 128, 128
    raw_k = torch.randn(b, h, t, kdim, dtype=torch.float16) * 0.02
    w = (torch.randn(b, hv, t, kdim, dtype=torch.float16) * 0.02).to(device)
    u = (torch.randn(b, hv, t, vdim, dtype=torch.float16) * 0.02).to(device)
    gk_steps = -(torch.rand(b, hv, t // 64, 64, kdim, dtype=torch.float32) * 0.01 + 0.002)
    gk_chunks = gk_steps.cumsum(dim=3)
    kg = (raw_k.float().reshape(b, h, t // 64, 64, kdim)
          * torch.exp2(gk_chunks[:, :, :, -1:, :] - gk_chunks)).reshape_as(raw_k)
    kg = kg.to(torch.float16)
    gk = gk_chunks.reshape(b, hv, t, kdim)

    def reference():
        state = torch.zeros(b, hv, kdim, vdim, dtype=torch.float32)
        h_ref = torch.zeros(b, hv, t // 64, kdim, vdim, dtype=torch.float32)
        v_new_ref = torch.zeros(b, hv, t, vdim, dtype=torch.float32)
        for chunk_idx in range(t // 64):
            start = chunk_idx * 64
            end = start + 64
            h_ref[:, :, chunk_idx] = state
            v_new = u.cpu().float()[:, :, start:end] - torch.einsum(
                "bhtk,bhkv->bhtv", w.cpu().float()[:, :, start:end], state,
            )
            v_new_ref[:, :, start:end] = v_new
            state = state * torch.exp2(gk[:, :, end - 1]).unsqueeze(-1) + torch.einsum(
                "bhtk,bhtv->bhkv", kg.float()[:, :, start:end], v_new,
            )
        return h_ref, v_new_ref, state

    h_ref, v_new_ref, state = reference()

    kg = kg.to(device)
    gk = gk.to(device)
    neutral_g = torch.zeros(b, hv, t, dtype=torch.float32, device=device)

    gk_only = fla_ascendc.chunk_gated_delta_rule_fwd_h(
        kg, w, u, gk=gk, output_final_state=True, chunk_size=64,
    )
    explicit_neutral = fla_ascendc.chunk_gated_delta_rule_fwd_h(
        kg, w, u, g=neutral_g, gk=gk, output_final_state=True, chunk_size=64,
    )
    for name, outputs in (("gk-only", gk_only), ("explicit-neutral", explicit_neutral)):
        assert outputs[2].dtype == torch.float32, f"{name} final_state must be float32 without initial_state"
        assert torch.isfinite(outputs[2]).all().item(), f"{name} final_state must be finite"
    for name, actual, expected in zip(("h", "v_new", "final_state"), gk_only, explicit_neutral):
        _assert_close(f"gk-only {name}", actual, expected, rtol=0, atol=0)
    _assert_close("gk h formula", gk_only[0], h_ref.to(torch.float16), rtol=2e-2, atol=2e-3)
    _assert_close("gk v_new formula", gk_only[1], v_new_ref.to(torch.float16), rtol=2e-2, atol=2e-3)
    _assert_close("gk final_state formula", gk_only[2], state, rtol=2e-2, atol=2e-3)


def test_chunk_gdn_fwd_h_a5_model_shape_is_bitwise_deterministic():
    device = _device()
    if device.type == "cpu":
        return

    torch.manual_seed(20260722)
    batch, heads, seqlen, kdim, vdim = 1, 12, 4096, 128, 128
    chunk_size = 64
    cu_seqlens = (0, seqlen // 3, seqlen)
    chunk_indices = tuple(
        value
        for seq_idx, (seq_start, seq_end) in enumerate(
            zip(cu_seqlens[:-1], cu_seqlens[1:])
        )
        for chunk_idx in range(
            (seq_end - seq_start + chunk_size - 1) // chunk_size
        )
        for value in (seq_idx, chunk_idx)
    )

    dtype = torch.bfloat16
    k = (torch.randn(batch, heads, seqlen, kdim, dtype=dtype) * 0.04).to(device)
    w = (torch.randn(batch, heads, seqlen, kdim, dtype=dtype) * 0.04).to(device)
    u = (torch.randn(batch, heads, seqlen, vdim, dtype=dtype) * 0.04).to(device)
    g = torch.zeros(batch, heads, seqlen, dtype=torch.float32, device=device)
    raw_gk = -torch.rand(
        batch, heads, seqlen, kdim, dtype=torch.float32
    ) * 0.04
    gk = torch.empty_like(raw_gk)
    rcp_ln2 = 1.4426950408889634
    for seq_start, seq_end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
        for chunk_start in range(seq_start, seq_end, chunk_size):
            chunk_end = min(chunk_start + chunk_size, seq_end)
            gk[:, :, chunk_start:chunk_end] = torch.cumsum(
                raw_gk[:, :, chunk_start:chunk_end] * rcp_ln2,
                dim=2,
            )
    gk = gk.to(device)
    initial_state = (
        torch.randn(2, heads, kdim, vdim, dtype=torch.float32) * 0.01
    ).to(device)

    def run_fwd_h():
        return fla_ascendc.chunk_gated_delta_rule_fwd_h(
            k,
            w,
            u,
            g=g,
            gk=gk,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
        )

    run_fwd_h()
    torch.npu.synchronize()
    expected = _snapshot_fwd_h_outputs(run_fwd_h())
    assert [tuple(output.shape) for output in expected] == [
        (1, 12, 65, 128, 128),
        (1, 12, 4096, 128),
        (2, 12, 128, 128),
    ]
    for repeat in range(1, FWD_H_DETERMINISM_REPEATS):
        actual = _snapshot_fwd_h_outputs(run_fwd_h())
        _assert_fwd_h_outputs_bitwise_equal(expected, actual, repeat)


def _run_single_test_in_subprocess(name):
    subprocess.run([sys.executable, __file__, "--single-test", name], check=True)


def profile_chunk_kda_fwd_from_manifest():
    manifest_path = pathlib.Path(os.environ["FLA_NPU_CASE_MANIFEST"])
    selected_ids = set(filter(None, os.environ.get("FLA_NPU_CASE_IDS", "").split(",")))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = [
        case for case in manifest["cases"]
        if case["id"] in selected_ids and "performance" in case["tags"]
    ]
    if len(cases) != 1:
        raise RuntimeError(f"expected exactly one performance case, got {len(cases)}")
    case = cases[0]
    shape, attrs = case["shape"], case["attrs"]
    if attrs["layout"] != "NTD":
        raise RuntimeError("the KDA performance runner currently requires NTD layout")

    device = _device()
    if device.type == "cpu":
        raise RuntimeError("KDA performance profiling requires an NPU")
    h = int(shape["H_k"])
    hv = int(shape["H_v"])
    t = int(shape["T"])
    kdim = int(shape["K"])
    vdim = int(shape["V"])
    chunk_size = int(shape["chunk_size"])
    nt = int(shape["N_c"])
    cu_seqlens = tuple(int(value) for value in case["optional_inputs"]["cu_seqlens"])
    expected_chunks = sum(
        (end - start + chunk_size - 1) // chunk_size
        for start, end in zip(cu_seqlens, cu_seqlens[1:])
    )
    if h != hv or cu_seqlens[0] != 0 or cu_seqlens[-1] != t or nt != expected_chunks:
        raise RuntimeError("invalid NTD varlen performance case")

    data_dtype = torch.float16
    q = torch.zeros((h, t, kdim), dtype=data_dtype).to(device)
    k = torch.zeros_like(q)
    v = torch.zeros((hv, t, vdim), dtype=data_dtype).to(device)
    raw_gate = torch.full((hv, t, kdim), -0.005 * math.log(2.0), dtype=torch.float32).to(device)
    beta = torch.ones((hv, t), dtype=torch.float32).to(device)

    for _ in range(8):
        outputs = fla_ascendc.chunk_kda_fwd(
            q, k, v, raw_gate, beta, float(attrs["scale"]), chunk_size,
            layout="NTD", cu_seqlens=cu_seqlens,
            output_final_state=bool(attrs["output_final_state"]),
            disable_recompute=bool(attrs["disable_recompute"]),
            return_intermediate_states=bool(attrs["return_intermediate_states"]),
            safe_gate=bool(attrs["safe_gate"]),
            use_gate_in_kernel=bool(attrs["use_gate_in_kernel"]),
        )
        torch.npu.synchronize()
        del outputs


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--single-test":
        globals()[sys.argv[2]]()
        raise SystemExit(0)

    if os.environ.get("FLA_NPU_PROFILE_ONLY") == "1":
        profile_chunk_kda_fwd_from_manifest()
        raise SystemExit(0)

    selected_operator = os.environ.get("FLA_NPU_OPERATOR")
    if selected_operator == "kda_gate_cumsum":
        test_kda_gate_cumsum_default_and_fwd_integration()
        test_kda_gate_cumsum_bnsd_direct_matches_reference()
        test_kda_gate_cumsum_ntd_direct_matches_reference()
        test_kda_gate_cumsum_safe_gate_matches_reference()
        test_chunk_kda_fwd_raw_gate_safe_modes_match_reference()
        test_kda_gate_cumsum_safe_gate_multitask_last_row_matches_reference()
        test_kda_gate_cumsum_layout_is_not_inferred_from_shape()
        raise SystemExit(0)
    if selected_operator == "chunk_kda_fwd":
        test_chunk_gdn_fwd_h_gk_only_matches_neutral_g()
        test_chunk_gdn_fwd_h_a5_model_shape_is_bitwise_deterministic()
        test_chunk_kda_fwd_matches_reference()
        test_chunk_kda_fwd_raw_gate_safe_modes_match_reference()
        test_chunk_kda_fwd_bf16_gate_matches_reference()
        test_chunk_kda_fwd_fp16_matches_reference()
        test_chunk_kda_fwd_fp16_safe_gate_large_span_matches_reference()
        test_chunk_kda_fwd_tnd_matches_reference()
        test_chunk_kda_fwd_tnd_multi_head_supported()
        test_chunk_kda_fwd_bnsd_direct_matches_reference()
        test_chunk_kda_fwd_ntd_direct_matches_reference()
        test_chunk_kda_fwd_vdim256_matches_reference()
        test_chunk_kda_fwd_chunk128_matches_reference()
        test_chunk_kda_fwd_state_v_first_modes_sequence_major_h_match_reference()
        test_chunk_kda_fwd_upper_triangle_dirty_zero()
        test_chunk_kda_fwd_without_intermediate_matches_export_and_reference()
        test_chunk_kda_fwd_optional_output_matrix()
        test_chunk_kda_fwd_invalid_head_mapping_rejected()
        test_chunk_kda_fwd_invalid_chunk_indices_rejected()
        raise SystemExit(0)

    test_chunk_kda_fwd_matches_reference()
    test_chunk_kda_fwd_bf16_gate_matches_reference()
    test_chunk_kda_fwd_fp16_matches_reference()
    test_chunk_kda_fwd_fp16_safe_gate_large_span_matches_reference()
    test_chunk_kda_fwd_tnd_matches_reference()
    test_chunk_kda_fwd_tnd_multi_head_supported()
    test_chunk_kda_fwd_bnsd_direct_matches_reference()
    test_chunk_kda_fwd_ntd_direct_matches_reference()
    test_kda_gate_cumsum_default_and_fwd_integration()
    test_kda_gate_cumsum_bnsd_direct_matches_reference()
    test_kda_gate_cumsum_ntd_direct_matches_reference()
    test_kda_gate_cumsum_safe_gate_matches_reference()
    test_chunk_kda_fwd_raw_gate_safe_modes_match_reference()
    test_kda_gate_cumsum_safe_gate_multitask_last_row_matches_reference()
    test_kda_gate_cumsum_layout_is_not_inferred_from_shape()
    test_chunk_kda_fwd_invalid_head_mapping_rejected()
    test_chunk_kda_fwd_invalid_chunk_indices_rejected()
    test_chunk_kda_fwd_varlen_sequence_count_capacity_rejected()
    test_chunk_kda_fwd_varlen_large_chunk_indices_not_prechecked()
    test_chunk_gdn_fwd_h_gk_only_matches_neutral_g()
    test_chunk_gdn_fwd_h_a5_model_shape_is_bitwise_deterministic()
    test_chunk_kda_fwd_upper_triangle_dirty_zero()
    test_chunk_kda_fwd_vdim256_matches_reference()
    test_chunk_kda_fwd_chunk128_matches_reference()
    test_chunk_kda_fwd_state_v_first_modes_sequence_major_h_match_reference()
    test_chunk_kda_fwd_bsnd_export_dependency_matches_reference()
    test_chunk_kda_fwd_without_intermediate_matches_export_and_reference()
    test_chunk_kda_fwd_optional_output_matrix()
    test_chunk_kda_fwd_model_shape_with_stats()
    test_chunk_kda_fwd_model_shape_initial_state_none_with_stats()

    for negative_test in (
        "test_chunk_kda_fwd_varlen_initial_state_shape_rejected",
        "test_chunk_kda_fwd_bf16_chunk32_rejected_as_unsupported",
        "test_chunk_kda_fwd_lowercase_layout_rejected",
        "test_chunk_kda_fwd_head_num_gt128_rejected",
        "test_chunk_kda_fwd_small_k_rejected_as_unsupported",
        "test_chunk_kda_fwd_float_q_rejected_as_unsupported",
    ):
        _run_single_test_in_subprocess(negative_test)
