# Copyright (c) 2026 Tianjin University, Ltd.

import os
import pathlib
import subprocess
import sys

import torch

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover - CPU fallback for syntax/smoke only
    torch_npu = None

from fla_npu.ops import ascendc as fla_ascendc


ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from tests.operators._shared.legacy_cases import find_legacy_case
from tests.reference.chunk_kda_reference import chunk_kda_forward_reference  # noqa: E402


MODEL_SHAPE_CASE = find_legacy_case(
    "chunk_kda_fwd", "model_shape_131072", "model_shape"
)["legacy"]["raw"]["config"]
MODEL_SHAPE_DUMP = pathlib.Path(os.environ.get("KDA_MODEL_SHAPE_DUMP", "/tmp/kda_model_shape_case.pt"))
RUN_MODEL_SHAPE_TEST_ENV = "RUN_KDA_MODEL_SHAPE_TEST"
CU_STATS_REL_EPS = 1e-6
CU_STATS_ABS_THRESHOLD = 2e-3
CU_STATS_REL_THRESHOLD = 2e-2
CU_STATS_SAMPLE_COUNT = 262144


def _device(device_id=None):
    if device_id is None:
        device_id = int(os.environ.get("TEST_DEVICE_ID", "0"))
    if torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device(f"npu:{device_id}")
    return torch.device("cpu")


def _stat(tensor, name):
    if hasattr(tensor, "is_npu") and tensor.is_npu:
        torch.npu.synchronize()
    flat = tensor.detach().flatten().float().cpu()
    has_nan = torch.isnan(flat).any().item()
    has_inf = torch.isinf(flat).any().item()
    nan_ratio = torch.isnan(flat).float().mean().item()
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        finite = torch.zeros(1, dtype=torch.float32)
    quant_src = finite
    if quant_src.numel() > 5_000_000:
        sample_idx = torch.randint(0, quant_src.numel(), (5_000_000,))
        quant_src = quant_src[sample_idx]
    return {
        "name": name,
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "min": finite.min().item(),
        "max": finite.max().item(),
        "mean": quant_src.mean().item(),
        "std": quant_src.std().item(),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "nan_ratio": nan_ratio,
        "percentile_1": torch.quantile(quant_src, 0.01).item(),
        "percentile_99": torch.quantile(quant_src, 0.99).item(),
    }


def _print_stat(stat_dict, prefix=""):
    s = stat_dict
    print(
        f"{prefix}{s['name']:18s} | "
        f"shape={s['shape']} {s['dtype']:12s} | "
        f"min={s['min']:10.4f} max={s['max']:10.4f} | "
        f"mean={s['mean']:8.4f} std={s['std']:8.4f} | "
        f"p1={s['percentile_1']:8.4f} p99={s['percentile_99']:8.4f} | "
        f"nan={s['has_nan']} inf={s['has_inf']} nan_ratio={s['nan_ratio']:.4f}"
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


def _cu_error_metrics(actual, expected):
    diff = (actual - expected).abs()
    rel = diff / expected.abs().clamp_min(CU_STATS_REL_EPS)
    flat_rel = rel.flatten()
    if flat_rel.numel() > CU_STATS_SAMPLE_COUNT:
        stride = (flat_rel.numel() + CU_STATS_SAMPLE_COUNT - 1) // CU_STATS_SAMPLE_COUNT
        rel_sample = flat_rel[::stride][:CU_STATS_SAMPLE_COUNT]
    else:
        rel_sample = flat_rel
    bad = (diff > CU_STATS_ABS_THRESHOLD) & (rel > CU_STATS_REL_THRESHOLD)
    return {
        "diff": diff,
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "l1_rel": (diff.sum() / expected.abs().sum().clamp_min(1e-12)).item(),
        "p99_rel": torch.quantile(rel_sample, 0.99).item(),
        "bad_ratio": bad.float().mean().item(),
    }


def _unravel_index(flat_index, shape):
    coordinates = []
    for size in reversed(shape):
        coordinates.append(flat_index % size)
        flat_index //= size
    return tuple(reversed(coordinates))


def _print_cu_layout(cu_seqlens, chunk_size):
    print("\n--- CU-Seqlens Layout ---")
    chunk_cursor = 0
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        length = end - start
        chunk_count = (length + chunk_size - 1) // chunk_size
        tail = length % chunk_size or chunk_size
        print(
            f"  seq={seq_idx:2d} T=[{start:6d},{end:6d}) len={length:6d} "
            f"chunks=[{chunk_cursor:4d},{chunk_cursor + chunk_count:4d}) "
            f"chunk_count={chunk_count:4d} tail={tail:2d}"
        )
        chunk_cursor += chunk_count


def _print_cu_token_stats(name, actual, expected, cu_seqlens, chunk_size):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    if actual.shape != expected.shape:
        print(f"  [CU][{name}] shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}")
        return
    if actual.dim() < 2 or actual.shape[1] != cu_seqlens[-1]:
        print(f"  [CU][{name}] cannot map T axis: shape={tuple(actual.shape)} total_t={cu_seqlens[-1]}")
        return

    print(f"\n  [CU token output] {name} shape={tuple(actual.shape)}")
    chunk_cursor = 0
    reduce_dims = tuple(dim for dim in range(actual.dim()) if dim != 1)
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        length = end - start
        chunk_count = (length + chunk_size - 1) // chunk_size
        tail = length % chunk_size or chunk_size
        actual_seq = actual[:, start:end]
        expected_seq = expected[:, start:end]
        metrics = _cu_error_metrics(actual_seq, expected_seq)
        diff = metrics["diff"]
        token_max = diff.amax(dim=reduce_dims)
        top_count = min(3, token_max.numel())
        top_values, top_indices = torch.topk(token_max, top_count)
        worst_local = int(top_indices[0].item())
        worst_global = start + worst_local
        worst_slice = diff[:, worst_local]
        worst_flat = int(worst_slice.argmax().item())
        worst_coord = _unravel_index(worst_flat, tuple(worst_slice.shape))
        actual_value = actual[:, worst_global].reshape(-1)[worst_flat].item()
        expected_value = expected[:, worst_global].reshape(-1)[worst_flat].item()
        first_chunk = diff[:, :min(chunk_size, length)].max().item()
        last_chunk = diff[:, -tail:].max().item()
        first_token = diff[:, 0].max().item()
        last_token = diff[:, -1].max().item()
        top_tokens = ", ".join(
            f"T={start + int(local)}(c={int(local) // chunk_size},off={int(local) % chunk_size}):{float(value):.3e}"
            for value, local in zip(top_values, top_indices)
        )
        print(
            f"    seq={seq_idx:2d} T=[{start:6d},{end:6d}) "
            f"chunks=[{chunk_cursor:4d},{chunk_cursor + chunk_count:4d}) tail={tail:2d} | "
            f"max_abs={metrics['max_abs']:.6e} mean_abs={metrics['mean_abs']:.6e} "
            f"l1_rel={metrics['l1_rel']:.6e} p99_rel={metrics['p99_rel']:.6e} "
            f"bad_ratio={metrics['bad_ratio']:.6e} | "
            f"first_token={first_token:.3e} last_token={last_token:.3e} "
            f"first_chunk={first_chunk:.3e} last_chunk={last_chunk:.3e} | "
            f"worst=T={worst_global} local={worst_local} "
            f"chunk={worst_local // chunk_size} off={worst_local % chunk_size} "
            f"coord={worst_coord} actual={actual_value:.6e} expected={expected_value:.6e} | "
            f"top3=[{top_tokens}]"
        )
        chunk_cursor += chunk_count


def _print_cu_chunk_stats(name, actual, expected, cu_seqlens, chunk_size):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    if actual.shape != expected.shape:
        print(f"  [CU][{name}] shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}")
        return

    print(f"\n  [CU chunk output] {name} shape={tuple(actual.shape)}")
    chunk_cursor = 0
    reduce_dims = tuple(dim for dim in range(actual.dim()) if dim != 1)
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        length = end - start
        chunk_count = (length + chunk_size - 1) // chunk_size
        chunk_end = chunk_cursor + chunk_count
        actual_seq = actual[:, chunk_cursor:chunk_end]
        expected_seq = expected[:, chunk_cursor:chunk_end]
        metrics = _cu_error_metrics(actual_seq, expected_seq)
        chunk_max = metrics["diff"].amax(dim=reduce_dims)
        top_count = min(3, chunk_max.numel())
        top_values, top_indices = torch.topk(chunk_max, top_count)
        top_chunks = ", ".join(
            f"global={chunk_cursor + int(local)}(local={int(local)}):{float(value):.3e}"
            for value, local in zip(top_values, top_indices)
        )
        print(
            f"    seq={seq_idx:2d} T=[{start:6d},{end:6d}) "
            f"chunks=[{chunk_cursor:4d},{chunk_end:4d}) | "
            f"max_abs={metrics['max_abs']:.6e} mean_abs={metrics['mean_abs']:.6e} "
            f"l1_rel={metrics['l1_rel']:.6e} p99_rel={metrics['p99_rel']:.6e} "
            f"bad_ratio={metrics['bad_ratio']:.6e} | top3=[{top_chunks}]"
        )
        chunk_cursor = chunk_end


def _print_cu_state_stats(name, actual, expected, cu_seqlens):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    if actual.shape != expected.shape:
        print(f"  [CU][{name}] shape mismatch: actual={tuple(actual.shape)} expected={tuple(expected.shape)}")
        return

    print(f"\n  [CU sequence output] {name} shape={tuple(actual.shape)}")
    for seq_idx, (start, end) in enumerate(zip(cu_seqlens[:-1], cu_seqlens[1:])):
        metrics = _cu_error_metrics(actual[seq_idx], expected[seq_idx])
        print(
            f"    seq={seq_idx:2d} T=[{start:6d},{end:6d}) len={end - start:6d} | "
            f"max_abs={metrics['max_abs']:.6e} mean_abs={metrics['mean_abs']:.6e} "
            f"l1_rel={metrics['l1_rel']:.6e} p99_rel={metrics['p99_rel']:.6e} "
            f"bad_ratio={metrics['bad_ratio']:.6e}"
        )


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


def _assert_close(name, actual, expected, rtol=2e-3, atol=2e-3):
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=rtol, atol=atol, msg=name)


def _kda_gate_cumsum_reference(g, chunk_size, A_log=None, dt_bias=None, cu_seqlens=None,
                               use_gate_in_kernel=False, safe_gate=False, lower_bound=-5.0):
    rcp_ln2 = 1.4426950408889634
    g_float = g.to(torch.float32)
    if use_gate_in_kernel:
        if not safe_gate:
            raise ValueError("test reference only covers safe_gate raw path")
        x = g_float
        if dt_bias is not None:
            bias = dt_bias.reshape(g.shape[-2], g.shape[-1]).to(torch.float32)
            if g.dim() == 4:
                x = x + bias[None, None, :, :]
            else:
                x = x + bias[None, :, :]
        a = torch.exp(A_log.to(torch.float32))
        if g.dim() == 4:
            x = x * a[None, None, :, None]
        else:
            x = x * a[None, :, None]
        gate = float(lower_bound) * torch.sigmoid(x)
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
                    out[0, start:end] = torch.cumsum(gate[0, start:end] * rcp_ln2, dim=0)
        else:
            for seq_idx in range(len(cu) - 1):
                seq_start, seq_end = int(cu[seq_idx]), int(cu[seq_idx + 1])
                for start in range(seq_start, seq_end, chunk_size):
                    end = min(start + chunk_size, seq_end)
                    out[start:end] = torch.cumsum(gate[start:end] * rcp_ln2, dim=0)
        return out

    if g.dim() == 4:
        for b in range(g.shape[0]):
            for start in range(0, g.shape[1], chunk_size):
                end = min(start + chunk_size, g.shape[1])
                out[b, start:end] = torch.cumsum(gate[b, start:end] * rcp_ln2, dim=0)
    else:
        for start in range(0, g.shape[0], chunk_size):
            end = min(start + chunk_size, g.shape[0])
            out[start:end] = torch.cumsum(gate[start:end] * rcp_ln2, dim=0)
    return out


def test_chunk_kda_fwd_model_shape_with_stats(dump_path=None, device_id=None):
    device = _device(device_id)
    if device.type == "cpu":
        print("skip model-shape stats test on CPU")
        return
    if os.environ.get(RUN_MODEL_SHAPE_TEST_ENV) != "1":
        print(f"skip model-shape stats test; set {RUN_MODEL_SHAPE_TEST_ENV}=1 to enable")
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
    # initial_state = data["initial_state"].to(device, non_blocking=True)

    scale = q.shape[-1] ** -0.5
    chunk_size = int(data.get("chunk_size", MODEL_SHAPE_CASE["chunk_size"]))
    safe_gate = data.get("safe_gate", True)
    lower_bound = float(data.get("lower_bound", -5.0))

    print("\n" + "=" * 80)
    print(f"=== KDA Forward Model-Shape Guard (device={device}) ===")
    print("=" * 80)
    print(f"[Meta] scale={scale:.6f}, chunk_size={chunk_size}")
    print(f"[Meta] cu_seqlens={cu_seqlens}")
    print(f"[Meta] safe_gate={safe_gate}, lower_bound={lower_bound}")
    print(f"[Meta] dump_path={dump_path}")

    print("\n--- Input Statistics ---")
    for name, tensor in [
        ("q", q),
        ("k", k),
        ("v", v),
        ("g_raw", g),
        ("beta", beta),
        ("A_log", a_log),
        ("dt_bias", dt_bias),
        # ("initial_state", initial_state),
    ]:
        _print_stat(_stat(tensor, name), "  ")

    q, q_rstd = _l2norm_fwd_torch(q)
    k, k_rstd = _l2norm_fwd_torch(k)
    print("\n--- After L2Norm ---")
    _print_stat(_stat(q, "q_norm"), "  ")
    _print_stat(_stat(k, "k_norm"), "  ")
    _print_stat(_stat(q_rstd, "q_rstd"), "  ")
    _print_stat(_stat(k_rstd, "k_rstd"), "  ")

    print("\n--- Gate Cumsum ---")
    gk = fla_ascendc.kda_gate_cumsum(
        g,
        chunk_size,
        A_log=a_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        use_gate_in_kernel=True,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
    )
    ref_gk = _kda_gate_cumsum_reference(
        g.detach().cpu(),
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
    gk_ntd = gk.squeeze(0).permute(1, 0, 2).contiguous()
    beta_nt = beta.squeeze(0).permute(1, 0).contiguous()

    print("\n--- Chunk KDA Forward ---")
    got = fla_ascendc.chunk_kda_fwd(
        q_ntd,
        k_ntd,
        v_ntd,
        gk_ntd,
        beta_nt,
        scale,
        chunk_size,
        layout="NTD",
        initial_state=None,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        return_intermediate=True,
    )
    o_npu, final_state_npu = got[0], got[1]
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
        # ("initial_state_out", got[11]),
    ]:
        if tensor.numel() == 0:
            continue
        stat = _stat(tensor, name)
        _print_stat(stat, "  ")
        if stat["has_nan"] or stat["has_inf"]:
            _print_first_nonfinite(tensor, name, "  ")
    assert torch.isfinite(o_npu).all().item(), "model shape o contains NaN or Inf"
    assert torch.isfinite(final_state_npu).all().item(), "model shape final_state contains NaN or Inf"

    print("\n--- CPU Reference ---")
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        ref_gk.cpu(),
        beta.detach().cpu(),
        scale=scale,
        chunk_size=chunk_size,
        initial_state=None,
        output_final_state=True,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int64),
    )
    ref_o_ntd = ref.o.squeeze(0).permute(1, 0, 2).contiguous()
    _print_stat(_stat(ref_o_ntd, "o_ref"), "  ")
    _print_stat(_stat(ref.final_state, "final_state_ref"), "  ")
    assert torch.isfinite(ref_o_ntd).all().item(), "model shape reference o contains NaN or Inf"
    assert torch.isfinite(ref.final_state).all().item(), "model shape reference final_state contains NaN or Inf"

    o_diff = (o_npu.detach().cpu() - ref_o_ntd).abs()
    fs_diff = (final_state_npu.detach().cpu() - ref.final_state).abs()
    print("\n--- Diff (NPU vs CPU Ref) ---")
    _print_stat(_stat(o_diff, "o_abs_diff"), "  ")
    _print_stat(_stat(fs_diff, "fs_abs_diff"), "  ")

    _print_cu_layout(cu_seqlens, chunk_size)
    print("\n--- CU-Seqlens Error Breakdown (NPU vs CPU Ref) ---")
    _print_cu_token_stats("gate_cumsum", gk, ref_gk, cu_seqlens, chunk_size)

    ref_gk_ntd = ref_gk.squeeze(0).permute(1, 0, 2)
    token_outputs = [
        ("o", o_npu, ref_o_ntd),
        ("g_out", got[2], ref_gk_ntd),
        ("Aqk", got[3], ref.Aqk.squeeze(0).permute(1, 0, 2)),
        ("Akk", got[4], ref.Akk.squeeze(0).permute(1, 0, 2)),
        ("w", got[5], ref.w.squeeze(0).permute(1, 0, 2)),
        ("u", got[6], ref.u.squeeze(0).permute(1, 0, 2)),
        ("qg", got[7], ref.qg.squeeze(0).permute(1, 0, 2)),
        ("kg", got[8], ref.kg.squeeze(0).permute(1, 0, 2)),
        ("v_new", got[9], ref.v_new.squeeze(0).permute(1, 0, 2)),
    ]
    for name, actual, expected in token_outputs:
        _print_cu_token_stats(name, actual, expected, cu_seqlens, chunk_size)

    ref_h_ntd = ref.h.squeeze(0).permute(1, 0, 2, 3)
    _print_cu_chunk_stats("h", got[10], ref_h_ntd, cu_seqlens, chunk_size)
    _print_cu_state_stats("final_state", final_state_npu, ref.final_state, cu_seqlens)

    # _assert_close("model shape o", o_npu.detach().cpu(), ref_o_ntd, rtol=5e-2, atol=5e-2)
    import ct
    ct.viz(o_npu.detach().cpu(),ref_o_ntd,sample_ratio=0.1,out_dir="./o_output")
    ct.viz(final_state_npu.detach().cpu(),ref.final_state,sample_ratio=0.1,out_dir="./finial_state_output")
    # _assert_close(
    #     "model shape final_state",
    #     final_state_npu.detach().cpu(),
    #     ref.final_state,
    #     rtol=5e-2,
    #     atol=5e-2,
    # )


def test_chunk_kda_fwd_from_dump_with_stats(dump_path: str, device_id=None):
    test_chunk_kda_fwd_model_shape_with_stats(dump_path=dump_path, device_id=device_id)


def test_chunk_kda_fwd_matches_reference():
    device = _device()
    q, k, v, gk, beta, initial_state = _make_inputs(device, h=1, hv=1, t=64)
    scale = q.shape[-1] ** -0.5

    got = fla_ascendc.chunk_kda_fwd(
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
        return_intermediate=True,
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

    _assert_close("o", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("final_state", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g", got[2], gk)
    _assert_close("Aqk", got[3], ref.Aqk, rtol=2e-2, atol=2e-2)
    _assert_close("Akk", got[4], ref.Akk, rtol=2e-2, atol=2e-2)
    _assert_close("w", got[5], ref.w, rtol=2e-2, atol=2e-2)
    _assert_close("u", got[6], ref.u, rtol=2e-2, atol=2e-2)
    _assert_close("qg", got[7], ref.qg, rtol=2e-2, atol=2e-2)
    _assert_close("kg", got[8], ref.kg, rtol=2e-2, atol=2e-2)
    _assert_close("v_new", got[9], ref.v_new, rtol=2e-2, atol=2e-2)
    _assert_close("initial_state", got[11], initial_state)


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

    got = fla_ascendc.chunk_kda_fwd(
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
        return_intermediate=True,
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
    aqk_npu = got[3].detach().float().cpu()[0, :, 0, :]
    akk_npu = got[4].detach().float().cpu()[0, :, 0, :]
    aqk_ref = ref.Aqk.detach().float()[0, :, 0, :]
    akk_ref = ref.Akk.detach().float()[0, :, 0, :]

    assert (aqk_npu[upper] == 0).all().item()
    assert (akk_npu[upper] == 0).all().item()
    assert (aqk_ref[upper] == 0).all().item()
    assert (akk_ref[upper] == 0).all().item()
    torch.testing.assert_close(akk_npu[diag, diag], torch.ones(t), rtol=0, atol=0)
    torch.testing.assert_close(akk_ref[diag, diag], torch.ones(t), rtol=0, atol=0)


def test_chunk_kda_fwd_chunk128_rejected_as_unsupported():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, _ = _make_inputs(device, h=1, hv=2, t=16)
    scale = q.shape[-1] ** -0.5
    cu_seqlens = [0, 6, 16]
    try:
        fla_ascendc.chunk_kda_fwd(
            q,
            k,
            v,
            gk,
            beta,
            scale,
            128,
            layout="BSND",
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            return_intermediate=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("chunk_size != 64 must be rejected as an unsupported KDA forward path")


def test_chunk_kda_fwd_varlen_initial_state_shape_rejected():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, bad_initial_state = _make_inputs(device, h=1, hv=2, t=16)
    scale = q.shape[-1] ** -0.5
    cu_seqlens = [0, 6, 16]
    try:
        fla_ascendc.chunk_kda_fwd(
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
        assert "initial_state must be [seq_num,Hv,K,V]" in str(exc)
    else:
        raise AssertionError("varlen initial_state with mismatched seq_num should be rejected")


def test_chunk_kda_fwd_bf16_chunk32_rejected_as_unsupported():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(device, h=1, hv=1, t=8, dtype=torch.bfloat16)
    scale = q.shape[-1] ** -0.5
    try:
        fla_ascendc.chunk_kda_fwd(
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
            return_intermediate=False,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("chunk_size != 64 must be rejected as an unsupported KDA forward path")


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

    got = fla_ascendc.chunk_kda_fwd(
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
        return_intermediate=False,
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
    _assert_close("g bf16 gate", got[2], gk)
    _assert_close("initial_state bf16 gate", got[11], initial_state)


def test_chunk_kda_fwd_fp16_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, initial_state = _make_inputs(
        device, h=1, hv=1, t=8, dtype=torch.float16
    )
    scale = q.shape[-1] ** -0.5

    got = fla_ascendc.chunk_kda_fwd(
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
        return_intermediate=True,
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
    _assert_close("g fp16", got[2], gk)
    _assert_close("Aqk fp16", got[3], ref.Aqk, rtol=2e-2, atol=2e-2)
    _assert_close("Akk fp16", got[4], ref.Akk, rtol=2e-2, atol=2e-2)
    _assert_close("w fp16", got[5], ref.w, rtol=2e-2, atol=2e-2)
    _assert_close("u fp16", got[6], ref.u, rtol=2e-2, atol=2e-2)
    _assert_close("qg fp16", got[7], ref.qg, rtol=2e-2, atol=2e-2)
    _assert_close("kg fp16", got[8], ref.kg, rtol=2e-2, atol=2e-2)
    _assert_close("v_new fp16", got[9], ref.v_new, rtol=2e-2, atol=2e-2)
    _assert_close("initial_state fp16", got[11], initial_state)


def test_chunk_kda_fwd_tnd_matches_reference():
    device = _device()
    q, k, v, gk, beta, initial_state = _make_inputs(device, b=1, h=1, hv=2, t=8)
    scale = q.shape[-1] ** -0.5

    got = fla_ascendc.chunk_kda_fwd(
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
        return_intermediate=True,
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
    _assert_close("g tnd", got[2], gk.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("Aqk tnd", got[3], ref.Aqk.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("Akk tnd", got[4], ref.Akk.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("w tnd", got[5], ref.w.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("u tnd", got[6], ref.u.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("qg tnd", got[7], ref.qg.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("kg tnd", got[8], ref.kg.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("v_new tnd", got[9], ref.v_new.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("h tnd", got[10], ref.h.squeeze(0), rtol=2e-2, atol=2e-2)
    _assert_close("initial_state tnd", got[11], initial_state)


def test_chunk_kda_fwd_tnd_multi_head_rejected():
    device = _device()
    if device.type == "cpu":
        return
    t, h, hv, kdim, vdim = 128, 2, 2, 128, 128
    q = torch.randn(t, h, kdim, device=device, dtype=torch.float16)
    k = torch.randn(t, h, kdim, device=device, dtype=torch.float16)
    v = torch.randn(t, hv, vdim, device=device, dtype=torch.float16)
    gk = torch.randn(t, hv, kdim, device=device, dtype=torch.float32)
    beta = torch.rand(t, hv, device=device, dtype=torch.float32)
    try:
        fla_ascendc.chunk_kda_fwd(
            q,
            k,
            v,
            gk,
            beta,
            kdim ** -0.5,
            64,
            layout="TND",
            output_final_state=True,
        )
    except RuntimeError as exc:
        assert "TND layout with H > 1 is not supported" in str(exc)
    else:
        raise AssertionError("multi-head TND rank3 input must be rejected before kernel launch")


def test_chunk_kda_fwd_lowercase_layout_rejected():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, _ = _make_inputs(device, h=1, hv=1, t=8)
    try:
        fla_ascendc.chunk_kda_fwd(
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
        assert "layout must be one of BSND, BNSD, TND, NTD" in str(exc)
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
        fla_ascendc.chunk_kda_fwd(
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
        assert "H and HV must be <= 128" in str(exc) or "H and HV must be less than or equal to 128" in str(exc)
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

    got = fla_ascendc.chunk_kda_fwd(
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
        return_intermediate=True,
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

    _assert_close("o bnsd", got[0], ref.o.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("final_state bnsd", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g bnsd", got[2], gk_bnsd, rtol=2e-2, atol=2e-2)
    _assert_close("Aqk bnsd", got[3], ref.Aqk.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("Akk bnsd", got[4], ref.Akk.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("w bnsd", got[5], ref.w.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("u bnsd", got[6], ref.u.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("qg bnsd", got[7], ref.qg.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("kg bnsd", got[8], ref.kg.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("v_new bnsd", got[9], ref.v_new.permute(0, 2, 1, 3), rtol=2e-2, atol=2e-2)
    _assert_close("h bnsd", got[10], ref.h.permute(0, 2, 1, 3, 4), rtol=2e-2, atol=2e-2)
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

    got = fla_ascendc.chunk_kda_fwd(
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
        return_intermediate=True,
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

    _assert_close("o ntd", got[0], ref.o.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("final_state ntd", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("g ntd", got[2], gk_ntd, rtol=2e-2, atol=2e-2)
    _assert_close("Aqk ntd", got[3], ref.Aqk.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("Akk ntd", got[4], ref.Akk.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("w ntd", got[5], ref.w.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("u ntd", got[6], ref.u.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("qg ntd", got[7], ref.qg.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("kg ntd", got[8], ref.kg.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("v_new ntd", got[9], ref.v_new.squeeze(0).permute(1, 0, 2), rtol=2e-2, atol=2e-2)
    _assert_close("h ntd", got[10], ref.h.squeeze(0).permute(1, 0, 2, 3), rtol=2e-2, atol=2e-2)
    _assert_close("initial_state ntd", got[11], initial_state)


def test_kda_gate_cumsum_default_and_fwd_integration():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, _, beta, initial_state = _make_inputs(device, h=1, hv=2, t=40, dtype=torch.float16)
    g_step = (torch.randn(1, 40, 2, 128, dtype=torch.bfloat16) * 0.001).to(device)
    gk = fla_ascendc.kda_gate_cumsum(g_step, 64)
    ref_gk = _kda_gate_cumsum_reference(g_step.detach().cpu(), 64)
    _assert_close("gate cumsum default", gk, ref_gk, rtol=2e-3, atol=2e-3)

    scale = q.shape[-1] ** -0.5
    got = fla_ascendc.chunk_kda_fwd(
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
        return_intermediate=False,
    )
    ref = chunk_kda_forward_reference(
        q.detach().cpu(),
        k.detach().cpu(),
        v.detach().cpu(),
        ref_gk,
        beta.detach().cpu(),
        scale=scale,
        chunk_size=64,
        initial_state=initial_state.detach().cpu(),
        output_final_state=True,
    )
    _assert_close("gate cumsum fwd o", got[0], ref.o, rtol=2e-2, atol=2e-2)
    _assert_close("gate cumsum fwd state", got[1], ref.final_state, rtol=2e-2, atol=2e-2)
    _assert_close("gate cumsum fwd g", got[2], gk, rtol=2e-2, atol=2e-2)
    _assert_close("gate cumsum fwd initial_state", got[11], initial_state)


def test_chunk_kda_fwd_small_k_rejected_as_unsupported():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, _ = _make_inputs(device, h=1, hv=1, t=8, kdim=8, vdim=128, dtype=torch.float16)
    try:
        fla_ascendc.chunk_kda_fwd(
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
        fla_ascendc.chunk_kda_fwd(
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
    got = fla_ascendc.kda_gate_cumsum(g_bnsd, 32, layout="BNSD")
    ref = _kda_gate_cumsum_reference(g_bsnd.detach().cpu(), 32).permute(0, 2, 1, 3)
    _assert_close("gate cumsum bnsd", got, ref, rtol=2e-3, atol=2e-3)


def test_kda_gate_cumsum_ntd_direct_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(7890)
    g_bsnd = (torch.randn(1, 40, 2, 8, dtype=torch.bfloat16) * 0.001).to(device)
    g_ntd = g_bsnd.squeeze(0).permute(1, 0, 2).contiguous()
    got = fla_ascendc.kda_gate_cumsum(g_ntd, 32, layout="NTD")
    ref = _kda_gate_cumsum_reference(g_bsnd.squeeze(0).detach().cpu(), 32).permute(1, 0, 2)
    _assert_close("gate cumsum ntd", got, ref, rtol=2e-3, atol=2e-3)


def test_kda_gate_cumsum_safe_gate_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(5678)
    raw = (torch.randn(1, 40, 2, 8, dtype=torch.bfloat16) * 0.5).to(device)
    a_log = (torch.randn(2, dtype=torch.float32) * 0.1).to(device)
    dt_bias = (torch.randn(2, 8, dtype=torch.float32) * 0.1).to(device)
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


def test_kda_gate_cumsum_safe_gate_multitask_last_row_matches_reference():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(20260707)
    chunk_size = 64
    raw = torch.randn(1, 1536, 2, 128, dtype=torch.bfloat16).to(device)
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
    got_bnsd = fla_ascendc.kda_gate_cumsum(g_bnsd, 32, layout="BNSD")
    ref_bnsd = _kda_gate_cumsum_reference(g_bsnd.detach().cpu(), 32).permute(0, 2, 1, 3)
    _assert_close("gate cumsum BNSD T<=HV", got_bnsd, ref_bnsd, rtol=2e-3, atol=2e-3)

    g_ntd = g_bsnd.squeeze(0).permute(1, 0, 2).contiguous()
    got_ntd = fla_ascendc.kda_gate_cumsum(g_ntd, 32, layout="NTD")
    ref_ntd = _kda_gate_cumsum_reference(g_bsnd.squeeze(0).detach().cpu(), 32).permute(1, 0, 2)
    _assert_close("gate cumsum NTD T<=HV", got_ntd, ref_ntd, rtol=2e-3, atol=2e-3)


def test_chunk_kda_fwd_invalid_head_mapping_rejected():
    device = _device()
    if device.type == "cpu":
        return
    for h, hv in ((2, 1), (2, 3)):
        q, k, v, gk, beta, _ = _make_inputs(device, h=h, hv=hv, t=64, dtype=torch.float16)
        try:
            fla_ascendc.chunk_kda_fwd(q, k, v, gk, beta, q.shape[-1] ** -0.5, 64, layout="BSND")
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid H/HV mapping H={h}, HV={hv} must be rejected")


def test_chunk_kda_fwd_invalid_chunk_indices_rejected():
    device = _device()
    if device.type == "cpu":
        return
    q, k, v, gk, beta, _ = _make_inputs(device, h=1, hv=2, t=128, dtype=torch.float16)
    invalid_indices = ((0, 0, 1), (0, 0), (0, 0, 0, 2), (2, 0, 0, 1))
    for indices in invalid_indices:
        try:
            fla_ascendc.chunk_kda_fwd(
                q, k, v, gk, beta, q.shape[-1] ** -0.5, 64, layout="BSND",
                cu_seqlens=(0, 64, 128), chunk_indices=indices,
            )
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"invalid chunk_indices={indices} must be rejected")


def test_chunk_gdn_fwd_h_gk_only_matches_neutral_g():
    device = _device()
    if device.type == "cpu":
        return
    torch.manual_seed(20260715)
    b, h, hv, t, kdim, vdim = 1, 1, 2, 64, 128, 128
    k = (torch.randn(b, h, t, kdim, dtype=torch.float16) * 0.02).to(device)
    w = (torch.randn(b, hv, t, kdim, dtype=torch.float16) * 0.02).to(device)
    u = (torch.randn(b, hv, t, vdim, dtype=torch.float16) * 0.02).to(device)
    gk = (torch.randn(b, hv, t, kdim, dtype=torch.float32) * 0.001).cumsum(dim=2).to(device)
    neutral_g = torch.zeros(b, hv, t, dtype=torch.float32, device=device)

    gk_only = fla_ascendc.chunk_gated_delta_rule_fwd_h(
        k, w, u, gk=gk, output_final_state=True, chunk_size=64,
    )
    explicit_neutral = fla_ascendc.chunk_gated_delta_rule_fwd_h(
        k, w, u, g=neutral_g, gk=gk, output_final_state=True, chunk_size=64,
    )
    for name, outputs in (("gk-only", gk_only), ("explicit-neutral", explicit_neutral)):
        assert outputs[2].dtype == torch.float32, f"{name} final_state must be float32 without initial_state"
        assert torch.isfinite(outputs[2]).all().item(), f"{name} final_state must be finite"
    for name, actual, expected in zip(("h", "v_new", "final_state"), gk_only, explicit_neutral):
        _assert_close(f"gk-only {name}", actual, expected, rtol=0, atol=0)


def _run_single_test_in_subprocess(name):
    subprocess.run([sys.executable, __file__, "--single-test", name], check=True)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--single-test":
        globals()[sys.argv[2]]()
        raise SystemExit(0)

    # test_chunk_kda_fwd_matches_reference()
    # test_chunk_kda_fwd_bf16_gate_matches_reference()
    # test_chunk_kda_fwd_fp16_matches_reference()
    # test_chunk_kda_fwd_tnd_matches_reference()
    # test_chunk_kda_fwd_bnsd_direct_matches_reference()
    # test_chunk_kda_fwd_ntd_direct_matches_reference()
    # test_kda_gate_cumsum_default_and_fwd_integration()
    # test_kda_gate_cumsum_bnsd_direct_matches_reference()
    # test_kda_gate_cumsum_ntd_direct_matches_reference()
    # test_kda_gate_cumsum_safe_gate_matches_reference()
    # test_kda_gate_cumsum_safe_gate_multitask_last_row_matches_reference()
    # test_kda_gate_cumsum_layout_is_not_inferred_from_shape()
    # test_chunk_kda_fwd_invalid_head_mapping_rejected()
    # test_chunk_kda_fwd_invalid_chunk_indices_rejected()
    # test_chunk_gdn_fwd_h_gk_only_matches_neutral_g()
    # test_chunk_kda_fwd_upper_triangle_dirty_zero()
    test_chunk_kda_fwd_model_shape_with_stats()

    # for negative_test in (
    #     "test_chunk_kda_fwd_chunk128_rejected_as_unsupported",
    #     "test_chunk_kda_fwd_varlen_initial_state_shape_rejected",
    #     "test_chunk_kda_fwd_bf16_chunk32_rejected_as_unsupported",
    #     "test_chunk_kda_fwd_tnd_multi_head_rejected",
    #     "test_chunk_kda_fwd_lowercase_layout_rejected",
    #     "test_chunk_kda_fwd_head_num_gt128_rejected",
    #     "test_chunk_kda_fwd_small_k_rejected_as_unsupported",
    #     "test_chunk_kda_fwd_float_q_rejected_as_unsupported",
    # ):
    #     _run_single_test_in_subprocess(negative_test)
