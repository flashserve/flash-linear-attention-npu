"""Ascend single-op tests for gdn_recoverssm_verify."""

import math

import pytest
import torch

from fla_npu.ops.ascendc import gdn_recoverssm_verify


def _cpu_reference(A_log, a, b, dt_bias, q, k, v, checkpoint, offsets, pages,
                   window, scale, beta, threshold, normalize):
    a_log, a, b, bias, q, k, v = (
        x.detach().cpu().double() for x in (A_log, a, b, dt_bias, q, k, v)
    )
    state = checkpoint.detach().cpu().double()
    ratio = v.shape[2] // k.shape[2]
    if normalize:
        q = q * torch.rsqrt(q.square().sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    q = q[0].repeat_interleave(ratio, dim=1) * scale
    k = k[0].repeat_interleave(ratio, dim=1)
    b = b.reshape(-1, v.shape[2])
    raw = a.reshape_as(b) + bias
    gate = -a_log.exp() * torch.nn.functional.softplus(raw, beta=beta, threshold=threshold)
    decay = gate.exp()
    beta_eff = b.sigmoid()
    out = torch.zeros_like(v)
    records = {}
    for row, page in enumerate(pages):
        if page <= 0:
            continue
        current = state[page].clone()
        token_records = []
        for token in range(offsets[row], offsets[row + 1]):
            current = current * decay[token, :, None, None]
            correction = beta_eff[token, :, None] * (
                v[0, token] - torch.einsum("hvk,hk->hv", current, k[token])
            )
            current = current + correction[:, :, None] * k[token][:, None, :]
            out[0, token] = torch.einsum("hvk,hk->hv", current, q[token])
            token_records.append(
                torch.cat((correction, k[token], decay[token, :, None]), dim=-1)
            )
        if token_records:
            records[page] = torch.stack(token_records, dim=1)
    return out, records


def _assert_close(actual, expected, label):
    observed = actual.detach().cpu().double()
    reference = expected.double()
    if actual.dtype == torch.bfloat16:
        rtol, atol, limit = 2**-6, 2**-6, 1.0
    elif actual.dtype == torch.float32:
        rtol, atol, limit = 2**-10, 2**-16, 0.01
    else:
        rtol, atol, limit = 2**-9, 2**-9, 0.1
    error = (observed - reference).abs()
    ratio = (error <= atol + rtol * reference.abs()).double().mean().item()
    maximum = error.max().item()
    assert ratio >= 0.99, f"{label}: ratio={ratio}, max={maximum}"
    assert maximum <= limit, f"{label}: max={maximum}, limit={limit}"


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_gdn_recoverssm_verify_small_ragged(state_dtype):
    torch.manual_seed(419)
    device = "npu"
    rows, window, hk, hv, kd, vd = 2, 4, 2, 6, 128, 128
    lengths = [4, 3]
    total = sum(lengths)
    offsets = torch.tensor([0, 4, 7], dtype=torch.int32, device=device)
    pages = torch.tensor([1, 2], dtype=torch.int32, device=device)
    q = torch.randn(1, total, hk, kd, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, total, hv, vd, dtype=torch.bfloat16, device=device)
    a = torch.randn(total, hv, dtype=torch.bfloat16, device=device)
    b = torch.randn(1, total, hv, dtype=torch.bfloat16, device=device)
    A_log = torch.linspace(-0.4, 0.3, hv, dtype=torch.float32, device=device)
    dt_bias = torch.linspace(-0.2, 0.2, hv, dtype=torch.float32, device=device)
    checkpoint_storage = torch.empty(
        4, hv, vd, kd, dtype=state_dtype, device=device
    ).normal_(0.0, 0.02)
    checkpoint = checkpoint_storage
    payload = vd + kd + 1
    replay_storage = torch.full(
        (4, hv, window, payload + 4), -7.0, dtype=torch.float32, device=device
    )
    replay = replay_storage[..., :payload]
    expected_out, expected_records = _cpu_reference(
        A_log, a, b, dt_bias, q, k, v, checkpoint, [0, 4, 7], [1, 2],
        window, 0.25, 0.5, 2.0, True,
    )
    before_checkpoint = checkpoint_storage.clone()
    supplied_out = torch.empty(
        1, total * 2, hv, vd, dtype=torch.bfloat16, device=device
    )[:, ::2]
    actual = gdn_recoverssm_verify(
        A_log, a, b, dt_bias, q, k, v, checkpoint, replay,
        offsets, pages, scale=0.25,
        softplus_beta=0.5, softplus_threshold=2.0,
        use_qk_l2norm_in_kernel=True, out=supplied_out,
    )
    assert actual is supplied_out
    _assert_close(actual, expected_out, "out")
    for page, records in expected_records.items():
        valid = records.shape[1]
        _assert_close(replay[page, :, :valid], records, f"replay/{page}")
    torch.testing.assert_close(checkpoint_storage, before_checkpoint, atol=0, rtol=0)
    torch.testing.assert_close(replay_storage[..., -4:], torch.full_like(
        replay_storage[..., -4:], -7.0), atol=0, rtol=0)


def test_gdn_recoverssm_verify_null_and_empty():
    device = "npu"
    hk, hv, kd, vd, window = 2, 4, 128, 128, 3
    q = torch.empty(1, 0, hk, kd, dtype=torch.bfloat16, device=device)
    k = torch.empty_like(q)
    v = torch.empty(1, 0, hv, vd, dtype=torch.bfloat16, device=device)
    a = torch.empty(0, hv, dtype=torch.bfloat16, device=device)
    b = torch.empty(1, 0, hv, dtype=torch.bfloat16, device=device)
    A_log = torch.zeros(hv, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(hv, dtype=torch.float32, device=device)
    checkpoint = torch.zeros(2, hv, vd, kd, dtype=torch.float32, device=device)
    replay = torch.full((2, hv, window, vd + kd + 1), -7.0,
                        dtype=torch.float32, device=device)
    before_checkpoint = checkpoint.clone()
    before_replay = replay.clone()
    out = gdn_recoverssm_verify(
        A_log, a, b, dt_bias, q, k, v, checkpoint, replay,
        torch.tensor([0], dtype=torch.int32, device=device),
        torch.empty(0, dtype=torch.int32, device=device),
    )
    assert out.numel() == 0
    torch.testing.assert_close(checkpoint, before_checkpoint, atol=0, rtol=0)
    torch.testing.assert_close(replay, before_replay, atol=0, rtol=0)


def test_gdn_recoverssm_verify_padded_requests():
    torch.manual_seed(1319)
    device = "npu"
    hk, hv, kd, vd, window = 8, 24, 128, 128, 4
    total = 7
    offsets = torch.tensor(
        [0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 7, 7, 7],
        dtype=torch.int32,
        device=device,
    )
    indices_storage = torch.full((48,), -1, dtype=torch.int32, device=device)
    indices_storage[::4].copy_(torch.arange(1, 13, dtype=torch.int32, device=device))
    pages = indices_storage[::4]

    q = torch.randn(1, total, hk, kd, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, total, hv, vd, dtype=torch.bfloat16, device=device)
    a = torch.randn(total, hv, dtype=torch.bfloat16, device=device)
    b = torch.randn(total, hv, dtype=torch.bfloat16, device=device)
    A_log = torch.randn(hv, dtype=torch.float32, device=device)
    dt_bias = torch.randn(hv, dtype=torch.float32, device=device)

    checkpoint_storage = torch.randn(
        13, hv, vd, kd + 4, dtype=torch.bfloat16, device=device
    )
    checkpoint = checkpoint_storage[..., :kd]
    payload = vd + kd + 1
    replay_storage = torch.full(
        (13, hv, window, payload + 3), -7.0, dtype=torch.float32, device=device
    )
    replay = replay_storage[..., :payload]

    expected_out, expected_records = _cpu_reference(
        A_log, a, b, dt_bias, q, k, v, checkpoint, offsets.tolist(),
        pages.tolist(), window, 0.25, 0.5, 2.0, True,
    )
    actual = gdn_recoverssm_verify(
        A_log, a, b, dt_bias, q, k, v, checkpoint, replay,
        offsets, pages, scale=0.25,
        softplus_beta=0.5, softplus_threshold=2.0,
        use_qk_l2norm_in_kernel=True,
    )
    _assert_close(actual, expected_out, "padded/out")
    for page, records in expected_records.items():
        valid = records.shape[1]
        _assert_close(
            replay[page, :, :valid], records, f"padded/replay/{page}"
        )
    torch.testing.assert_close(
        replay_storage[..., payload:],
        torch.full_like(replay_storage[..., payload:], -7.0),
        atol=0,
        rtol=0,
    )
