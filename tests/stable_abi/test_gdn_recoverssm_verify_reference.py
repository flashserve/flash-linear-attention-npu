"""CPU tests for the gdn_recoverssm_verify golden."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

operator_root = (
    Path(__file__).resolve().parents[2]
    / "fla/ops/ascendc/gdn/recurrent_gdn/gdn_recoverssm_verify"
)
sys.path.insert(0, str(operator_root))

from reference.reference import gdn_recoverssm_verify_reference


def _upstream_cpu_reference(
    inputs,
    checkpoint,
    offsets,
    pages,
    *,
    normalize=True,
    scale=None,
    sp_beta=1.0,
    threshold=20.0,
):
    """Pinned PR 20485 CPU reference, kept separate for alignment testing."""

    a_log, a, b, bias, q, k, v = (
        tensor.detach().cpu().double() for tensor in inputs
    )
    initial = checkpoint.detach().cpu().double()
    ratio = v.shape[2] // k.shape[2]
    scale = q.shape[-1] ** -0.5 if scale is None else scale
    if normalize:
        q = q / (q.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        k = k / (k.square().sum(-1, keepdim=True) + 1e-6).sqrt()
    q = (q[0] * scale).repeat_interleave(ratio, dim=1)
    k = k[0].repeat_interleave(ratio, dim=1)
    x = a.reshape_as(b[0]) + bias
    softplus = torch.nn.functional.softplus(x, beta=sp_beta, threshold=threshold)
    decay = (-a_log.exp() * softplus).exp()
    beta = b[0].sigmoid()
    output = torch.zeros_like(v)
    records = torch.full_like(
        torch.zeros(
            checkpoint.shape[0],
            v.shape[2],
            max(
                offsets[row + 1] - offsets[row]
                for row in range(len(pages))
            ),
            v.shape[-1] + k.shape[-1] + 1,
            dtype=torch.float64,
        ),
        float("nan"),
    )
    for row, page in enumerate(pages):
        if page <= 0:
            continue
        state = initial[page]
        for token, position in zip(
            range(offsets[row], offsets[row + 1]),
            range(offsets[row + 1] - offsets[row]),
        ):
            decayed = state * decay[token, :, None, None]
            key = k[token]
            residual = beta[token, :, None] * (
                v[0, token] - torch.einsum("hvk,hk->hv", decayed, key)
            )
            state = decayed + torch.einsum("hv,hk->hvk", residual, key)
            output[0, token] = torch.einsum("hvk,hk->hv", state, q[token])
            records[page, :, position] = torch.cat(
                (residual, key, decay[token, :, None]), dim=-1
            )
    return output, records


def _make_inputs(
    *,
    rows: int = 2,
    window: int = 4,
    hk: int = 2,
    hv: int = 6,
    key: int = 7,
    value: int = 5,
    state_dtype: torch.dtype = torch.float32,
    checkpoint_padding: int = 3,
    replay_padding: int = 4,
    seed: int = 419,
):
    generator = torch.Generator().manual_seed(seed)
    lengths = [window] * (rows - 1) + [window - 1]
    total = sum(lengths)
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int64
    )
    pages = torch.arange(1, rows + 1, dtype=torch.int64)
    q = torch.randn(1, total, hk, key, generator=generator, dtype=torch.bfloat16)
    k = torch.randn(1, total, hk, key, generator=generator, dtype=torch.bfloat16)
    v = torch.randn(1, total, hv, value, generator=generator, dtype=torch.bfloat16)
    a = torch.randn(total, hv, generator=generator, dtype=torch.bfloat16)
    b = torch.randn(1, total, hv, generator=generator, dtype=torch.bfloat16)
    a_log = torch.linspace(-0.4, 0.3, hv, dtype=torch.float32)
    dt_bias = torch.linspace(-0.2, 0.2, hv, dtype=torch.float32)
    pages_count = rows + 2
    state_storage = torch.empty(
        pages_count,
        hv,
        value,
        key + checkpoint_padding,
        dtype=state_dtype,
    )
    state_storage.normal_(0.0, 0.02, generator=generator)
    checkpoint = state_storage[..., :key]
    payload = value + key + 1
    replay_storage = torch.full(
        (pages_count, hv, window, payload + replay_padding),
        -7.0,
        dtype=torch.float32,
    )
    replay = replay_storage[..., :payload]
    return {
        "A_log": a_log,
        "a": a,
        "b": b,
        "dt_bias": dt_bias,
        "q": q,
        "k": k,
        "v": v,
        "checkpoint_state": checkpoint,
        "replay_cache": replay,
        "query_start_loc": offsets,
        "state_indices": pages,
    }, state_storage, replay_storage


def test_manual_single_token() -> None:
    q = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    k = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)
    v = torch.tensor([[[[2.0]]]], dtype=torch.float32)
    a = torch.zeros(1, 1, dtype=torch.float32)
    b = torch.zeros(1, 1, dtype=torch.float32)
    a_log = torch.zeros(1, dtype=torch.float32)
    bias = torch.zeros(1, dtype=torch.float32)
    state = torch.zeros(2, 1, 1, 2, dtype=torch.float32)
    replay = torch.full((2, 1, 1, 4), -3.0, dtype=torch.float32)
    out, actual_replay = gdn_recoverssm_verify_reference(
        a_log,
        a,
        b,
        bias,
        q,
        k,
        v,
        state,
        replay,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        scale=1.0,
        softplus_beta=1.0,
        softplus_threshold=20.0,
        use_qk_l2norm_in_kernel=False,
    )
    expected_decay = torch.exp(
        -torch.exp(torch.tensor(0.0, dtype=torch.float64))
        * torch.log(torch.tensor(2.0, dtype=torch.float64))
    )
    expected_correction = 0.5 * 2.0
    torch.testing.assert_close(
        out, torch.tensor([[[[0.0]]]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        actual_replay[1, 0, 0, 0],
        torch.tensor(expected_correction, dtype=torch.float64),
    )
    torch.testing.assert_close(
        actual_replay[1, 0, 0, 1:3],
        torch.tensor([0.0, 1.0], dtype=torch.float64),
    )
    torch.testing.assert_close(actual_replay[1, 0, 0, 3], expected_decay)
    torch.testing.assert_close(actual_replay[0], replay[0].double())


def test_ragged_gqa_padding_and_read_only_checkpoint() -> None:
    kwargs, state_storage, replay_storage = _make_inputs()
    before_state = state_storage.clone()
    before_replay_padding = replay_storage[..., -4:].clone()
    out, replay = gdn_recoverssm_verify_reference(**kwargs)
    assert out.shape == kwargs["v"].shape
    assert replay.shape == kwargs["replay_cache"].shape
    torch.testing.assert_close(state_storage, before_state, atol=0, rtol=0)
    torch.testing.assert_close(
        replay_storage[..., -4:], before_replay_padding, atol=0, rtol=0
    )
    assert torch.isfinite(out).all() and torch.isfinite(replay).all()


def test_null_and_empty_rows() -> None:
    kwargs, state_storage, replay_storage = _make_inputs(rows=2, window=2)
    kwargs["query_start_loc"] = torch.tensor([0, 0, 2], dtype=torch.int32)
    kwargs["state_indices"] = torch.tensor([0, -1], dtype=torch.int32)
    kwargs["a"] = kwargs["a"][:2]
    kwargs["b"] = kwargs["b"][:, :2]
    kwargs["q"] = kwargs["q"][:, :2]
    kwargs["k"] = kwargs["k"][:, :2]
    kwargs["v"] = kwargs["v"][:, :2]
    before = (state_storage.clone(), replay_storage.clone())
    out, replay = gdn_recoverssm_verify_reference(**kwargs)
    assert torch.count_nonzero(out) == 0
    torch.testing.assert_close(state_storage, before[0], atol=0, rtol=0)
    torch.testing.assert_close(replay_storage, before[1], atol=0, rtol=0)
    torch.testing.assert_close(replay, replay_storage[..., :13].double())


def test_bf16_checkpoint_and_normalization_off() -> None:
    kwargs, _, _ = _make_inputs(state_dtype=torch.bfloat16, seed=7)
    out, replay = gdn_recoverssm_verify_reference(
        **kwargs,
        scale=0.25,
        softplus_beta=0.5,
        softplus_threshold=2.0,
        use_qk_l2norm_in_kernel=False,
    )
    assert out.dtype == torch.float64 and replay.dtype == torch.float64
    assert torch.isfinite(out).all() and torch.isfinite(replay).all()


def test_reference_cases_are_deterministic() -> None:
    kwargs, _, _ = _make_inputs(seed=123)
    first_out, first_replay = gdn_recoverssm_verify_reference(**kwargs)
    second_out, second_replay = gdn_recoverssm_verify_reference(**kwargs)
    torch.testing.assert_close(first_out, second_out, atol=0, rtol=0)
    torch.testing.assert_close(first_replay, second_replay, atol=0, rtol=0)


def test_alignment_with_pinned_upstream_reference() -> None:
    kwargs, _, _ = _make_inputs(hk=2, hv=6, key=33, value=47, window=4, seed=20485)
    out, replay = gdn_recoverssm_verify_reference(**kwargs)
    upstream_out, upstream_records = _upstream_cpu_reference(
        (
            kwargs["A_log"],
            kwargs["a"],
            kwargs["b"],
            kwargs["dt_bias"],
            kwargs["q"],
            kwargs["k"],
            kwargs["v"],
        ),
        kwargs["checkpoint_state"],
        kwargs["query_start_loc"].tolist(),
        kwargs["state_indices"].tolist(),
    )
    torch.testing.assert_close(out, upstream_out)
    for row, page in enumerate(kwargs["state_indices"].tolist()):
        length = int(
            kwargs["query_start_loc"][row + 1] - kwargs["query_start_loc"][row]
        )
        torch.testing.assert_close(
            replay[page, :, :length], upstream_records[page, :, :length]
        )


if __name__ == "__main__":
    tests = (
        test_manual_single_token,
        test_ragged_gqa_padding_and_read_only_checkpoint,
        test_null_and_empty_rows,
        test_bf16_checkpoint_and_normalization_off,
        test_reference_cases_are_deterministic,
        test_alignment_with_pinned_upstream_reference,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
