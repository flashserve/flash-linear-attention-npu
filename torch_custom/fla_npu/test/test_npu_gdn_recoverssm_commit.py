"""Ascend single-op tests for gdn_recoverssm_commit."""

import math
import sys
from pathlib import Path

import pytest
import torch

from fla_npu.ops.ascendc import gdn_recoverssm_commit


def _load_reference():
    root = (
        Path(__file__).resolve().parents[3]
        / "fla/ops/ascendc/gdn/recurrent_gdn/gdn_recoverssm_commit"
    )
    sys.path.insert(0, str(root))
    from reference.reference import gdn_recoverssm_commit_reference

    return gdn_recoverssm_commit_reference


gdn_recoverssm_commit_reference = _load_reference()

PR20485_WINDOW_CASES = (
    (4, 8, 128, 128, 1, torch.float32),
    (4, 8, 128, 128, 2, torch.bfloat16),
    (4, 8, 128, 128, 5, torch.float32),
    (8, 24, 128, 128, 9, torch.bfloat16),
    (1, 3, 128, 128, 5, torch.float32),
    (2, 6, 33, 47, 4, torch.float32),
)


def _make_plan(device):
    return {
        "state_indices": torch.tensor([1, 2, 3], dtype=torch.int32, device=device),
        "commit_lens": torch.tensor([5, 3, 1], dtype=torch.int32, device=device),
        "final_state_indices": torch.tensor([4, 2, 5], dtype=torch.int32, device=device),
        "boundary_state_indices": torch.tensor([0, 6, 0], dtype=torch.int32, device=device),
        "boundary_recovery_lens": torch.tensor([0, 2, 0], dtype=torch.int32, device=device),
    }


def _make_table(state_layers, replay_layers):
    table = torch.empty(
        (2, 2, len(state_layers)),
        dtype=torch.int64,
        device=state_layers[0].device,
    )
    for layer, (state, replay) in enumerate(zip(state_layers, replay_layers)):
        table[0, 0, layer] = state.data_ptr()
        table[0, 1, layer] = state.stride(0)
        table[1, 0, layer] = replay.data_ptr()
        table[1, 1, layer] = replay.stride(0)
    return table


def _assert_close(actual, expected, name):
    error = (actual.double() - expected.double()).abs()
    tolerance = 1.0 / 64.0 + (1.0 / 1024.0) * expected.double().abs()
    assert error.le(tolerance).double().mean().item() >= 0.999, name
    assert error.max().item() <= 1.0, name


def _assert_pr20485_close(actual, expected, name):
    tolerances = {
        torch.float16: (2**-9, 2**-9, 0.1),
        torch.bfloat16: (2**-6, 2**-6, 1.0),
        torch.float32: (2**-10, 2**-16, 0.01),
    }
    rtol, atol, floor = tolerances[actual.dtype]
    actual_cpu = actual.detach().to(device="cpu", dtype=torch.float64)
    expected_cpu = expected.detach().to(device="cpu", dtype=torch.float64)
    error = (actual_cpu - expected_cpu).abs()
    passed = (error <= atol + rtol * expected_cpu.abs()).double().mean().item()
    limit = max(floor, 32 * torch.finfo(actual.dtype).eps)
    assert passed >= 0.99, name
    assert error.max().item() <= limit, name


@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_commit_multilayer_align(state_dtype):
    if not torch.npu.is_available():
        pytest.skip("requires torch_npu")
    torch.manual_seed(20485)
    device = "npu"
    pages, heads, dim, window = 7, 4, 128, 5
    state_layers = [
        torch.empty(pages, heads, dim, dim, dtype=state_dtype, device=device).normal_(
            0.0, 0.02
        )
        for _ in range(2)
    ]
    replay_layers = [
        torch.empty(
            pages, heads, window, 2 * dim + 1, dtype=torch.float32, device=device
        ).normal_(0.0, 0.03)
        for _ in range(2)
    ]
    for replay in replay_layers:
        replay[..., 2 * dim].abs_().add_(0.5).clamp_max_(1.0)
    table = _make_table(state_layers, replay_layers)
    plan = _make_plan(device)
    replay_before = [replay.clone() for replay in replay_layers]
    expected = [
        gdn_recoverssm_commit_reference(
            state.detach().cpu(),
            replay.detach().cpu(),
            plan["state_indices"].cpu(),
            plan["commit_lens"].cpu(),
            plan["final_state_indices"].cpu(),
            plan["boundary_state_indices"].cpu(),
            plan["boundary_recovery_lens"].cpu(),
            spec_query_len=window,
            align_mode=True,
        )
        for state, replay in zip(state_layers, replay_layers)
    ]

    result = gdn_recoverssm_commit(
        state_layers[0],
        replay_layers[0],
        table,
        plan["state_indices"],
        plan["commit_lens"],
        plan["final_state_indices"],
        plan["boundary_state_indices"],
        plan["boundary_recovery_lens"],
        spec_query_len=window,
        align_mode=True,
    )
    torch.npu.synchronize()
    assert result is None
    for index, (actual, golden) in enumerate(zip(state_layers, expected)):
        _assert_close(actual.cpu(), golden, f"layer{index}")
        assert torch.equal(replay_layers[index], replay_before[index])


def test_commit_non_align_and_empty_batch():
    if not torch.npu.is_available():
        pytest.skip("requires torch_npu")
    device = "npu"
    state = torch.zeros(2, 2, 128, 128, dtype=torch.float32, device=device)
    replay = torch.zeros(2, 2, 3, 257, dtype=torch.float32, device=device)
    table = _make_table([state], [replay])
    empty = torch.empty(0, dtype=torch.int32, device=device)
    result = gdn_recoverssm_commit(
        state,
        replay,
        table,
        empty,
        empty,
        empty,
        empty,
        empty,
        spec_query_len=3,
        align_mode=False,
    )
    assert result is None
    assert torch.count_nonzero(state) == 0


@pytest.mark.parametrize("_hk,hv,key,value,window,state_dtype", PR20485_WINDOW_CASES)
def test_commit_pr20485_window_matrix(
    _hk, hv, key, value, window, state_dtype
):
    if not torch.npu.is_available():
        pytest.skip("requires torch_npu")
    torch.manual_seed(419)
    device = "npu"
    batch = max(3, math.ceil(96 / hv)) if key == value == 128 and hv % 2 == 0 else 3
    pages = 3 * batch + 1
    state = torch.empty(
        pages, hv, value, key, dtype=state_dtype, device=device
    ).normal_(0.0, 0.02)
    replay = torch.empty(
        pages, hv, window, value + key + 1, dtype=torch.float32, device=device
    ).normal_(0.0, 0.03)
    replay[..., value + key].abs_().add_(0.5).clamp_max_(1.0)
    table = _make_table([state], [replay])
    source = torch.arange(1, batch + 1, dtype=torch.int32, device=device)
    final = torch.arange(
        batch + 1, 2 * batch + 1, dtype=torch.int32, device=device
    )
    lengths = torch.full(
        (batch,), window, dtype=torch.int32, device=device
    )
    boundary = torch.zeros(batch, dtype=torch.int32, device=device)
    boundary[1] = 2 * batch + 1
    recovery = torch.zeros(batch, dtype=torch.int32, device=device)
    recovery[1] = max(1, window // 2)

    if key != 128 or value != 128:
        with pytest.raises(RuntimeError, match="shape contract"):
            gdn_recoverssm_commit(
                state,
                replay,
                table,
                source,
                lengths,
                final,
                boundary,
                recovery,
                spec_query_len=window,
                align_mode=True,
            )
        return

    replay_before = replay.clone()
    expected = gdn_recoverssm_commit_reference(
        state.detach().cpu(),
        replay.detach().cpu(),
        source.cpu(),
        lengths.cpu(),
        final.cpu(),
        boundary.cpu(),
        recovery.cpu(),
        spec_query_len=window,
        align_mode=True,
    )
    result = gdn_recoverssm_commit(
        state,
        replay,
        table,
        source,
        lengths,
        final,
        boundary,
        recovery,
        spec_query_len=window,
        align_mode=True,
    )
    torch.npu.synchronize()
    assert result is None
    final_cpu = final.cpu()
    boundary_cpu = boundary.cpu()
    for row in range(batch):
        final_page = int(final_cpu[row])
        _assert_pr20485_close(
            state[final_page], expected[final_page], f"final/{row}"
        )
        boundary_page = int(boundary_cpu[row])
        if boundary_page != 0:
            _assert_pr20485_close(
                state[boundary_page],
                expected[boundary_page],
                f"boundary/{row}",
            )
    assert torch.equal(replay, replay_before)
