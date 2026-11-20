"""Ascend single-op tests for gdn_recoverssm_commit."""

import sys
from pathlib import Path

import pytest
import torch

from fla_npu.ops.ascendc import gdn_recoverssm_commit


def _load_reference():
    root = (
        Path(__file__).resolve().parents[2]
        / "fla/ops/ascendc/gdn/recurrent_gdn/gdn_recoverssm_commit"
    )
    sys.path.insert(0, str(root))
    from reference.reference import gdn_recoverssm_commit_reference

    return gdn_recoverssm_commit_reference


gdn_recoverssm_commit_reference = _load_reference()


def _make_plan(device):
    return {
        "state_indices": torch.tensor([1, 2, 3], dtype=torch.int32, device=device),
        "commit_lens": torch.tensor([5, 3, 0], dtype=torch.int32, device=device),
        "final_state_indices": torch.tensor([4, 2, 5], dtype=torch.int32, device=device),
        "boundary_state_indices": torch.tensor([0, 6, 0], dtype=torch.int32, device=device),
        "boundary_recovery_lens": torch.tensor([0, 2, 0], dtype=torch.int32, device=device),
    }


def _make_table(state_layers, replay_layers):
    rows = []
    for state, replay in zip(state_layers, replay_layers):
        rows.append(
            [
                [state.data_ptr(), state.stride(0)],
                [replay.data_ptr(), replay.stride(0)],
            ]
        )
    return torch.tensor(
        rows, dtype=torch.int64, device=state_layers[0].device
    ).permute(1, 2, 0).contiguous()


def _assert_close(actual, expected, name):
    error = (actual.double() - expected.double()).abs()
    tolerance = 1.0 / 64.0 + (1.0 / 1024.0) * expected.double().abs()
    assert error.le(tolerance).double().mean().item() >= 0.999, name
    assert error.max().item() <= 1.0, name


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
    before = [state.clone() for state in state_layers]
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
        unchanged = plan["commit_lens"] == 0
        assert torch.equal(
            actual[plan["final_state_indices"][unchanged]],
            before[index][plan["final_state_indices"][unchanged]],
        )
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
