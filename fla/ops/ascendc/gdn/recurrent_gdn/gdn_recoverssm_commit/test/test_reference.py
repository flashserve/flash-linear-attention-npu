"""CPU tests for the gdn_recoverssm_commit golden."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.reference import gdn_recoverssm_commit_reference


def _make_case(
    *,
    state_dtype: torch.dtype = torch.float32,
    seed: int = 20485,
):
    generator = torch.Generator().manual_seed(seed)
    pages, heads, dim, window = 7, 4, 128, 5
    state = torch.empty(
        pages, heads, dim, dim, dtype=state_dtype
    ).normal_(0.0, 0.02, generator=generator)
    replay = torch.empty(
        pages, heads, window, 2 * dim + 1, dtype=torch.float32
    ).normal_(0.0, 0.03, generator=generator)
    replay[..., 2 * dim].abs_().add_(0.5).clamp_max_(1.0)
    return {
        "state": state,
        "replay": replay,
        "state_indices": torch.tensor([1, 2, 3], dtype=torch.int32),
        "commit_lens": torch.tensor([5, 3, 0], dtype=torch.int32),
        "final_state_indices": torch.tensor([4, 2, 5], dtype=torch.int32),
        "boundary_state_indices": torch.tensor([0, 6, 0], dtype=torch.int32),
        "boundary_recovery_lens": torch.tensor([0, 2, 0], dtype=torch.int32),
        "spec_query_len": window,
        "align_mode": True,
    }


def _forward_records(state, replay, source_page, commit_len):
    current = state[source_page].double()
    for position in range(commit_len):
        record = replay[source_page, :, position].double()
        correction = record[..., :128]
        normalized_key = record[..., 128:256]
        decay = record[..., 256]
        current = (
            current * decay[:, None, None]
            + correction[..., None] * normalized_key[:, None, :]
        )
    return current


def test_reverse_composition_matches_forward_records() -> None:
    kwargs = _make_case(seed=7)
    committed = gdn_recoverssm_commit_reference(**kwargs)
    for row, (source_page, length) in enumerate(
        zip(kwargs["state_indices"].tolist(), kwargs["commit_lens"].tolist())
    ):
        if length == 0:
            continue
        expected = _forward_records(
            kwargs["state"], kwargs["replay"], source_page, length
        )
        torch.testing.assert_close(
            committed[kwargs["final_state_indices"][row]].double(),
            expected,
        )


def test_align_boundary_uses_accepted_prefix() -> None:
    kwargs = _make_case(seed=11)
    committed = gdn_recoverssm_commit_reference(**kwargs)
    boundary_page = int(kwargs["boundary_state_indices"][1])
    expected = _forward_records(
        kwargs["state"], kwargs["replay"], 2, int(kwargs["boundary_recovery_lens"][1])
    )
    torch.testing.assert_close(committed[boundary_page].double(), expected)
    assert not torch.equal(
        committed[boundary_page],
        committed[kwargs["final_state_indices"][1]],
    )


def test_zero_length_and_null_pages_are_unchanged() -> None:
    kwargs = _make_case(seed=19)
    before = kwargs["state"].clone()
    committed = gdn_recoverssm_commit_reference(**kwargs)
    torch.testing.assert_close(committed[5], before[5].double())
    torch.testing.assert_close(committed[0], before[0].double())


def test_same_final_and_boundary_page_writes_once() -> None:
    kwargs = _make_case(seed=23)
    kwargs["final_state_indices"][1] = 6
    kwargs["boundary_recovery_lens"][1] = kwargs["commit_lens"][1]
    committed = gdn_recoverssm_commit_reference(**kwargs)
    expected = _forward_records(kwargs["state"], kwargs["replay"], 2, 3)
    torch.testing.assert_close(committed[6].double(), expected)


def test_bf16_state_converts_only_on_store() -> None:
    kwargs = _make_case(state_dtype=torch.bfloat16, seed=29)
    committed = gdn_recoverssm_commit_reference(**kwargs)
    assert committed.dtype == torch.float64
    assert torch.isfinite(committed).all()


def test_reference_is_deterministic() -> None:
    kwargs = _make_case(seed=31)
    first = gdn_recoverssm_commit_reference(**kwargs)
    second = gdn_recoverssm_commit_reference(**kwargs)
    torch.testing.assert_close(first, second, atol=0, rtol=0)


if __name__ == "__main__":
    tests = (
        test_reverse_composition_matches_forward_records,
        test_align_boundary_uses_accepted_prefix,
        test_zero_length_and_null_pages_are_unchanged,
        test_same_final_and_boundary_page_writes_once,
        test_bf16_state_converts_only_on_store,
        test_reference_is_deterministic,
    )
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
