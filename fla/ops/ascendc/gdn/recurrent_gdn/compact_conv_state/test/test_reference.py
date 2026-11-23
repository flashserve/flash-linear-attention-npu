"""CPU tests for compact_conv_state page routing and copy semantics."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reference.reference import compact_conv_state_reference


def _plan(dtype, align):
    source = torch.tensor([1, 2, 0, 5], dtype=dtype)
    lengths = torch.tensor([4, 1, 3, 0], dtype=dtype)
    finals = torch.tensor([3, 2, 6, 5], dtype=dtype)
    boundary = torch.tensor([4, 0, 0, 0], dtype=dtype)
    recovery = torch.tensor([2, 0, 0, 0], dtype=dtype)
    if not align:
        boundary.zero_()
        recovery.zero_()
    return source, lengths, finals, boundary, recovery


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("align", [False, True])
def test_compact_conv_state_exact_copy(dtype, align):
    torch.manual_seed(419)
    actual_views = []
    golden_views = []
    actual_backings = []
    golden_backings = []
    for layer in range(2):
        storage = torch.empty(
            7, 127 * 7 + 64 + layer * 32
        ).uniform_(-3.0, 3.0)
        golden = storage.clone()
        actual_views.append(
            storage.as_strided((7, 127, 7), (storage.stride(0), 7, 1))
        )
        golden_views.append(
            golden.as_strided((7, 127, 7), (golden.stride(0), 7, 1))
        )
        actual_backings.append(storage)
        golden_backings.append(golden)

    compact_conv_state_reference(
        actual_views,
        *_plan(dtype, align),
        conv_history_len=3,
        align_mode=align,
    )
    compact_conv_state_reference(
        golden_views,
        *_plan(dtype, align),
        conv_history_len=3,
        align_mode=align,
    )
    for actual, golden in zip(actual_backings, golden_backings):
        torch.testing.assert_close(actual, golden, atol=0, rtol=0)


def test_source_and_boundary_are_loaded_before_writes():
    storage = torch.arange(7 * 5 * 7, dtype=torch.float32).reshape(
        7, 5, 7
    )
    view = storage.clone()
    before = view.clone()
    compact_conv_state_reference(
        [view],
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        conv_history_len=3,
        align_mode=True,
    )
    torch.testing.assert_close(
        view[1, :, :3], before[1, :, 1:4], atol=0, rtol=0
    )
    torch.testing.assert_close(
        view[3, :, :3], before[1, :, 3:6], atol=0, rtol=0
    )
