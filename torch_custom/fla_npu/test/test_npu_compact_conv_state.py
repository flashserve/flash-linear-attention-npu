"""Ascend single-op tests for compact_conv_state."""

import sys
from pathlib import Path

import pytest
import torch

from fla_npu.ops.ascendc import compact_conv_state


def _load_reference():
    root = (
        Path(__file__).resolve().parents[3]
        / "fla/ops/ascendc/gdn/recurrent_gdn/compact_conv_state"
    )
    sys.path.insert(0, str(root))
    from reference.reference import compact_conv_state_reference

    return compact_conv_state_reference


compact_conv_state_reference = _load_reference()


def _logical_view(backing, channels, span, channel_first):
    physical = (channels, span) if channel_first else (span, channels)
    view = backing.as_strided(
        (backing.shape[0], *physical),
        (backing.stride(0), physical[1], 1),
    )
    return view if channel_first else view.transpose(-1, -2)


def _plan(device, dtype, align):
    source = torch.tensor([1, 2, 0, 3], dtype=dtype, device=device)
    lengths = torch.tensor([4, 1, 3, 0], dtype=dtype, device=device)
    finals = torch.tensor([3, 2, 5, 4], dtype=dtype, device=device)
    boundary = torch.tensor([0, 0, 0, 0], dtype=dtype, device=device)
    recovery = torch.tensor([0, 0, 0, 0], dtype=dtype, device=device)
    if align:
        boundary[0] = 4
        recovery[0] = 2
    return source, lengths, finals, boundary, recovery


@pytest.mark.parametrize(
    "channels,channel_first,align,state_dtype,dim_stride_mode",
    [
        (127, False, False, torch.bfloat16, "shared"),
        (2048, False, False, torch.bfloat16, None),
        (2560, True, True, torch.bfloat16, "shared"),
        (5120, False, True, torch.bfloat16, "shared"),
        (128, False, True, torch.float32, "shared"),
    ],
)
def test_compact_conv_state_page_routing(
    channels, channel_first, align, state_dtype, dim_stride_mode
):
    if not torch.npu.is_available():
        pytest.skip("requires torch_npu")
    torch.manual_seed(419)
    device = torch.device("npu")
    pages, span, layers = 7, 7, 2

    actual_backings = []
    golden_backings = []
    actual_views = []
    golden_views = []
    for layer in range(layers):
        row_elements = channels * span + 128 + 128 * layer
        backing = torch.empty(
            pages, row_elements, dtype=state_dtype
        ).uniform_(-3.0, 3.0)
        actual_backings.append(backing.npu())
        golden_backings.append(backing.clone())
        actual_views.append(
            _logical_view(actual_backings[-1], channels, span, channel_first)
        )
        golden_views.append(
            _logical_view(golden_backings[-1], channels, span, channel_first)
        )

    source, lengths, finals, boundary, recovery = _plan(
        device, torch.int32, align
    )
    addresses = torch.tensor(
        [view.data_ptr() for view in actual_views],
        dtype=torch.int64,
        device=device,
    )
    block_strides, dim_strides, token_strides = (
        torch.tensor(
            [view.stride(axis) for view in actual_views],
            dtype=torch.int64,
            device=device,
        )
        for axis in range(3)
    )
    dim_stride = (
        actual_views[0].stride(1)
        if dim_stride_mode == "shared"
        else None
    )

    result = compact_conv_state(
        actual_views[0],
        addresses,
        block_strides,
        dim_strides,
        token_strides,
        source,
        lengths,
        finals,
        boundary,
        recovery,
        conv_history_len=3,
        align_mode=align,
        dim_stride=dim_stride,
        null_block_id=0,
    )
    compact_conv_state_reference(
        golden_views,
        source.cpu(),
        lengths.cpu(),
        finals.cpu(),
        boundary.cpu(),
        recovery.cpu(),
        conv_history_len=3,
        align_mode=align,
        null_block_id=0,
    )

    assert result is None
    for actual, golden in zip(actual_backings, golden_backings):
        assert torch.equal(actual.cpu(), golden)


def test_compact_conv_state_int64_metadata():
    if not torch.npu.is_available():
        pytest.skip("requires torch_npu")
    torch.manual_seed(20485)
    device = torch.device("npu")
    channels, pages, span = 257, 7, 7
    backing = torch.empty(
        pages, channels * span + 256, dtype=torch.bfloat16
    ).normal_()
    actual = backing.npu()
    golden = backing.clone()
    actual_view = _logical_view(actual, channels, span, True)
    golden_view = _logical_view(golden, channels, span, True)
    source, lengths, finals, boundary, recovery = _plan(
        device, torch.int64, True
    )
    source = torch.tensor(
        [int(source[0]), 99], dtype=torch.int64, device=device
    )[::2]
    stride = lambda axis: torch.tensor(
        [actual_view.stride(axis)], dtype=torch.int64, device=device
    )
    compact_conv_state(
        actual_view,
        torch.tensor(
            [actual_view.data_ptr()], dtype=torch.int64, device=device
        ),
        stride(0),
        stride(1),
        stride(2),
        source[:1],
        lengths[:1],
        finals[:1],
        boundary[:1],
        recovery[:1],
        conv_history_len=3,
        align_mode=True,
        dim_stride=None,
        null_block_id=0,
    )
    compact_conv_state_reference(
        [golden_view],
        source[:1].cpu(),
        lengths[:1].cpu(),
        finals[:1].cpu(),
        boundary[:1].cpu(),
        recovery[:1].cpu(),
        conv_history_len=3,
        align_mode=True,
        null_block_id=0,
    )
    assert torch.equal(actual.cpu(), golden)
