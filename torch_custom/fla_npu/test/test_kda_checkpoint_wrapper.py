"""CPU-only guards and backward-compatible Stable dispatcher selection."""

import importlib.util
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def backend(monkeypatch):
    path = Path(__file__).parents[1] / "fla_npu/ops/ascendc/_stable.py"
    spec = importlib.util.spec_from_file_location(
        "kda_checkpoint_stable_under_test", path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    run = Mock(return_value=tuple(range(11)))
    dispatch = Mock(return_value=run)
    monkeypatch.setattr(mod, "_op", dispatch)
    monkeypatch.setattr(mod, "_current_stream_ptr", lambda: 17)
    return mod, dispatch, run


def inputs():
    xs = [torch.zeros(1, 65, 1, 128, dtype=torch.bfloat16) for _ in range(4)]
    xs.append(torch.ones(1, 65, 1))
    return xs, torch.zeros(1, 1, 128, 128), torch.tensor([64], dtype=torch.int32)


def call(mod, xs, state, **kwargs):
    return mod.npu_chunk_kda_fwd(
        *xs,
        128**-0.5,
        cu_seqlens=[0, 65],
        initial_state=state,
        state_v_first=True,
        **kwargs,
    )


def test_default_call_keeps_legacy_schema_and_return_slots(backend):
    mod, dispatch, run = backend
    xs, state, offsets = inputs()
    result = call(mod, xs, state)
    dispatch.assert_called_once_with("npu_chunk_kda_fwd")
    original_count = len(run.call_args.args)
    assert run.call_args.args[-1] == 17
    assert len(result) == 12 and result[-1] is state
    cp = torch.empty_like(state)
    result = call(mod, xs, state, checkpoint_offsets=offsets, checkpoint_state=cp)
    dispatch.assert_called_with("npu_chunk_kda_fwd_checkpoint")
    assert len(run.call_args.args) == original_count + 2
    assert run.call_args.args[-3] is offsets
    assert run.call_args.args[-2] is cp
    assert run.call_args.args[-1] == 17
    assert len(result) == 12 and result[-1] is state


@pytest.mark.parametrize(
    "kind",
    [
        "missing_offsets",
        "missing_state",
        "offset_dtype",
        "state_dtype",
        "non_contiguous",
        "alias",
        "offset_alias",
    ],
)
def test_invalid_buffers_never_dispatch(backend, kind):
    mod, dispatch, _ = backend
    xs, state, offsets = inputs()
    cp = torch.empty_like(state)
    if kind == "missing_offsets":
        offsets = None
    if kind == "missing_state":
        cp = None
    if kind == "offset_dtype":
        offsets = offsets.long()
    if kind == "state_dtype":
        cp = cp.bfloat16()
    if kind == "non_contiguous":
        cp = cp.transpose(-1, -2)
    if kind == "alias":
        cp = state
    if kind == "offset_alias":
        offsets = cp.view(torch.int32).flatten()[:1]
    with pytest.raises(ValueError):
        call(mod, xs, state, checkpoint_offsets=offsets, checkpoint_state=cp)
    dispatch.assert_not_called()
