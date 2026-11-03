"""Stable KDA intra wrapper forwarding; NPU tests own dtype/shape validation.

The old ctypes workspace segmentation no longer exists: the Stable adapter
passes BSND/BNSD/TND directly to the kernel. Keep input identity, metadata,
stream, layout and reserved-flag contracts under direct test.
"""
from __future__ import annotations
import importlib.util
from pathlib import Path
from unittest import mock
import pytest

path = Path(__file__).resolve().parents[1] / "fla_npu/ops/ascendc/_stable.py"
spec = importlib.util.spec_from_file_location("intra_stable", path)
STABLE = importlib.util.module_from_spec(spec)
spec.loader.exec_module(STABLE)


@pytest.mark.parametrize("layout,code", [("BSND", 0), ("BNSD", 1), ("TND", 2)])
@pytest.mark.parametrize("metadata", [None, [0, 64, 129]])
def test_layout_metadata_identity_and_single_dispatch(layout, code, metadata):
    tensors = tuple(object() for _ in range(10))
    outputs = tuple(object() for _ in range(4))
    launch = mock.Mock(return_value=outputs)
    indices = [0, 0, 1, 0, 1, 1] if metadata else None
    with mock.patch.object(STABLE, "_op", return_value=launch), \
            mock.patch.object(STABLE, "_host_ints", side_effect=lambda x: x), \
            mock.patch.object(STABLE, "_current_stream_ptr", return_value=123):
        actual = STABLE.npu_chunk_kda_bwd_intra(
            *tensors, cu_seqlens=metadata, chunk_indices=indices, layout=layout)
    launch.assert_called_once_with(*tensors, metadata, indices, 64, True, code, 123)
    assert actual is outputs
    assert launch.call_args.args[3] is tensors[3]  # beta must never be cast in Python.


def test_reserved_safe_gate_never_launches():
    with mock.patch.object(STABLE, "_op") as launch:
        with pytest.raises(RuntimeError, match="safe_gate=False"):
            STABLE.npu_chunk_kda_bwd_intra(*(object() for _ in range(10)), safe_gate=False)
        launch.assert_not_called()


def test_unsupported_layout_never_executes():
    launch = mock.Mock()
    with mock.patch.object(STABLE, "_op", return_value=launch), \
            mock.patch.object(STABLE, "_host_ints", side_effect=lambda x: x):
        with pytest.raises((ValueError, RuntimeError), match="layout"):
            STABLE.npu_chunk_kda_bwd_intra(*(object() for _ in range(10)), layout="NTD")
        launch.assert_not_called()
