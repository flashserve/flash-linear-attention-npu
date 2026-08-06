import math
import sys
import types
from unittest import mock

from fla_npu.ops.ascendc import _aclnn_ctypes as ascendc_ctypes


_BFLOAT16 = object()
_FLOAT16 = object()
_FLOAT32 = object()
_FAKE_TORCH = types.SimpleNamespace(
    bfloat16=_BFLOAT16,
    float16=_FLOAT16,
    float32=_FLOAT32,
)
_ELEMENT_BYTES = {
    _BFLOAT16: 2,
    _FLOAT16: 2,
    _FLOAT32: 4,
}


class _FakeTensor:
    def __init__(
        self,
        shape,
        dtype,
        *,
        device="npu:0",
        strides=None,
        storage_offset=0,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._storage_offset = int(storage_offset)
        if strides is None:
            strides = []
            stride = 1
            for dim in reversed(self.shape):
                strides.append(stride)
                stride *= dim
            strides.reverse()
        self._strides = tuple(strides)

    def numel(self):
        return math.prod(self.shape)

    def element_size(self):
        return _ELEMENT_BYTES[self.dtype]

    def is_contiguous(self):
        return True

    def storage_offset(self):
        return self._storage_offset

    def narrow(self, dim, start, length):
        shape = list(self.shape)
        shape[dim] = int(length)
        return _FakeTensor(
            shape,
            self.dtype,
            device=self.device,
            strides=self._strides,
            storage_offset=self._storage_offset + int(start) * self._strides[dim],
        )


def _make_inputs(
    batch=1,
    seqlen=129,
    heads=2,
    head_dim=128,
    beta_dtype=_BFLOAT16,
):
    vector_shape = (batch, seqlen, heads, head_dim)
    scalar_shape = (batch, seqlen, heads)
    matrix_shape = (batch, seqlen, heads, 64)
    return (
        _FakeTensor(vector_shape, _BFLOAT16),
        _FakeTensor(vector_shape, _BFLOAT16),
        _FakeTensor(vector_shape, _FLOAT32),
        _FakeTensor(scalar_shape, beta_dtype),
        _FakeTensor(matrix_shape, _FLOAT32),
        _FakeTensor(matrix_shape, _FLOAT32),
        _FakeTensor(vector_shape, _FLOAT32),
        _FakeTensor(vector_shape, _FLOAT32),
        _FakeTensor(scalar_shape, _FLOAT32),
        _FakeTensor(vector_shape, _FLOAT32),
    )


def _make_bnsd_inputs(
    batch=1,
    seqlen=129,
    heads=2,
    head_dim=128,
    beta_dtype=_BFLOAT16,
):
    vector_shape = (batch, heads, seqlen, head_dim)
    scalar_shape = (batch, heads, seqlen)
    matrix_shape = (batch, heads, seqlen, 64)
    return (
        _FakeTensor(vector_shape, _BFLOAT16),
        _FakeTensor(vector_shape, _BFLOAT16),
        _FakeTensor(vector_shape, _FLOAT32),
        _FakeTensor(scalar_shape, beta_dtype),
        _FakeTensor(matrix_shape, _FLOAT32),
        _FakeTensor(matrix_shape, _FLOAT32),
        _FakeTensor(vector_shape, _FLOAT32),
        _FakeTensor(vector_shape, _FLOAT32),
        _FakeTensor(scalar_shape, _FLOAT32),
        _FakeTensor(vector_shape, _FLOAT32),
    )


class _FakeCallContext:
    def __init__(self):
        self.tensors = {}

    def tensor(self, tensor, name, **kwargs):
        self.tensors[name] = (tensor, kwargs)
        return tensor

    def int_array(self, values):
        return values


def _capture_aclnn_calls():
    calls = []

    def fake_call(name, build_args, outputs, **kwargs):
        ctx = _FakeCallContext()
        args = build_args(ctx)
        calls.append(
            {
                "name": name,
                "args": args,
                "outputs": outputs,
                "tensors": ctx.tensors,
                "kwargs": kwargs,
            }
        )
        return outputs

    return calls, fake_call


def _empty_like(tensor):
    return _FakeTensor(tensor.shape, tensor.dtype, device=tensor.device)


def test_bsnd_workspace_segment_length_is_chunk_aligned():
    for beta_dtype in (_BFLOAT16, _FLOAT32):
        inputs = _make_inputs(
            seqlen=18432,
            heads=96,
            beta_dtype=beta_dtype,
        )
        segment_tokens = ascendc_ctypes._chunk_kda_bwd_intra_bsnd_segment_tokens(
            inputs,
            18432,
            64,
        )

        assert segment_tokens == 3392
        assert segment_tokens % 64 == 0


def test_small_bsnd_workspace_keeps_single_launch():
    inputs = _make_inputs(seqlen=8192, heads=32)
    segment_tokens = ascendc_ctypes._chunk_kda_bwd_intra_bsnd_segment_tokens(
        inputs,
        8192,
        64,
    )

    assert segment_tokens == 8192


def test_fp32_beta_is_accepted_without_casting():
    inputs = _make_inputs(beta_dtype=_FLOAT32)
    calls, fake_call = _capture_aclnn_calls()

    with mock.patch.dict(sys.modules, {"torch": _FAKE_TORCH}):
        with mock.patch.object(ascendc_ctypes, "_empty_like", _empty_like):
            with mock.patch.object(ascendc_ctypes, "_call_aclnn", fake_call):
                ascendc_ctypes.npu_chunk_kda_bwd_intra(
                    *inputs,
                    chunk_size=64,
                    safe_gate=True,
                    layout="BSND",
                )

    assert len(calls) == 1
    assert calls[0]["tensors"]["beta"][0] is inputs[3]
    assert calls[0]["tensors"]["beta"][0].dtype is _FLOAT32


def test_fp16_beta_is_rejected():
    inputs = _make_inputs(beta_dtype=_FLOAT16)

    with mock.patch.dict(sys.modules, {"torch": _FAKE_TORCH}):
        with mock.patch.object(ascendc_ctypes, "_empty_like", _empty_like):
            try:
                ascendc_ctypes.npu_chunk_kda_bwd_intra(
                    *inputs,
                    chunk_size=64,
                    safe_gate=True,
                    layout="BSND",
                )
            except RuntimeError as error:
                assert "beta must be torch.bfloat16 or torch.float32" in str(error)
            else:
                raise AssertionError("FP16 beta must be rejected")


def test_long_bsnd_launches_chunk_aligned_views_into_full_outputs():
    inputs = _make_inputs()
    calls, fake_call = _capture_aclnn_calls()
    one_chunk_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in inputs
    ) * 64 // 129

    with mock.patch.dict(sys.modules, {"torch": _FAKE_TORCH}):
        with mock.patch.object(ascendc_ctypes, "_empty_like", _empty_like):
            with mock.patch.object(
                ascendc_ctypes,
                "_KDA_BSND_TRANSPOSE_WORKSPACE_BUDGET_BYTES",
                one_chunk_bytes,
            ), mock.patch.object(ascendc_ctypes, "_call_aclnn", fake_call):
                outputs = ascendc_ctypes.npu_chunk_kda_bwd_intra(
                    *inputs,
                    chunk_size=64,
                    safe_gate=True,
                    layout="BSND",
                )

    assert len(calls) == 3
    assert [call["tensors"]["q"][0].shape[1] for call in calls] == [64, 64, 1]
    assert [call["tensors"]["q"][0].storage_offset() for call in calls] == [
        0,
        64 * 2 * 128,
        128 * 2 * 128,
    ]
    assert [call["tensors"]["beta"][0].storage_offset() for call in calls] == [
        0,
        64 * 2,
        128 * 2,
    ]
    assert [
        call["tensors"]["dAqk"][0].storage_offset() for call in calls
    ] == [
        0,
        64 * 2 * 64,
        128 * 2 * 64,
    ]
    assert [
        call["tensors"]["dq_out"][0].storage_offset() for call in calls
    ] == [
        0,
        64 * 2 * 128,
        128 * 2 * 128,
    ]
    assert all(
        "use_allocator_stream_lifetime" not in call["kwargs"] for call in calls
    )
    assert all(
        tuple(metadata["storage_shape_override"]) == tuple(tensor.shape)
        for call in calls
        for tensor, metadata in call["tensors"].values()
    )
    assert [tuple(output.shape) for output in outputs] == [
        (1, 129, 2, 128),
        (1, 129, 2, 128),
        (1, 129, 2),
        (1, 129, 2, 128),
    ]


def test_multi_batch_bsnd_keeps_single_launch():
    inputs = _make_inputs(batch=2)
    calls, fake_call = _capture_aclnn_calls()

    with mock.patch.dict(sys.modules, {"torch": _FAKE_TORCH}):
        with mock.patch.object(ascendc_ctypes, "_empty_like", _empty_like):
            with mock.patch.object(
                ascendc_ctypes,
                "_KDA_BSND_TRANSPOSE_WORKSPACE_BUDGET_BYTES",
                1,
            ), mock.patch.object(ascendc_ctypes, "_call_aclnn", fake_call):
                ascendc_ctypes.npu_chunk_kda_bwd_intra(
                    *inputs,
                    chunk_size=64,
                    safe_gate=True,
                    layout="BSND",
                )

    assert len(calls) == 1
    assert calls[0]["tensors"]["q"][0] is inputs[0]


def test_native_bnsd_keeps_single_launch():
    inputs = _make_bnsd_inputs()
    calls, fake_call = _capture_aclnn_calls()

    with mock.patch.dict(sys.modules, {"torch": _FAKE_TORCH}):
        with mock.patch.object(ascendc_ctypes, "_empty_like", _empty_like):
            with mock.patch.object(
                ascendc_ctypes,
                "_KDA_BSND_TRANSPOSE_WORKSPACE_BUDGET_BYTES",
                1,
            ), mock.patch.object(ascendc_ctypes, "_call_aclnn", fake_call):
                ascendc_ctypes.npu_chunk_kda_bwd_intra(
                    *inputs,
                    chunk_size=64,
                    safe_gate=True,
                    layout="BNSD",
                )

    assert len(calls) == 1
    assert calls[0]["tensors"]["q"][0] is inputs[0]


def test_varlen_bsnd_keeps_single_launch():
    inputs = _make_inputs()
    calls, fake_call = _capture_aclnn_calls()

    with mock.patch.dict(sys.modules, {"torch": _FAKE_TORCH}):
        with mock.patch.object(ascendc_ctypes, "_empty_like", _empty_like):
            with mock.patch.object(
                ascendc_ctypes,
                "_KDA_BSND_TRANSPOSE_WORKSPACE_BUDGET_BYTES",
                1,
            ), mock.patch.object(ascendc_ctypes, "_call_aclnn", fake_call):
                ascendc_ctypes.npu_chunk_kda_bwd_intra(
                    *inputs,
                    cu_seqlens=[0, 64, 129],
                    chunk_size=64,
                    safe_gate=True,
                    layout="BSND",
                )

    assert len(calls) == 1
    assert calls[0]["tensors"]["q"][0] is inputs[0]
